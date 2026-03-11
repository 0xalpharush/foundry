use crate::{
    executors::{
        DURATION_BETWEEN_METRICS_REPORT, EarlyExit, EvmError, Executor, FuzzTestTimer,
        RawCallResult,
        corpus::{GlobalCorpusMetrics, SyncStats, WorkerCorpus},
    },
    inspectors::Fuzzer,
};
use alloy_primitives::{
    Address, Bytes, FixedBytes, I256, Selector, U256, keccak256,
    map::{AddressMap, HashMap},
};
use alloy_sol_types::{SolCall, sol};
use eyre::{ContextCompat, Result, eyre};
use foundry_common::{
    TestFunctionExt,
    contracts::{ContractsByAddress, ContractsByArtifact},
    sh_println,
};
use foundry_config::InvariantConfig;
use foundry_evm_core::{
    constants::{
        CALLER, CHEATCODE_ADDRESS, DEFAULT_CREATE2_DEPLOYER, HARDHAT_CONSOLE_ADDRESS, MAGIC_ASSUME,
    },
    precompiles::PRECOMPILES,
};
use foundry_evm_fuzz::{
    BasicTxDetails, FuzzCase, FuzzFixtures, FuzzedCases,
    invariant::{
        ArtifactFilters, FuzzRunIdentifiedContracts, InvariantContract, RandomCallGenerator,
        SenderFilters, TargetedContract, TargetedContracts,
    },
    strategies::{EvmFuzzState, invariant_strat, override_call_strat},
};
use foundry_evm_traces::{CallTraceArena, SparsedTraceArena};
use indicatif::ProgressBar;
use parking_lot::{Mutex, RwLock};
use proptest::{
    strategy::Strategy,
    test_runner::{RngAlgorithm, TestRng, TestRunner},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use result::{assert_after_invariant, process_call_result};
use revm::state::Account;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{HashMap as Map, btree_map::Entry},
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Corpus syncs across workers every `SYNC_INTERVAL`.
const SYNC_INTERVAL: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// Minimum number of runs per worker to justify spawning.
const MIN_RUNS_PER_WORKER: u32 = 1000;

mod error;
pub use error::{FailureKey, InvariantFailures, InvariantFuzzError};
use foundry_evm_coverage::HitMaps;

mod replay;
pub use replay::{generate_counterexample, replay_error, replay_run};

mod result;
pub use result::InvariantFuzzTestResult;

mod shrink;
use crate::executors::invariant::result::invariant_preflight_check;
pub use shrink::{FailureTarget, check_sequence, check_sequence_value};

sol! {
    interface IInvariantTest {
        #[derive(Default)]
        struct FuzzSelector {
            address addr;
            bytes4[] selectors;
        }

        #[derive(Default)]
        struct FuzzArtifactSelector {
            string artifact;
            bytes4[] selectors;
        }

        #[derive(Default)]
        struct FuzzInterface {
            address addr;
            string[] artifacts;
        }

        function afterInvariant() external;

        #[derive(Default)]
        function excludeArtifacts() public view returns (string[] memory excludedArtifacts);

        #[derive(Default)]
        function excludeContracts() public view returns (address[] memory excludedContracts);

        #[derive(Default)]
        function excludeSelectors() public view returns (FuzzSelector[] memory excludedSelectors);

        #[derive(Default)]
        function excludeSenders() public view returns (address[] memory excludedSenders);

        #[derive(Default)]
        function targetArtifacts() public view returns (string[] memory targetedArtifacts);

        #[derive(Default)]
        function targetArtifactSelectors() public view returns (FuzzArtifactSelector[] memory targetedArtifactSelectors);

        #[derive(Default)]
        function targetContracts() public view returns (address[] memory targetedContracts);

        #[derive(Default)]
        function targetSelectors() public view returns (FuzzSelector[] memory targetedSelectors);

        #[derive(Default)]
        function targetSenders() public view returns (address[] memory targetedSenders);

        #[derive(Default)]
        function targetInterfaces() public view returns (FuzzInterface[] memory targetedInterfaces);
    }
}

/// Contains invariant metrics for a single fuzzed selector.
#[derive(Default, Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct InvariantMetrics {
    // Count of fuzzed selector calls.
    pub calls: usize,
    // Count of fuzzed selector reverts.
    pub reverts: usize,
    // Count of fuzzed selector discards (through assume cheatcodes).
    pub discards: usize,
}

/// Contains data collected during invariant test runs.
struct InvariantTestData {
    // Consumed gas and calldata of every successful fuzz call.
    fuzz_cases: Vec<FuzzedCases>,
    // Data related to reverts or failed assertions of the test.
    failures: InvariantFailures,
    // Calldata in the last invariant run.
    last_run_inputs: Vec<BasicTxDetails>,
    // Additional traces for gas report.
    gas_report_traces: Vec<Vec<CallTraceArena>>,
    // Line coverage information collected from all fuzzed calls.
    line_coverage: Option<HitMaps>,
    // Metrics for each fuzzed selector.
    metrics: Map<String, InvariantMetrics>,

    // Proptest runner to query for random values.
    // The strategy only comes with the first `input`. We fill the rest of the `inputs`
    // until the desired `depth` so we can use the evolving fuzz dictionary
    // during the run.
    branch_runner: TestRunner,

    // Optimization mode state: tracks the best (maximum) value and the sequence that produced it.
    // Only used when invariant function returns int256.
    optimization_best_value: Option<I256>,
    optimization_best_sequence: Vec<BasicTxDetails>,
}

/// Contains invariant test data.
struct InvariantTest {
    // Fuzz state of invariant test.
    fuzz_state: EvmFuzzState,
    // Contracts fuzzed by the invariant test.
    targeted_contracts: FuzzRunIdentifiedContracts,
    // Sender filters (targeted/excluded senders).
    sender_filters: SenderFilters,
    // Data collected during invariant runs.
    test_data: InvariantTestData,
}

impl InvariantTest {
    /// Instantiates an invariant test.
    fn new(
        fuzz_state: EvmFuzzState,
        targeted_contracts: FuzzRunIdentifiedContracts,
        sender_filters: SenderFilters,
        failures: InvariantFailures,
        branch_runner: TestRunner,
    ) -> Self {
        let test_data = InvariantTestData {
            fuzz_cases: vec![],
            failures,
            last_run_inputs: vec![],
            gas_report_traces: vec![],
            line_coverage: None,
            metrics: Map::default(),
            branch_runner,
            optimization_best_value: None,
            optimization_best_sequence: vec![],
        };
        Self { fuzz_state, targeted_contracts, sender_filters, test_data }
    }

    /// Returns number of invariant test reverts.
    fn reverts(&self) -> usize {
        self.test_data.failures.reverts
    }

    /// Set invariant test error.
    fn set_error(&mut self, key: FailureKey, error: InvariantFuzzError) {
        self.test_data.failures.record_failure(key, error);
    }

    /// Set last invariant run call sequence.
    fn set_last_run_inputs(&mut self, inputs: &Vec<BasicTxDetails>) {
        self.test_data.last_run_inputs.clone_from(inputs);
    }

    /// Merge current collected line coverage with the new coverage from last fuzzed call.
    fn merge_line_coverage(&mut self, new_coverage: Option<HitMaps>) {
        HitMaps::merge_opt(&mut self.test_data.line_coverage, new_coverage);
    }

    /// Update metrics for a fuzzed selector, extracted from tx details.
    /// Always increments number of calls; discarded runs (through assume cheatcodes) are tracked
    /// separated from reverts.
    fn record_metrics(&mut self, tx_details: &BasicTxDetails, reverted: bool, discarded: bool) {
        if let Some(metric_key) =
            self.targeted_contracts.targets.lock().fuzzed_metric_key(tx_details)
        {
            let test_metrics = &mut self.test_data.metrics;
            let invariant_metrics = test_metrics.entry(metric_key).or_default();
            invariant_metrics.calls += 1;
            if discarded {
                invariant_metrics.discards += 1;
            } else if reverted {
                invariant_metrics.reverts += 1;
            }
        }
    }

    /// End invariant test run by collecting results, cleaning collected artifacts and reverting
    /// created fuzz state.
    fn end_run(&mut self, run: InvariantTestRun, gas_samples: usize) {
        // We clear all the targeted contracts created during this run.
        self.targeted_contracts.clear_created_contracts(run.created_contracts);

        if self.test_data.gas_report_traces.len() < gas_samples {
            self.test_data
                .gas_report_traces
                .push(run.run_traces.into_iter().map(|arena| arena.arena).collect());
        }
        self.test_data.fuzz_cases.push(FuzzedCases::new(run.fuzz_runs));

        // Revert state to not persist values between runs.
        self.fuzz_state.revert();
    }

    /// Updates the optimization state if the new value is better (higher) than the current best.
    fn update_optimization_value(&mut self, value: I256, sequence: &[BasicTxDetails]) {
        if self.test_data.optimization_best_value.is_none_or(|best| value > best) {
            self.test_data.optimization_best_value = Some(value);
            self.test_data.optimization_best_sequence = sequence.to_vec();
        }
    }
}

/// Contains data for an invariant test run.
struct InvariantTestRun {
    // Invariant run call sequence.
    inputs: Vec<BasicTxDetails>,
    // Current invariant run executor.
    executor: Executor,
    // Invariant run stat reports (eg. gas usage).
    fuzz_runs: Vec<FuzzCase>,
    // Contracts created during current invariant run.
    created_contracts: Vec<Address>,
    // Traces of each call of the invariant run call sequence.
    run_traces: Vec<SparsedTraceArena>,
    // Current depth of invariant run.
    depth: u32,
    // Current assume rejects of the invariant run.
    rejects: u32,
    // Whether new coverage was discovered during this run.
    new_coverage: bool,
}

impl InvariantTestRun {
    /// Instantiates an invariant test run.
    fn new(first_input: BasicTxDetails, executor: Executor, depth: usize) -> Self {
        Self {
            inputs: vec![first_input],
            executor,
            fuzz_runs: Vec::with_capacity(depth),
            created_contracts: vec![],
            run_traces: vec![],
            depth: 0,
            rejects: 0,
            new_coverage: false,
        }
    }
}

/// Per-worker counters that can be read by worker 0 for aggregation.
struct WorkerCounters {
    calls: AtomicU64,
    gas: AtomicU64,
    failures: AtomicU64,
}

impl WorkerCounters {
    fn new() -> Self {
        Self { calls: AtomicU64::new(0), gas: AtomicU64::new(0), failures: AtomicU64::new(0) }
    }

    fn add_call(&self, gas: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.gas.fetch_add(gas, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }
}

/// Shared state for coordinating parallel invariant workers.
struct SharedInvariantState {
    total_runs: Arc<AtomicU32>,
    /// Per-worker counters; worker 0 sums across all for the global view.
    worker_counters: Vec<Arc<WorkerCounters>>,
    timer: FuzzTestTimer,
    /// Global test suite early exit (ctrl+C).
    early_exit: EarlyExit,
    /// Local early exit (failure triggered).
    local_early_exit: EarlyExit,
    /// Global corpus metrics.
    global_corpus_metrics: GlobalCorpusMetrics,
}

impl SharedInvariantState {
    fn new(timeout: Option<u32>, early_exit: EarlyExit, num_workers: usize) -> Self {
        let worker_counters = (0..num_workers).map(|_| Arc::new(WorkerCounters::new())).collect();
        Self {
            total_runs: Arc::new(AtomicU32::new(0)),
            worker_counters,
            timer: FuzzTestTimer::new(timeout),
            early_exit,
            local_early_exit: EarlyExit::new(true),
            global_corpus_metrics: GlobalCorpusMetrics::default(),
        }
    }

    fn increment_runs(&self) {
        self.total_runs.fetch_add(1, Ordering::Relaxed);
    }

    /// Sum calls/gas/failures across all workers.
    fn global_totals(&self) -> (u64, u64, u64) {
        let mut calls = 0u64;
        let mut gas = 0u64;
        let mut failures = 0u64;
        for wc in &self.worker_counters {
            calls += wc.calls.load(Ordering::Relaxed);
            gas += wc.gas.load(Ordering::Relaxed);
            failures += wc.failures.load(Ordering::Relaxed);
        }
        (calls, gas, failures)
    }

    /// Returns `true` if the worker should continue running.
    fn should_continue(&self) -> bool {
        !(self.early_exit.should_stop()
            || self.local_early_exit.should_stop()
            || self.timer.is_timed_out())
    }
}

/// Per-worker accumulation struct for invariant testing.
struct InvariantWorkerState {
    /// Worker identifier.
    id: usize,
    /// Failures found by this worker.
    failures: InvariantFailures,
    /// Consumed gas and calldata of every successful fuzz call.
    fuzz_cases: Vec<FuzzedCases>,
    /// Calldata in the last invariant run.
    last_run_inputs: Vec<BasicTxDetails>,
    /// Additional traces for gas report.
    gas_report_traces: Vec<Vec<CallTraceArena>>,
    /// Line coverage information.
    line_coverage: Option<HitMaps>,
    /// Metrics for each fuzzed selector.
    metrics: Map<String, InvariantMetrics>,
    /// Number of runs completed.
    runs: u32,
    /// Optimization mode state.
    optimization_best_value: Option<I256>,
    optimization_best_sequence: Vec<BasicTxDetails>,
    /// Failed corpus replays.
    failed_corpus_replays: usize,
}

impl InvariantWorkerState {
    fn new(id: usize) -> Self {
        Self {
            id,
            failures: InvariantFailures::new(),
            fuzz_cases: vec![],
            last_run_inputs: vec![],
            gas_report_traces: vec![],
            line_coverage: None,
            metrics: Map::default(),
            runs: 0,
            optimization_best_value: None,
            optimization_best_sequence: vec![],
            failed_corpus_replays: 0,
        }
    }
}

/// Wrapper around any [`Executor`] implementer which provides fuzzing support using [`proptest`].
///
/// After instantiation, calling `invariant_fuzz` will proceed to hammer the deployed smart
/// contracts with inputs, until it finds a counterexample sequence. The provided [`TestRunner`]
/// contains all the configuration which can be overridden via [environment
/// variables](proptest::test_runner::Config)
pub struct InvariantExecutor<'a> {
    pub executor: Executor,
    /// Proptest runner.
    runner: TestRunner,
    /// The invariant configuration
    config: InvariantConfig,
    /// Contracts deployed with `setUp()`
    setup_contracts: &'a ContractsByAddress,
    /// Contracts that are part of the project but have not been deployed yet. We need the bytecode
    /// to identify them from the stateset changes.
    project_contracts: &'a ContractsByArtifact,
    /// Filters contracts to be fuzzed through their artifact identifiers.
    artifact_filters: ArtifactFilters,
    /// The number of parallel workers.
    num_workers: usize,
}

impl<'a> InvariantExecutor<'a> {
    /// Instantiates a fuzzed executor EVM given a testrunner
    pub fn new(
        executor: Executor,
        runner: TestRunner,
        config: InvariantConfig,
        setup_contracts: &'a ContractsByAddress,
        project_contracts: &'a ContractsByArtifact,
        num_invariant_contracts: usize,
    ) -> Self {
        // Divide the thread pool evenly among concurrent invariant contracts.
        // TODO: consider work-stealing so contracts that finish early can donate workers
        // to contracts still running, rather than leaving threads idle.
        let available_threads =
            Ord::max(1, rayon::current_num_threads() / Ord::max(1, num_invariant_contracts));
        let max_workers =
            if config.runs == 0 { 0 } else { Ord::max(1, config.runs / MIN_RUNS_PER_WORKER) };
        let num_workers = Ord::min(available_threads, max_workers as usize);
        Self {
            executor,
            runner,
            config,
            setup_contracts,
            project_contracts,
            artifact_filters: ArtifactFilters::default(),
            num_workers,
        }
    }

    pub fn config(self) -> InvariantConfig {
        self.config
    }

    /// Determines the number of runs per worker.
    fn runs_per_worker(&self, worker_id: usize) -> u32 {
        let worker_id = worker_id as u32;
        let total_runs = self.config.runs;
        let n = self.num_workers as u32;
        let runs = total_runs / n;
        let remainder = total_runs % n;
        if worker_id < remainder { runs + 1 } else { runs }
    }

    /// Fuzzes any deployed contract and checks any broken invariant at `invariant_address`.
    pub fn invariant_fuzz(
        &mut self,
        invariant_contract: InvariantContract<'_>,
        fuzz_fixtures: &FuzzFixtures,
        fuzz_state: EvmFuzzState,
        progress: Option<&ProgressBar>,
        early_exit: &EarlyExit,
        target_name: &str,
        tokio_handle: &tokio::runtime::Handle,
    ) -> Result<InvariantFuzzTestResult> {
        // Throw an error to abort test run if the invariant function accepts input params
        if !invariant_contract.invariant_fn.inputs.is_empty() {
            return Err(eyre!("Invariant test function should have no inputs"));
        }

        // Phase 1: Single-threaded preparation.
        let (fuzz_state, targeted_senders, targeted_contracts, failures) =
            self.prepare_test(&invariant_contract, fuzz_state)?;

        let shared_state =
            SharedInvariantState::new(self.config.timeout, early_exit.clone(), self.num_workers);

        let _ = sh_println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "start",
                "target": target_name,
                "workers": self.num_workers,
                "total_runs": self.config.runs,
            }))?
        );

        // Phase 2: Parallel worker dispatch.
        // Each worker creates its own strategy (BoxedStrategy is not Send/Sync).
        let workers = (0..self.num_workers)
            .into_par_iter()
            .map(|worker_id| {
                let _guard = tokio_handle.enter();
                let _guard = info_span!("invariant_worker", id = worker_id).entered();
                let timer = Instant::now();
                let r = self.run_invariant_worker(
                    worker_id,
                    &invariant_contract,
                    &fuzz_state,
                    &targeted_senders,
                    &targeted_contracts,
                    &failures,
                    fuzz_fixtures,
                    &shared_state,
                    progress,
                    target_name,
                );
                debug!(worker_id, elapsed = ?timer.elapsed(), "invariant worker finished");
                r
            })
            .collect::<Result<Vec<_>>>()?;

        // Phase 3: Aggregate results.
        Ok(self.aggregate_invariant_results(workers, &fuzz_state))
    }

    /// Runs a single invariant fuzzing worker.
    #[allow(clippy::too_many_arguments)]
    fn run_invariant_worker(
        &self,
        worker_id: usize,
        invariant_contract: &InvariantContract<'_>,
        fuzz_state: &EvmFuzzState,
        targeted_senders: &SenderFilters,
        targeted_contracts: &FuzzRunIdentifiedContracts,
        initial_failures: &InvariantFailures,
        fuzz_fixtures: &FuzzFixtures,
        shared_state: &SharedInvariantState,
        progress: Option<&ProgressBar>,
        target_name: &str,
    ) -> Result<InvariantWorkerState> {
        let mut worker = InvariantWorkerState::new(worker_id);
        let counters = &shared_state.worker_counters[worker_id];

        // Each worker clones the executor (gets own inspector chain + RandomCallGenerator).
        let mut executor = self.executor.clone();

        // Each worker gets its own TargetedContracts and FuzzDictionary to avoid contention.
        let targeted_contracts = FuzzRunIdentifiedContracts {
            targets: Arc::new(Mutex::new(targeted_contracts.targets.lock().clone())),
            is_updatable: targeted_contracts.is_updatable,
        };
        let fuzz_state = fuzz_state.fork(self.num_workers);

        // Each worker creates its own strategy (BoxedStrategy is not Send/Sync due to Rc).
        let strategy = invariant_strat(
            fuzz_state.clone(),
            targeted_senders.clone(),
            targeted_contracts.clone(),
            self.config.clone(),
            fuzz_fixtures.clone(),
        )
        .no_shrink()
        .boxed();

        // Set up call_override for this worker's executor.
        if self.config.call_override {
            let target_contract_ref = Arc::new(RwLock::new(Address::ZERO));
            let handler_addresses: std::collections::HashSet<Address> =
                targeted_contracts.targets.lock().keys().copied().collect();

            // TODO: each worker should have a unique RNG so they don't generate identical
            // call sequences. Currently all workers clone self.runner with the same seed.
            let call_generator = RandomCallGenerator::new(
                invariant_contract.address,
                handler_addresses,
                self.runner.clone(),
                override_call_strat(
                    fuzz_state.clone(),
                    targeted_contracts.clone(),
                    target_contract_ref.clone(),
                    fuzz_fixtures.clone(),
                ),
                target_contract_ref,
            );

            if let Some(fuzzer) = executor.inspector_mut().fuzzer.as_mut() {
                fuzzer.call_generator = Some(call_generator);
            }
        }

        // Update the inspector's fuzz_state to the forked (per-worker) copy so that
        // collect_values() in the inspector step() doesn't contend on the shared lock.
        if let Some(fuzzer) = executor.inspector_mut().fuzzer.as_mut() {
            fuzzer.fuzz_state = fuzz_state.clone();
        }

        // Create own WorkerCorpus (only worker 0 replays corpus).
        let mut corpus_manager = WorkerCorpus::new(
            worker_id,
            self.config.corpus.clone(),
            strategy,
            if worker_id == 0 { Some(&executor) } else { None },
            None,
            Some(&targeted_contracts),
            self.config.max_time_delay,
            self.config.max_block_delay,
            self.config.gen_weight,
        )?;

        // Create own TestRunner. Worker 0 clones the runner as-is for determinism;
        // other workers derive a unique seed from the worker id.
        let branch_runner = if worker_id == 0 {
            self.runner.clone()
        } else {
            let seed_bytes = keccak256(worker_id.to_be_bytes());
            let rng = TestRng::from_seed(RngAlgorithm::ChaCha, seed_bytes.as_ref());
            TestRunner::new_with_rng(self.runner.config().clone(), rng)
        };

        // Create own InvariantTest.
        let mut invariant_test = InvariantTest::new(
            fuzz_state.clone(),
            targeted_contracts.clone(),
            targeted_senders.clone(),
            initial_failures.clone(),
            branch_runner,
        );

        let edge_coverage_enabled = self.config.corpus.collect_edge_coverage();
        let worker_runs = self.runs_per_worker(worker_id);
        debug!(worker_id, worker_runs, "invariant worker starting");

        // Stagger syncs: worker 0 syncs immediately to export corpus first,
        // then other workers sync at evenly spaced offsets so they don't all
        // hit the filesystem at once. Worker i syncs at i * (interval / num_workers).
        let sync_offset = if worker_id == 0 {
            SYNC_INTERVAL
        } else {
            SYNC_INTERVAL * worker_id as u32 / self.num_workers as u32
        };
        let mut last_sync = Instant::now() - sync_offset;
        let mut last_metrics_report = Instant::now();
        let mut last_reported_failures = std::collections::HashSet::<FailureKey>::new();
        let mut last_metrics_calls: u64 = 0;
        let mut last_metrics_gas: u64 = 0;
        let (init_global_calls, init_global_gas, _) = shared_state.global_totals();
        let mut last_global_calls: u64 = init_global_calls;
        let mut last_global_gas: u64 = init_global_gas;

        'stop: while shared_state.should_continue() && worker.runs < worker_runs {
            if last_sync.elapsed() >= SYNC_INTERVAL {
                let timer = Instant::now();
                let SyncStats { imported, exported, calibrate_elapsed, export_elapsed } =
                    corpus_manager.sync(
                        self.num_workers,
                        &executor,
                        None,
                        Some(&targeted_contracts),
                        &shared_state.global_corpus_metrics,
                    )?;
                let total_elapsed = timer.elapsed();
                last_sync = Instant::now();

                // Emit separate import/export events in JSON log from all workers.
                if edge_coverage_enabled {
                    corpus_manager.sync_metrics(&shared_state.global_corpus_metrics);
                    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                    if imported > 0 {
                        let event = json!({
                            "worker_id": worker_id,
                            "timestamp": ts,
                            "event": "corpus_import",
                            "target": target_name,
                            "imported": imported,
                            "elapsed_ms": calibrate_elapsed.as_millis() as u64,
                        });
                        let _ = sh_println!("{}", serde_json::to_string(&event)?);
                    }
                    if exported > 0 {
                        let event = json!({
                            "worker_id": worker_id,
                            "timestamp": ts,
                            "event": "corpus_export",
                            "target": target_name,
                            "exported": exported,
                            "elapsed_ms": export_elapsed.as_millis() as u64,
                        });
                        let _ = sh_println!("{}", serde_json::to_string(&event)?);
                    }
                    trace!(
                        worker_id,
                        imported,
                        exported,
                        total_elapsed_ms = total_elapsed.as_millis() as u64,
                        "corpus sync complete"
                    );
                }
            }

            let initial_seq = corpus_manager.new_inputs(
                &mut invariant_test.test_data.branch_runner,
                &invariant_test.fuzz_state,
                &invariant_test.targeted_contracts,
                Some(&invariant_test.sender_filters),
            )?;

            // Create current invariant run data.
            let mut current_run = InvariantTestRun::new(
                initial_seq[0].clone(),
                // Before each run, we must reset the backend state.
                executor.clone(),
                self.config.depth as usize,
            );

            // We stop the run immediately if we have reverted, and `fail_on_revert` is set.
            if self.config.fail_on_revert && invariant_test.reverts() > 0 {
                shared_state.local_early_exit.record_failure();
                break 'stop;
            }

            let mut cumulative_edges = Vec::new();
            while current_run.depth < self.config.depth {
                // Check if the timeout has been reached or if we should stop early (ctrl+C).
                if shared_state.timer.is_timed_out() || shared_state.early_exit.should_stop() {
                    break 'stop;
                }

                let tx = current_run
                    .inputs
                    .last()
                    .ok_or_else(|| eyre!("no input generated to call fuzzed target."))?;

                // Execute call from the randomly generated sequence without committing state.
                // State is committed only if call is not a magic assume.
                let mut call_result = execute_tx(&mut current_run.executor, tx)?;
                // Flush inspector-buffered stack values to the per-worker dictionary.
                if let Some(fuzzer) = current_run.executor.inspector_mut().fuzzer.as_mut() {
                    fuzzer.flush_collected_values();
                }
                let discarded = call_result.result.as_ref() == MAGIC_ASSUME;
                if self.config.show_metrics {
                    invariant_test.record_metrics(tx, call_result.reverted, discarded);
                }

                // Collect line coverage from last fuzzed call.
                invariant_test.merge_line_coverage(call_result.line_coverage.clone());
                // Collect edge coverage and set the flag in the current run.
                let (new_cov, edges) = corpus_manager.merge_edge_coverage(&mut call_result);
                cumulative_edges.extend(edges);
                current_run.new_coverage |= new_cov;

                // Save corpus immediately when new coverage is found, capturing
                // the minimal sequence up to this point rather than the full depth.
                if new_cov {
                    corpus_manager.process_inputs(
                        &current_run.inputs,
                        true,
                        std::mem::take(&mut cumulative_edges),
                    );
                }

                // Count all calls (including discards) in throughput metrics.
                counters.add_call(call_result.gas_used);

                if discarded {
                    current_run.inputs.pop();
                    current_run.rejects += 1;
                    if current_run.rejects > self.config.max_assume_rejects {
                        invariant_test.set_error(
                            FailureKey::new(target_name, &invariant_contract.invariant_fn.name),
                            InvariantFuzzError::MaxAssumeRejects(self.config.max_assume_rejects),
                        );
                        shared_state.local_early_exit.record_failure();
                        break 'stop;
                    }
                } else {
                    // Commit executed call result.
                    current_run.executor.commit(&mut call_result);

                    // Collect data for fuzzing from the state changeset.
                    let mut state_changeset = std::mem::take(&mut call_result.state_changeset);
                    if !call_result.reverted {
                        collect_data(
                            &invariant_test,
                            &mut state_changeset,
                            tx,
                            &call_result,
                            self.config.depth,
                        );
                    }

                    // Collect created contracts and add to fuzz targets only if targeted contracts
                    // are updatable.
                    if let Err(error) =
                        &invariant_test.targeted_contracts.collect_created_contracts(
                            &state_changeset,
                            self.project_contracts,
                            self.setup_contracts,
                            &self.artifact_filters,
                            &mut current_run.created_contracts,
                        )
                    {
                        warn!(target: "forge::test", "{error}");
                    }

                    // Emit pulse/metrics at regular intervals (inside depth loop
                    // so pulses fire even during long runs).
                    if edge_coverage_enabled
                        && last_metrics_report.elapsed() > DURATION_BETWEEN_METRICS_REPORT
                    {
                        let failures = &invariant_test.test_data.failures;

                        // Emit failure events for any new unique failures.
                        for (key, error) in &failures.errors {
                            if !last_reported_failures.contains(key) {
                                let failure_event = json!({
                                    "worker_id": worker_id,
                                    "timestamp": SystemTime::now()
                                        .duration_since(UNIX_EPOCH)?
                                        .as_secs(),
                                    "event": "failure",
                                    "target": key,
                                    "type": error.failure_type(),
                                });
                                let _ = sh_println!("{}", serde_json::to_string(&failure_event)?);
                                // TODO: persist failure immediately so it survives Ctrl+C.
                                last_reported_failures.insert(key.clone());
                            }
                        }

                        // Emit metrics event.
                        let elapsed = last_metrics_report.elapsed().as_secs_f64();

                        // All workers emit local pulse.
                        let local_calls = counters.calls.load(Ordering::Relaxed);
                        let local_gas = counters.gas.load(Ordering::Relaxed);
                        let local_failures = counters.failures.load(Ordering::Relaxed);
                        let delta_calls = local_calls - last_metrics_calls;
                        let delta_gas = local_gas - last_metrics_gas;
                        let tx_per_sec = delta_calls as f64 / elapsed;
                        let gas_per_sec = delta_gas as f64 / elapsed;
                        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                        let pulse = json!({
                            "worker_id": worker_id,
                            "timestamp": ts,
                            "event": "pulse",
                            "target": target_name,
                            "metrics": {
                                "unique_failures": failures.unique_failures(),
                                "failures": local_failures,
                                "corpus_count": corpus_manager.corpus_count(),
                                "tx/s": tx_per_sec as u64,
                                "gas/s": gas_per_sec as u64,
                            },
                        });
                        let _ = sh_println!("{}", serde_json::to_string(&pulse)?);
                        last_metrics_calls = local_calls;
                        last_metrics_gas = local_gas;

                        // Worker 0 also emits global_metrics aggregated across all workers.
                        if worker_id == 0 {
                            corpus_manager.sync_metrics(&shared_state.global_corpus_metrics);
                            let (global_calls, global_gas, global_failures) =
                                shared_state.global_totals();
                            let global_delta_calls = global_calls - last_global_calls;
                            let global_delta_gas = global_gas - last_global_gas;
                            let global_event = json!({
                                "timestamp": ts,
                                "event": "global_metrics",
                                "target": target_name,
                                "metrics": {
                                    "failures": global_failures,
                                    "tx/s": if elapsed > 0.0 { (global_delta_calls as f64 / elapsed) as u64 } else { 0 },
                                    "gas/s": if elapsed > 0.0 { (global_delta_gas as f64 / elapsed) as u64 } else { 0 },
                                },
                            });
                            let _ = sh_println!("{}", serde_json::to_string(&global_event)?);
                            last_global_calls = global_calls;
                            last_global_gas = global_gas;
                        }
                        last_metrics_report = Instant::now();
                    }
                    current_run
                        .fuzz_runs
                        .push(FuzzCase { gas: call_result.gas_used, stipend: call_result.stipend });

                    // Determine if test can continue or should exit.
                    // Check invariants based on check_interval to improve deep run performance.
                    let is_last_call = current_run.depth == self.config.depth - 1;
                    let should_check_invariant = if self.config.check_interval == 0 {
                        is_last_call
                    } else {
                        self.config.check_interval == 1
                            || (current_run.depth + 1).is_multiple_of(self.config.check_interval)
                            || is_last_call
                    };

                    if should_check_invariant {
                        let failures_before = invariant_test.test_data.failures.total_failures;
                        process_call_result(
                            &invariant_contract,
                            &mut invariant_test,
                            &mut current_run,
                            &self.config,
                            call_result,
                            &state_changeset,
                            target_name,
                        )
                        .map_err(|e| eyre!(e.to_string()))?;
                        let new_failures =
                            invariant_test.test_data.failures.total_failures - failures_before;
                        for _ in 0..new_failures {
                            counters.record_failure();
                        }
                    } else {
                        // Skip invariant check but still detect assertion failures
                        if self.config.fail_on_assert
                            && (call_result.is_assert_failure()
                                || current_run.executor.has_global_failure(&state_changeset))
                        {
                            let handler_name = current_run
                                .inputs
                                .last()
                                .and_then(|last_input| {
                                    invariant_test
                                        .targeted_contracts
                                        .targets
                                        .lock()
                                        .fuzzed_metric_key(last_input)
                                        .map(|metric_key| {
                                            metric_key
                                                .rsplit('.')
                                                .next()
                                                .unwrap_or(metric_key.as_str())
                                                .to_string()
                                        })
                                })
                                .unwrap_or_else(|| "unknown".to_string());
                            let case_data = error::FailedInvariantCaseData::new(
                                &invariant_contract,
                                self.config.shrink_run_limit,
                                self.config.fail_on_revert,
                                self.config.fail_on_assert,
                                &invariant_test.targeted_contracts,
                                &current_run.inputs,
                                &call_result,
                                &[],
                            );
                            invariant_test.test_data.failures.revert_reason =
                                Some(case_data.revert_reason.clone());
                            invariant_test.test_data.failures.record_failure(
                                FailureKey::new(target_name, &handler_name),
                                InvariantFuzzError::BrokenAssertion(case_data),
                            );
                            counters.record_failure();
                        }
                        // Track reverts
                        if call_result.reverted {
                            invariant_test.test_data.failures.reverts += 1;
                            if self.config.fail_on_revert {
                                for (invariant, fail_on_revert) in &invariant_contract.invariant_fns
                                {
                                    if *fail_on_revert {
                                        let case_data = error::FailedInvariantCaseData::new(
                                            &invariant_contract,
                                            self.config.shrink_run_limit,
                                            *fail_on_revert,
                                            self.config.fail_on_assert,
                                            &invariant_test.targeted_contracts,
                                            &current_run.inputs,
                                            &call_result,
                                            &[],
                                        );
                                        invariant_test.test_data.failures.record_failure(
                                            FailureKey::new(target_name, &invariant.name),
                                            InvariantFuzzError::Revert(case_data),
                                        );
                                        counters.record_failure();
                                    }
                                }
                            } else if !invariant_contract.is_optimization() {
                                current_run.inputs.pop();
                            }
                        }
                    }
                    if current_run.depth == self.config.depth - 1 {
                        invariant_test.set_last_run_inputs(&current_run.inputs);
                    }
                    current_run.depth += 1;
                }

                current_run.inputs.push(corpus_manager.generate_next_input(
                    &mut invariant_test.test_data.branch_runner,
                    &initial_seq,
                    discarded,
                    current_run.depth as usize,
                )?);
            }

            // Call `afterInvariant` if it is declared.
            if invariant_contract.call_after_invariant {
                assert_after_invariant(
                    &invariant_contract,
                    &mut invariant_test,
                    &current_run,
                    &self.config,
                    target_name,
                )
                .map_err(|_| eyre!("Failed to call afterInvariant"))?;
            }

            // End current invariant test run.
            invariant_test.end_run(current_run, self.config.gas_report_samples as usize);

            shared_state.increment_runs();
            worker.runs += 1;

            if let Some(progress) = progress {
                // If running with progress then increment completed runs.
                progress.inc(1);

                if worker_id == 0 {
                    let failures = &invariant_test.test_data.failures;
                    let mut parts = Vec::new();
                    // Add failures if present
                    if !failures.errors.is_empty() {
                        parts.push(format!("{failures}"));
                    }
                    // Add edge coverage metrics if enabled
                    if edge_coverage_enabled {
                        corpus_manager.sync_metrics(&shared_state.global_corpus_metrics);
                        parts.push(format!("{}", shared_state.global_corpus_metrics));
                    }
                    // Add throughput metrics.
                    let elapsed = last_metrics_report.elapsed().as_secs_f64();
                    let (global_calls, global_gas, _) = shared_state.global_totals();
                    let delta_calls = global_calls - last_global_calls;
                    let delta_gas = global_gas - last_global_gas;
                    let tx_s = delta_calls as f64 / elapsed;
                    let gas_s = delta_gas as f64 / elapsed;
                    parts.push(format!("\n      {tx_s:.0} tx/s, {gas_s:.0} gas/s"));
                    progress.set_message(parts.join(""));
                }
            }
        }

        // Collect results from this worker's InvariantTest into the worker state.
        let test_data = invariant_test.test_data;
        worker.failures = test_data.failures;
        worker.fuzz_cases = test_data.fuzz_cases;
        worker.last_run_inputs = test_data.last_run_inputs;
        worker.gas_report_traces = test_data.gas_report_traces;
        worker.line_coverage = test_data.line_coverage;
        worker.metrics = test_data.metrics;
        worker.optimization_best_value = test_data.optimization_best_value;
        worker.optimization_best_sequence = test_data.optimization_best_sequence;

        if worker_id == 0 {
            worker.failed_corpus_replays = corpus_manager.failed_replays;
        }

        invariant_test.fuzz_state.log_stats();

        // Compress corpus files that were written uncompressed during the run.
        corpus_manager.compress_corpus();

        Ok(worker)
    }

    /// Aggregates results from all workers into a single result.
    fn aggregate_invariant_results(
        &self,
        workers: Vec<InvariantWorkerState>,
        fuzz_state: &EvmFuzzState,
    ) -> InvariantFuzzTestResult {
        let mut result = InvariantFuzzTestResult {
            errors: std::collections::HashMap::default(),
            cases: vec![],
            reverts: 0,
            last_run_inputs: vec![],
            gas_report_traces: vec![],
            line_coverage: None,
            metrics: Map::default(),
            failed_corpus_replays: 0,
            optimization_best_value: None,
            optimization_best_sequence: vec![],
        };

        let mut has_error = false;
        for mut worker in workers {
            // Merge errors: first error per key wins.
            if !worker.failures.errors.is_empty() {
                for (key, error) in worker.failures.errors {
                    if !result.errors.contains_key(&key) {
                        if !has_error {
                            result.last_run_inputs = std::mem::take(&mut worker.last_run_inputs);
                            has_error = true;
                        }
                        result.errors.insert(key, error);
                    }
                }
            }

            result.reverts += worker.failures.reverts;
            result.cases.extend(worker.fuzz_cases);
            result.gas_report_traces.extend(worker.gas_report_traces);
            HitMaps::merge_opt(&mut result.line_coverage, worker.line_coverage);

            // Merge metrics per selector.
            for (key, m) in worker.metrics {
                let entry = result.metrics.entry(key).or_default();
                entry.calls += m.calls;
                entry.reverts += m.reverts;
                entry.discards += m.discards;
            }

            // Keep the best optimization value.
            if let Some(value) = worker.optimization_best_value {
                if result.optimization_best_value.is_none_or(|best| value > best) {
                    result.optimization_best_value = Some(value);
                    result.optimization_best_sequence = worker.optimization_best_sequence;
                }
            }

            // Only set replays from worker 0.
            if worker.id == 0 {
                result.failed_corpus_replays = worker.failed_corpus_replays;
                // If no error, use worker 0's last run inputs.
                if !has_error {
                    result.last_run_inputs = worker.last_run_inputs;
                }
            }
        }

        trace!("aggregated invariant results");
        fuzz_state.log_stats();

        result
    }

    /// Prepares shared state for invariant testing:
    /// * Selects contracts and senders
    /// * Runs the initial invariant assertion (preflight check)
    ///
    /// Returns the components needed by each worker to run independently.
    /// Each worker creates its own strategy (since `BoxedStrategy` is not `Send`/`Sync`).
    #[allow(clippy::type_complexity)]
    fn prepare_test(
        &mut self,
        invariant_contract: &InvariantContract<'_>,
        fuzz_state: EvmFuzzState,
    ) -> Result<(EvmFuzzState, SenderFilters, FuzzRunIdentifiedContracts, InvariantFailures)> {
        // Finds out the chosen deployed contracts and/or senders.
        self.select_contract_artifacts(invariant_contract.address)?;
        let (targeted_senders, targeted_contracts) =
            self.select_contracts_and_senders(invariant_contract.address)?;

        // If any of the targeted contracts have the storage layout enabled then we can sample
        // mapping values. To accomplish, we need to record the mapping storage slots and keys.
        let fuzz_state =
            if targeted_contracts.targets.lock().iter().any(|(_, t)| t.storage_layout.is_some()) {
                fuzz_state.with_mapping_slots(AddressMap::default())
            } else {
                fuzz_state
            };

        // Set up fuzzer WITHOUT call_generator initially.
        // We defer call_override until after the initial invariant check to avoid
        // injecting random calls during setup which would break the invariant assertion.
        self.executor.inspector_mut().set_fuzzer(Fuzzer::new(fuzz_state.clone(), None));

        // Let's make sure the invariant is sound before actually starting the run:
        // We'll assert the invariant in its initial state, and if it fails, we'll
        // already know if we can early exit the invariant run.
        // This does not count as a fuzz run. It will just register the revert.
        let mut failures = InvariantFailures::new();
        invariant_preflight_check(
            invariant_contract,
            &self.config,
            &targeted_contracts,
            &self.executor,
            &[],
            &mut failures,
            "", // preflight check doesn't need target name
        )?;
        if !failures.errors.is_empty() {
            let error = failures.errors.values().next().unwrap();
            return Err(eyre!(error.revert_reason().unwrap_or_default()));
        }

        Ok((fuzz_state, targeted_senders, targeted_contracts, failures))
    }

    /// Fills the `InvariantExecutor` with the artifact identifier filters (in `path:name` string
    /// format). They will be used to filter contracts after the `setUp`, and more importantly,
    /// during the runs.
    ///
    /// Also excludes any contract without any mutable functions.
    ///
    /// Priority:
    ///
    /// targetArtifactSelectors > excludeArtifacts > targetArtifacts
    pub fn select_contract_artifacts(&mut self, invariant_address: Address) -> Result<()> {
        let targeted_artifact_selectors = self
            .executor
            .call_sol_default(invariant_address, &IInvariantTest::targetArtifactSelectorsCall {});

        // Insert them into the executor `targeted_abi`.
        for IInvariantTest::FuzzArtifactSelector { artifact, selectors } in
            targeted_artifact_selectors
        {
            let identifier = self.validate_selected_contract(artifact, &selectors)?;
            self.artifact_filters.targeted.entry(identifier).or_default().extend(selectors);
        }

        let targeted_artifacts = self
            .executor
            .call_sol_default(invariant_address, &IInvariantTest::targetArtifactsCall {});
        let excluded_artifacts = self
            .executor
            .call_sol_default(invariant_address, &IInvariantTest::excludeArtifactsCall {});

        // Insert `excludeArtifacts` into the executor `excluded_abi`.
        for contract in excluded_artifacts {
            let identifier = self.validate_selected_contract(contract, &[])?;

            if !self.artifact_filters.excluded.contains(&identifier) {
                self.artifact_filters.excluded.push(identifier);
            }
        }

        // Exclude any artifact without mutable functions.
        for (artifact, contract) in self.project_contracts.iter() {
            if contract
                .abi
                .functions()
                .filter(|func| {
                    !matches!(
                        func.state_mutability,
                        alloy_json_abi::StateMutability::Pure
                            | alloy_json_abi::StateMutability::View
                    )
                })
                .count()
                == 0
                && !self.artifact_filters.excluded.contains(&artifact.identifier())
            {
                self.artifact_filters.excluded.push(artifact.identifier());
            }
        }

        // Insert `targetArtifacts` into the executor `targeted_abi`, if they have not been seen
        // before.
        for contract in targeted_artifacts {
            let identifier = self.validate_selected_contract(contract, &[])?;

            if !self.artifact_filters.targeted.contains_key(&identifier)
                && !self.artifact_filters.excluded.contains(&identifier)
            {
                self.artifact_filters.targeted.insert(identifier, vec![]);
            }
        }
        Ok(())
    }

    /// Makes sure that the contract exists in the project. If so, it returns its artifact
    /// identifier.
    fn validate_selected_contract(
        &mut self,
        contract: String,
        selectors: &[FixedBytes<4>],
    ) -> Result<String> {
        if let Some((artifact, contract_data)) =
            self.project_contracts.find_by_name_or_identifier(&contract)?
        {
            // Check that the selectors really exist for this contract.
            for selector in selectors {
                contract_data
                    .abi
                    .functions()
                    .find(|func| func.selector().as_slice() == selector.as_slice())
                    .wrap_err(format!("{contract} does not have the selector {selector:?}"))?;
            }

            return Ok(artifact.identifier());
        }
        eyre::bail!(
            "{contract} not found in the project. Allowed format: `contract_name` or `contract_path:contract_name`."
        );
    }

    /// Selects senders and contracts based on the contract methods `targetSenders() -> address[]`,
    /// `targetContracts() -> address[]` and `excludeContracts() -> address[]`.
    pub fn select_contracts_and_senders(
        &self,
        to: Address,
    ) -> Result<(SenderFilters, FuzzRunIdentifiedContracts)> {
        let targeted_senders =
            self.executor.call_sol_default(to, &IInvariantTest::targetSendersCall {});
        let mut excluded_senders =
            self.executor.call_sol_default(to, &IInvariantTest::excludeSendersCall {});
        // Extend with default excluded addresses - https://github.com/foundry-rs/foundry/issues/4163
        excluded_senders.extend([
            CHEATCODE_ADDRESS,
            HARDHAT_CONSOLE_ADDRESS,
            DEFAULT_CREATE2_DEPLOYER,
        ]);
        // Extend with precompiles - https://github.com/foundry-rs/foundry/issues/4287
        excluded_senders.extend(PRECOMPILES);
        let sender_filters = SenderFilters::new(targeted_senders, excluded_senders);

        let selected = self.executor.call_sol_default(to, &IInvariantTest::targetContractsCall {});
        let excluded = self.executor.call_sol_default(to, &IInvariantTest::excludeContractsCall {});

        let contracts = self
            .setup_contracts
            .iter()
            .filter(|&(addr, (identifier, _))| {
                // Include to address if explicitly set as target.
                if *addr == to && selected.contains(&to) {
                    return true;
                }

                *addr != to
                    && *addr != CHEATCODE_ADDRESS
                    && *addr != HARDHAT_CONSOLE_ADDRESS
                    && (selected.is_empty() || selected.contains(addr))
                    && (excluded.is_empty() || !excluded.contains(addr))
                    && self.artifact_filters.matches(identifier)
            })
            .map(|(addr, (identifier, abi))| {
                (
                    *addr,
                    TargetedContract::new(identifier.clone(), abi.clone())
                        .with_project_contracts(self.project_contracts),
                )
            })
            .collect();
        let mut contracts = TargetedContracts { inner: contracts };

        self.target_interfaces(to, &mut contracts)?;

        self.select_selectors(to, &mut contracts)?;

        // There should be at least one contract identified as target for fuzz runs.
        if contracts.is_empty() {
            eyre::bail!("No contracts to fuzz.");
        }

        Ok((sender_filters, FuzzRunIdentifiedContracts::new(contracts, selected.is_empty())))
    }

    /// Extends the contracts and selectors to fuzz with the addresses and ABIs specified in
    /// `targetInterfaces() -> (address, string[])[]`. Enables targeting of addresses that are
    /// not deployed during `setUp` such as when fuzzing in a forked environment. Also enables
    /// targeting of delegate proxies and contracts deployed with `create` or `create2`.
    pub fn target_interfaces(
        &self,
        invariant_address: Address,
        targeted_contracts: &mut TargetedContracts,
    ) -> Result<()> {
        let interfaces = self
            .executor
            .call_sol_default(invariant_address, &IInvariantTest::targetInterfacesCall {});

        // Since `targetInterfaces` returns a tuple array there is no guarantee
        // that the addresses are unique this map is used to merge functions of
        // the specified interfaces for the same address. For example:
        // `[(addr1, ["IERC20", "IOwnable"])]` and `[(addr1, ["IERC20"]), (addr1, ("IOwnable"))]`
        // should be equivalent.
        let mut combined = TargetedContracts::new();

        // Loop through each address and its associated artifact identifiers.
        // We're borrowing here to avoid taking full ownership.
        for IInvariantTest::FuzzInterface { addr, artifacts } in &interfaces {
            // Identifiers are specified as an array, so we loop through them.
            for identifier in artifacts {
                // Try to find the contract by name or identifier in the project's contracts.
                if let Some((_, contract_data)) =
                    self.project_contracts.iter().find(|(artifact, _)| {
                        &artifact.name == identifier || &artifact.identifier() == identifier
                    })
                {
                    let abi = &contract_data.abi;
                    combined
                        // Check if there's an entry for the given key in the 'combined' map.
                        .entry(*addr)
                        // If the entry exists, extends its ABI with the function list.
                        .and_modify(|entry| {
                            // Extend the ABI's function list with the new functions.
                            entry.abi.functions.extend(abi.functions.clone());
                        })
                        // Otherwise insert it into the map.
                        .or_insert_with(|| {
                            let mut contract =
                                TargetedContract::new(identifier.to_string(), abi.clone());
                            contract.storage_layout =
                                contract_data.storage_layout.as_ref().map(Arc::clone);
                            contract
                        });
                }
            }
        }

        targeted_contracts.extend(combined.inner);

        Ok(())
    }

    /// Selects the functions to fuzz based on the contract method `targetSelectors()` and
    /// `targetArtifactSelectors()`.
    pub fn select_selectors(
        &self,
        address: Address,
        targeted_contracts: &mut TargetedContracts,
    ) -> Result<()> {
        for (address, (identifier, _)) in self.setup_contracts {
            if let Some(selectors) = self.artifact_filters.targeted.get(identifier) {
                self.add_address_with_functions(*address, selectors, false, targeted_contracts)?;
            }
        }

        let mut target_test_selectors = vec![];
        let mut excluded_test_selectors = vec![];

        // Collect contract functions marked as target for fuzzing campaign.
        let selectors =
            self.executor.call_sol_default(address, &IInvariantTest::targetSelectorsCall {});
        for IInvariantTest::FuzzSelector { addr, selectors } in selectors {
            if addr == address {
                target_test_selectors = selectors.clone();
            }
            self.add_address_with_functions(addr, &selectors, false, targeted_contracts)?;
        }

        // Collect contract functions excluded from fuzzing campaign.
        let excluded_selectors =
            self.executor.call_sol_default(address, &IInvariantTest::excludeSelectorsCall {});
        for IInvariantTest::FuzzSelector { addr, selectors } in excluded_selectors {
            if addr == address {
                // If fuzz selector address is the test contract, then record selectors to be
                // later excluded if needed.
                excluded_test_selectors = selectors.clone();
            }
            self.add_address_with_functions(addr, &selectors, true, targeted_contracts)?;
        }

        if target_test_selectors.is_empty()
            && let Some(target) = targeted_contracts.get(&address)
        {
            // If test contract is marked as a target and no target selector explicitly set, then
            // include only state-changing functions that are not reserved and selectors that are
            // not explicitly excluded.
            let selectors: Vec<_> = target
                .abi
                .functions()
                .filter_map(|func| {
                    if matches!(
                        func.state_mutability,
                        alloy_json_abi::StateMutability::Pure
                            | alloy_json_abi::StateMutability::View
                    ) || func.is_reserved()
                        || excluded_test_selectors.contains(&func.selector())
                    {
                        None
                    } else {
                        Some(func.selector())
                    }
                })
                .collect();
            self.add_address_with_functions(address, &selectors, false, targeted_contracts)?;
        }

        Ok(())
    }

    /// Adds the address and fuzzed or excluded functions to `TargetedContracts`.
    fn add_address_with_functions(
        &self,
        address: Address,
        selectors: &[Selector],
        should_exclude: bool,
        targeted_contracts: &mut TargetedContracts,
    ) -> eyre::Result<()> {
        // Do not add address in target contracts if no function selected.
        if selectors.is_empty() {
            return Ok(());
        }

        let contract = match targeted_contracts.entry(address) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let (identifier, abi) = self.setup_contracts.get(&address).ok_or_else(|| {
                    eyre::eyre!(
                        "[{}] address does not have an associated contract: {}",
                        if should_exclude { "excludeSelectors" } else { "targetSelectors" },
                        address
                    )
                })?;
                entry.insert(
                    TargetedContract::new(identifier.clone(), abi.clone())
                        .with_project_contracts(self.project_contracts),
                )
            }
        };
        contract.add_selectors(selectors.iter().copied(), should_exclude)?;
        Ok(())
    }
}

/// Collects data from call for fuzzing. However, it first verifies that the sender is not an EOA
/// before inserting it into the dictionary. Otherwise, we flood the dictionary with
/// randomly generated addresses.
fn collect_data(
    invariant_test: &InvariantTest,
    state_changeset: &mut HashMap<Address, Account>,
    tx: &BasicTxDetails,
    call_result: &RawCallResult,
    run_depth: u32,
) {
    // Verify it has no code.
    let mut has_code = false;
    if let Some(Some(code)) =
        state_changeset.get(&tx.sender).map(|account| account.info.code.as_ref())
    {
        has_code = !code.is_empty();
    }

    // We keep the nonce changes to apply later.
    let mut sender_changeset = None;
    if !has_code {
        sender_changeset = state_changeset.remove(&tx.sender);
    }

    // Collect values from fuzzed call result and add them to fuzz dictionary.
    invariant_test.fuzz_state.collect_values_from_call(
        &invariant_test.targeted_contracts,
        tx,
        &call_result.result,
        &call_result.logs,
        &*state_changeset,
        run_depth,
    );

    // Re-add changes
    if let Some(changed) = sender_changeset {
        state_changeset.insert(tx.sender, changed);
    }
}

/// Calls the `afterInvariant()` function on a contract.
/// Returns call result and if call succeeded.
/// The state after the call is not persisted.
pub(crate) fn call_after_invariant_function(
    executor: &Executor,
    to: Address,
) -> Result<(RawCallResult, bool), EvmError> {
    let calldata = Bytes::from_static(&IInvariantTest::afterInvariantCall::SELECTOR);
    let mut call_result = executor.call_raw(CALLER, to, calldata, U256::ZERO)?;
    let success = executor.is_raw_call_mut_success(to, &mut call_result, false);
    Ok((call_result, success))
}

/// Calls the invariant function and returns call result and if succeeded.
pub(crate) fn call_invariant_function(
    executor: &Executor,
    address: Address,
    calldata: Bytes,
) -> Result<(RawCallResult, bool)> {
    let mut call_result = executor.call_raw(CALLER, address, calldata, U256::ZERO)?;
    let success = executor.is_raw_call_mut_success(address, &mut call_result, false);
    Ok((call_result, success))
}

/// Executes a fuzz call and returns the result.
/// Applies any block timestamp (warp), block number (roll), and balance (deal) adjustments before
/// the call.
pub(crate) fn execute_tx(executor: &mut Executor, tx: &BasicTxDetails) -> Result<RawCallResult> {
    let warp = tx.warp.unwrap_or_default();
    let roll = tx.roll.unwrap_or_default();

    if warp > 0 || roll > 0 {
        // Apply pre-call block adjustments to the executor's env.
        executor.env_mut().evm_env.block_env.timestamp += warp;
        executor.env_mut().evm_env.block_env.number += roll;

        // Also update the inspector's cheatcodes.block if set.
        // The inspector's block may override the env during interpreter initialization,
        // so we need to add our warp/roll on top of any existing cheatcode-set values.
        let block_env = executor.env().evm_env.block_env.clone();
        if let Some(cheatcodes) = executor.inspector_mut().cheatcodes.as_mut() {
            if let Some(block) = cheatcodes.block.as_mut() {
                block.timestamp += warp;
                block.number += roll;
            } else {
                cheatcodes.block = Some(block_env);
            }
        }
    }

    let requested_value = tx.call_details.value.unwrap_or(U256::ZERO);

    // If no value requested, skip balance checks and deal logic.
    let value = if requested_value.is_zero() {
        U256::ZERO
    } else {
        // Apply deal (increase sender balance) if specified.
        if let Some(deal) = tx.deal {
            let current_balance = executor.get_balance(tx.sender)?;
            executor.set_balance(tx.sender, current_balance + deal)?;
        }

        // Only use value if sender has sufficient balance (after deal), otherwise fall back to 0.
        let sender_balance = executor.get_balance(tx.sender)?;
        if sender_balance >= requested_value { requested_value } else { U256::ZERO }
    };

    executor
        .call_raw(tx.sender, tx.call_details.target, tx.call_details.calldata.clone(), value)
        .map_err(|e| eyre!(format!("Could not make raw evm call: {e}")))
}
