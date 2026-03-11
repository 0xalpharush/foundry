use super::{
    FailureKey, InvariantFailures, InvariantFuzzError, InvariantMetrics, InvariantTest,
    InvariantTestRun, call_after_invariant_function, call_invariant_function,
    error::FailedInvariantCaseData,
};
use crate::executors::{Executor, RawCallResult};
use alloy_dyn_abi::JsonAbiExt;
use alloy_primitives::I256;
use eyre::Result;
use foundry_config::InvariantConfig;
use foundry_evm_core::utils::StateChangeset;
use foundry_evm_coverage::HitMaps;
use foundry_evm_fuzz::{
    BasicTxDetails, FuzzedCases,
    invariant::{FuzzRunIdentifiedContracts, InvariantContract},
};
use revm_inspectors::tracing::CallTraceArena;
use std::{borrow::Cow, collections::HashMap};

/// The outcome of an invariant fuzz test
#[derive(Debug)]
pub struct InvariantFuzzTestResult {
    /// Errors recorded per invariant.
    pub errors: HashMap<FailureKey, InvariantFuzzError>,
    /// Every successful fuzz test case
    pub cases: Vec<FuzzedCases>,
    /// Number of reverted fuzz calls
    pub reverts: usize,
    /// The entire inputs of the last run of the invariant campaign, used for
    /// replaying the run for collecting traces.
    pub last_run_inputs: Vec<BasicTxDetails>,
    /// Additional traces used for gas report construction.
    pub gas_report_traces: Vec<Vec<CallTraceArena>>,
    /// The coverage info collected during the invariant test runs.
    pub line_coverage: Option<HitMaps>,
    /// Fuzzed selectors metrics collected during the invariant test runs.
    pub metrics: HashMap<String, InvariantMetrics>,
    /// Number of failed replays from persisted corpus.
    pub failed_corpus_replays: usize,
    /// For optimization mode (int256 return): the best (maximum) value achieved.
    /// None means standard invariant check mode.
    pub optimization_best_value: Option<I256>,
    /// For optimization mode: the call sequence that produced the best value.
    pub optimization_best_sequence: Vec<BasicTxDetails>,
}

/// Given the executor state, asserts that no invariant has been broken. Otherwise, it fills the
/// external `invariant_failures.failed_invariant` map and returns a generic error.
/// Either returns the call result if successful, or nothing if there was an error.
pub(crate) fn invariant_preflight_check(
    invariant_contract: &InvariantContract<'_>,
    invariant_config: &InvariantConfig,
    targeted_contracts: &FuzzRunIdentifiedContracts,
    executor: &Executor,
    calldata: &[BasicTxDetails],
    invariant_failures: &mut InvariantFailures,
    target_name: &str,
) -> Result<()> {
    let (call_result, success) = call_invariant_function(
        executor,
        invariant_contract.address,
        invariant_contract.invariant_fn.abi_encode_input(&[])?.into(),
    )?;
    if !success {
        invariant_failures.record_failure(
            FailureKey::new(target_name, &invariant_contract.invariant_fn.name),
            InvariantFuzzError::BrokenInvariant(FailedInvariantCaseData::new(
                invariant_contract,
                invariant_config.shrink_run_limit,
                invariant_config.fail_on_revert,
                invariant_config.fail_on_assert,
                targeted_contracts,
                calldata,
                &call_result,
                &invariant_inner_sequence(executor),
            )),
        );
    }

    Ok(())
}

/// Given the executor state, asserts that no invariant has been broken. Otherwise, it fills the
/// external `invariant_failures.failed_invariant` map and returns a generic error.
/// Either returns the call result if successful, or nothing if there was an error.
pub(crate) fn assert_invariants(
    invariant_contract: &InvariantContract<'_>,
    invariant_config: &InvariantConfig,
    targeted_contracts: &FuzzRunIdentifiedContracts,
    executor: &Executor,
    calldata: &[BasicTxDetails],
    invariant_failures: &mut InvariantFailures,
    target_name: &str,
) -> Result<()> {
    let inner_sequence = invariant_inner_sequence(executor);
    for (invariant, fail_on_revert) in &invariant_contract.invariant_fns {
        let (call_result, success) = call_invariant_function(
            executor,
            invariant_contract.address,
            invariant.abi_encode_input(&[])?.into(),
        )?;
        if !success {
            invariant_failures.record_failure(
                FailureKey::new(target_name, &invariant.name),
                InvariantFuzzError::BrokenInvariant(FailedInvariantCaseData::new(
                    invariant_contract,
                    invariant_config.shrink_run_limit,
                    *fail_on_revert,
                    invariant_config.fail_on_assert,
                    targeted_contracts,
                    calldata,
                    &call_result,
                    &inner_sequence,
                )),
            );
        }
    }

    Ok(())
}

/// Helper function to initialize invariant inner sequence.
fn invariant_inner_sequence(executor: &Executor) -> Vec<Option<BasicTxDetails>> {
    let mut seq = vec![];
    if let Some(fuzzer) = &executor.inspector().fuzzer
        && let Some(call_generator) = &fuzzer.call_generator
    {
        seq.extend(call_generator.last_sequence.read().iter().cloned());
    }
    seq
}

/// Returns if invariant test can continue and last successful call result of the invariant test
/// function (if it can continue).
///
/// For optimization mode (int256 return), tracks the max value but never fails on invariant.
/// For check mode, asserts the invariant and fails if broken.
/// Processes the result of a handler call: detects assertion failures, checks invariants,
/// and records any errors found during the campaign.
pub(crate) fn process_call_result(
    invariant_contract: &InvariantContract<'_>,
    invariant_test: &mut InvariantTest,
    invariant_run: &mut InvariantTestRun,
    invariant_config: &InvariantConfig,
    call_result: RawCallResult,
    state_changeset: &StateChangeset,
    target_name: &str,
) -> Result<()> {
    let is_optimization = invariant_contract.is_optimization();

    let handlers_succeeded = || {
        invariant_test.targeted_contracts.targets.lock().keys().all(|address| {
            invariant_run.executor.is_success(
                *address,
                false,
                Cow::Borrowed(state_changeset),
                false,
            )
        })
    };

    // When fail_on_assert is enabled, detect handler-level assertion failures.
    if invariant_config.fail_on_assert
        && (call_result.is_assert_failure()
            || invariant_run.executor.has_global_failure(state_changeset))
    {
        let handler_name = invariant_run
            .inputs
            .last()
            .and_then(|last_input| {
                invariant_test.targeted_contracts.targets.lock().fuzzed_metric_key(last_input).map(
                    |metric_key| {
                        metric_key.rsplit('.').next().unwrap_or(metric_key.as_str()).to_string()
                    },
                )
            })
            .unwrap_or_else(|| "unknown".to_string());
        invariant_test.test_data.failures.reverts += 1;
        let case_data = FailedInvariantCaseData::new(
            invariant_contract,
            invariant_config.shrink_run_limit,
            invariant_config.fail_on_revert,
            invariant_config.fail_on_assert,
            &invariant_test.targeted_contracts,
            &invariant_run.inputs,
            &call_result,
            &[],
        );
        invariant_test.test_data.failures.revert_reason = Some(case_data.revert_reason.clone());
        invariant_test.test_data.failures.record_failure(
            FailureKey::new(target_name, &handler_name),
            InvariantFuzzError::BrokenAssertion(case_data),
        );
        return Ok(());
    }

    // Assert invariants if the call did not revert and the handlers did not fail.
    if !call_result.reverted && handlers_succeeded() {
        if let Some(traces) = call_result.traces {
            invariant_run.run_traces.push(traces);
        }

        if is_optimization {
            // Optimization mode: call invariant and track max value, never fail.
            let (inv_result, success) = call_invariant_function(
                &invariant_run.executor,
                invariant_contract.address,
                invariant_contract.invariant_fn.abi_encode_input(&[])?.into(),
            )?;
            if success
                && inv_result.result.len() >= 32
                && let Some(value) = I256::try_from_be_slice(&inv_result.result[..32])
            {
                invariant_test.update_optimization_value(value, &invariant_run.inputs);
            }
        } else {
            // Check mode: assert invariants and fail if broken.
            assert_invariants(
                invariant_contract,
                invariant_config,
                &invariant_test.targeted_contracts,
                &invariant_run.executor,
                &invariant_run.inputs,
                &mut invariant_test.test_data.failures,
                target_name,
            )?;
        }
    } else {
        // Increase the amount of reverts.
        invariant_test.test_data.failures.reverts += 1;
        // If fail on revert is set, record invariant failure.
        for (invariant, fail_on_revert) in &invariant_contract.invariant_fns {
            if *fail_on_revert {
                let case_data = FailedInvariantCaseData::new(
                    invariant_contract,
                    invariant_config.shrink_run_limit,
                    *fail_on_revert,
                    invariant_config.fail_on_assert,
                    &invariant_test.targeted_contracts,
                    &invariant_run.inputs,
                    &call_result,
                    &[],
                );
                invariant_test.test_data.failures.record_failure(
                    FailureKey::new(target_name, &invariant.name),
                    InvariantFuzzError::Revert(case_data),
                );
            }
        }
        // Remove last reverted call from inputs.
        // This improves shrinking performance as irrelevant calls won't be checked again.
        if !is_optimization {
            // In optimization mode, we keep reverted calls to preserve warp/roll values
            // for correct replay during shrinking.
            invariant_run.inputs.pop();
        }
    }
    Ok(())
}

/// Given the executor state, asserts conditions within `afterInvariant` function.
/// If call fails then the invariant test is considered failed.
pub(crate) fn assert_after_invariant(
    invariant_contract: &InvariantContract<'_>,
    invariant_test: &mut InvariantTest,
    invariant_run: &InvariantTestRun,
    invariant_config: &InvariantConfig,
    target_name: &str,
) -> Result<bool> {
    let (call_result, success) =
        call_after_invariant_function(&invariant_run.executor, invariant_contract.address)?;
    // Fail the test case if `afterInvariant` doesn't succeed.
    if !success {
        let case_data = FailedInvariantCaseData::new(
            invariant_contract,
            invariant_config.shrink_run_limit,
            invariant_config.fail_on_revert,
            invariant_config.fail_on_assert,
            &invariant_test.targeted_contracts,
            &invariant_run.inputs,
            &call_result,
            &[],
        );
        invariant_test.set_error(
            FailureKey::new(target_name, &invariant_contract.invariant_fn.name),
            InvariantFuzzError::BrokenInvariant(case_data),
        );
    }
    Ok(success)
}
