use super::{
    calldata::CalldataStrategy,
    fuzz_calldata, fuzz_msg_value, fuzz_param, fuzz_param_from_state,
    param::{MsgValueStrategy, ParamStrategy},
};
use crate::{
    BasicTxDetails, CallDetails, FuzzFixtures,
    invariant::{FuzzRunIdentifiedContracts, SenderFilters},
    strategies::{EvmFuzzState, FuzzStateReader, InvariantFuzzState, fuzz_calldata_from_state},
};
use abi_fuzz::generators::sampler::sample_uint;
use alloy_json_abi::Function;
use alloy_primitives::{Address, Selector, U256};
use foundry_config::InvariantConfig;
use parking_lot::RwLock;
use rand::{Rng, RngCore, seq::IteratorRandom};
use std::{collections::HashMap, rc::Rc, sync::Arc};

/// Closure-style "strategy" yielding a [`CallDetails`] per call.
pub type CallDetailsStrategy = Box<dyn FnMut(&mut dyn RngCore) -> CallDetails>;

/// Closure-style "strategy" yielding a [`CallDetails`] per call from shared call override state.
pub type SharedCallDetailsStrategy = Box<dyn FnMut(&mut dyn RngCore) -> CallDetails + Send + Sync>;

/// Closure-style "strategy" yielding a [`BasicTxDetails`] per call.
pub type TxDetailsStrategy = Box<dyn FnMut(&mut dyn RngCore) -> BasicTxDetails>;

/// Closure-style "strategy" yielding an `Address` per call.
pub type SenderStrategy = Box<dyn FnMut(&mut dyn RngCore) -> Address>;

/// Given a target address, we generate random calldata.
pub fn override_call_strat(
    fuzz_state: EvmFuzzState,
    contracts: Vec<(Address, Vec<Function>)>,
    target: Arc<RwLock<Address>>,
    fuzz_fixtures: FuzzFixtures,
) -> SharedCallDetailsStrategy {
    let contracts = Arc::new(contracts);
    Box::new(move |rng: &mut dyn RngCore| {
        // 80/20: original target vs. random targeted contract.
        let target_address = if rng.random_ratio(80, 100) {
            *target.read()
        } else {
            contracts[rng.random_range(0..contracts.len())].0
        };

        let (actual_target, func) = {
            let (actual_target, fuzzed_functions) = contracts
                .iter()
                .find(|(address, _)| *address == target_address)
                .unwrap_or_else(|| &contracts[rng.random_range(0..contracts.len())]);
            let func = fuzzed_functions[rng.random_range(0..fuzzed_functions.len())].clone();
            (*actual_target, func)
        };

        let mut inner =
            fuzz_contract_with_calldata(&fuzz_state, &fuzz_fixtures, actual_target, func);
        inner(rng)
    })
}

/// Creates the invariant generator.
///
/// Given the known and future contracts, it generates the next call by fuzzing the `caller`,
/// `calldata` and `target`. The generated data is evaluated lazily for every single call to fully
/// leverage the evolving fuzz dictionary.
///
/// The fuzzed parameters can be filtered through different methods implemented in the test
/// contract:
///
/// `targetContracts()`, `targetSenders()`, `excludeContracts()`, `targetSelectors()`
pub fn invariant_strat(
    fuzz_state: InvariantFuzzState,
    senders: SenderFilters,
    contracts: FuzzRunIdentifiedContracts,
    config: InvariantConfig,
    fuzz_fixtures: FuzzFixtures,
) -> TxDetailsStrategy {
    let senders = Rc::new(senders);
    let dictionary_weight = config.dictionary.dictionary_weight;
    let max_time_delay = config.max_time_delay;
    let max_block_delay = config.max_block_delay;
    let max_deal = config.max_deal;

    let target_functions = collect_target_functions(&contracts);
    let mut call_generators = {
        let targets = contracts.targets();
        targets
            .fuzzed_functions()
            .map(|(target, function)| {
                (
                    (*target, function.selector()),
                    ContractCallGenerator::new(
                        &fuzz_state,
                        &fuzz_fixtures,
                        *target,
                        function.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>()
    };
    let mut sender_strat = select_random_sender(&fuzz_state, senders.clone(), dictionary_weight);

    Box::new(move |rng: &mut dyn RngCore| {
        // Pick a random (target_address, target_function) pair.
        let (target_address, selector, target_function) = if !contracts.is_updatable
            && !target_functions.is_empty()
        {
            let (target, selector) = target_functions[rng.random_range(0..target_functions.len())];
            (target, selector, None)
        } else {
            let contracts = contracts.targets();
            let (target, function) =
                contracts.fuzzed_functions().choose(rng).expect("at least one target function");
            (*target, function.selector(), Some(function.clone()))
        };

        // TODO cycle b/w [0x10000, 0x20000, defaultDeployerAddr]?
        let sender = sender_strat(rng);

        if let Some(target_function) = target_function {
            call_generators.entry((target_address, selector)).or_insert_with(|| {
                ContractCallGenerator::new(
                    &fuzz_state,
                    &fuzz_fixtures,
                    target_address,
                    target_function,
                )
            });
        }
        let call_details = call_generators
            .get_mut(&(target_address, selector))
            .expect("target function generator is cached")
            .generate(rng);

        let warp = max_time_delay.map(|m| sample_uint(256, rng) % U256::from(m));
        let roll = max_block_delay.map(|m| sample_uint(256, rng) % U256::from(m));
        let deal = max_deal.map(|m| sample_uint(256, rng) % U256::from(m));

        BasicTxDetails { warp, roll, deal, sender, call_details }
    })
}

fn collect_target_functions(contracts: &FuzzRunIdentifiedContracts) -> Vec<(Address, Selector)> {
    let contracts = contracts.targets();
    contracts.fuzzed_functions().map(|(target, function)| (*target, function.selector())).collect()
}

/// Strategy to select a sender address:
/// * If `senders` is empty, draw an address either uniformly random (`100 - dictionary_weight`%) or
///   from the EVM fuzz dictionary (`dictionary_weight`%).
/// * If `senders` is not empty, pick uniformly from the targeted list.
///
/// Excluded-sender fixup happens once at the run-loop boundary
/// ([`SenderFilters::resolve`] applied before pushing onto `current_run.inputs`),
/// so this strategy is free to return any address.
fn select_random_sender<S: FuzzStateReader>(
    fuzz_state: &S,
    senders: Rc<SenderFilters>,
    dictionary_weight: u32,
) -> SenderStrategy {
    if senders.targeted.is_empty() {
        assert!(dictionary_weight <= 100, "dictionary_weight must be <= 100");
        let total = 100u32;
        let dict_w = dictionary_weight;
        let mut random_strat: ParamStrategy = fuzz_param(&alloy_dyn_abi::DynSolType::Address);
        let mut state_strat: ParamStrategy =
            fuzz_param_from_state(&alloy_dyn_abi::DynSolType::Address, fuzz_state);
        Box::new(move |rng: &mut dyn RngCore| {
            let value =
                if rng.random_ratio(dict_w, total) { state_strat(rng) } else { random_strat(rng) };
            value.as_address().unwrap()
        })
    } else {
        Box::new(move |rng: &mut dyn RngCore| {
            senders.targeted[rng.random_range(0..senders.targeted.len())]
        })
    }
}

/// Given a function, it returns a generator which produces valid abi-encoded calldata
/// for that function's input types.
pub fn fuzz_contract_with_calldata<S: FuzzStateReader>(
    fuzz_state: &S,
    fuzz_fixtures: &FuzzFixtures,
    target: Address,
    func: Function,
) -> CallDetailsStrategy {
    let is_payable = func.state_mutability == alloy_json_abi::StateMutability::Payable;
    // 60/40: fixtures-aware calldata vs. dictionary-state calldata.
    let mut a: CalldataStrategy = fuzz_calldata(func.clone(), fuzz_fixtures);
    let mut b: CalldataStrategy = fuzz_calldata_from_state(func, fuzz_state);
    // For payable functions, generate random value using shared strategy. Otherwise, always None.
    let mut value_strategy: MsgValueStrategy =
        if is_payable { fuzz_msg_value() } else { Box::new(|_| None) };
    Box::new(move |rng: &mut dyn RngCore| {
        let calldata = if rng.random_ratio(60, 100) { a(rng) } else { b(rng) };
        let value = value_strategy(rng);
        trace!(input=?calldata, ?value);
        CallDetails { target, calldata, value }
    })
}

struct ContractCallGenerator {
    target: Address,
    fixtures_calldata: CalldataStrategy,
    state_calldata: CalldataStrategy,
    value_strategy: Option<MsgValueStrategy>,
}

impl ContractCallGenerator {
    fn new<S: FuzzStateReader>(
        fuzz_state: &S,
        fuzz_fixtures: &FuzzFixtures,
        target: Address,
        func: Function,
    ) -> Self {
        let is_payable = func.state_mutability == alloy_json_abi::StateMutability::Payable;
        Self {
            target,
            fixtures_calldata: fuzz_calldata(func.clone(), fuzz_fixtures),
            state_calldata: fuzz_calldata_from_state(func, fuzz_state),
            value_strategy: is_payable.then(fuzz_msg_value),
        }
    }

    fn generate(&mut self, rng: &mut dyn RngCore) -> CallDetails {
        let calldata = if rng.random_ratio(60, 100) {
            (self.fixtures_calldata)(rng)
        } else {
            (self.state_calldata)(rng)
        };
        let value = self.value_strategy.as_mut().and_then(|strategy| strategy(rng));
        trace!(input=?calldata, ?value);
        CallDetails { target: self.target, calldata, value }
    }
}
