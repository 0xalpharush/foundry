use super::{fuzz_calldata, fuzz_msg_value, fuzz_param_from_state};
use crate::{
    BasicTxDetails, CallDetails, FuzzFixtures,
    invariant::{FuzzRunIdentifiedContracts, SenderFilters},
    strategies::{
        EvmFuzzState, FuzzStateReader, InvariantFuzzState, fuzz_calldata_from_state, fuzz_param,
    },
};
use alloy_json_abi::{Function, StateMutability};
use alloy_primitives::{Address, Bytes, U256};
use foundry_config::InvariantConfig;
use parking_lot::RwLock;
use proptest::prelude::*;
use rand::seq::IteratorRandom;
use std::{rc::Rc, sync::Arc};

/// Given a target address, we generate random calldata.
pub fn override_call_strat(
    fuzz_state: EvmFuzzState,
    contracts: Vec<(Address, Vec<Function>)>,
    target: Arc<RwLock<Address>>,
    fuzz_fixtures: FuzzFixtures,
) -> impl Strategy<Value = CallDetails> + Send + Sync + 'static {
    let contracts = Arc::new(contracts);
    let contracts_ref = contracts.clone();
    proptest::prop_oneof![
        80 => proptest::strategy::LazyJust::new(move || *target.read()),
        20 => any::<prop::sample::Selector>()
            .prop_map(move |selector| {
                let (target, _) = selector.select(contracts_ref.iter());
                *target
            }),
    ]
    .prop_flat_map(move |target_address| {
        let fuzz_state = fuzz_state.clone();
        let fuzz_fixtures = fuzz_fixtures.clone();
        let contracts = contracts.clone();

        let (actual_target, func) = {
            // If the target address is in the contracts map, use it directly.
            // Otherwise, fall back to a random contract from the targeted contracts.
            // This can happen when call_override sets target_reference to a contract
            // that is not in targetContracts (e.g., the protocol contract during reentrancy).
            let (actual_target, fuzzed_functions) = contracts
                .iter()
                .find(|(address, _)| *address == target_address)
                .map(|(address, functions)| (*address, functions.clone()))
                .unwrap_or_else(|| {
                    let (address, functions) = contracts
                        .iter()
                        .choose(&mut rand::rng())
                        .expect("at least one target contract");
                    (*address, functions.clone())
                });
            (
                actual_target,
                any::<prop::sample::Index>()
                    .prop_map(move |index| index.get(&fuzzed_functions).clone()),
            )
        };

        func.prop_flat_map(move |func| {
            fuzz_contract_with_calldata(&fuzz_state, &fuzz_fixtures, actual_target, func)
        })
    })
}

/// Creates the invariant strategy.
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
) -> impl Strategy<Value = BasicTxDetails> {
    enum Call {
        Function(Function),
        Fallback { payable: bool },
        Receive,
    }

    let senders = Rc::new(senders);
    let dictionary_weight = config.dictionary.dictionary_weight;

    // Strategy to generate values for tx warp and roll.
    let warp_roll_strat = |cond: bool| {
        if cond { any::<U256>().prop_map(Some).boxed() } else { Just(None).boxed() }
    };

    any::<prop::sample::Selector>()
        .prop_flat_map(move |selector| {
            let contracts = contracts.targets();
            let calls = contracts.iter().flat_map(|(address, contract)| {
                let functions = contract
                    .abi_fuzzed_functions()
                    .cloned()
                    .map(|func| (*address, Call::Function(func)));
                let special = contract.targeted_functions.is_empty().then(|| {
                    contract.abi.receive.map(|_| (*address, Call::Receive)).into_iter().chain(
                        contract.abi.fallback.map(|fallback| {
                            (
                                *address,
                                Call::Fallback {
                                    payable: fallback.state_mutability == StateMutability::Payable,
                                },
                            )
                        }),
                    )
                });
                functions.chain(special.into_iter().flatten())
            });
            let (target_address, target_call) = selector.select(calls);

            let sender = select_random_sender(&fuzz_state, senders.clone(), dictionary_weight);

            let call_details = match target_call {
                Call::Function(func) => {
                    fuzz_contract_with_calldata(&fuzz_state, &fuzz_fixtures, target_address, func)
                        .boxed()
                }
                Call::Fallback { payable } => {
                    let value = if payable { fuzz_msg_value().boxed() } else { Just(None).boxed() };
                    value
                        .prop_map(move |value| CallDetails {
                            target: target_address,
                            calldata: Bytes::from_static(&[0]),
                            value,
                        })
                        .boxed()
                }
                Call::Receive => fuzz_msg_value()
                    .prop_map(move |value| CallDetails {
                        target: target_address,
                        calldata: Bytes::new(),
                        value,
                    })
                    .boxed(),
            };

            let warp = warp_roll_strat(config.max_time_delay.is_some());
            let roll = warp_roll_strat(config.max_block_delay.is_some());

            (warp, roll, sender, call_details)
        })
        .prop_map(move |(warp, roll, sender, call_details)| {
            let warp =
                warp.map(|time| time % U256::from(config.max_time_delay.unwrap_or_default()));
            let roll =
                roll.map(|block| block % U256::from(config.max_block_delay.unwrap_or_default()));
            BasicTxDetails { warp, roll, sender, call_details }
        })
}

/// Strategy to select a sender address:
/// * If `senders` is empty, then it's either a random address (10%) or from the dictionary (90%).
/// * If `senders` is not empty, a random address is chosen from the list of senders.
fn select_random_sender<S: FuzzStateReader>(
    fuzz_state: &S,
    senders: Rc<SenderFilters>,
    dictionary_weight: u32,
) -> impl Strategy<Value = Address> + use<S> {
    if senders.targeted.is_empty() {
        assert!(dictionary_weight <= 100, "dictionary_weight must be <= 100");
        proptest::prop_oneof![
            100 - dictionary_weight => fuzz_param(&alloy_dyn_abi::DynSolType::Address),
            dictionary_weight => fuzz_param_from_state(&alloy_dyn_abi::DynSolType::Address, fuzz_state),
        ]
        .prop_map(move |addr| {
            let mut addr = addr.as_address().unwrap();
            // Make sure the selected address is not in the list of excluded senders.
            // We don't use proptest's filter to avoid reaching the `PROPTEST_MAX_LOCAL_REJECTS`
            // max rejects and exiting test before all runs completes.
            // See <https://github.com/foundry-rs/foundry/issues/11369>.
            loop {
                if !senders.excluded.contains(&addr) {
                    break;
                }
                addr = Address::random();
            }
            addr
        })
        .boxed()
    } else {
        any::<prop::sample::Index>().prop_map(move |index| *index.get(&senders.targeted)).boxed()
    }
}

/// Given a function, it returns a proptest strategy which generates valid abi-encoded calldata
/// for that function's input types.
pub fn fuzz_contract_with_calldata<S: FuzzStateReader>(
    fuzz_state: &S,
    fuzz_fixtures: &FuzzFixtures,
    target: Address,
    func: Function,
) -> impl Strategy<Value = CallDetails> + use<S> {
    let is_payable = func.state_mutability == alloy_json_abi::StateMutability::Payable;

    // We need to compose all the strategies generated for each parameter in all possible
    // combinations.
    // `prop_oneof!` / `TupleUnion` `Arc`s for cheap cloning.
    let calldata_strategy = prop_oneof![
        60 => fuzz_calldata(func.clone(), fuzz_fixtures),
        40 => fuzz_calldata_from_state(func, fuzz_state),
    ];

    // For payable functions, generate random value using shared strategy.
    let value_strategy = if is_payable { fuzz_msg_value().boxed() } else { Just(None).boxed() };

    (calldata_strategy, value_strategy).prop_map(move |(calldata, value)| {
        trace!(input=?calldata, ?value);
        CallDetails { target, calldata, value }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariant::{TargetedContract, TargetedContracts};
    use alloy_json_abi::{Fallback, JsonAbi, Receive};
    use proptest::{
        strategy::ValueTree,
        test_runner::{Config, TestRunner},
    };

    fn tx_for_abi(abi: JsonAbi) -> BasicTxDetails {
        let target = Address::repeat_byte(0x11);
        let mut targets = TargetedContracts::new();
        targets.inner.insert(target, TargetedContract::new("Target".to_string(), abi));
        let contracts = FuzzRunIdentifiedContracts::new(targets, /* is_updatable */ false);
        let mut runner =
            TestRunner::new(Config { failure_persistence: None, ..Default::default() });

        invariant_strat(
            EvmFuzzState::test().into(),
            SenderFilters::new(
                /* targeted */ vec![Address::repeat_byte(0x22)],
                /* excluded */ vec![],
            ),
            contracts,
            InvariantConfig::default(),
            FuzzFixtures::default(),
        )
        .new_tree(&mut runner)
        .unwrap()
        .current()
    }

    #[test]
    fn invariant_strat_can_call_receive() {
        let mut abi = JsonAbi::new();
        abi.receive = Some(Receive { state_mutability: StateMutability::Payable });
        let tx = tx_for_abi(abi);

        assert_eq!(tx.call_details.target, Address::repeat_byte(0x11));
        assert!(tx.call_details.calldata.is_empty());
    }

    #[test]
    fn invariant_strat_can_call_fallback() {
        let mut abi = JsonAbi::new();
        abi.fallback = Some(Fallback { state_mutability: StateMutability::NonPayable });
        let tx = tx_for_abi(abi);

        assert_eq!(tx.call_details.target, Address::repeat_byte(0x11));
        assert_eq!(tx.call_details.calldata, Bytes::from_static(&[0]));
        assert_eq!(tx.call_details.value, None);
    }
}
