use crate::{
    FuzzFixtures,
    strategies::{FuzzStateReader, fuzz_param_from_state, fuzz_param_with_fixtures},
};
use alloy_dyn_abi::JsonAbiExt;
use alloy_json_abi::Function;
use alloy_primitives::Bytes;
use rand::RngCore;

/// Closure-style "strategy": yields a fresh calldata blob per call.
pub type CalldataStrategy = Box<dyn FnMut(&mut dyn RngCore) -> Bytes>;

/// Given a function, it returns a generator which produces valid calldata
/// for that function's input types, following declared test fixtures.
pub fn fuzz_calldata(func: Function, fuzz_fixtures: &FuzzFixtures) -> CalldataStrategy {
    let mut strats = func
        .inputs
        .iter()
        .map(|input| {
            fuzz_param_with_fixtures(
                &input.selector_type().parse().unwrap(),
                fuzz_fixtures.param_fixtures(&input.name),
                &input.name,
            )
        })
        .collect::<Vec<_>>();
    Box::new(move |rng| {
        let values: Vec<_> = strats.iter_mut().map(|s| s(rng)).collect();
        func.abi_encode_input(&values)
            .unwrap_or_else(|_| {
                panic!(
                    "Fuzzer generated invalid arguments for function `{}` with inputs {:?}: {:?}",
                    func.name, func.inputs, values
                )
            })
            .into()
    })
}

/// Given a function and some state, it returns a generator which produces valid calldata for the
/// given function's input types, based on state taken from the EVM.
pub fn fuzz_calldata_from_state<S: FuzzStateReader>(func: Function, state: &S) -> CalldataStrategy {
    let mut strats = func
        .inputs
        .iter()
        .map(|input| fuzz_param_from_state(&input.selector_type().parse().unwrap(), state))
        .collect::<Vec<_>>();
    Box::new(move |rng| {
        let values: Vec<_> = strats.iter_mut().map(|s| s(rng)).collect();
        func.abi_encode_input(&values)
            .unwrap_or_else(|_| {
                panic!(
                    "Fuzzer generated invalid arguments for function `{}` with inputs {:?}: {:?}",
                    func.name, func.inputs, values
                )
            })
            .into()
    })
}

#[cfg(test)]
mod tests {
    use crate::{FuzzFixtures, strategies::fuzz_calldata};
    use abi_fuzz::Runner;
    use alloy_dyn_abi::{DynSolValue, JsonAbiExt};
    use alloy_json_abi::Function;
    use alloy_primitives::{Address, map::HashMap};

    #[test]
    fn can_fuzz_with_fixtures() {
        let function = Function::parse("test_fuzzed_address(address addressFixture)").unwrap();

        let address_fixture = DynSolValue::Address(Address::random());
        let mut fixtures = HashMap::default();
        // FuzzFixtures lowercases the lookup key (see `normalize_fixture`),
        // so the inserted key must already be lowercase.
        fixtures.insert(
            "addressfixture".to_string(),
            DynSolValue::Array(vec![address_fixture.clone()]),
        );

        let expected = function.abi_encode_input(&[address_fixture]).unwrap();
        let mut strategy = fuzz_calldata(function, &FuzzFixtures::new(fixtures));
        let mut runner = Runner::seeded([0u8; 32]);
        // The fixture is picked ~50% of the time; sampling many times must hit it.
        let saw_fixture = (0..256).any(|_| strategy(runner.rng()) == expected);
        assert!(saw_fixture, "fixture was never selected from the configured pool");
    }
}
