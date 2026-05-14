use super::state::{FuzzDictionary, FuzzStateReader};
use crate::invariant::SenderFilters;
use abi_fuzz::{
    Generator, Mutator,
    generators::{IntDistribution, IntGenerator, RandomGenerator, UintDistribution, UintGenerator},
    mutators::{self as fz, RecursiveMutator},
};
use alloy_dyn_abi::{DynSolType, DynSolValue};
use alloy_primitives::{Address, B256, I256, U256};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};

/// The max length of arrays we fuzz for is 256.
const MAX_ARRAY_LEN: usize = 256;
/// The max length of bytes/strings we fuzz for is 32.
const MAX_BYTES_LEN: usize = 32;

/// Closure-style "strategy": yields a fresh [`DynSolValue`] per call given an RNG.
pub type ParamStrategy = Box<dyn FnMut(&mut dyn RngCore) -> DynSolValue>;

/// Given a parameter type, returns a generator for values of that type.
pub fn fuzz_param(param: &DynSolType) -> ParamStrategy {
    fuzz_param_inner(param, None)
}

/// Given a parameter type and configured fixtures for param name, returns a generator for values of
/// that type.
///
/// Fixtures can be currently generated for uint, int, address, bytes and
/// string types and are defined for parameter name.
/// For example, fixtures for parameter `owner` of type `address` can be defined in a function with
/// a `function fixture_owner() public returns (address[] memory)` signature.
///
/// Fixtures are matched on parameter name, hence fixtures defined in
/// `fixture_owner` function can be used in a fuzzed test function with a signature like
/// `function testFuzz_ownerAddress(address owner, uint amount)`.
///
/// Logs an error if all the fixture types are not of the same type as the input parameter, and
/// falls back to plain random generation in that case.
///
/// Works with ABI Encoder v2 tuples.
pub fn fuzz_param_with_fixtures(
    param: &DynSolType,
    fixtures: Option<&[DynSolValue]>,
    name: &str,
) -> ParamStrategy {
    fuzz_param_inner(param, fixtures.map(|f| (f, name)))
}

fn fuzz_param_inner(
    param: &DynSolType,
    mut fuzz_fixtures: Option<(&[DynSolValue], &str)>,
) -> ParamStrategy {
    if let Some((fixtures, name)) = fuzz_fixtures
        && !fixtures.iter().all(|f| f.matches(param))
    {
        error!("fixtures for {name:?} do not match type {param}");
        fuzz_fixtures = None;
    }
    let fixtures: Option<Vec<DynSolValue>> = fuzz_fixtures.map(|(f, _)| f.to_vec());

    // Configure the generator's int/uint distribution from the param + fixtures. For non-int
    // types the distribution is unused but the configuration is uniform anyway.
    let (int_dist, uint_dist) = match *param {
        DynSolType::Int(n @ 8..=256) => (
            IntDistribution::Mixed(IntGenerator::new(n, int_fixtures(n, fixtures.as_deref()))),
            UintDistribution::Uniform,
        ),
        DynSolType::Uint(n @ 8..=256) => (
            IntDistribution::Uniform,
            UintDistribution::Mixed(UintGenerator::new(n, uint_fixtures(n, fixtures.as_deref()))),
        ),
        _ => (IntDistribution::Uniform, UintDistribution::Uniform),
    };
    let mut g = RandomGenerator {
        max_array_len: MAX_ARRAY_LEN,
        max_bytes_len: MAX_BYTES_LEN,
        int: int_dist,
        uint: uint_dist,
        ..Default::default()
    };
    let param = param.clone();
    Box::new(move |rng: &mut dyn RngCore| {
        let value = if let Some(ref f) = fixtures
            && !f.is_empty()
            && rng.random_ratio(50, 100)
        {
            f[rng.random_range(0..f.len())].clone()
        } else {
            g.generate(&param, rng)
        };
        if let DynSolValue::String(s) = value {
            return DynSolValue::String(s.trim().trim_end_matches('\0').to_string());
        }
        value
    })
}

/// Filter `fixtures` to those that decode as `Int(bits)`. Mismatches log an error.
fn int_fixtures(bits: usize, fixtures: Option<&[DynSolValue]>) -> Vec<I256> {
    fixtures
        .into_iter()
        .flatten()
        .filter_map(|v| match v.as_int() {
            Some((i, w)) if w == bits => Some(i),
            _ => {
                error!("{v:?} is not a valid {} fixture", DynSolType::Int(bits));
                None
            }
        })
        .collect()
}

/// Filter `fixtures` to those that decode as `Uint(bits)`. Mismatches log an error.
fn uint_fixtures(bits: usize, fixtures: Option<&[DynSolValue]>) -> Vec<U256> {
    fixtures
        .into_iter()
        .flatten()
        .filter_map(|v| match v.as_uint() {
            Some((u, w)) if w == bits => Some(u),
            _ => {
                error!("{v:?} is not a valid {} fixture", DynSolType::Uint(bits));
                None
            }
        })
        .collect()
}

/// Given a parameter type, returns a generator for values of that type, given some EVM
/// fuzz state.
///
/// Composites recurse through abi-fuzz's [`RandomGenerator`], which consults the attached
/// [`EvmFuzzStateDict`] at every leaf. The dict is responsible for type-routing: see
/// [`EvmFuzzStateDict::sample`] for AST literal / typed-bucket / raw-state-value selection
/// logic.
///
/// Note: this preserves foundry's legacy "always consult dict, fall to random only on
/// `None`" behavior (`dict_bias = 100`). Echidna's default by comparison is `dictFreq = 0.40`
/// (40% dict / 60% random); switching to that requires lowering `dict_bias` here.
///
/// Works with ABI Encoder v2 tuples.
pub fn fuzz_param_from_state<S: FuzzStateReader>(param: &DynSolType, state: &S) -> ParamStrategy {
    let mut g = StateBackedGenerator::new(state);
    let param = param.clone();
    Box::new(move |rng: &mut dyn RngCore| {
        let value = g.generate(&param, rng);
        // Foundry-side post-process: trim whitespace + trailing NULs from random Strings.
        // (Dict-sourced Strings already pass through untouched per AST harvesting.)
        if let DynSolValue::String(s) = value {
            return DynSolValue::String(s.trim().trim_end_matches('\0').to_string());
        }
        value
    })
}

/// Foundry's dictionary adapter over [`EvmFuzzState`].
///
/// Routes per-type into the right pool (Echidna: leaf-level oracle):
/// - `Address`/`FixedBytes(n)`/`Uint(n)`/`Int(n)` → 50/50 typed-samples vs. raw state words, with
///   appropriate decoding (deployed-libs avoidance for `Address`, modular wrap for `Uint`,
///   sign-extension for `Int`, hi-byte mask for `FixedBytes`).
/// - `String` → 30% chance of an AST string literal; otherwise `None` → fall to random gen.
/// - `Bytes` → 10% AST string + 20% AST bytes + raw word fallback.
/// - `Bool`/`Function`/composites → `None` (defer to random generation; composites get their leaves
///   dict-biased recursively via the generator).
#[derive(Clone)]
struct EvmFuzzStateDict<S: FuzzStateReader> {
    state: S,
}

impl<S: FuzzStateReader> EvmFuzzStateDict<S> {
    fn sample(&self, ty: &DynSolType, rng: &mut dyn RngCore) -> Option<DynSolValue> {
        self.state.with_dictionary(|dict| match ty {
            DynSolType::Address => {
                let word = sample_dict_word(dict, ty, rng)?;
                let mut addr = Address::from_word(word);
                if self.state.deployed_libs().contains(&addr) {
                    let mut local_rng = StdRng::seed_from_u64(0x1337);
                    loop {
                        addr.randomize_with(&mut local_rng);
                        if !self.state.deployed_libs().contains(&addr) {
                            break;
                        }
                    }
                }
                Some(DynSolValue::Address(addr))
            }
            DynSolType::FixedBytes(size @ 1..=32) => {
                let mut word = sample_dict_word(dict, ty, rng)?;
                word.0[*size..].fill(0);
                Some(DynSolValue::FixedBytes(word, *size))
            }
            DynSolType::String => {
                if rng.random_ratio(3, 10) {
                    let strings = dict.ast_strings();
                    if !strings.is_empty() {
                        let s = &strings.as_slice()[rng.random_range(0..strings.len())];
                        return Some(DynSolValue::String(s.clone()));
                    }
                }
                None
            }
            DynSolType::Bytes => {
                if rng.random_ratio(1, 10) {
                    let strings = dict.ast_strings();
                    if !strings.is_empty() {
                        let s = &strings.as_slice()[rng.random_range(0..strings.len())];
                        return Some(DynSolValue::Bytes(s.as_bytes().to_vec()));
                    }
                }
                if rng.random_ratio(2, 10) {
                    let bytes = dict.ast_bytes();
                    if !bytes.is_empty() {
                        let b = &bytes.as_slice()[rng.random_range(0..bytes.len())];
                        return Some(DynSolValue::Bytes(b.to_vec()));
                    }
                }
                let word = sample_dict_word(dict, ty, rng)?;
                Some(DynSolValue::Bytes(word.0.into()))
            }
            DynSolType::Int(n @ 8..=256) => {
                let n = *n;
                let word = sample_dict_word(dict, ty, rng)?;
                let value = if n / 8 == 32 {
                    I256::from_raw(U256::from_be_bytes(word.0))
                } else {
                    let uint_n = U256::from_be_bytes(word.0) % U256::from(1).wrapping_shl(n);
                    let sign_bit = U256::from(1) << (n - 1);
                    if uint_n >= sign_bit {
                        let modulus = U256::from(1) << n;
                        I256::from_raw(uint_n.wrapping_sub(modulus))
                    } else {
                        I256::from_raw(uint_n)
                    }
                };
                Some(DynSolValue::Int(value, n))
            }
            DynSolType::Uint(n @ 8..=256) => {
                let n = *n;
                let word = sample_dict_word(dict, ty, rng)?;
                let value = if n / 8 == 32 {
                    U256::from_be_bytes(word.0)
                } else {
                    U256::from_be_bytes(word.0) % U256::from(1).wrapping_shl(n)
                };
                Some(DynSolValue::Uint(value, n))
            }
            // Bool/Function/composites: defer to random generation.
            _ => None,
        })
    }
}

/// 50/50 typed-samples vs. raw state-values; returns `None` when both pools are empty.
fn sample_dict_word(dict: &FuzzDictionary, ty: &DynSolType, rng: &mut dyn RngCore) -> Option<B256> {
    let bias = rng.random_ratio(1, 2);
    let values =
        if bias { dict.samples(ty) } else { None }.unwrap_or_else(|| dict.values()).as_slice();
    if values.is_empty() { None } else { Some(values[rng.random_range(0..values.len())]) }
}

/// Selects a random address for mutation, biased toward targeted senders when
/// available.
///
/// Priority:
/// 1. If `senders` has targeted addresses, pick uniformly from those.
/// 2. Otherwise, pick uniformly from the EVM fuzz dictionary's state values.
///
/// Returns `None` when the only candidate equals `current` (so callers know to
/// retry / fall back). Excluded-sender fixup is **not** done here — it happens
/// once at the run-loop boundary via [`SenderFilters::resolve`].
pub(crate) fn select_random_address(
    current: Address,
    test_runner: &mut abi_fuzz::Runner,
    state: &impl FuzzStateReader,
    senders: Option<&SenderFilters>,
) -> Option<Address> {
    if let Some(senders) = senders
        && !senders.targeted.is_empty()
    {
        let index = test_runner.rng().random_range(0..senders.targeted.len());
        let addr = senders.targeted[index];
        return (addr != current).then_some(addr);
    }

    state.with_dictionary(|dict| {
        let values = dict.values();
        if values.is_empty() {
            return None;
        }
        let index = test_runner.rng().random_range(0..values.len());
        let addr = Address::from_word(values[index]);
        (addr != current).then_some(addr)
    })
}

/// Mutates the current value of the given parameter type and value.
pub fn mutate_param_value(
    param: &DynSolType,
    value: DynSolValue,
    test_runner: &mut abi_fuzz::Runner,
    state: &impl FuzzStateReader,
) -> DynSolValue {
    mutate_param_value_inner(param, value, test_runner, state, None)
}

/// Mutates the current value of the given parameter type and value, with optional sender filters.
///
/// When `senders` is provided and has targeted addresses, address mutations will prefer
/// selecting from those targeted addresses (similar to `select_random_sender` behavior).
pub fn mutate_param_value_with_senders(
    param: &DynSolType,
    value: DynSolValue,
    test_runner: &mut abi_fuzz::Runner,
    state: &impl FuzzStateReader,
    senders: &SenderFilters,
) -> DynSolValue {
    mutate_param_value_inner(param, value, test_runner, state, Some(senders))
}

/// Foundry-side fallback [`Generator`] for the recursive mutator: a thin
/// wrapper over [`RandomGenerator`] that uses Foundry's array-length cap and
/// the EVM dictionary for "splice in a fresh value" fallbacks.
struct StateBackedGenerator<S: FuzzStateReader> {
    dict: EvmFuzzStateDict<S>,
    inner: RandomGenerator,
}

impl<S: FuzzStateReader> StateBackedGenerator<S> {
    fn new(state: &S) -> Self {
        Self {
            dict: EvmFuzzStateDict { state: state.clone() },
            inner: RandomGenerator {
                max_array_len: MAX_ARRAY_LEN,
                max_bytes_len: MAX_BYTES_LEN,
                int: IntDistribution::Uniform,
                uint: UintDistribution::Uniform,
                ..Default::default()
            },
        }
    }
}

impl<S: FuzzStateReader> Generator for StateBackedGenerator<S> {
    fn generate(&mut self, ty: &DynSolType, rng: &mut dyn RngCore) -> DynSolValue {
        if let Some(value) = self.dict.sample(ty, rng) {
            return value;
        }
        self.inner.generate(ty, rng)
    }
}

fn mutate_param_value_inner(
    param: &DynSolType,
    value: DynSolValue,
    test_runner: &mut abi_fuzz::Runner,
    state: &impl FuzzStateReader,
    senders: Option<&SenderFilters>,
) -> DynSolValue {
    // Top-level Address gets foundry's dict + senders bias; everything else
    // falls through to the abi-fuzz recursive mutator (which itself bottoms out
    // in `StateBackedGenerator` for "splice in a fresh value" fallbacks).
    if let DynSolValue::Address(val) = value {
        return match test_runner.rng().random_range(0..=5u32) {
            // Replace with a random address from targeted senders or dictionary.
            4 => select_random_address(val, test_runner, state, senders),
            5 => None,
            _ => fz::mutate_address(val, test_runner.rng()), /* TODO cycle through fixed list
                                                              * most of the time? */
        }
        .map(DynSolValue::Address)
        .unwrap_or_else(|| StateBackedGenerator::new(state).generate(param, test_runner.rng()));
    }

    let mut mutator = RecursiveMutator {
        generator: StateBackedGenerator::new(state),
        max_array_len: MAX_ARRAY_LEN,
    };
    mutator.mutate(param, value, test_runner.rng())
}

/// 0.001 ETH in wei.
const MILLI_ETH: u64 = 1_000_000_000_000_000;
/// 1 ETH in wei.
const ONE_ETH: u64 = 1_000_000_000_000_000_000;

/// Closure-style "strategy" yielding an optional `msg.value` per call.
pub type MsgValueStrategy = Box<dyn FnMut(&mut dyn RngCore) -> Option<U256>>;

/// Returns a closure yielding random `msg.value` for payable functions, biased
/// toward smaller values to avoid balance issues.
///
/// Distribution:
/// - 85% chance: no value (None)
/// - 10% chance: small values (0-1000 wei)
/// - 4% chance: medium values (up to 0.001 ETH)
/// - 1% chance: larger values (up to 1 ETH)
pub fn fuzz_msg_value() -> MsgValueStrategy {
    Box::new(|rng: &mut dyn RngCore| match rng.random_range(0..100u32) {
        // 85% chance: no value.
        0..=84 => None,
        // 10% chance: small values (0-1000 wei).
        85..=94 => Some(U256::from(rng.random_range(0u64..=1000))),
        // 4% chance: medium values (up to 0.001 ETH).
        95..=98 => Some(U256::from(rng.random_range(0u64..=MILLI_ETH))),
        // 1% chance: larger values (up to 1 ETH).
        _ => Some(U256::from(rng.random_range(0u64..=ONE_ETH))),
    })
}

/// Generates a random `msg.value` for payable functions, biased toward smaller
/// values to avoid balance issues.
///
/// Distribution:
/// - 60% chance: small values (0-1000 wei)
/// - 30% chance: medium values (up to 0.001 ETH)
/// - 9% chance: larger values (up to 1 ETH)
/// - 1% chance: max value (edge case)
pub fn generate_msg_value(test_runner: &mut abi_fuzz::Runner) -> U256 {
    let rng = test_runner.rng();
    match rng.random_range(0..=10) {
        // Small values (0-1000 wei) - 60% chance.
        0..=5 => U256::from(rng.random_range(0u64..=1000)),
        // Medium values (up to 0.001 ETH) - 30% chance.
        6..=8 => U256::from(rng.random_range(0u64..=MILLI_ETH)),
        // Larger values (up to 1 ETH) - 9% chance.
        9 => U256::from(rng.random_range(0u64..=ONE_ETH)),
        // Edge case (max) - 1% chance.
        _ => U256::MAX,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        FuzzFixtures,
        strategies::{EvmFuzzState, fuzz_calldata, fuzz_calldata_from_state},
    };
    use abi_fuzz::Runner;
    use alloy_primitives::B256;
    use foundry_common::abi::get_func;
    use rand::Rng;
    use std::collections::HashSet;

    #[test]
    fn can_fuzz_array() {
        let f = "testArray(uint64[2] calldata values)";
        let func = get_func(f).unwrap();
        let state = EvmFuzzState::test();
        let mut runner = Runner::seeded([0u8; 32]);
        let mut a = fuzz_calldata(func.clone(), &FuzzFixtures::default());
        let mut b = fuzz_calldata_from_state(func, &state);
        for _ in 0..32 {
            if runner.rng().random_ratio(60, 100) {
                let _ = a(runner.rng());
            } else {
                let _ = b(runner.rng());
            }
        }
    }

    #[test]
    fn can_fuzz_string_and_bytes_with_ast_literals_and_hashes() {
        use super::fuzz_param_from_state;
        use crate::strategies::LiteralMaps;
        use alloy_dyn_abi::DynSolType;
        use alloy_primitives::keccak256;

        // Seed dict with string values and their hashes --> mimic `CheatcodeAnalysis` behavior.
        let mut literals = LiteralMaps::default();
        literals.strings.insert("hello".to_string());
        literals.strings.insert("world".to_string());
        literals.words.entry(DynSolType::FixedBytes(32)).or_default().insert(keccak256("hello"));
        literals.words.entry(DynSolType::FixedBytes(32)).or_default().insert(keccak256("world"));

        let mut state = EvmFuzzState::test();
        state.seed_literals(literals);

        let mut runner = Runner::seeded([0u8; 32]);

        // Verify generators produce the seeded AST literals
        let mut generated_bytes = HashSet::new();
        let mut generated_hashes = HashSet::new();
        let mut generated_strings = HashSet::new();
        let mut bytes_strategy = fuzz_param_from_state(&DynSolType::Bytes, &state);
        let mut string_strategy = fuzz_param_from_state(&DynSolType::String, &state);
        let mut bytes32_strategy = fuzz_param_from_state(&DynSolType::FixedBytes(32), &state);

        for _ in 0..256 {
            let v = bytes_strategy(runner.rng());
            if let Some(bytes) = v.as_bytes()
                && let Ok(s) = std::str::from_utf8(bytes)
            {
                generated_bytes.insert(s.to_string());
            }

            let v = string_strategy(runner.rng());
            if let Some(s) = v.as_str() {
                generated_strings.insert(s.to_string());
            }

            let v = bytes32_strategy(runner.rng());
            if let Some((bytes, size)) = v.as_fixed_bytes()
                && size == 32
            {
                generated_hashes.insert(B256::from_slice(bytes));
            }
        }

        assert!(generated_bytes.contains("hello"));
        assert!(generated_bytes.contains("world"));
        assert!(generated_strings.contains("hello"));
        assert!(generated_strings.contains("world"));
        assert!(generated_hashes.contains(&keccak256("hello")));
        assert!(generated_hashes.contains(&keccak256("world")));
    }

    #[test]
    fn mutate_address_can_select_from_dictionary() {
        use super::mutate_param_value;
        use alloy_dyn_abi::{DynSolType, DynSolValue};
        use alloy_primitives::Address;

        let mut state = EvmFuzzState::test();

        // Add addresses to dictionary via state values.
        let addr1 = Address::repeat_byte(0x11);
        let addr2 = Address::repeat_byte(0x22);
        let addr3 = Address::repeat_byte(0x33);
        state.collect_values([addr1.into_word(), addr2.into_word(), addr3.into_word()]);

        let mut runner = Runner::seeded([0u8; 32]);

        // Mutate an address many times and verify we can get addresses from the dictionary.
        let original = Address::repeat_byte(0xff);
        let mut got_addr1 = false;
        let mut got_addr2 = false;
        let mut got_addr3 = false;

        for _ in 0..1000 {
            let mutated = mutate_param_value(
                &DynSolType::Address,
                DynSolValue::Address(original),
                &mut runner,
                &state,
            );
            if let DynSolValue::Address(addr) = mutated {
                if addr == addr1 {
                    got_addr1 = true;
                }
                if addr == addr2 {
                    got_addr2 = true;
                }
                if addr == addr3 {
                    got_addr3 = true;
                }
            }
            if got_addr1 && got_addr2 && got_addr3 {
                break;
            }
        }

        // We should have seen at least one dictionary address in 1000 iterations.
        assert!(
            got_addr1 || got_addr2 || got_addr3,
            "Address mutation should select addresses from dictionary"
        );
    }

    #[test]
    fn mutate_address_prefers_targeted_senders() {
        use super::select_random_address;
        use crate::invariant::SenderFilters;
        use alloy_primitives::Address;

        let mut state = EvmFuzzState::test();

        // Add addresses to dictionary (these should NOT be selected when targeted is set).
        let dict_addr = Address::repeat_byte(0xdd);
        state.collect_values([dict_addr.into_word()]);

        // Set up targeted senders.
        let targeted1 = Address::repeat_byte(0x11);
        let targeted2 = Address::repeat_byte(0x22);
        let senders = SenderFilters::new(vec![targeted1, targeted2], vec![]);

        let mut runner = Runner::seeded([0u8; 32]);

        // Call select_random_address directly to verify it uses targeted senders.
        let original = Address::repeat_byte(0xff);
        let mut got_targeted1 = false;
        let mut got_targeted2 = false;
        let mut got_dict = false;

        for _ in 0..100 {
            if let Some(addr) = select_random_address(original, &mut runner, &state, Some(&senders))
            {
                if addr == targeted1 {
                    got_targeted1 = true;
                }
                if addr == targeted2 {
                    got_targeted2 = true;
                }
                if addr == dict_addr {
                    got_dict = true;
                }
            }
        }

        // Should see targeted addresses, never dictionary address.
        assert!(
            got_targeted1 || got_targeted2,
            "select_random_address should select from targeted senders"
        );
        assert!(
            !got_dict,
            "select_random_address should not select from dictionary when targeted senders are set"
        );
    }

    #[test]
    fn excluded_senders_are_remapped_to_fallback_at_execution_boundary() {
        use crate::invariant::{FALLBACK_SENDER, SenderFilters};
        use alloy_primitives::Address;

        let allowed = Address::repeat_byte(0x11);
        let excluded = Address::repeat_byte(0xee);
        let filters = SenderFilters::new(vec![], vec![excluded]);

        // Allowed addresses pass through unchanged…
        assert_eq!(filters.resolve(allowed), allowed);
        // …while excluded addresses fall back to the deterministic 0x30000.
        assert_eq!(filters.resolve(excluded), FALLBACK_SENDER);
        // address(0) is excluded by default per `SenderFilters::new`.
        assert_eq!(filters.resolve(Address::ZERO), FALLBACK_SENDER);
    }
}
