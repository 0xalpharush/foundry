use super::InvariantContract;
use crate::executors::RawCallResult;
use alloy_primitives::{Address, Bytes};
use foundry_evm_core::decode::RevertDecoder;
use foundry_evm_fuzz::{BasicTxDetails, Reason, invariant::FuzzRunIdentifiedContracts};
use proptest::test_runner::TestError;
use serde::Serialize;
use std::{collections::HashMap, fmt};

/// Type-safe key for invariant/assertion failures in format `"ContractName:function_name"`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct FailureKey(String);

impl FailureKey {
    /// Create a new failure key from contract name and function name.
    pub fn new(contract_name: &str, fn_name: &str) -> Self {
        Self(format!("{contract_name}:{fn_name}"))
    }

    /// The contract name portion (before the `:`).
    pub fn contract_name(&self) -> &str {
        self.0.split(':').next().unwrap_or(&self.0)
    }

    /// The function name portion (after the last `:`).
    pub fn function_name(&self) -> &str {
        self.0.rsplit(':').next().unwrap_or(&self.0)
    }

    /// Whether this key ends with the given function name suffix.
    pub fn has_function(&self, fn_name: &str) -> bool {
        self.0.ends_with(&format!(":{fn_name}"))
    }

    /// The full key string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FailureKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stores information about failures and reverts of the invariant tests.
#[derive(Clone, Default)]
pub struct InvariantFailures {
    /// Total number of reverts.
    pub reverts: usize,
    /// The latest revert reason of a run.
    pub revert_reason: Option<String>,
    /// Maps a failure target key to its error.
    /// Unique failures are deduplicated by this key.
    pub errors: HashMap<FailureKey, InvariantFuzzError>,
    /// Total number of failures encountered (including duplicates).
    pub total_failures: usize,
}

impl InvariantFailures {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_inner(self) -> (usize, HashMap<FailureKey, InvariantFuzzError>) {
        (self.reverts, self.errors)
    }

    /// Record a failure with a typed key.
    pub fn record_failure(&mut self, key: FailureKey, failure: InvariantFuzzError) {
        self.total_failures += 1;
        self.errors.insert(key, failure);
    }

    /// Number of unique failures (deduplicated by target key).
    pub fn unique_failures(&self) -> usize {
        self.errors.len()
    }
}

impl fmt::Display for InvariantFailures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "      Failures: {} (unique: {})", self.total_failures, self.errors.len())?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum InvariantFuzzError {
    Revert(FailedInvariantCaseData),
    BrokenInvariant(FailedInvariantCaseData),
    BrokenAssertion(FailedInvariantCaseData),
    MaxAssumeRejects(u32),
}

impl InvariantFuzzError {
    pub fn revert_reason(&self) -> Option<String> {
        match self {
            Self::BrokenInvariant(case_data)
            | Self::BrokenAssertion(case_data)
            | Self::Revert(case_data) => {
                (!case_data.revert_reason.is_empty()).then(|| case_data.revert_reason.clone())
            }
            Self::MaxAssumeRejects(allowed) => {
                Some(format!("`vm.assume` rejected too many inputs ({allowed} allowed)"))
            }
        }
    }

    pub fn failure_type(&self) -> &'static str {
        match self {
            Self::BrokenInvariant(_) => "invariant",
            Self::BrokenAssertion(_) => "assertion",
            Self::Revert(_) => "revert",
            Self::MaxAssumeRejects(_) => "assume_rejects",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FailedInvariantCaseData {
    /// The proptest error occurred as a result of a test case.
    pub test_error: TestError<Vec<BasicTxDetails>>,
    /// The return reason of the offending call.
    pub return_reason: Reason,
    /// The revert string of the offending call.
    pub revert_reason: String,
    /// Address of the invariant asserter.
    pub addr: Address,
    /// Function calldata for invariant check.
    pub calldata: Bytes,
    /// Inner fuzzing Sequence coming from overriding calls.
    pub inner_sequence: Vec<Option<BasicTxDetails>>,
    /// Shrink run limit
    pub shrink_run_limit: u32,
    /// Fail on revert, used to check sequence when shrinking.
    pub fail_on_revert: bool,
    /// Fail on Solidity assert failures, used to check sequence when shrinking.
    pub fail_on_assert: bool,
}

impl FailedInvariantCaseData {
    pub fn new(
        invariant_contract: &InvariantContract<'_>,
        shrink_run_limit: u32,
        fail_on_revert: bool,
        fail_on_assert: bool,
        targeted_contracts: &FuzzRunIdentifiedContracts,
        calldata: &[BasicTxDetails],
        call_result: &RawCallResult,
        inner_sequence: &[Option<BasicTxDetails>],
    ) -> Self {
        // Collect abis of fuzzed and invariant contracts to decode custom error.
        let revert_reason = RevertDecoder::new()
            .with_abis(targeted_contracts.targets.lock().values().map(|c| &c.abi))
            .with_abi(invariant_contract.abi)
            .decode(call_result.result.as_ref(), call_result.exit_reason);

        let func = invariant_contract.invariant_fn;
        debug_assert!(func.inputs.is_empty());
        let origin = func.name.as_str();
        Self {
            test_error: TestError::Fail(
                format!("{origin}, reason: {revert_reason}").into(),
                calldata.to_vec(),
            ),
            return_reason: "".into(),
            revert_reason,
            addr: invariant_contract.address,
            calldata: func.selector().to_vec().into(),
            inner_sequence: inner_sequence.to_vec(),
            shrink_run_limit,
            fail_on_revert,
            fail_on_assert,
        }
    }
}
