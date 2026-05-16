use crate::{BasicTxDetails, CallDetails};
use abi_fuzz::Runner;
use alloy_primitives::Address;
use parking_lot::{Mutex, RwLock};
use rand::{Rng, RngCore};
use std::{collections::HashSet, sync::Arc};

/// Closure-style strategy yielding an `Option<CallDetails>` per draw.
type WeightedCallStrategy = Box<dyn FnMut(&mut dyn RngCore) -> Option<CallDetails> + Send + Sync>;

/// Given a Runner and a strategy, it generates calls. Used inside the Fuzzer inspector to
/// override external calls to test for potential reentrancy vulnerabilities.
///
/// The key insight is that we only override calls TO handler contracts (targeted contracts).
/// This simulates a malicious contract that reenters when receiving ETH via its receive() function.
#[derive(Clone)]
pub struct RandomCallGenerator {
    /// Address of the test contract.
    pub test_address: Address,
    /// Addresses of handler contracts that can be reentered.
    /// We only inject callbacks when the call target is one of these.
    pub handler_addresses: Arc<RwLock<HashSet<Address>>>,
    /// Runner that drives the strategy.
    pub runner: Arc<Mutex<Runner>>,
    /// Strategy to be used to generate calls from `target_reference`. Stored
    /// behind a mutex so the cloneable [`RandomCallGenerator`] can mutate the
    /// underlying closure state when sampling.
    pub strategy: Arc<Mutex<WeightedCallStrategy>>,
    /// Reference to which contract we want a fuzzed calldata from.
    pub target_reference: Arc<RwLock<Address>>,
    /// Tracks the call depth when an override is active. When > 0, we're inside an overridden
    /// call and should not override nested calls. Incremented when we override a call,
    /// decremented when any call ends while inside an override.
    pub override_depth: usize,
    /// If set to `true`, consumes the next call from `last_sequence`, otherwise queries it from
    /// the strategy.
    pub replay: bool,
    /// Saves the sequence of generated calls that can be replayed later on.
    pub last_sequence: Arc<RwLock<Vec<Option<BasicTxDetails>>>>,
}

impl std::fmt::Debug for RandomCallGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomCallGenerator")
            .field("test_address", &self.test_address)
            .field("override_depth", &self.override_depth)
            .field("replay", &self.replay)
            .finish_non_exhaustive()
    }
}

impl RandomCallGenerator {
    pub fn new(
        test_address: Address,
        handler_addresses: HashSet<Address>,
        runner: Runner,
        mut strategy: Box<dyn FnMut(&mut dyn RngCore) -> CallDetails + Send + Sync>,
        target_reference: Arc<RwLock<Address>>,
    ) -> Self {
        // 90% Some(call), 10% None — same shape as the prior `weighted(0.9, ..)` adapter.
        let weighted: WeightedCallStrategy =
            Box::new(
                move |rng: &mut dyn RngCore| {
                    if rng.random_ratio(9, 10) { Some(strategy(rng)) } else { None }
                },
            );
        Self {
            test_address,
            handler_addresses: Arc::new(RwLock::new(handler_addresses)),
            runner: Arc::new(Mutex::new(runner)),
            strategy: Arc::new(Mutex::new(weighted)),
            target_reference,
            last_sequence: Arc::default(),
            replay: false,
            override_depth: 0,
        }
    }

    /// Check if the given address is a handler that can be reentered.
    pub fn is_handler(&self, address: Address) -> bool {
        self.handler_addresses.read().contains(&address)
    }

    /// All `self.next()` calls will now pop `self.last_sequence`. Used to replay an invariant
    /// failure.
    pub fn set_replay(&mut self, status: bool) {
        self.replay = status;
        if status {
            // So it can later be popped.
            self.last_sequence.write().reverse();
        }
    }

    /// Gets the next call. Random if replay is not set. Otherwise, it pops from `last_sequence`.
    pub fn next(
        &mut self,
        original_caller: Address,
        original_target: Address,
    ) -> Option<BasicTxDetails> {
        if self.replay {
            self.last_sequence.write().pop().expect(
                "to have same size as the number of (unsafe) external calls of the sequence.",
            )
        } else {
            // TODO: Do we want it to be 80% chance only too ?
            let sender = original_target;

            // Set which contract we mostly (80% chance) want to generate calldata from.
            *self.target_reference.write() = original_caller;

            // `original_caller` has a 80% chance of being the `new_target`.
            let mut runner = self.runner.lock();
            let mut strategy = self.strategy.lock();
            let choice = strategy(runner.rng()).map(|call_details| BasicTxDetails {
                warp: None,
                roll: None,
                deal: None,
                sender,
                call_details,
            });
            drop(strategy);
            drop(runner);

            self.last_sequence.write().push(choice.clone());
            choice
        }
    }
}
