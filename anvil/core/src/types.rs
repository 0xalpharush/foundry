use ethers_core::types::{H256, U256, U64};
use revm::SpecId;

#[cfg(feature = "serde")]
use serde::{de::Error, Deserializer, Serializer};

/// Represents the params to set forking which can take various forms
///  - untagged
///  - tagged `forking`
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Forking {
    pub json_rpc_url: Option<String>,
    pub block_number: Option<u64>,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Forking {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct ForkOpts {
            pub json_rpc_url: Option<String>,
            pub block_number: Option<u64>,
        }

        #[derive(serde::Deserialize)]
        struct Tagged {
            forking: ForkOpts,
        }
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum ForkingVariants {
            Tagged(Tagged),
            Fork(ForkOpts),
        }
        let f = match ForkingVariants::deserialize(deserializer)? {
            ForkingVariants::Fork(ForkOpts { json_rpc_url, block_number }) => {
                Forking { json_rpc_url, block_number }
            }
            ForkingVariants::Tagged(f) => Forking {
                json_rpc_url: f.forking.json_rpc_url,
                block_number: f.forking.block_number,
            },
        };
        Ok(f)
    }
}

/// Additional `evm_mine` options
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum EvmMineOptions {
    Options {
        timestamp: Option<u64>,
        // If `blocks` is given, it will mine exactly blocks number of blocks, regardless of any
        // other blocks mined or reverted during it's operation
        blocks: Option<u64>,
    },
    /// The timestamp the block should be mined with
    Timestamp(Option<u64>),
}

impl Default for EvmMineOptions {
    fn default() -> Self {
        EvmMineOptions::Options { timestamp: None, blocks: None }
    }
}

/// Represents the result of `eth_getWork`
/// This may or may not include the block number
#[derive(Debug, PartialEq, Eq, Default)]
pub struct Work {
    pub pow_hash: H256,
    pub seed_hash: H256,
    pub target: H256,
    pub number: Option<u64>,
}

#[cfg(feature = "serde")]
impl serde::Serialize for Work {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(num) = self.number {
            (&self.pow_hash, &self.seed_hash, &self.target, U256::from(num)).serialize(s)
        } else {
            (&self.pow_hash, &self.seed_hash, &self.target).serialize(s)
        }
    }
}

/// A hex encoded or decimal index
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Index(usize);

impl From<Index> for usize {
    fn from(idx: Index) -> Self {
        idx.0
    }
}

#[cfg(feature = "serde")]
impl<'a> serde::Deserialize<'a> for Index {
    fn deserialize<D>(deserializer: D) -> Result<Index, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        use std::fmt;

        struct IndexVisitor;

        impl<'a> serde::de::Visitor<'a> for IndexVisitor {
            type Value = Index;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "hex-encoded or decimal index")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Index(value as usize))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                if let Some(val) = value.strip_prefix("0x") {
                    usize::from_str_radix(val, 16).map(Index).map_err(|e| {
                        Error::custom(format!("Failed to parse hex encoded index value: {e}"))
                    })
                } else {
                    value
                        .parse::<usize>()
                        .map(Index)
                        .map_err(|e| Error::custom(format!("Failed to parse numeric index: {e}")))
                }
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: Error,
            {
                self.visit_str(value.as_ref())
            }
        }

        deserializer.deserialize_any(IndexVisitor)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeInfo {
    pub current_block_number: U64,
    pub current_block_timestamp: u64,
    pub current_block_hash: H256,
    pub hard_fork: SpecId,
    pub transaction_order: String,
    pub environment: NodeEnvironment,
    pub fork_config: NodeForkConfig,
}

#[derive(Debug, Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeEnvironment {
    pub base_fee: U256,
    pub chain_id: U256,
    pub gas_limit: U256,
    pub gas_price: U256,
}

#[derive(Debug, Clone, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeForkConfig {
    pub fork_url: Option<String>,
    pub fork_block_number: Option<u64>,
    pub fork_retry_backoff: Option<u128>,
}
