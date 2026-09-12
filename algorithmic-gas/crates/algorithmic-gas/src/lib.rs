//! Independent Algorithmic Gas. Public contracts do not expose Burn types.
pub mod batch;
pub mod boundary;
pub mod checkpoint;
pub mod cloning;
pub mod compute;
pub mod domain;
pub mod donor;
pub mod engine;
pub mod error;
pub mod extraction;
pub mod fitness;
pub mod geometry;
pub mod kinetic;
pub mod memory;
pub mod noise;
pub mod operators;
pub mod random;
pub mod scalar;
pub use batch::*;
pub use compute::{BackendKind, ComputeBackend, ExecutionContext};
pub use engine::{
    AlgorithmicGas, CancellationToken, Checkpoint, GasBuilder, GasConfig, InvalidRewardPolicy,
    StepReport,
};
pub use error::{GasError, Result};
pub use operators::GasOperators;
pub use scalar::{Precision, Real};
