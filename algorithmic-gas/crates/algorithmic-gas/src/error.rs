use thiserror::Error;
pub type Result<T> = std::result::Result<T, GasError>;
#[derive(Debug, Error)]
pub enum GasError {
    #[error("invalid configuration: {0}")]
    Configuration(String),
    #[error("shape contract: {0}")]
    Shape(String),
    #[error("missing field: {0}")]
    MissingField(String),
    #[error("unsupported capability: {0}")]
    Capability(String),
    #[error("invalid numerical result: {0}")]
    Numerical(String),
    #[error("population extinct: no eligible revival donor")]
    Extinction,
    #[error("incompatible donor topology: {0}")]
    Topology(String),
    #[error("domain consistency: {0}")]
    Domain(String),
    #[error("checkpoint contract: {0}")]
    Checkpoint(String),
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("step cancelled before commit")]
    Cancelled,
}
pub(crate) fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(GasError::Configuration(message.into()))
    }
}
