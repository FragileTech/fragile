use algorithmic_gas::physics::partvi::{ExperimentRequest, analyze};
use serde_json::json;
#[test]
fn every_public_experiment_requires_executed_evidence() {
    for experiment in 1..=66 {
        assert!(
            analyze(&ExperimentRequest {
                experiment,
                parameters: json!({})
            })
            .is_err(),
            "{experiment}"
        );
    }
}
