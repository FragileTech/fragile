use algorithmic_gas_benchmarks::convergence_stored_chapter04::validate;
use serde_json::json;

#[test]
fn a_numeric_transport_split_violation_is_detected() {
    let good = json!({"cases":[{"seed_trajectories":[{"points":[{"step":0,
        "observables":{"alive_wasserstein_squared":3.,"alive_barycenter_error":1.,
        "alive_structural_error":2.}}]}]}]});
    let report = validate(&[("saved-good.json".into(), good.clone())]).unwrap();
    assert_eq!(report["summary"]["comparisons_failed"], 0);
    let mut bad = good;
    bad["cases"][0]["seed_trajectories"][0]["points"][0]["observables"]["alive_structural_error"] =
        json!(3.);
    let report = validate(&[("saved-bad.json".into(), bad)]).unwrap();
    assert_eq!(report["summary"]["comparisons_failed"], 1);
}

#[test]
fn partial_alive_input_is_not_promoted_to_all_alive_chapter_reset() {
    let input = json!({"cases":[{"walkers":4,"alive":1,"replicas":[{}],"constants":{}}]});
    let report = validate(&[("saved-singleton.json".into(), input)]).unwrap();
    assert_eq!(report["summary"]["comparisons"], 0);
    assert_eq!(report["summary"]["complete_required_estimates"], false);
    assert!(report["gaps"].as_array().unwrap().iter().any(|g| {
        g["missing_required_input"]
            .as_str()
            .unwrap_or("")
            .contains("QSD")
    }));
}

#[test]
fn saved_postproposal_reset_failure_is_not_hidden_by_old_passed_flags() {
    let input = json!({"cases":[{"walkers":4,"alive":4,"dimensions":1,
        "initial_positions":[[-1.],[-0.5],[0.5],[1.]],
        "constants":{"jitter_sigma":0.,"D_x_squared":4.,"B_x":2.,"lambda_v":1.,"b":0.5},
        "replicas":[{"proposal_position_variance":20.,"independent_pair_proposal_transport":1.},{"proposal_position_variance":20.,"independent_pair_proposal_transport":1.}]}]});
    let report = validate(&[("saved-bad-reset.json".into(), input)]).unwrap();
    assert_eq!(report["summary"]["comparisons_failed"], 1);
    assert!(report["comparisons"].as_array().unwrap().iter().any(|c| {
        c["id"]
            .as_str()
            .unwrap()
            .contains("two_swarm_expected_reset")
            && c["passed"] == false
    }));
}

#[test]
fn missing_numeric_source_data_is_rejected_instead_of_defaulting_to_zero() {
    let input = json!({"cases":[{"seed_trajectories":[{"points":[{"step":0,
        "observables":{"alive_wasserstein_squared":3.,"alive_barycenter_error":1.}}]}]}]});
    assert!(validate(&[("incomplete.json".into(), input)]).is_err());
}

#[test]
fn a_loose_upper_surrogate_is_inconclusive_not_a_theory_violation() {
    let input = json!({"cases":[{"walkers":4,"alive":4,"dimensions":1,
        "initial_positions":[[-1.],[-0.5],[0.5],[1.]],
        "constants":{"jitter_sigma":0.,"D_x_squared":4.,"B_x":2.,"lambda_v":1.,"b":0.5},
        "replicas":[{"proposal_position_variance":1.,"independent_pair_proposal_transport":100.},
                    {"proposal_position_variance":1.,"independent_pair_proposal_transport":100.}]}]});
    let report = validate(&[("saved-loose-surrogate.json".into(), input)]).unwrap();
    assert_eq!(report["summary"]["comparisons_failed"], 0);
    assert!(
        report["comparisons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["status"] == "inconclusive_upper_certificate" && c["passed"].is_null())
    );
}
