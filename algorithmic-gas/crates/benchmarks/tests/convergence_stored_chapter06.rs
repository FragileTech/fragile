use algorithmic_gas_benchmarks::convergence_stored_chapter06::validate;
use serde_json::{Value, json};

fn control() -> Value {
    json!({"cases":[{
      "profile":"linear_quadratic_noiseless","walkers":4,"dimensions":1,"pair_randomness":"independent",
      "native_config":{"boundary":{"kind":"unbounded"},"fitness":{"reward_exponent":0.0,"diversity_exponent":0.0},"clone_transform":{"jitter_amplitude":0.0,"restitution":null},"qft":{"viscosity":null,"graph_viscosity":null},"kinetic":{"velocity_cap":null,"integrator":{"kind":"baoab","dt":0.04,"friction":1.0},"noise":{"geometry":{"scale":{"values":[0.0]}}},"position_diffusion":0.0}},
      "reference_linear_constants":{"map":[[0.999215684224339,0.03921578878304646],[-0.03920010246753324,0.9600051233766622]],"lyapunov_matrix":[[1.0,0.3291078207098271],[0.3291078207098271,0.6714625206006498]],"innovation_covariance":[[0.0,0.0],[0.0,0.0]],"contraction_factor":0.9781330784692426,"squared_error_decay_rate":0.552738653746043},
      "seed_trajectories":[{"points":[{"observables":{"alive_counts":[4,4]},"cloned":[0,0],"revived":[0,0]}]}],
      "ensemble_trajectory":[{"step":0,"barycenter_error":{"mean":1.0,"standard_error":0.0},"alive_error":{"mean":1.0,"standard_error":0.0},"exact_barycenter_noise_floor":0.0,"exact_barycenter_expectation":1.0,"n_independent_geometric_noise_envelope":1.0}]
    }]})
}
#[test]
fn rejects_a_false_saved_rate() {
    let mut report = control();
    assert_eq!(
        validate(&[("test.json".into(), report.clone())]).unwrap()["summary"]["comparisons_failed"],
        0
    );
    report["cases"][0]["reference_linear_constants"]["squared_error_decay_rate"] = json!(20.0);
    let out = validate(&[("test.json".into(), report)]).unwrap();
    assert_eq!(out["summary"]["comparisons_failed"], 1);
}
#[test]
fn rejects_a_false_saved_noise_offset() {
    let mut report = control();
    report["cases"][0]["ensemble_trajectory"][0]["exact_barycenter_noise_floor"] = json!(1.0);
    let out = validate(&[("test.json".into(), report)]).unwrap();
    assert_eq!(out["summary"]["comparisons_failed"], 1);
}
#[test]
fn unavailable_premises_cannot_be_enabled_by_an_old_pass_flag() {
    let mut report = control();
    report["cases"][0]["native_config"]["fitness"]["reward_exponent"] = json!(1.0);
    report["cases"][0]["linear_control_theorem_applicable"] = json!(true);
    let out = validate(&[("test.json".into(), report)]).unwrap();
    assert_eq!(out["summary"]["comparisons"], 0);
    assert_eq!(out["summary"]["complete_required_estimates"], false);
    assert!(out["coverage"]["gaps"].as_array().unwrap().len() > 10);
}
#[test]
fn converts_alive_normalization_without_charging_dead_slots() {
    let report = json!({"runs":[{"config":{"walkers":4,"gas":{"kinetic":{"velocity_cap":2.0}}},"trajectory":[{"step":1,"moments":{"alive":1,"velocity_second_moment":4.0}}]}]});
    let out = validate(&[("test.json".into(), report)]).unwrap();
    assert_eq!(out["comparisons"][0]["observed"], 1.0);
    assert_eq!(out["comparisons"][0]["passed"], true);
}

#[test]
fn uses_exact_gaussian_uncertainty_when_retained_sample_variance_is_small() {
    let mut report = control();
    let c = &mut report["cases"][0];
    c["profile"] = json!("linear_quadratic_independent_noise");
    c["native_config"]["kinetic"]["noise"]["geometry"]["scale"]["values"][0] = json!(0.2);
    c["native_config"]["kinetic"]["position_diffusion"] = json!(0.01);
    c["reference_linear_constants"]["innovation_covariance"] = json!([
        [4.615069228906915e-6, 3.074116006076756e-5],
        [3.074116006076756e-5, 0.0015364431798371625]
    ]);
    c["ensemble_trajectory"].as_array_mut().unwrap().push(json!({"step":1,"barycenter_error":{"mean":0.9752101582785355,"standard_error":0.0},"alive_error":{"mean":0.9736819015825808,"standard_error":0.0},"exact_barycenter_noise_floor":0.000528256695954714,"exact_barycenter_expectation":0.9742101582785355,"n_independent_geometric_noise_envelope":0.9802461052530614}));
    let out = validate(&[("test.json".into(), report)]).unwrap();
    assert_eq!(out["summary"]["comparisons_failed"], 0);
    let measured = out["comparisons"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["id"] == "measured_barycenter_expectation_case0_step1")
        .unwrap();
    assert!(
        measured["hypotheses"]["barycenter_analytic_standard_error"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}
