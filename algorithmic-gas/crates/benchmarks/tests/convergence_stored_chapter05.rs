use algorithmic_gas_benchmarks::{
    convergence_kinetic::{KineticValidationConfig, kinetic_constants},
    convergence_stored_chapter05::{
        gaussian_quadratic_cost_variance, linear_control_applicable, quadratic_cap_constants,
        validate,
    },
};
use serde_json::json;

#[test]
fn canonical_cap_constants_and_invalid_timestep() {
    let c = quadratic_cap_constants(0.04, 1., 1., 2.).unwrap();
    assert!(c["eta"].as_f64().unwrap() > 0.0184);
    assert!(c["delta"].as_f64().unwrap() > 6.03e-6);
    assert!(quadratic_cap_constants(2., 1., 1., 2.).is_err());
    assert!(quadratic_cap_constants(0.04, 1., 0., 2.).is_err());
}

#[test]
fn exact_gaussian_variance_counts_coordinates_and_noncentral_mean() {
    let p = [[1., 0.], [0., 1.]];
    let covariance = [[0.5, 0.], [0., 0.75]];
    let mean_gram = [[1., 0.], [0., 0.]];
    // Four independent Gaussian scalar coordinates; two x and two v, one
    // unit mean displacement. Var(sum Z_j²)=sum(2σ_j⁴+4μ_j²σ_j²).
    assert_eq!(
        gaussian_quadratic_cost_variance(p, covariance, mean_gram, 2),
        5.25
    );
    assert_eq!(
        gaussian_quadratic_cost_variance(p, [[0.; 2]; 2], mean_gram, 2),
        0.
    );
}

#[test]
fn stored_wrong_noise_constant_is_rejected_without_new_runs() {
    let cfg = KineticValidationConfig::default();
    let mut c = serde_json::to_value(kinetic_constants(&cfg).unwrap()).unwrap();
    c["thermal_variance"] = json!(0.5);
    let reports = vec![(
        "synthetic-stored.json".into(),
        json!({"kinetic":[{
                "landscape_assumptions":cfg.landscape.assumptions().unwrap(),
                "config":cfg,"constants":c,"experiments":[]
        }]}),
    )];
    let report = validate(&reports).unwrap();
    assert_eq!(report["summary"]["comparisons_failed"], 1);
    assert!(report["gaps"].as_array().unwrap().len() >= 6);
}

#[test]
fn selection_or_cap_revokes_linear_control_applicability() {
    let mut case = json!({"profile":"linear_quadratic_noiseless","native_config":{
        "boundary":{"kind":"unbounded"},"kinetic":{"velocity_cap":null},
        "fitness":{"reward_exponent":0.,"diversity_exponent":0.},
        "clone_transform":{"jitter_amplitude":0.}
    }});
    assert!(linear_control_applicable(&case));
    case["native_config"]["kinetic"]["velocity_cap"] = json!(2.);
    assert!(!linear_control_applicable(&case));
    case["native_config"]["kinetic"]["velocity_cap"] = json!(null);
    case["native_config"]["fitness"]["diversity_exponent"] = json!(1.);
    assert!(!linear_control_applicable(&case));
}
