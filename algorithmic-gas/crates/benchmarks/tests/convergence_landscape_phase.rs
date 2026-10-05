use algorithmic_gas::GasConfig;
use algorithmic_gas_benchmarks::convergence_landscape_phase::*;
fn reference(d: usize) -> RegionalParameters {
    RegionalParameters {
        dimension: d,
        timestep: 0.04,
        friction: 1.,
        clone_jitter: 0.1,
        ou_amplitude: (-(-0.08_f64).exp_m1() / 2.).sqrt(),
        position_amplitude: 0.02,
        velocity_cap: 2.,
        restitution: 0.5,
        viscosity: 0.3,
        bandwidth: 1.,
        normalization: "count".into(),
        core_radius: 1. / 16.,
        largest_well: 2,
    }
}
#[test]
fn existing_kur_reference_and_population_normalized_constants() {
    let p = reference(3);
    let r = regional_bound(&p).unwrap();
    assert!((r["positional_coefficient"].as_f64().unwrap() - 0.7989420407893824).abs() < 3e-14);
    assert!((r["conservative_floor"].as_f64().unwrap() - 0.20613357968212598).abs() < 3e-14);
    let mut row = p.clone();
    row.normalization = "row".into();
    assert_eq!(
        regional_bound(&row).unwrap()["conservative_floor"],
        r["conservative_floor"]
    );
    let mut bad = p.clone();
    bad.viscosity = 51.;
    assert_eq!(regional_bound(&bad).unwrap()["applicable"], false);
    bad = p.clone();
    bad.core_radius = 0.2;
    assert_eq!(
        regional_bound(&bad).unwrap()["physical_multiplier_rate"],
        serde_json::Value::Null
    );
    bad = p;
    bad.timestep = 1.;
    assert_eq!(regional_bound(&bad).unwrap()["applicable"], false);
}
#[test]
fn actual_native_parameters_and_source_core_gate() {
    let cfg = GasConfig::euclidean(1, 0.04).unwrap();
    let mut json = serde_json::to_value(cfg).unwrap();
    json["kinetic"]["velocity_cap"] = serde_json::json!(2.);
    let p = native_parameters(&json, 1).unwrap();
    assert_eq!(p.timestep, 0.04);
    assert!((p.position_amplitude - 0.02).abs() < 1e-16);
    let a = source_core_eligibility(&[0., 0.01], 1, &p).unwrap();
    let b = source_core_eligibility(&[0.01, 0., 0.01, 0.], 1, &p).unwrap();
    assert_eq!(a["all_sources_in_one_core"], b["all_sources_in_one_core"]);
    assert_eq!(a["all_sources_in_one_core"], true);
    assert_eq!(
        source_core_eligibility(&[0.5, 0.], 1, &p).unwrap()["all_sources_in_one_core"],
        false
    );
    let profile = rastrigin_regional_profile(-2, 2).unwrap();
    let root = profile["stable_roots"][3]["value"].as_f64().unwrap();
    assert_eq!(
        source_core_eligibility(&[0., root], 1, &p).unwrap()["all_in_declared_cores"],
        true
    );
    assert_eq!(
        source_core_eligibility(&[0., root], 1, &p).unwrap()["all_sources_in_one_core"],
        false
    );
    json["qft"]["curl"] = serde_json::json!({"beta_curl":1.});
    assert!(native_parameters(&json, 1).is_err());
}
#[test]
fn native_restitution_domain_is_enforced_at_every_regional_entry_point() {
    let cfg = GasConfig::euclidean(1, 0.04).unwrap();
    let mut native = serde_json::to_value(cfg).unwrap();
    native["kinetic"]["velocity_cap"] = serde_json::json!(2.);
    for restitution in [-0.01, 1.01, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        let mut p = reference(1);
        p.restitution = restitution;
        assert!(regional_bound(&p).is_err());
        assert!(tightened_regional_bound(&p).is_err());
        assert!(source_core_eligibility(&[0., 0.01], 1, &p).is_err());
        if restitution.is_finite() {
            native["clone_transform"]["restitution"] = serde_json::json!(restitution);
            assert!(native_parameters(&native, 1).is_err());
        }
    }
    for restitution in [0., 1.] {
        let mut p = reference(1);
        p.restitution = restitution;
        assert_eq!(regional_bound(&p).unwrap()["applicable"], true);
        assert_eq!(tightened_regional_bound(&p).unwrap()["applicable"], true);
        assert_eq!(
            source_core_eligibility(&[0., 0.01], 1, &p).unwrap()["all_sources_in_one_core"],
            true
        );
        native["clone_transform"]["restitution"] = serde_json::json!(restitution);
        assert_eq!(
            native_parameters(&native, 1).unwrap().restitution,
            restitution
        );
    }
}
#[test]
fn existing_python_independent_fixture_agreement() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/landscape-phase-python.json")).unwrap();
    for f in fixtures["cases"].as_array().unwrap() {
        let p = serde_json::from_value(f["parameters"].clone()).unwrap();
        let r = regional_bound(&p).unwrap();
        for key in [
            "mean_map_squared",
            "accepted_coordinate_variance",
            "gate_refresh_bound",
            "positional_coefficient",
            "conservative_floor",
        ] {
            let x = r[key].as_f64().unwrap();
            let y = f["expected"][key].as_f64().unwrap();
            assert!(
                (x - y).abs() < 2e-12 * (1. + x.abs() + y.abs()),
                "{} {key}",
                f["id"]
            );
        }
    }
    for f in fixtures["cases"].as_array().unwrap() {
        let p: RegionalParameters = serde_json::from_value(f["parameters"].clone()).unwrap();
        let refined = tightened_regional_bound(&p).unwrap();
        for key in [
            "accepted_coordinate_variance",
            "positional_coefficient",
            "conservative_floor",
        ] {
            let x = refined[key].as_f64().unwrap();
            let y = f["expected_tightened"][key].as_f64().unwrap();
            assert!((x - y).abs() < 3e-12, "{} {key}", f["id"]);
        }
        if p.clone_jitter == 0. {
            assert_eq!(
                refined["accepted_coordinate_variance"],
                serde_json::json!(0.)
            );
        }
    }
    assert!(rastrigin_regional_profile(-22, 2).is_err());
}
