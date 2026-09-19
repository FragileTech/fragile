use algorithmic_gas::{
    GasConfig, GasError, ObservationBatch, Population, Precision, Real, TensorBatch,
    kinetic::ViscousForceConfig,
    variants::{Variant, catalog, viscous_euclidean::reference_viscosity},
};
use std::collections::BTreeSet;

/// Serialized before `GasConfig::euclidean` and `GasConfig::einstein_hilbert`
/// moved into `variants/`.
const EUCLIDEAN_D2: &str = include_str!("fixtures/variants/euclidean_d2_dt0.04.json");
const EUCLIDEAN_D3: &str = include_str!("fixtures/variants/euclidean_d3_dt0.01.json");
const EINSTEIN_HILBERT: &str =
    include_str!("fixtures/variants/einstein_hilbert_t0.33_dt0.002.json");

fn pretty(config: &GasConfig) -> String {
    let mut text = serde_json::to_string_pretty(config).unwrap();
    text.push('\n');
    text
}

/// Distinct positions inside the Euclidean box, at rest.
fn population<T: Real>(walkers: usize, dimensions: usize) -> Population<T> {
    let positions = (0..walkers * dimensions)
        .map(|k| T::from_f64(-1. + 2. * ((k * 7 + 3) % 97) as f64 / 97.))
        .collect();
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(walkers, dimensions, positions).unwrap());
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(walkers, dimensions, vec![T::ZERO; walkers * dimensions]).unwrap(),
    );
    Population::new(observations).unwrap()
}

#[test]
fn euclidean_constructor_serializes_to_the_bytes_pinned_before_the_move() {
    assert_eq!(
        pretty(&GasConfig::euclidean(2, 0.04).unwrap()),
        EUCLIDEAN_D2
    );
    assert_eq!(
        pretty(&GasConfig::euclidean(3, 0.01).unwrap()),
        EUCLIDEAN_D3
    );
}

#[test]
fn einstein_hilbert_constructor_serializes_to_the_bytes_pinned_before_the_move() {
    assert_eq!(
        pretty(&GasConfig::einstein_hilbert(0.33, 0.002).unwrap()),
        EINSTEIN_HILBERT
    );
}

#[test]
fn pinned_snapshots_deserialize_to_the_constructed_configurations() {
    let euclidean: GasConfig = serde_json::from_str(EUCLIDEAN_D2).unwrap();
    assert_eq!(euclidean, GasConfig::euclidean(2, 0.04).unwrap());
    let einstein_hilbert: GasConfig = serde_json::from_str(EINSTEIN_HILBERT).unwrap();
    assert_eq!(
        einstein_hilbert,
        GasConfig::einstein_hilbert(0.33, 0.002).unwrap()
    );
}

#[test]
fn constructors_reject_invalid_arguments() {
    assert!(GasConfig::euclidean(0, 0.04).is_err());
    assert!(GasConfig::euclidean(257, 0.04).is_err());
    assert!(GasConfig::euclidean(2, 0.).is_err());
    assert!(GasConfig::einstein_hilbert(0., 0.002).is_err());
    assert!(GasConfig::einstein_hilbert(0.33, f64::NAN).is_err());
    assert!(GasConfig::viscous_euclidean(0, 0.04, reference_viscosity()).is_err());
    for (coefficient, bandwidth) in [(-1., 1.), (f64::NAN, 1.), (0.3, 0.), (0.3, f64::INFINITY)] {
        let viscosity = ViscousForceConfig {
            coefficient,
            bandwidth,
            row_normalized: false,
        };
        assert!(
            GasConfig::viscous_euclidean(3, 0.04, viscosity).is_err(),
            "coefficient {coefficient}, bandwidth {bandwidth}"
        );
    }
}

#[test]
fn viscous_euclidean_differs_from_euclidean_only_in_the_dense_viscosity() {
    let viscosity = ViscousForceConfig {
        coefficient: 0.7,
        bandwidth: 0.4,
        row_normalized: true,
    };
    let viscous = GasConfig::viscous_euclidean(3, 0.01, viscosity.clone()).unwrap();
    assert_eq!(viscous.qft.viscosity, Some(viscosity));
    let euclidean = GasConfig::euclidean(3, 0.01).unwrap();
    assert_eq!(euclidean.qft.viscosity, None);

    let mut stripped = viscous.clone();
    stripped.qft.viscosity = None;
    assert_eq!(stripped, euclidean);

    let mut value = serde_json::to_value(&viscous).unwrap();
    assert_eq!(
        value["qft"]["viscosity"],
        serde_json::json!({"coefficient": 0.7, "bandwidth": 0.4, "row_normalized": true})
    );
    value["qft"]["viscosity"] = serde_json::Value::Null;
    assert_eq!(value, serde_json::to_value(&euclidean).unwrap());
}

#[test]
fn reference_viscosity_is_the_momentum_conserving_gaussian_coupling() {
    let reference = reference_viscosity();
    assert!(reference.coefficient > 0. && reference.bandwidth > 0.);
    assert!(!reference.row_normalized);
    assert_eq!(
        Variant::ViscousEuclidean.config(3, 0.04).unwrap(),
        GasConfig::viscous_euclidean(3, 0.04, reference).unwrap()
    );
}

#[test]
fn registry_names_are_unique_snake_case_and_equal_to_the_wire_names() {
    let mut names = BTreeSet::new();
    let mut labels = BTreeSet::new();
    let mut titles = BTreeSet::new();
    for &variant in Variant::all() {
        let name = variant.name();
        assert!(
            !name.is_empty()
                && name.starts_with(|c: char| c.is_ascii_lowercase())
                && !name.ends_with('_')
                && !name.contains("__")
                && name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "{name} is not snake_case"
        );
        assert!(names.insert(name), "duplicate name {name}");
        assert!(labels.insert(variant.book_label()));
        assert!(titles.insert(variant.title()));
        assert!(variant.book_label().starts_with("def-variant-"));
        assert!(!variant.summary().is_empty());

        assert_eq!(
            serde_json::to_value(variant).unwrap(),
            serde_json::Value::String(name.into())
        );
        let parsed: Variant = serde_json::from_value(serde_json::json!(name)).unwrap();
        assert_eq!(parsed, variant);
        assert_eq!(Variant::from_name(name).unwrap(), variant);
        assert_eq!(
            Variant::from_name(&name.replace('_', "-")).unwrap(),
            variant
        );
    }
    assert_eq!(names.len(), 6);
    assert!(matches!(
        Variant::from_name("algorithmic"),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn catalog_rows_follow_the_registry() {
    let rows = catalog();
    assert_eq!(rows.len(), Variant::all().len());
    for (row, &variant) in rows.iter().zip(Variant::all()) {
        assert_eq!(row, &variant.info());
        assert_eq!(row.name, variant.name());
        assert_eq!(row.implemented, variant.implemented());
        assert_eq!(row.reference.is_some(), variant.implemented());
    }
    let value = serde_json::to_value(&rows).unwrap();
    assert_eq!(value[2]["name"], "einstein_hilbert");
    assert_eq!(value[2]["book_label"], "def-variant-einstein-hilbert");
    assert_eq!(value[2]["reference"]["walkers"], 500);
    assert!(value[5]["reference"].is_null());
}

#[test]
fn implemented_variants_validate_and_round_trip_through_serde() {
    for &variant in Variant::all().iter().filter(|v| v.implemented()) {
        let reference = variant.reference().unwrap();
        let config = variant.default_config().unwrap();
        assert_eq!(
            config,
            variant.config(reference.dimensions, reference.dt).unwrap()
        );
        let text = serde_json::to_string(&config).unwrap();
        let back: GasConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, config, "{}", variant.name());
        assert_eq!(serde_json::to_string(&back).unwrap(), text);

        // Every implemented variant runs BAOAB and so needs a force provider.
        let valid = match config.precision {
            Precision::F32 => config.validate(&population::<f32>(12, reference.dimensions), true),
            Precision::F64 => config.validate(&population::<f64>(12, reference.dimensions), true),
        };
        valid.unwrap_or_else(|e| panic!("{}: {e}", variant.name()));
    }
}

#[test]
fn reference_instances_reproduce_the_named_constructors() {
    assert_eq!(
        Variant::Euclidean.default_config().unwrap(),
        GasConfig::euclidean(2, 0.04).unwrap()
    );
    assert_eq!(
        Variant::ViscousEuclidean.default_config().unwrap(),
        GasConfig::viscous_euclidean(3, 0.04, reference_viscosity()).unwrap()
    );
    assert_eq!(
        Variant::EinsteinHilbert.default_config().unwrap(),
        GasConfig::einstein_hilbert(0.33, 0.002).unwrap()
    );
}

#[test]
fn book_only_variants_are_a_capability_error_not_a_configuration() {
    for &variant in Variant::all().iter().filter(|v| !v.implemented()) {
        assert!(variant.reference().is_none());
        for result in [variant.config(3, 0.01), variant.default_config()] {
            match result {
                Err(GasError::Capability(message)) => {
                    assert!(message.contains(variant.book_label()), "{message}");
                }
                other => panic!("{}: unexpected {other:?}", variant.name()),
            }
        }
    }
}

#[test]
fn book_labels_are_defined_in_the_variants_chapter_when_it_exists() {
    let chapter = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/source/2_fractal_gas/1_the_algorithm/04_gas_variants.md"
    );
    // The crate is also built outside the book's repository.
    let Ok(text) = std::fs::read_to_string(chapter) else {
        return;
    };
    for &variant in Variant::all() {
        let label = format!(":label: {}", variant.book_label());
        assert!(
            text.lines().any(|line| line.trim() == label),
            "{} is not defined in the variants chapter",
            variant.book_label()
        );
    }
}
