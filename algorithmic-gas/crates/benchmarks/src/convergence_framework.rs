//! Chapter 1 diagnostics. Auxiliary uniform-companion estimates retain their scope.
use algorithmic_gas::{
    GasConfig, GasError, ObservationBatch, Result, TensorBatch,
    cloning::CloneDecision,
    fitness::{PositiveMap, PositiveMapping, Standardizer},
    geometry::{AlgorithmicDistance, Distance},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundCheck {
    pub id: String,
    pub source_labels: Vec<String>,
    pub scope: String,
    pub observed: f64,
    pub bound: f64,
    pub slack: f64,
    pub passed: bool,
}
impl BoundCheck {
    pub fn upper(id: &str, labels: &[&str], scope: &str, observed: f64, bound: f64) -> Self {
        let tolerance = 1e-10 * (1. + bound.abs());
        Self {
            id: id.into(),
            source_labels: labels.iter().map(|s| (*s).into()).collect(),
            scope: scope.into(),
            observed,
            bound,
            slack: bound - observed,
            passed: observed.is_finite() && bound.is_finite() && observed <= bound + tolerance,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameworkReport {
    pub walkers: usize,
    pub dimensions: usize,
    pub sigma_min: f64,
    pub constants: BTreeMap<String, f64>,
    pub checks: Vec<BoundCheck>,
    pub scope_notes: Vec<String>,
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
}
fn second(v: &[f64]) -> f64 {
    mean(&v.iter().map(|x| x * x).collect::<Vec<_>>())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HermitePatch {
    pub z_max: f64,
    pub coefficients: [f64; 4],
    pub independently_solved_coefficients: [f64; 4],
    pub normalized_endpoint_matrix_determinant: f64,
    pub derivative_vertex: f64,
    pub derivative_maximizer: f64,
    pub exact_derivative_maximum: f64,
    pub exact_derivative_minimum: f64,
    pub exact_rescale_lipschitz: f64,
    pub uniform_derivative_bound: f64,
}

/// Auxiliary chapter 1 cubic, in its stable unit-interval coordinate. The
/// native algorithm continues to use its configured logistic map.
pub fn hermite_patch(z_max: f64) -> Result<HermitePatch> {
    if !z_max.is_finite() || z_max <= 1. {
        return Err(GasError::Configuration(
            "Hermite patch needs finite z_max>1".into(),
        ));
    }
    let x = 1. / z_max;
    let change = x.ln_1p();
    let coefficients = [x - 2. * change, 3. * change - 2. * x, x, z_max.ln() + 1.];
    let endpoint_targets = [0., x, change, 0.];
    // Solve Q-D to preserve the small value difference log(1+1/z_max).
    // Rows are (Q-D)(0), Q'(0), (Q-D)(1), Q'(1), columns are A,B,C,D0.
    // Elimination computes the determinant and solves independently of the
    // closed coefficient formulas above.
    let mut matrix = [
        [0., 0., 0., 1., endpoint_targets[0]],
        [0., 0., 1., 0., endpoint_targets[1]],
        [1., 1., 1., 1., endpoint_targets[2]],
        [3., 2., 1., 0., endpoint_targets[3]],
    ];
    let mut determinant = 1.;
    for col in 0..4 {
        let pivot = matrix
            .iter()
            .enumerate()
            .skip(col)
            .max_by(|(_, a), (_, b)| a[col].abs().total_cmp(&b[col].abs()))
            .map(|(i, _)| i)
            .unwrap_or(col);
        if pivot != col {
            matrix.swap(col, pivot);
            determinant = -determinant;
        }
        let diagonal = matrix[col][col];
        if diagonal == 0. {
            return Err(GasError::Numerical(
                "singular normalized Hermite endpoint matrix".into(),
            ));
        }
        determinant *= diagonal;
        let pivot_row = matrix[col];
        for row in matrix.iter_mut().skip(col + 1) {
            let factor = row[col] / diagonal;
            for (entry, pivot_entry) in row.iter_mut().zip(pivot_row).skip(col) {
                *entry -= factor * pivot_entry;
            }
        }
    }
    let mut solved = [0.; 4];
    for i in (0..4).rev() {
        let tail = matrix[i]
            .iter()
            .zip(solved)
            .skip(i + 1)
            .take(3 - i)
            .map(|(m, coefficient)| m * coefficient)
            .sum::<f64>();
        solved[i] = (matrix[i][4] - tail) / matrix[i][i];
    }
    solved[3] += coefficients[3];
    let [a, b, c, _] = coefficients;
    let vertex = -b / (3. * a);
    let derivative = |s: f64| (3. * a * s + 2. * b) * s + c;
    let mut candidates = vec![(0., derivative(0.)), (1., derivative(1.))];
    if (0. ..=1.).contains(&vertex) {
        candidates.push((vertex, derivative(vertex)));
    }
    let (maximizer, maximum) = candidates
        .iter()
        .copied()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    let minimum = candidates
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::INFINITY, f64::min);
    let ln2 = 2_f64.ln();
    let uniform_bound = 1. + (3. * ln2 - 2.).powi(2) / (3. * (2. * ln2 - 1.));
    let result = HermitePatch {
        z_max,
        coefficients,
        independently_solved_coefficients: solved,
        normalized_endpoint_matrix_determinant: determinant,
        derivative_vertex: vertex,
        derivative_maximizer: maximizer,
        exact_derivative_maximum: maximum,
        exact_derivative_minimum: minimum,
        exact_rescale_lipschitz: 1_f64.max(maximum),
        uniform_derivative_bound: uniform_bound,
    };
    if coefficients
        .iter()
        .chain(solved.iter())
        .chain([
            &result.derivative_vertex,
            &result.exact_derivative_maximum,
            &result.exact_derivative_minimum,
            &result.exact_rescale_lipschitz,
        ])
        .any(|value| !value.is_finite())
    {
        return Err(GasError::Numerical(
            "Hermite coefficient/extremum calculation overflowed".into(),
        ));
    }
    Ok(result)
}

/// Compares the actual native constructor against the explicit canonical
/// constants, separately from whichever profile supplied the main fixture.
pub fn canonical_rust_contract(dimensions: usize) -> Result<Vec<BoundCheck>> {
    let config = GasConfig::euclidean(dimensions, 0.04)?;
    let serialized = serde_json::to_value(&config).map_err(|e| {
        GasError::Configuration(format!("canonical configuration serialization: {e}"))
    })?;
    let scope = format!(
        "Native GasConfig::euclidean({dimensions},0.04) fixed canonical fixture; validates constructor fields against def-eg-canonical-rust, independently of varied experiment profiles."
    );
    let mut checks = vec![];
    for (path, expected) in [
        ("/fitness/reward_standardizer/sigma_min", 0.1),
        ("/fitness/diversity_standardizer/sigma_min", 0.1),
        ("/fitness/reward_map/amplitude", 2.),
        ("/fitness/reward_map/floor", 0.1),
        ("/fitness/diversity_map/amplitude", 2.),
        ("/fitness/diversity_map/floor", 0.1),
        ("/fitness/reward_exponent", 1.),
        ("/fitness/diversity_exponent", 1.),
        ("/fitness/distance_floor", 1e-3),
        ("/clone_decision/epsilon", 1e-6),
        ("/clone_decision/saturation", 1.),
        ("/distance_donors/kernel/width", 2.),
        ("/cloning_donors/kernel/width", 2.),
        ("/distance_donors/distance/position_radius", 2.),
        ("/distance_donors/distance/velocity_radius", 2.),
        ("/distance_donors/distance/lambda", 1.),
        ("/cloning_donors/distance/position_radius", 2.),
        ("/cloning_donors/distance/velocity_radius", 2.),
        ("/cloning_donors/distance/lambda", 1.),
        ("/kinetic/integrator/dt", 0.04),
        ("/kinetic/integrator/friction", 1.),
        ("/kinetic/noise/geometry/scale/values/0", 1.),
        ("/kinetic/position_diffusion", 0.1),
        ("/kinetic/velocity_cap", 2.),
        ("/clone_transform/jitter_amplitude", 0.1),
        ("/clone_transform/restitution", 0.5),
    ] {
        let actual = serialized
            .pointer(path)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                GasError::Configuration(format!("missing canonical numeric field {path}"))
            })?;
        checks.push(BoundCheck::upper(
            "canonical_rust_numeric_parameter",
            &["def-eg-canonical-rust"],
            &format!("{scope} Parameter {path}: actual={actual}, declared={expected}"),
            (actual - expected).abs(),
            0.,
        ));
    }
    for (path, expected) in [
        ("/distance_donors/law", "independent"),
        ("/cloning_donors/law", "independent"),
        ("/distance_donors/kernel/kind", "gaussian"),
        ("/cloning_donors/kernel/kind", "gaussian"),
        ("/fitness/reward_standardizer/kind", "global"),
        ("/fitness/diversity_standardizer/kind", "global"),
        ("/fitness/reward_map/kind", "logistic"),
        ("/fitness/diversity_map/kind", "logistic"),
        ("/kinetic/integrator/kind", "baoab"),
        ("/kinetic/noise/innovation", "gaussian"),
        ("/kinetic/boundary_schedule", "end_of_step"),
        ("/boundary/kind", "absorbing_box"),
    ] {
        let actual = serialized
            .pointer(path)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                GasError::Configuration(format!("missing canonical module field {path}"))
            })?;
        checks.push(BoundCheck::upper(
            "canonical_rust_native_module",
            &["def-eg-canonical-rust"],
            &format!("{scope} Module {path}: actual={actual}, declared={expected}"),
            u8::from(actual != expected) as f64,
            0.,
        ));
    }
    for donor in [&config.distance_donors, &config.cloning_donors] {
        checks.push(BoundCheck::upper(
            "canonical_rust_current_frame_independent_pool",
            &["def-eg-canonical-rust"],
            &scope,
            donor.history_window as f64,
            0.,
        ));
    }
    for (path, expected) in [
        ("/boundary/domain/lower", -2.),
        ("/boundary/domain/upper", 2.),
    ] {
        let values = serialized
            .pointer(path)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                GasError::Configuration(format!("missing canonical domain field {path}"))
            })?;
        checks.push(BoundCheck::upper(
            "canonical_rust_absorbing_box_dimension",
            &["def-eg-canonical-rust"],
            &scope,
            (values.len() as f64 - dimensions as f64).abs(),
            0.,
        ));
        for value in values {
            let actual = value.as_f64().ok_or_else(|| {
                GasError::Configuration("invalid canonical box coordinate".into())
            })?;
            checks.push(BoundCheck::upper(
                "canonical_rust_absorbing_box_coordinate",
                &["def-eg-canonical-rust"],
                &scope,
                (actual - expected).abs(),
                0.,
            ));
        }
    }
    Ok(checks)
}

/// Tests explicit chapter 1 bounds on paired bounded swarms. The distance
/// calculation uses the real `Distance` provider; expectations enumerate its
/// auxiliary uniform donor law exactly. No bound on Gaussian proposals is inferred.
pub fn validate_framework(
    walkers: usize,
    dimensions: usize,
    sigma_min: f64,
    seed: u64,
) -> Result<FrameworkReport> {
    if !(2..=4096).contains(&walkers)
        || !(1..=256).contains(&dimensions)
        || !sigma_min.is_finite()
        || sigma_min <= 0.
    {
        return Err(GasError::Configuration(
            "invalid framework diagnostic inputs".into(),
        ));
    }
    let mut rng = RandomStream::new(seed, 0, Stream::Initialize, 0, 0);
    let x: Vec<f64> = (0..walkers * dimensions)
        .map(|_| 2. * rng.uniform::<f64>() - 1.)
        .collect();
    let y: Vec<f64> = x.iter().map(|&x| (x + 0.02).clamp(-1., 1.)).collect();
    let a = ObservationBatch::positions(TensorBatch::vectors(walkers, dimensions, x.clone())?);
    let b = ObservationBatch::positions(TensorBatch::vectors(walkers, dimensions, y.clone())?);
    let raw: Vec<f64> = x.chunks(dimensions).map(mean).collect();
    let altered: Vec<f64> = raw
        .iter()
        .enumerate()
        .map(|(i, &v)| (v + 0.03 * (i as f64).sin()).clamp(-1., 1.))
        .collect();
    let v_max = 1.;
    let k = walkers as f64;
    let diameter = 2. * (dimensions as f64).sqrt();
    let delta = raw
        .iter()
        .zip(&altered)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let standardizer = Standardizer::Global { sigma_min };
    let alive = vec![true; walkers];
    let (z, stats) = standardizer.apply(&raw, &alive, &a)?;
    let (w, changed_stats) = standardizer.apply(&altered, &alive, &a)?;
    let mut reduced = alive.clone();
    reduced[walkers - 1] = false;
    let reduced_values = &raw[..walkers - 1];
    let n_changed = 1.;
    let mut checks = vec![
        BoundCheck::upper(
            "empirical_mean_lipschitz",
            &[
                "lem-empirical-moments-lipschitz",
                "lem-sasaki-aggregator-value",
            ],
            "bounded raw values; fixed alive set",
            (mean(&raw) - mean(&altered)).abs(),
            delta / k.sqrt(),
        ),
        BoundCheck::upper(
            "empirical_second_moment_lipschitz",
            &[
                "lem-empirical-moments-lipschitz",
                "lem-sasaki-aggregator-value",
            ],
            "|v_i| <= 1; fixed alive set",
            (second(&raw) - second(&altered)).abs(),
            2. * v_max * delta / k.sqrt(),
        ),
        BoundCheck::upper(
            "empirical_variance_lipschitz",
            &[
                "lem-empirical-aggregator-properties",
                "lem-lipschitz-bound-for-the-variance-functional",
            ],
            "|v_i| <= 1; fixed alive set",
            (variance(&raw) - variance(&altered)).abs(),
            4. * v_max * delta / k.sqrt(),
        ),
        BoundCheck::upper(
            "empirical_structural_mean",
            &[
                "lem-empirical-aggregator-properties",
                "lem-sasaki-aggregator-structural",
            ],
            "bounded fixed values; one death; k2 > 0",
            (mean(&raw) - mean(reduced_values)).abs(),
            2. * v_max * n_changed / (k - 1.),
        ),
        BoundCheck::upper(
            "empirical_structural_second_moment",
            &[
                "lem-empirical-aggregator-properties",
                "lem-sasaki-aggregator-structural",
            ],
            "bounded fixed values; one death; k2 > 0",
            (second(&raw) - second(reduced_values)).abs(),
            2. * v_max.powi(2) * n_changed / (k - 1.),
        ),
        BoundCheck::upper(
            "variance_deviation_identity",
            &[
                "axiom-bounded-deviation-variance",
                "lem-empirical-aggregator-properties",
            ],
            "empirical aggregator kappa_var = 1",
            (z.iter().map(|z| z * z).sum::<f64>() - k * variance(&raw) / stats.scale[0].powi(2))
                .abs(),
            0.,
        ),
        BoundCheck::upper(
            "range_to_variance",
            &[
                "axiom-bounded-variance-production",
                "lem-empirical-aggregator-properties",
            ],
            "bounded raw values kappa_range = 1",
            variance(&raw),
            v_max.powi(2),
        ),
        BoundCheck::upper(
            "mean_respects_range_upper",
            &["axiom-range-respecting-mean"],
            "empirical aggregator",
            mean(&raw),
            raw.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ),
        BoundCheck::upper(
            "mean_respects_range_lower",
            &["axiom-range-respecting-mean"],
            "empirical aggregator",
            raw.iter().copied().fold(f64::INFINITY, f64::min),
            mean(&raw),
        ),
        BoundCheck::upper(
            "standardization_denominator_floor",
            &[
                "def-statistical-properties-measurement",
                "lem-sigma-reg-derivative-bounds",
            ],
            "canonical quadratic floor sigma_min > 0",
            sigma_min,
            stats.scale[0],
        ),
        BoundCheck::upper(
            "standardization_scale_lipschitz",
            &["lem-stats-value-continuity", "cor-chain-rule-sigma-reg-var"],
            "canonical quadratic floor, |v_i| <= 1",
            (stats.scale[0] - changed_stats.scale[0]).abs(),
            2. * v_max * delta / (sigma_min * k.sqrt()),
        ),
        BoundCheck::upper(
            "zscore_norm",
            &["thm-z-score-norm-bound"],
            "empirical mean and quadratic variance floor",
            z.iter().map(|z| z * z).sum(),
            k,
        ),
        BoundCheck::upper(
            "standardization_value_continuity",
            &[
                "thm-standardization-value-error-mean-square",
                "def-value-error-coefficients",
            ],
            "quadratic floor a = sigma_min; bounded deterministic values",
            z.iter().zip(&w).map(|(x, y)| (x - y).powi(2)).sum(),
            3. * (2. / sigma_min.powi(2) + 16. * v_max.powi(4) / sigma_min.powi(6)) * delta.powi(2),
        ),
    ];
    checks.extend(canonical_rust_contract(dimensions)?);
    // The singleton branch is explicitly separate from k>=2 distance bounds.
    let distance = Distance::default();
    let expected = |obs: &ObservationBatch<f64>, mask: &[bool]| -> Result<(Vec<f64>, f64)> {
        let mut means = vec![0.; walkers];
        let mut total_variance = 0.;
        for i in 0..walkers {
            if !mask[i] {
                continue;
            }
            let mut values = vec![];
            for (j, &active) in mask.iter().enumerate() {
                if active && j != i {
                    values.push(distance.compare(obs, i, obs, j)?);
                }
            }
            if values.is_empty() {
                values.push(0.);
            }
            means[i] = mean(&values);
            total_variance += variance(&values);
        }
        Ok((means, total_variance))
    };
    let (d1, distance_var) = expected(&a, &alive)?;
    let (d2, distance_var2) = expected(&b, &reduced)?;
    let displacement = x.iter().zip(&y).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
    let distance_error = d1
        .iter()
        .zip(&d2)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>();
    let expected_bound = 12. * displacement
        + diameter.powi(2) * n_changed
        + 8. * k * diameter.powi(2) / (k - 1.).powi(2) * n_changed.powi(2);
    checks.push(BoundCheck::upper(
        "expected_raw_distance",
        &["thm-expected-raw-distance-bound"],
        "auxiliary uniform companions; paired box [-1,1]^d; k1>=2",
        distance_error,
        expected_bound,
    ));
    checks.push(BoundCheck::upper(
        "sampled_raw_distance_mean_square",
        &["thm-distance-operator-mean-square-continuity"],
        "independent outputs; auxiliary uniform law; bounded fixture",
        distance_error + distance_var + distance_var2,
        6. * k * diameter.powi(2) + 3. * expected_bound,
    ));
    checks.push(BoundCheck::upper(
        "raw_distance_variance",
        &["thm-distance-operator-satisfies-bounded-variance-axiom"],
        "uniform companions in bounded fixture",
        distance_var,
        k * diameter.powi(2),
    ));
    let mut singleton = vec![false; walkers];
    singleton[0] = true;
    checks.push(BoundCheck::upper(
        "singleton_raw_distance",
        &["thm-expected-raw-distance-k1"],
        "single surviving walker",
        expected(&a, &singleton)?.0.iter().map(|x| x * x).sum(),
        0.,
    ));
    let logistic = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let decision = CloneDecision::default();
    let mut max_logistic_slope: f64 = 0.;
    let mut max_gate_error: f64 = 0.;
    for index in -2000..2000 {
        let x = index as f64 / 100.;
        let y = x + 1e-4;
        max_logistic_slope =
            max_logistic_slope.max((logistic.map(y)? - logistic.map(x)?).abs() / (y - x));
        let own = 0.01 + (index + 2000) as f64 / 1000.;
        let donor = 4.41 - (index + 2000) as f64 / 1000.;
        let formula =
            ((donor - own) / (own + decision.epsilon) / decision.saturation).clamp(0., 1.);
        max_gate_error =
            max_gate_error.max((decision.acceptance_probability(0, own, donor) - formula).abs());
    }
    checks.push(BoundCheck::upper(
        "logistic_lipschitz",
        &["thm-canonical-logistic-validity"],
        "actual Rust amplitude 2 logistic map",
        max_logistic_slope,
        0.5,
    ));
    checks.push(BoundCheck::upper(
        "clone_gate_identity",
        &[
            "def-cloning-probability-function",
            "lem-cloning-probability-lipschitz",
        ],
        "actual Rust living gate epsilon=1e-6,p_max=1",
        max_gate_error,
        0.,
    ));
    let ln2 = 2_f64.ln();
    let l_patch = 1. + (3. * ln2 - 2.).powi(2) / (3. * (2. * ln2 - 1.));
    let mut patch_max: f64 = 0.;
    let mut patch_min: f64 = 0.;
    let mut hermite_constants = BTreeMap::new();
    for &zmax in &[1.000001, 1.1, 2., 10., 100.] {
        let x: f64 = 1. / zmax;
        let patch = hermite_patch(zmax)?;
        let [a, b, c, d] = patch.coefficients;
        let scope = format!(
            "Auxiliary chapter 1 patched rescale; z_max={zmax}; normalized coordinate s=z-(z_max-1), s in [0,1]. Exact quadratic extrema include both endpoints and its interior vertex; native logistic unchanged."
        );
        for (name, value) in [
            ("A", a),
            ("B", b),
            ("C", c),
            ("D", d),
            (
                "endpoint_determinant",
                patch.normalized_endpoint_matrix_determinant,
            ),
            ("derivative_vertex", patch.derivative_vertex),
            ("derivative_maximizer", patch.derivative_maximizer),
            (
                "exact_patch_derivative_maximum",
                patch.exact_derivative_maximum,
            ),
            (
                "exact_patch_derivative_minimum",
                patch.exact_derivative_minimum,
            ),
            ("exact_L_gA", patch.exact_rescale_lipschitz),
        ] {
            hermite_constants.insert(format!("hermite_zmax_{zmax:.6}_{name}"), value);
        }
        checks.push(BoundCheck::upper(
            "hermite_endpoint_matrix_determinant",
            &["lem-cubic-patch-uniqueness"],
            &scope,
            (patch.normalized_endpoint_matrix_determinant - 1.).abs(),
            0.,
        ));
        checks.push(BoundCheck::upper(
            "hermite_coefficient_linear_system_identity",
            &["lem-cubic-patch-coefficients", "lem-cubic-patch-uniqueness"],
            &scope,
            patch
                .coefficients
                .iter()
                .zip(patch.independently_solved_coefficients)
                .map(|(formula, solved)| (formula - solved).abs())
                .fold(0., f64::max),
            0.,
        ));
        let polynomial = |s: f64| ((a * s + b) * s + c) * s + d;
        let derivative = |s: f64| (3. * a * s + 2. * b) * s + c;
        for (id, actual, expected) in [
            (
                "hermite_start_value_endpoint",
                polynomial(0.),
                zmax.ln() + 1.,
            ),
            ("hermite_start_slope_endpoint", derivative(0.), x),
            (
                "hermite_end_value_endpoint",
                polynomial(1.),
                zmax.ln_1p() + 1.,
            ),
            ("hermite_end_slope_endpoint", derivative(1.), 0.),
        ] {
            checks.push(BoundCheck::upper(
                id,
                &[
                    "lem-cubic-patch-coefficients",
                    "lem-cubic-patch-uniqueness",
                    "def-asymmetric-rescale-function",
                ],
                &scope,
                (actual - expected).abs(),
                0.,
            ));
        }
        checks.push(BoundCheck::upper(
            "hermite_exact_patch_derivative_bound",
            &[
                "lem-cubic-patch-derivative",
                "lem-cubic-patch-derivative-bounds",
            ],
            &scope,
            patch.exact_derivative_maximum,
            l_patch,
        ));
        checks.push(BoundCheck::upper(
            "hermite_exact_global_rescale_lipschitz",
            &["thm-rescale-function-lipschitz"],
            &scope,
            patch.exact_rescale_lipschitz,
            l_patch,
        ));
        let log_increment = x.ln_1p();
        let source_vertex = (3. * log_increment - 2. * x) / (3. * (2. * log_increment - x));
        let source_exact_modulus = 1_f64.max(x + (3. * log_increment - 2. * x) * source_vertex);
        checks.push(BoundCheck::upper(
            "hermite_exact_fixed_parameter_modulus_identity",
            &["thm-rescale-function-lipschitz"],
            &scope,
            (patch.exact_rescale_lipschitz - source_exact_modulus).abs(),
            0.,
        ));
        checks.push(BoundCheck::upper(
            "hermite_exact_patch_monotonicity",
            &["lem-polynomial-patch-monotonicity"],
            &scope,
            -patch.exact_derivative_minimum,
            0.,
        ));
        let mut derivative_identity_residual: f64 = 0.;
        let mut exact_max_residual: f64 = 0.;
        for index in 0..=1000 {
            let s = index as f64 / 1000.;
            let q = 3. * (x - 2. * x.ln_1p()) * s * s + 2. * (3. * x.ln_1p() - 2. * x) * s + x;
            patch_max = patch_max.max(q);
            patch_min = patch_min.min(q);
            let [solved_a, solved_b, solved_c, _] = patch.independently_solved_coefficients;
            derivative_identity_residual = derivative_identity_residual
                .max((q - (3. * solved_a * s + 2. * solved_b) * s - solved_c).abs());
            exact_max_residual = exact_max_residual.max(q - patch.exact_derivative_maximum);
        }
        checks.push(BoundCheck::upper(
            "hermite_explicit_derivative_identity",
            &["lem-cubic-patch-derivative"],
            &scope,
            derivative_identity_residual,
            0.,
        ));
        checks.push(BoundCheck::upper(
            "hermite_grid_below_exact_vertex_maximum",
            &["lem-cubic-patch-derivative-bounds"],
            &scope,
            exact_max_residual,
            0.,
        ));
    }
    checks.push(BoundCheck::upper(
        "hermite_patch_derivative_upper",
        &[
            "lem-cubic-patch-derivative-bounds",
            "lem-cubic-patch-derivative",
        ],
        "auxiliary patched rescale; z_max > 1, s in [0,1]",
        patch_max,
        l_patch,
    ));
    checks.push(BoundCheck::upper(
        "hermite_patch_monotonicity",
        &["lem-polynomial-patch-monotonicity"],
        "auxiliary patched rescale",
        -patch_min,
        0.,
    ));
    let mut constants = BTreeMap::from([
        ("L_mu_M".into(), 1. / k.sqrt()),
        ("L_m2_M".into(), 2. * v_max / k.sqrt()),
        ("L_var_M".into(), 4. * v_max / k.sqrt()),
        ("L_sigma_reg".into(), 1. / (2. * sigma_min)),
        (
            "sigma_reg_second_derivative".into(),
            1. / (4. * sigma_min.powi(3)),
        ),
        (
            "sigma_reg_third_derivative".into(),
            3. / (8. * sigma_min.powi(5)),
        ),
        ("L_sigma_M".into(), 2. * v_max / (sigma_min * k.sqrt())),
        ("L_P_auxiliary".into(), l_patch),
        ("L_logistic".into(), 0.5),
        ("V_pot_min_logistic_alpha1_beta1".into(), 0.01),
        ("V_pot_max_logistic_alpha1_beta1".into(), 4.41),
        ("L_pi_companion".into(), 1. / decision.epsilon),
        (
            "L_pi_own".into(),
            (4.41 + decision.epsilon) / decision.epsilon.powi(2),
        ),
        ("C_struct_pi".into(), 2. / (k - 1.)),
        ("C_pos_d".into(), 12.),
        ("C_status_d1".into(), diameter.powi(2)),
        (
            "C_status_d2".into(),
            8. * k * diameter.powi(2) / (k - 1.).powi(2),
        ),
        ("D_fixture".into(), diameter),
        ("kappa_var".into(), 1.),
        ("kappa_range".into(), 1.),
        ("C_value_direct".into(), 1. / sigma_min.powi(2)),
        ("C_value_mu".into(), 1. / sigma_min.powi(2)),
        (
            "C_value_sigma".into(),
            16. * v_max.powi(4) / sigma_min.powi(6),
        ),
        (
            "C_value_total".into(),
            3. * (2. / sigma_min.powi(2) + 16. * v_max.powi(4) / sigma_min.powi(6)),
        ),
    ]);
    constants.extend(hermite_constants);
    if constants.values().any(|v: &f64| !v.is_finite()) {
        return Err(GasError::Numerical(
            "framework constants overflowed; enlarge sigma_min".into(),
        ));
    }
    Ok(FrameworkReport { walkers, dimensions, sigma_min, constants, checks, scope_notes: vec![
        "Bounded fixture estimates do not certify unbounded physical proposal diameters or global reward regularity.".into(),
        "Uniform-companion and Hermite patch tests are auxiliary chapter 1 law checks; native Euclidean runs retain Gaussian companions and logistic rescaling.".into(),
        "The canonical sigma_min is the combined denominator floor; no arbitrary split into kappa_var_min and epsilon_std is inferred.".into(),
    ] })
}

/// Regression of the repaired chapter 2 bound: identical coordinates can
/// still change entering measurement support. Revival is measured downstream.
pub fn stable_distance_regressions() -> Result<Vec<BoundCheck>> {
    let mut obs: ObservationBatch<f64> =
        ObservationBatch::positions(TensorBatch::vectors(3, 1, vec![0., 1., 2.])?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(3, 1, vec![0.; 3])?,
    );
    let distance = Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius: 2.,
        velocity_radius: 2.,
        lambda: 1.,
    };
    let mut witnesses = vec![];
    for gaussian in [false] {
        let mut squared_error = 0.;
        for i in 0..2 {
            let mut total = 0.;
            let mut normalization = 0.;
            for j in 0..3 {
                if j == i {
                    continue;
                }
                let d = distance.compare(&obs, i, &obs, j)?;
                let weight = if gaussian {
                    (-0.5 * (d / 2.).powi(2)).exp()
                } else {
                    1.
                };
                total += weight * d;
                normalization += weight;
            }
            let before = total / normalization;
            let after = distance.compare(&obs, i, &obs, 1 - i)?;
            squared_error += (before - after).powi(2);
        }
        witnesses.push(BoundCheck::upper(
            "stable_sasaki_changed_support_uniform",
            &["lem-sasaki-total-squared-error-stable", "thm-sasaki-distance-ms"],
            "repaired auxiliary uniform bound: x=(0,1,2),v=0; alive 111 -> 110; identical coordinates; positional displacement=0; structural term retained",
            squared_error, 8. * 2. * 32. / 4.,
        ));
    }
    Ok(witnesses)
}
