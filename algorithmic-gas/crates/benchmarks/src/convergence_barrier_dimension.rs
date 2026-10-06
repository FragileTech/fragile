//! Dimension-explicit logarithmic barrier bounds for the actual final Gaussian.
//!
//! All certificates condition on the complete preparation before the independent
//! final position Gaussian. Means can be anywhere in R^d. No compact support is
//! imposed on the noise, and every swarm observable is averaged over N.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};

fn invalid(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn positive_upper_floor(value: f64) -> f64 {
    // Mathematical factors are strictly positive. A normal-positive floor keeps
    // numerical underflow from being mistaken for a zero-probability certificate.
    value.max(f64::MIN_POSITIVE)
}
fn parameters(d: usize, half_width: f64, noise_std: f64) -> Result<()> {
    if d == 0
        || !half_width.is_finite()
        || half_width <= 0.
        || !noise_std.is_finite()
        || noise_std <= 0.
    {
        return Err(invalid(
            "positive dimension, box half-width and Gaussian standard deviation required",
        ));
    }
    let a = 2. * half_width / ((2. * std::f64::consts::PI).sqrt() * noise_std);
    if !a.is_finite()
        || a <= 0.
        || !(4. * half_width).is_finite()
        || !(1. / ((2. * std::f64::consts::PI).sqrt() * noise_std)).is_finite()
        || (2. * std::f64::consts::PI).sqrt() * noise_std == f64::INFINITY
    {
        return Err(invalid(
            "Gaussian/box density ratio outside finite numerical range",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DimensionBarrierBounds {
    pub dimensions: usize,
    pub half_width: f64,
    pub noise_std: f64,
    pub density_ratio: f64,
    pub uniform_coordinate_survival_upper: f64,
    /// Former full-box density bound. None means floating-point overflow.
    pub old_bound: Option<f64>,
    /// Dropping only the other coordinate survival factors gives this linear bound.
    pub linear_bound: f64,
    pub one_coordinate_layer_cake: f64,
    /// Layer-cake coordinate bound times d, before optional survival factors.
    pub layer_cake_linear_bound: f64,
    /// Uses all remaining coordinate survival factors; always <= linear_bound.
    pub sharp_bound: f64,
    pub log_sharp_bound: f64,
    pub sharp_bound_positive_floor_applied: bool,
}

/// Closed-form uniform bounds, valid for arbitrary conditional means and all N.
/// Floating evaluations are of analytic formulas; no numerical quadrature is used.
pub fn dimension_barrier_bounds(
    d: usize,
    half_width: f64,
    noise_std: f64,
) -> Result<DimensionBarrierBounds> {
    parameters(d, half_width, noise_std)?;
    let a = 2. * half_width / ((2. * std::f64::consts::PI).sqrt() * noise_std);
    let q = a.min(1.);
    let coefficient = 2. * (1. - 2_f64.ln());
    let old = (d as f64 * coefficient) * a.powf(d as f64);
    let layer = if a <= 1. {
        coefficient * a
    } else {
        // Equivalent to 2log(a)-log(2a-1)+2+2a log(1-1/(2a)).
        // This form avoids cancellation in the threshold and keeps large a finite.
        a.ln() - (2. - 1. / a).ln() + 2. + 2. * (a * (-0.5 / a).ln_1p())
    };
    let linear_bound = d as f64 * coefficient * a;
    if !linear_bound.is_finite() || !layer.is_finite() || layer <= 0. {
        return Err(invalid(
            "dimension bound outside supported finite numerical range",
        ));
    }
    let log_sharp_bound = (d as f64).ln() + layer.ln() + (d - 1) as f64 * q.ln();
    let sharp_bound = positive_upper_floor(log_sharp_bound.exp());
    Ok(DimensionBarrierBounds {
        dimensions: d,
        half_width,
        noise_std,
        density_ratio: a,
        uniform_coordinate_survival_upper: q,
        old_bound: old.is_finite().then_some(positive_upper_floor(old)),
        linear_bound,
        one_coordinate_layer_cake: layer,
        layer_cake_linear_bound: d as f64 * layer,
        sharp_bound,
        log_sharp_bound,
        sharp_bound_positive_floor_applied: log_sharp_bound < f64::MIN_POSITIVE.ln(),
    })
}

/// Integral from 0 to r of -log(1-(x/L)^2), including the integrable endpoint.
fn barrier_primitive(r: f64, half_width: f64) -> f64 {
    let u = (r / half_width).clamp(0., 1.);
    let remaining = 1. - u;
    let endpoint_term = if remaining == 0. {
        0.
    } else {
        remaining * remaining.ln()
    };
    half_width * (2. * u - (1. + u) * u.ln_1p() + endpoint_term)
}
fn omitted_product(values: &[f64], first: usize, second: Option<usize>) -> f64 {
    positive_upper_floor(
        values
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != first && Some(*j) != second)
            .map(|(_, value)| value)
            .product(),
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedBarrierBounds {
    pub dimensions: usize,
    pub coordinate_survival_upper: Vec<f64>,
    pub coordinate_barrier_upper: Vec<f64>,
    pub coordinate_barrier_second_moment_upper: Vec<f64>,
    /// E[sum_j phi1(X_j) 1_box(X) | preparation].
    pub mean_upper: f64,
    /// E[(sum_j phi1(X_j) 1_box(X))^2 | preparation].
    pub second_moment_upper: f64,
    pub global_mean_upper: f64,
}

/// State-aware analytic bounds using exact barrier integrals and Gaussian density
/// envelopes on an interior interval and its complementary tails. No quadrature.
pub fn prepared_barrier_bounds(
    means: &[f64],
    half_width: f64,
    noise_std: f64,
) -> Result<PreparedBarrierBounds> {
    let global = dimension_barrier_bounds(means.len(), half_width, noise_std)?;
    if means.iter().any(|x| !x.is_finite()) {
        return Err(invalid("prepared Gaussian means must be finite"));
    }
    let density = 1. / ((2. * std::f64::consts::PI).sqrt() * noise_std);
    let integral = 4. * half_width * (1. - 2_f64.ln());
    let mut survival = vec![];
    let mut first = vec![];
    let mut second = vec![];
    for &mu in means {
        let abs_mu = mu.abs();
        let box_gap = (abs_mu - half_width).max(0.);
        let local_density = positive_upper_floor(
            density * positive_upper_floor((-0.5 * (box_gap / noise_std).powi(2)).exp()),
        );
        let q = positive_upper_floor((2. * half_width * local_density).min(1.));
        let mut g = global
            .one_coordinate_layer_cake
            .min(local_density * integral);
        // phi1(x)^2 <= [-log(1-|x|/L)]^2; integral of the latter is 4L.
        let mut g2 = positive_upper_floor(4. * half_width * local_density);
        if !g2.is_finite() {
            return Err(invalid(
                "second-moment envelope outside finite numerical range",
            ));
        }
        let mut candidates: Vec<f64> = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999,
        ]
        .into_iter()
        .map(|fraction| fraction * half_width)
        .collect();
        candidates.extend(
            [2., 3., 4., 5., 6., 8.]
                .into_iter()
                .map(|k| abs_mu + k * noise_std),
        );
        for r in candidates {
            if r <= abs_mu || r >= half_width {
                continue;
            }
            if r / half_width >= 1. {
                continue;
            }
            let phi = -(-((r / half_width).powi(2))).ln_1p();
            let tail_density = positive_upper_floor(
                density * positive_upper_floor((-0.5 * ((r - abs_mu) / noise_std).powi(2)).exp()),
            );
            let tail_integral = (integral - 2. * barrier_primitive(r, half_width)).max(0.);
            let t = 1. - r / half_width;
            let log_t = t.ln();
            let tail_second_integral_upper =
                2. * half_width * t * (log_t * log_t - 2. * log_t + 2.);
            g = g.min(phi * q + tail_density * tail_integral);
            g2 = g2.min(phi * phi * q + tail_density * tail_second_integral_upper);
        }
        survival.push(q);
        first.push(positive_upper_floor(g));
        second.push(positive_upper_floor(g2));
    }
    let mean: f64 = (0..means.len())
        .map(|j| first[j] * omitted_product(&survival, j, None))
        .sum();
    let mut second_moment: f64 = (0..means.len())
        .map(|j| second[j] * omitted_product(&survival, j, None))
        .sum();
    for j in 0..means.len() {
        for k in j + 1..means.len() {
            second_moment += 2. * first[j] * first[k] * omitted_product(&survival, j, Some(k));
        }
    }
    Ok(PreparedBarrierBounds {
        dimensions: means.len(),
        coordinate_survival_upper: survival,
        coordinate_barrier_upper: first,
        coordinate_barrier_second_moment_upper: second,
        mean_upper: positive_upper_floor(mean.min(global.sharp_bound)),
        second_moment_upper: positive_upper_floor(second_moment),
        global_mean_upper: global.sharp_bound,
    })
}

/// Jensen over swarm rows preserves the normalization 1/N even with correlated
/// cloning. Coordinate independence is needed within a prepared row only.
pub fn prepared_swarm_barrier_bounds(
    means: &[Vec<f64>],
    half_width: f64,
    noise_std: f64,
) -> Result<(f64, f64)> {
    if means.is_empty() || means.iter().any(|row| row.len() != means[0].len()) {
        return Err(invalid("positive swarm with a common dimension required"));
    }
    let mut mean = 0.;
    let mut second = 0.;
    for row in means {
        let bound = prepared_barrier_bounds(row, half_width, noise_std)?;
        mean += bound.mean_upper / means.len() as f64;
        second += bound.second_moment_upper / means.len() as f64;
    }
    Ok((positive_upper_floor(mean), positive_upper_floor(second)))
}

/// One-sided Cantelli allowance for the average of independently seeded swarms.
/// The sum is of conditional second-moment upper bounds, including absorbed zeros.
pub fn independent_seed_allowance(
    second_moment_sum: f64,
    requested: usize,
    delta: f64,
) -> Result<f64> {
    if requested == 0
        || !second_moment_sum.is_finite()
        || second_moment_sum < 0.
        || !delta.is_finite()
        || delta <= 0.
        || delta >= 1.
    {
        return Err(invalid(
            "finite second moment, independent sample count and delta in (0,1) required",
        ));
    }
    Ok(((1. - delta) / delta * second_moment_sum).sqrt() / requested as f64)
}

fn read_json_gzip(path: &Path) -> Result<Value> {
    let file = fs::File::open(path).map_err(|e| invalid(&e.to_string()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    serde_json::from_reader(decoder).map_err(|e| invalid(&e.to_string()))
}
fn checksum(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| invalid(&e.to_string()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| invalid(&e.to_string()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Reanalysis of retained Chapter 6 manifests only; no engine steps are run.
pub fn validate_retained_dimension_bounds(input: &Path) -> Result<Value> {
    let index: Value = serde_json::from_reader(
        fs::File::open(input.join("archive-index.json")).map_err(|e| invalid(&e.to_string()))?,
    )
    .map_err(|e| invalid(&e.to_string()))?;
    let entries = index["entries"]
        .as_array()
        .ok_or_else(|| invalid("archive entries missing"))?;
    let mut provenance = vec![];
    let mut cases = vec![];
    let mut dimension_rows = vec![];
    for d in [1, 2, 4, 8] {
        dimension_rows.push(json!({"d":d,"bounds":dimension_barrier_bounds(d,2.,0.02)?,"old_bound":dimension_barrier_bounds(d,2.,0.02)?.old_bound,"linear_bound":dimension_barrier_bounds(d,2.,0.02)?.linear_bound,"sharp_bound":dimension_barrier_bounds(d,2.,0.02)?.sharp_bound,"scope":"Analytic dimension evaluation at fixed box/noise parameters. Native trajectory cases and their own seed counts are recorded separately."}));
    }
    let mut comparisons = vec![];
    let manifest_entries: Vec<_> = entries
        .iter()
        .filter(|entry| {
            entry["tag"].as_str().is_some_and(|tag| {
                tag.starts_with("chapter06-case") && tag.ends_with("-trajectory-manifest")
            })
        })
        .collect();
    if manifest_entries.is_empty() {
        return Err(invalid("completed case manifests missing"));
    }
    let case_count = manifest_entries.len();
    let delta = 0.01 / (case_count as f64 * 2. * 2. * 9.);
    for entry in manifest_entries {
        let tag = entry["tag"].as_str().unwrap();
        let relative = entry["path"]
            .as_str()
            .ok_or_else(|| invalid("manifest path missing"))?;
        let path = input.join(relative);
        let sha = checksum(&path)?;
        if entry["sha256"].as_str() != Some(&sha) {
            return Err(invalid("retained manifest checksum mismatch"));
        }
        provenance.push(json!({"tag":tag,"path":path,"sha256":sha}));
        let manifest = read_json_gzip(&path)?;
        let case = manifest["case"].as_u64().unwrap();
        let cfg = &manifest["native_config"];
        let d = cfg["dimensions"].as_u64().unwrap() as usize;
        let n = cfg["walkers"].as_u64().unwrap() as usize;
        let half_width = cfg["gas"]["boundary"]["domain"]["upper"][0]
            .as_f64()
            .unwrap();
        let h = cfg["gas"]["kinetic"]["integrator"]["dt"].as_f64().unwrap();
        let noise_std = cfg["gas"]["kinetic"]["position_diffusion"]
            .as_f64()
            .unwrap()
            * h.sqrt();
        let global = dimension_barrier_bounds(d, half_width, noise_std)?;
        let trajectories = manifest["trajectories"].as_array().unwrap();
        let times = manifest["checkpoints"].as_array().unwrap();
        let mut frames = vec![];
        for time in times
            .iter()
            .filter_map(Value::as_u64)
            .filter(|time| *time > 0)
        {
            for convention in ["native_zero_alive", "theorem_fewer_than_two"] {
                let death_key = if convention == "native_zero_alive" {
                    "native_extinction_step"
                } else {
                    "theorem_cemetery_step"
                };
                for distribution in 0..2 {
                    let group: Vec<_> = trajectories
                        .iter()
                        .filter(|t| t["distribution"] == distribution)
                        .collect();
                    let requested = group.len();
                    let mut observed_sum = 0.;
                    let mut prediction_sum = 0.;
                    let mut second_sum = 0.;
                    let mut survivors = 0;
                    let mut entering = 0;
                    for trajectory in group {
                        let absorbed = trajectory[death_key].as_u64();
                        if absorbed.is_none_or(|at| at > time) {
                            let state = trajectory["checkpoints"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|s| s["step"] == time)
                                .ok_or_else(|| invalid("retained live checkpoint missing"))?;
                            observed_sum += state["normalized_log_barrier"].as_f64().unwrap();
                            survivors += 1;
                        }
                        if absorbed.is_some_and(|at| at < time) {
                            continue;
                        }
                        let prediction = trajectory["death_predictions"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|p| p["step"] == time);
                        let Some(prediction) = prediction else {
                            continue;
                        };
                        let rows: Vec<Vec<f64>> = prediction["gaussian_means"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|row| {
                                row.as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|x| x.as_f64().unwrap())
                                    .collect()
                            })
                            .collect();
                        if rows.len() != n || rows.iter().any(|row| row.len() != d) {
                            return Err(invalid(
                                "prepared mean dimension differs from retained configuration",
                            ));
                        }
                        let (first, second) =
                            prepared_swarm_barrier_bounds(&rows, half_width, noise_std)?;
                        prediction_sum += first;
                        second_sum += second;
                        entering += 1;
                    }
                    let observed = observed_sum / requested as f64;
                    let state_bound = prediction_sum / requested as f64;
                    let allowance = independent_seed_allowance(second_sum, requested, delta)?;
                    let passed = observed <= state_bound + allowance + 1e-12;
                    let conditional_observed =
                        (survivors > 0).then(|| observed_sum / survivors as f64);
                    let frame = json!({"step":time,"distribution":distribution,"cemetery_convention":convention,"requested_independent_seeds":requested,"entering_survivors":entering,"survivors":survivors,"observed_unconditional_barrier":observed,"observed_conditional_barrier":conditional_observed,"old_bound":global.old_bound,"linear_bound":global.linear_bound,"sharp_bound":global.sharp_bound,"prepared_state_mean_upper":state_bound,"conditional_second_moment_sum":second_sum,"independent_seed_Cantelli_allowance":allowance,"delta":delta,"passed":passed,"scope":"Conditional tensor Gaussian expectation upper bound; actual observation averages all requested seeds with absorbed zero. Conditional survivor mean is separate and is not compared directly with the unconditional bound."});
                    comparisons.push(frame.clone());
                    frames.push(frame);
                }
            }
        }
        cases.push(json!({"case":case,"N":n,"d":d,"bounds":global,"frames":frames}));
    }
    let failures = comparisons
        .iter()
        .filter(|row| row["passed"] != true)
        .count();
    Ok(
        json!({"chapter":6,"title":"Dimension-explicit Gaussian barrier tightening","dimension_rows":dimension_rows,"cases":cases,"comparisons":comparisons,"summary":{"cases":case_count,"comparisons":comparisons.len(),"comparisons_failed":failures,"new_simulation_steps":0,"family_error_budget":0.01},"provenance":{"input_directory":input,"manifest_sha256":provenance,"source_path":"docs/source/2_fractal_gas/convergence_program/06_convergence.md","source_labels":["lem-convergence-dimension-log-barrier","lem-convergence-state-log-barrier"],"source_sha256":checksum(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source/2_fractal_gas/convergence_program/06_convergence.md"))?,"rust_api_sha256":checksum(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/convergence_barrier_dimension.rs"))?,"executed_binary_sha256":checksum(&std::env::current_exe().map_err(|e|invalid(&e.to_string()))?)?,"integration_method":"Closed-form integrals and analytic Gaussian density envelopes; no numerical quadrature.","floating_evaluation_scope":"Analytic real inequalities evaluated in f64; not arbitrary-precision machine interval arithmetic. Strictly positive factors and final bounds receive a normal-positive floor on underflow; unsupported overflow ranges return errors. Log sharp bounds are retained."}}),
    )
}
