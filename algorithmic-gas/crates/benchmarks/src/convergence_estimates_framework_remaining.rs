//! Explicit operand witnesses for the framework's remaining inline contracts.
//! The index file contains only exact source text. Every emitted expression has
//! an explicit evaluator or a named comparison of the same numerical operands.
use super::*;
use std::collections::BTreeSet;
#[path = "convergence_estimates_framework_remaining_noise.rs"]
mod noise;

const EXPRESSIONS: &str =
    include_str!("../../../proof-validation/chapter01-remaining-source-expressions.json");
const SCOPE: &str = "Explicit finite operand contract. Production scalar statistics and native configuration are retained; analytic distribution and continuous generator definitions are distinguished from the discrete native transition. No finite witness establishes a global hypothesis, limiting assertion or uniform law convergence.";

fn source() -> Result<Value> {
    serde_json::from_str(EXPRESSIONS).map_err(|e| GasError::Configuration(e.to_string()))
}
fn emit(
    suite: &mut EstimateSuite,
    source: &Value,
    indices: &[&str],
    inputs: &Value,
    checks: &[BoundCheck],
    scope: &str,
) -> Result<()> {
    require(
        !checks.is_empty(),
        "empty remaining framework expression comparison",
    )?;
    for index in indices {
        let entry = &source[*index];
        let formula = entry["formula"].as_str().ok_or_else(|| {
            GasError::Configuration(format!("unknown explicit inline index {index}"))
        })?;
        let labels = entry["label"].as_str().into_iter().collect::<Vec<_>>();
        inline_evidence(
            suite,
            &labels,
            formula,
            inputs.clone(),
            checks.to_vec(),
            scope,
        )?;
    }
    Ok(())
}
fn eq(id: &str, a: f64, b: f64) -> BoundCheck {
    identity(id, "chapter01-explicit-inline-operands", a, b)
}
fn le(id: &str, a: f64, b: f64) -> BoundCheck {
    check(id, "chapter01-explicit-inline-operands", a, b)
}
fn truth(id: &str, p: bool) -> BoundCheck {
    le(id, if p { 0. } else { 1. }, 0.)
}
fn numeric_contracts(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let config = GasConfig::euclidean(1, 0.04)?;
    let raw = [-0.8_f64, 0.3, 0.9, -0.1];
    let other = [-0.76_f64, 0.2, 0.82, 0.04];
    let a = [true, true, true, true];
    let b = [true, false, true, true];
    let floor = 0.1_f64;
    let native = Standardizer::Global { sigma_min: floor };
    let obs = observations(&raw)?;
    let (z1, s1) = native.apply(&raw, &a, &obs)?;
    let (z2, s2) = native.apply(&raw, &b, &obs)?;
    let (zv, sv) = native.apply(&other, &a, &obs)?;
    let (zi, si) = native.apply(&other, &b, &obs)?;
    let m1 = s1.mean[0];
    let m2 = s2.mean[0];
    let sc1 = s1.scale[0];
    let sc2 = s2.scale[0];
    let nc = a.iter().zip(b).filter(|(x, y)| **x != *y).count() as f64;
    let stable = a.iter().zip(b).filter(|(x, y)| **x && *y).count();
    let inputs = json!({"actual_native_config":config,"raw":raw,"changed":other,"alive1":a,"alive2":b,"native_z1":z1,"native_z2":z2,"native_z_changed_same_support":zv,"native_z_changed_support":zi,"native_statistics1":s1,"native_statistics2":s2,"native_statistics_changed_same_support":sv,"native_statistics_changed_support":si,"sigma_min":floor,"actual_native_statistics_provider":native,"status_changes":nc,"stable_count":stable,"comparison_plan":"finite admissible coordinate coupling; scalar conclusions do not assign intrinsic walker identities"});
    emit(
        suite,
        src,
        &["0923", "0924"],
        &inputs,
        &[eq("inline_actual_raw_law_mean", m1, mean(&raw))],
        SCOPE,
    )?;
    let var = variance(&raw);
    let vmax = 1_f64;
    emit(
        suite,
        src,
        &["0925"],
        &inputs,
        &[eq(
            "inline_actual_raw_law_regularized_variance",
            sc1,
            (var + floor.powi(2)).sqrt(),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0926", "0932", "0695", "1135"],
        &inputs,
        &[
            le("inline_actual_positive_regularized_scale", floor, sc1),
            truth("inline_actual_floor_strict", floor > 0.),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0958"],
        &inputs,
        &[eq(
            "inline_variance_second_minus_mean",
            var,
            second(&raw) - m1.powi(2),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1010"],
        &inputs,
        &[eq(
            "inline_alive_count",
            4.,
            a.iter().filter(|v| **v).count() as f64,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1094"],
        &inputs,
        &[eq(
            "inline_alive_intersection",
            stable as f64,
            (0..4).filter(|i| a[*i] && b[*i]).count() as f64,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1095", "1141"],
        &inputs,
        &[le(
            "inline_actual_raw_envelope",
            raw.iter().copied().map(f64::abs).fold(0., f64::max),
            vmax,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1096"],
        &inputs,
        &[
            le("inline_first_scale_floor", floor, sc1),
            le("inline_second_scale_floor", floor, sc2),
            truth("inline_floor_positive", floor > 0.),
        ],
        SCOPE,
    )?;
    // The explicit support estimate for the empirical mean and second moment.
    let lmu = 2. * vmax / 4.;
    let lmsecond = 2. * vmax.powi(2) / 4.;
    let ls = (lmsecond + 2. * vmax * lmu) / (2. * floor);
    emit(
        suite,
        src,
        &["1097"],
        &inputs,
        &[le(
            "inline_fixed_raw_mean_status_modulus",
            (m1 - m2).abs(),
            lmu * nc,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1098"],
        &inputs,
        &[le(
            "inline_fixed_raw_scale_status_modulus",
            (sc1 - sc2).abs(),
            ls * nc,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1020"],
        &inputs,
        &[le(
            "inline_variance_range_constant",
            raw.iter().fold(f64::NEG_INFINITY, |v, x| v.max(*x))
                - raw.iter().fold(f64::INFINITY, |v, x| v.min(*x)),
            2. * vmax,
        )],
        "The chosen raw envelope Vmax=1 gives C=2; its range comparison is independently computed from the retained scalar endpoints (no assertion that C is the sharp fixture range).",
    )?;
    for (ids, lhs, rhs) in [
        (
            vec!["1138"],
            nc / 4.,
            a.iter().zip(b).filter(|(x, y)| **x != *y).count() as f64 / 4.,
        ),
        (
            vec!["1139"],
            3. / 4.,
            b.iter().filter(|x| **x).count() as f64 / 4.,
        ),
        (
            vec!["1140"],
            stable as f64 / 4.,
            (0..4).filter(|i| a[*i] && b[*i]).count() as f64 / 4.,
        ),
        (vec!["0929"], floor, (0.0064_f64 + 0.06_f64.powi(2)).sqrt()),
        (vec!["0936"], floor, (0_f64 + floor.powi(2)).sqrt()),
        (vec!["1201"], 1. / sc1.powi(2), 1. / floor.powi(2)),
    ] {
        let checks = if ids[0] == "1201" {
            vec![le("inline_reciprocal_floor", lhs, rhs)]
        } else {
            vec![eq(&format!("inline_operand_{}", ids[0]), lhs, rhs)]
        };
        emit(suite, src, &ids, &inputs, &checks, SCOPE)?;
    }
    let total_z = zv.iter().map(|v| v * v).sum::<f64>();
    emit(
        suite,
        src,
        &["1071", "1210"],
        &inputs,
        &[le(
            "inline_value_standardized_norm",
            total_z,
            4. * (2. * vmax / floor).powi(2),
        )],
        SCOPE,
    )?;
    for (ids, values) in [
        (vec!["1110", "1237"], &z2),
        (vec!["1024"], &z1),
        (vec!["1025"], &z2),
        (vec!["1027"], &zv),
        (vec!["1103"], &z1),
        (vec!["1104"], &z2),
    ] {
        let checks = if ids[0] == "1110" {
            vec![le(
                "inline_structural_standardized_scalar_range",
                values.iter().copied().map(f64::abs).fold(0., f64::max),
                2. * vmax / floor,
            )]
        } else {
            values
                .iter()
                .enumerate()
                .map(|(i, z)| {
                    let (val, mask, mu, scale) = if ids[0] == "1027" {
                        (other[i], a[i], sv.mean[i], sv.scale[i])
                    } else if ids[0] == "1025" || ids[0] == "1104" {
                        (raw[i], b[i], m2, sc2)
                    } else {
                        (raw[i], a[i], m1, sc1)
                    };
                    eq(
                        &format!("inline_native_standardizer_output_{}_{}", ids[0], i),
                        *z,
                        if mask { (val - mu) / scale } else { 0. },
                    )
                })
                .collect()
        };
        emit(suite, src, &ids, &inputs, &checks, SCOPE)?;
    }
    let diff = z1.iter().zip(&z2).map(|(a, b)| a - b).collect::<Vec<_>>();
    emit(
        suite,
        src,
        &["1105", "1233"],
        &inputs,
        &diff
            .iter()
            .enumerate()
            .map(|(i, x)| {
                eq(
                    &format!("inline_actual_structural_difference_{i}"),
                    *x,
                    z1[i] - z2[i],
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1229"],
        &inputs,
        &[eq(
            "inline_native_structural_error_norm",
            squared(&diff),
            z1.iter().zip(&z2).map(|(a, b)| (a - b).powi(2)).sum(),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1235"],
        &inputs,
        &(0..4)
            .filter(|i| a[*i] && b[*i])
            .map(|i| {
                eq(
                    &format!("inline_stable_status_{i}"),
                    f64::from(a[i]),
                    f64::from(b[i]),
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1241"],
        &inputs,
        &[le(
            "inline_structural_mean_square_modulus",
            (m1 - m2).powi(2),
            lmu.powi(2) * nc.powi(2),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1244"],
        &inputs,
        &[le(
            "inline_structural_scale_square_modulus",
            (sc1 - sc2).powi(2),
            ls.powi(2) * nc.powi(2),
        )],
        SCOPE,
    )?;
    let dz = z1.iter().zip(&zv).map(|(a, b)| a - b).collect::<Vec<_>>();
    let dv = raw
        .iter()
        .zip(other)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    emit(
        suite,
        src,
        &["1198"],
        &inputs,
        &[eq(
            "inline_native_value_error_norm",
            squared(&dz),
            z1.iter().zip(&zv).map(|(a, b)| (a - b).powi(2)).sum(),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1204"],
        &inputs,
        &[eq(
            "inline_native_direct_error_scaling",
            dv.iter().map(|v| (v / sc1).powi(2)).sum(),
            squared(&dv) / sc1.powi(2),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1063", "1207"],
        &inputs,
        &[le(
            "inline_native_mean_value_modulus",
            (sv.mean[0] - m1).powi(2),
            squared(&dv) / 4.,
        )],
        SCOPE,
    )?;
    let lscale = 2. * vmax / (floor * 2_f64); // 2 Vmax /(sigma_min sqrt(k)) at k=4.
    emit(
        suite,
        src,
        &["1072", "1211"],
        &inputs,
        &[le(
            "inline_native_scale_value_modulus",
            (sv.scale[0] - sc1).powi(2),
            lscale.powi(2) * squared(&dv),
        )],
        SCOPE,
    )?;
    let mean_shift = [(sv.mean[0] - m1) / sc1; 4];
    emit(
        suite,
        src,
        &["1060"],
        &inputs,
        &mean_shift
            .iter()
            .enumerate()
            .map(|(i, x)| {
                eq(
                    &format!("inline_native_mean_shift_component_{i}"),
                    *x,
                    (sv.mean[0] - m1) / sc1,
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    let fluct = zv
        .iter()
        .map(|v| v * (sv.scale[0] - sc1) / sc1)
        .collect::<Vec<_>>();
    emit(
        suite,
        src,
        &["1067"],
        &inputs,
        &fluct
            .iter()
            .zip(&zv)
            .enumerate()
            .map(|(i, (x, z))| {
                eq(
                    &format!("inline_native_fluctuation_component_{i}"),
                    *x,
                    *z * (sv.scale[0] - sc1) / sc1,
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0964", "0965"],
        &inputs,
        &[
            eq(
                "inline_variance_difference_expansion",
                (variance(&raw) - variance(&other)).abs(),
                ((second(&raw) - mean(&raw).powi(2)) - (second(&other) - mean(&other).powi(2)))
                    .abs(),
            ),
            le(
                "inline_variance_two_moment_triangle",
                (variance(&raw) - variance(&other)).abs(),
                (second(&raw) - second(&other)).abs()
                    + (mean(&raw).powi(2) - mean(&other).powi(2)).abs(),
            ),
        ],
        SCOPE,
    )?;
    for v in [0_f64, 1., 1e2, 1e4, 1e8] {
        let scale = (v + floor.powi(2)).sqrt();
        let deriv = 0.5 / scale;
        let inp = json!({"variance":v,"sigma_min":floor,"regularized_scale":scale,"actual_native_standardizer":native});
        emit(
            suite,
            src,
            &["0959"],
            &inp,
            &[eq(
                "inline_scale_derivative",
                deriv,
                (0.25 / (v + floor.powi(2))).sqrt(),
            )],
            SCOPE,
        )?;
        emit(
            suite,
            src,
            &["0306", "0954", "0960"],
            &inp,
            &[
                eq(
                    "inline_scale_derivative_supremum",
                    0.5 / floor,
                    0.5 / (0_f64 + floor.powi(2)).sqrt(),
                ),
                le("inline_scale_derivative_bound", deriv, 0.5 / floor),
            ],
            SCOPE,
        )?;
        if v > 0. {
            let approx = v.sqrt() + floor.powi(2) / (2. * v.sqrt());
            let rem = floor.powi(4) / (8. * v.powf(1.5));
            emit(
                suite,
                src,
                &["0934"],
                &inp,
                &[le(
                    "inline_scale_asymptotic_remainder",
                    (scale - approx).abs(),
                    rem + 2. * f64::EPSILON * scale,
                )],
                "Finite asymptotic expansion tests retain the rigorous Taylor remainder sigma_min^4/(8 V^(3/2)); no finite sequence proves a limit.",
            )?;
        }
        emit(
            suite,
            src,
            &["0935"],
            &inp,
            &[le(
                "inline_scale_divergence_lower_envelope",
                v.sqrt(),
                scale,
            )],
            "Each retained V checks sigma_reg(V)>=sqrt(V), the explicit diverging lower envelope in the analytic limit argument; this is not an empirical observation of infinity.",
        )?;
    }
    emit(
        suite,
        src,
        &["0970"],
        &inputs,
        &[le(
            "inline_scale_variance_chain",
            (sc1 - sv.scale[0]).abs(),
            (0.5 / floor) * (variance(&raw) - variance(&other)).abs(),
        )],
        SCOPE,
    )?;
    Ok(())
}

fn pool(mask: &[bool], i: usize) -> Vec<usize> {
    let alive = (0..mask.len()).filter(|j| mask[*j]).collect::<Vec<_>>();
    if !mask[i] || alive.len() < 2 {
        alive
    } else {
        alive.into_iter().filter(|j| *j != i).collect()
    }
}
fn uniform_expectation(values: &[f64], support: &[usize]) -> Option<f64> {
    (!support.is_empty())
        .then(|| support.iter().map(|j| values[*j]).sum::<f64>() / support.len() as f64)
}
fn support_contracts(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let config = GasConfig::euclidean(1, 0.04)?;
    let x = [-0.9_f64, -0.2, 0.35, 0.85];
    let obs = observations(&x)?;
    let mut distances = vec![vec![0.; 4]; 4];
    for (i, row) in distances.iter_mut().enumerate() {
        for (j, d) in row.iter_mut().enumerate() {
            *d = config.distance_donors.distance.compare(&obs, i, &obs, j)?;
        }
    }
    let diameter = 2_f64;
    let scope = "Exhaustively enumerated finite uniform companion supports, including the empty, singleton self, dead recipient and nonself live branches. Native AlgorithmicDistance supplies the raw observable. Conditional uniform laws are the reference specialization; weighted Gaussian donor laws are not substituted into a uniform-support lemma. All N-growing vector sums are auxiliary quantities; primary physical errors use division by N.";
    for ma in 0_usize..16 {
        let a = (0..4).map(|i| ma & (1 << i) != 0).collect::<Vec<_>>();
        let k = a.iter().filter(|v| **v).count();
        for i in 0..4 {
            let s = pool(&a, i);
            let p = if s.is_empty() {
                0.
            } else {
                1. / s.len() as f64
            };
            let input = json!({"actual_native_config":config,"positions":x,"alive":a,"recipient":i,"companion_support":s,"reference_kernel":"uniform independent draws","probability_per_element":p,"native_pair_distances":distances,"D":diameter});
            if k == 0 {
                emit(
                    suite,
                    src,
                    &["0459", "0460"],
                    &input,
                    &[
                        eq("inline_empty_companion_support", s.len() as f64, 0.),
                        eq("inline_empty_alive_count", k as f64, 0.),
                    ],
                    scope,
                )?;
                continue;
            }
            emit(
                suite,
                src,
                &["0461", "0464", "0740", "0742"],
                &input,
                &[
                    eq("inline_uniform_support_mass", p * s.len() as f64, 1.),
                    truth("inline_nonempty_companion_support", !s.is_empty()),
                ],
                scope,
            )?;
            if a[i] && k >= 2 {
                let expected = (0..4).filter(|j| a[*j] && *j != i).collect::<Vec<_>>();
                emit(
                    suite,
                    src,
                    &["0451", "0453", "0741", "0752"],
                    &input,
                    &[
                        truth("inline_nonself_set_equality", s == expected),
                        eq("inline_nonself_size", s.len() as f64, (k - 1) as f64),
                        truth("inline_nonself_alive_count", k >= 2),
                    ],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0726"],
                    &input,
                    &[eq("inline_live_raw_recipient", f64::from(a[i]), 1.)],
                    scope,
                )?;
            } else if !a[i] {
                let expected = (0..4).filter(|j| a[*j]).collect::<Vec<_>>();
                emit(
                    suite,
                    src,
                    &["0454", "0455"],
                    &input,
                    &[
                        truth("inline_dead_recipient_live_set", s == expected),
                        truth("inline_dead_recipient_live_count", k >= 1),
                    ],
                    scope,
                )?;
            } else {
                emit(
                    suite,
                    src,
                    &["0456", "0457", "0746", "0747", "0857", "0862"],
                    &input,
                    &[
                        truth("inline_singleton_self_set", s == vec![i]),
                        eq("inline_singleton_live_count", k as f64, 1.),
                    ],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0863"],
                    &input,
                    &[eq("inline_singleton_self_companion", s[0] as f64, i as f64)],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0865", "0873", "0874"],
                    &input,
                    &[eq(
                        "inline_singleton_native_self_raw_zero",
                        distances[i][i],
                        0.,
                    )],
                    scope,
                )?;
            }
            let values = s.iter().map(|j| distances[i][*j]).collect::<Vec<_>>();
            let mean = values.iter().sum::<f64>() * p;
            let second = values.iter().map(|v| v * v).sum::<f64>() * p;
            emit(
                suite,
                src,
                &["0462", "0780", "0818"],
                &input,
                &[eq(
                    "inline_uniform_expectation_mean",
                    mean,
                    values.iter().sum::<f64>() / s.len() as f64,
                )],
                scope,
            )?;
            emit(
                suite,
                src,
                &["0877"],
                &input,
                &[
                    eq(
                        "inline_categorical_variance_identity",
                        variance(&values),
                        second - mean.powi(2),
                    ),
                    le(
                        "inline_categorical_variance_second_bound",
                        variance(&values),
                        second,
                    ),
                ],
                scope,
            )?;
            emit(
                suite,
                src,
                &["0879", "0880"],
                &input,
                &[
                    le(
                        "inline_native_raw_distance_range",
                        values.iter().copied().fold(0., f64::max),
                        diameter,
                    ),
                    le(
                        "inline_native_raw_distance_square_range",
                        values.iter().map(|v| v * v).fold(0., f64::max),
                        diameter.powi(2),
                    ),
                    le(
                        "inline_native_raw_distance_nonnegative",
                        0.,
                        values.iter().copied().fold(f64::INFINITY, f64::min),
                    ),
                ],
                scope,
            )?;
            emit(
                suite,
                src,
                &["0738", "0751"],
                &input,
                &[le(
                    "inline_native_raw_distance_Mf",
                    values.iter().copied().map(f64::abs).fold(0., f64::max),
                    diameter,
                )],
                scope,
            )?;
        }
        for mb in 0_usize..16 {
            let b = (0..4).map(|i| mb & (1 << i) != 0).collect::<Vec<_>>();
            let nc = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let unstable = (0..4).filter(|i| a[*i] != b[*i]).collect::<Vec<_>>();
            let input = json!({"positions":x,"alive1":a,"alive2":b,"unstable_set":unstable,"native_distances":distances,"status_changes":nc,"D":diameter});
            let bit_norm = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>();
            emit(
                suite,
                src,
                &["0367", "0764", "0767", "0771", "0812", "0840"],
                &input,
                &[
                    eq(
                        "inline_symmetric_difference_status_bits",
                        unstable.len() as f64,
                        bit_norm,
                    ),
                    eq("inline_status_change_count", nc as f64, bit_norm),
                ],
                scope,
            )?;
            for i in 0..4 {
                let sa = pool(&a, i);
                let sb = pool(&b, i);
                let m = |support: &[usize], alive: bool| {
                    if !alive || support.is_empty() {
                        0.
                    } else {
                        support.iter().map(|j| distances[i][*j]).sum::<f64>() / support.len() as f64
                    }
                };
                let ea = m(&sa, a[i]);
                let eb = m(&sb, b[i]);
                if !sa.is_empty() && sa == sb {
                    emit(
                        suite,
                        src,
                        &["0496"],
                        &json!({"fixture":input,"recipient":i,"first_companion_support":sa,"second_companion_support":sb,"source_function_values":distances[i],"raw_receiver_masked_expectation1":ea,"raw_receiver_masked_expectation2":eb,"unmasked_companion_expectation1":uniform_expectation(&distances[i],&sa),"unmasked_companion_expectation2":uniform_expectation(&distances[i],&sb)}),
                        &[eq(
                            "inline_identical_support_expectation_error_zero",
                            (uniform_expectation(&distances[i], &sa).unwrap()
                                - uniform_expectation(&distances[i], &sb).unwrap())
                            .abs(),
                            0.,
                        )],
                        scope,
                    )?;
                }
                emit(
                    suite,
                    src,
                    &["0497"],
                    &input,
                    &[le("inline_own_raw_status_range", (ea - eb).abs(), diameter)],
                    scope,
                )?;
                let row_input = json!({"fixture":input,"recipient":i,"companion_support1":sa,"companion_support2":sb,"native_raw_expectation1":ea,"native_raw_expectation2":eb});
                if a[i] && !b[i] {
                    emit(
                        suite,
                        src,
                        &["0756", "0757"],
                        &row_input,
                        &[
                            eq("inline_dead_second_raw_expectation", eb, 0.),
                            truth("inline_status_live_to_dead", a[i] && !b[i]),
                        ],
                        scope,
                    )?;
                }
                if !a[i] && b[i] {
                    emit(
                        suite,
                        src,
                        &["0760", "0761", "0762"],
                        &row_input,
                        &[
                            eq("inline_dead_first_raw_expectation", ea, 0.),
                            le("inline_revived_second_raw_range", eb, diameter),
                            truth("inline_status_dead_to_live", !a[i] && b[i]),
                        ],
                        scope,
                    )?;
                }
                if a[i] && b[i] && k >= 2 && b.iter().filter(|x| **x).count() == 1 {
                    let delta = sa.iter().filter(|j| !sb.contains(j)).count()
                        + sb.iter().filter(|j| !sa.contains(j)).count();
                    emit(
                        suite,
                        src,
                        &["0748", "0749"],
                        &row_input,
                        &[
                            le(
                                "inline_singleton_collapse_status_lower",
                                (k - 1) as f64,
                                nc as f64,
                            ),
                            eq(
                                "inline_singleton_support_difference",
                                delta as f64,
                                k as f64,
                            ),
                        ],
                        scope,
                    )?;
                }
                if sa.is_empty() || sb.is_empty() {
                    continue;
                }
                let f = x.map(|v| v.sin());
                let mf = 1_f64;
                let sum_a = sa.iter().map(|j| f[*j]).sum::<f64>();
                let sum_b = sb.iter().map(|j| f[*j]).sum::<f64>();
                let onlya = sa
                    .iter()
                    .copied()
                    .filter(|j| !sb.contains(j))
                    .collect::<BTreeSet<_>>();
                let onlyb = sb
                    .iter()
                    .copied()
                    .filter(|j| !sa.contains(j))
                    .collect::<BTreeSet<_>>();
                let intersection = sa
                    .iter()
                    .copied()
                    .filter(|j| sb.contains(j))
                    .collect::<BTreeSet<_>>();
                let fulla = onlya.union(&intersection).copied().collect::<Vec<_>>();
                let fullb = onlyb.union(&intersection).copied().collect::<Vec<_>>();
                let na = sa.len() as f64;
                let nb = sb.len() as f64;
                let status_diff = onlya.len() + onlyb.len();
                let scalar_input = json!({"fixture":row_input,"raw_function":"sin(x)","function_values":f,"Mf":mf,"sum1":sum_a,"sum2":sum_b,"only1":onlya,"only2":onlyb,"intersection":intersection});
                emit(
                    suite,
                    src,
                    &["0469", "0482", "0489"],
                    &scalar_input,
                    &x.iter()
                        .zip(f)
                        .enumerate()
                        .map(|(j, (x, f))| eq(&format!("inline_function_value_{j}"), f, x.sin()))
                        .collect::<Vec<_>>(),
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0470", "0483", "0490"],
                    &scalar_input,
                    &[
                        truth("inline_first_support_nonempty", na > 0.),
                        truth("inline_second_support_nonempty", nb > 0.),
                    ],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0471", "0477"],
                    &scalar_input,
                    &[le(
                        "inline_pointwise_function_envelope",
                        f.iter().copied().map(f64::abs).fold(0., f64::max),
                        mf,
                    )],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0474", "0475"],
                    &scalar_input,
                    &[
                        truth("inline_first_set_partition", fulla == sa),
                        truth("inline_second_set_partition", fullb == sb),
                    ],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0479"],
                    &scalar_input,
                    &[eq(
                        "inline_symmetric_difference_union_count",
                        status_diff as f64,
                        sa.iter().filter(|j| !sb.contains(j)).count() as f64
                            + sb.iter().filter(|j| !sa.contains(j)).count() as f64,
                    )],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0473"],
                    &scalar_input,
                    &[le(
                        "inline_normalized_set_sum_absolute",
                        (sum_a - sum_b).abs() / na,
                        mf * status_diff as f64 / na,
                    )],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0485"],
                    &scalar_input,
                    &[le(
                        "inline_normalization_reciprocal_product",
                        (1. / na - 1. / nb).abs() * sum_b.abs(),
                        mf * (na - nb).abs() / na,
                    )],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0486"],
                    &scalar_input,
                    &[
                        le(
                            "inline_support_sum_absolute_triangle",
                            sum_b.abs(),
                            sb.iter().map(|j| f[*j].abs()).sum(),
                        ),
                        le(
                            "inline_support_absolute_sum_envelope",
                            sb.iter().map(|j| f[*j].abs()).sum(),
                            nb * mf,
                        ),
                    ],
                    scope,
                )?;
                emit(
                    suite,
                    src,
                    &["0491"],
                    &scalar_input,
                    &[eq(
                        "inline_two_uniform_expectations_error",
                        (sum_a / na - sum_b / nb).abs(),
                        (sa.iter().map(|j| f[*j] / na).sum::<f64>()
                            - sb.iter().map(|j| f[*j] / nb).sum::<f64>())
                        .abs(),
                    )],
                    scope,
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn append(suite: &mut EstimateSuite) -> Result<()> {
    let src = source()?;
    numeric_contracts(suite, &src)?;
    support_contracts(suite, &src)?;
    revival_contracts(suite, &src)?;
    empirical_law_contracts(suite, &src)?;
    branch_contracts(suite, &src)?;
    exact_aliases(suite, &src)?;
    noise::append(suite, &src)?;
    Ok(())
}

fn revival_contracts(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    use algorithmic_gas::fitness::{PositiveMap, PositiveMapping};
    let mut cfg = GasConfig::euclidean(1, 0.04)?;
    let eta = 0.1_f64;
    let amplitude = 2_f64;
    cfg.fitness.reward_map = PositiveMap::Logistic {
        amplitude,
        floor: eta,
    };
    cfg.fitness.diversity_map = cfg.fitness.reward_map.clone();
    let alpha = cfg.fitness.reward_exponent;
    let beta = cfg.fitness.diversity_exponent;
    let epsilon = cfg.clone_decision.epsilon;
    let pmax = cfg.clone_decision.saturation;
    let lower = eta.powf(alpha + beta);
    let upper = (amplitude + eta).powf(alpha + beta);
    let ratio = lower / (epsilon * pmax);
    let z = [-3_f64, -0.4, 0.7, 2.];
    let fitness = cfg.fitness.combine(&z, &z, &[true; 4])?;
    let input = json!({"actual_native_config":cfg,"standardized_reward":z,"standardized_diversity":z,"actual_native_fitness":fitness,"eta":eta,"alpha":alpha,"beta":beta,"epsilon":epsilon,"pmax":pmax,"lower":lower,"upper":upper,"ratio":ratio,"scope":"Generic threshold sufficient condition evaluated on native fitness operands; actual native dead-recipient branch revives unconditionally from live companions."});
    emit(
        suite,
        src,
        &["0122", "0236", "1634"],
        &input,
        &[
            eq(
                "inline_revival_ratio",
                ratio,
                eta.powf(alpha) * eta.powf(beta) / epsilon / pmax,
            ),
            truth("inline_revival_ratio_strict", ratio > 1.),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0123", "0130", "0237", "1608"],
        &input,
        &[truth("inline_positive_revival_ratio", ratio > 1.)],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0137", "1462"],
        &input,
        &[truth(
            "inline_revival_floor_exceeds_threshold_product",
            epsilon * pmax < lower,
        )],
        SCOPE,
    )?;
    let lower_checks = fitness
        .iter()
        .enumerate()
        .map(|(i, v)| le(&format!("inline_native_live_fitness_lower_{i}"), lower, *v))
        .collect::<Vec<_>>();
    emit(suite, src, &["0147", "1459"], &input, &lower_checks, SCOPE)?;
    let scores = fitness.iter().map(|v| v / epsilon).collect::<Vec<_>>();
    emit(
        suite,
        src,
        &["0065", "0073", "0142", "0152"],
        &input,
        &scores
            .iter()
            .enumerate()
            .map(|(i, s)| truth(&format!("inline_strict_dead_generic_score_{i}"), *s > pmax))
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1456"],
        &input,
        &scores
            .iter()
            .zip(&fitness)
            .enumerate()
            .map(|(i, (s, v))| {
                eq(
                    &format!("inline_dead_generic_score_definition_{i}"),
                    *s,
                    *v / epsilon,
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1629", "1945"],
        &input,
        &[eq(
            "inline_potential_floor_factorization",
            lower,
            eta.powf(alpha) * eta.powf(beta),
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1944"],
        &input,
        &[eq(
            "inline_potential_ceiling_factorization",
            upper,
            (amplitude + eta).powf(alpha) * (amplitude + eta).powf(beta),
        )],
        SCOPE,
    )?;
    let range = fitness
        .iter()
        .enumerate()
        .flat_map(|(i, v)| {
            [
                le(&format!("inline_potential_lower_{i}"), 0., *v),
                le(&format!("inline_potential_upper_{i}"), *v, upper),
            ]
        })
        .collect::<Vec<_>>();
    emit(suite, src, &["0225"], &input, &range, SCOPE)?;
    let subcritical_epsilon = 2. * lower / pmax;
    let subratio = lower / (subcritical_epsilon * pmax);
    let sub = json!({"eta":eta,"alpha":alpha,"beta":beta,"epsilon":subcritical_epsilon,"pmax":pmax,"revival_ratio":subratio,"generic_floor_acceptance":(subratio).min(1.),"native_dead_recipient_branch":"unconditional, distinct from this generic threshold regime"});
    emit(
        suite,
        src,
        &["0127", "0131"],
        &sub,
        &[
            truth("inline_subcritical_revival_ratio", subratio < 1.),
            eq("inline_subcritical_floor_acceptance", subratio, 0.5),
        ],
        "Explicit generic-threshold subcritical parameter case. The axiom's strict sufficient revival condition is not satisfied; this does not predict persistence of native deaths.",
    )?;
    let mut rng = RandomStream::new(730441, 0, Stream::Accept, 0, 0);
    let thresholds = (0..64)
        .map(|_| pmax * rng.uniform::<f64>())
        .collect::<Vec<_>>();
    let threshold_input = json!({"thresholds":thresholds,"pmax":pmax,"seed":730441,"address":[0,"Accept",0,0],"rule":"strict score > threshold","native_random_provider":"RandomStream.uniform"});
    let threshold_checks = thresholds
        .iter()
        .enumerate()
        .flat_map(|(i, t)| {
            [
                le(&format!("inline_uniform_threshold_lower_{i}"), 0., *t),
                le(&format!("inline_uniform_threshold_upper_{i}"), *t, pmax),
                truth(&format!("inline_singleton_clone_false_{i}"), 0_f64 <= *t),
            ]
        })
        .collect::<Vec<_>>();
    emit(
        suite,
        src,
        &["0153", "1620", "1621"],
        &threshold_input,
        &threshold_checks,
        "Retained native uniform thresholds and exact singleton score 0. The source event 0>T is FALSE for every nonnegative threshold, including equality T=0 under the strict acceptance rule. T=0 is recorded as an endpoint event, not asserted for sampled thresholds or credited as a positive-probability atom.",
    )?;
    let boundary_threshold = 0_f64;
    let singleton_score = cfg
        .clone_decision
        .acceptance_probability(0, fitness[0], fitness[0]);
    emit(
        suite,
        src,
        &["1623"],
        &json!({"threshold":0.,"singleton_score":0.,"strict_acceptance":false,"scope":"endpoint operand case, not a sampled atom of positive mass"}),
        &[
            eq("inline_threshold_zero_endpoint", 0., 0.),
            truth(
                "inline_zero_score_zero_threshold_does_not_clone",
                boundary_threshold >= singleton_score,
            ),
        ],
        "Explicit zero-threshold endpoint of the strict singleton rule. This is a boundary operand evaluation, not an assertion that continuous Uniform thresholds hit an endpoint with positive probability.",
    )?;
    let boundary = [0_f64, pmax];
    emit(
        suite,
        src,
        &["0069"],
        &json!({"fixture":input,"threshold_endpoints":boundary}),
        &boundary
            .iter()
            .enumerate()
            .map(|(i, t)| {
                truth(
                    &format!("inline_dead_score_threshold_endpoint_{i}"),
                    scores[0] > pmax && pmax >= *t,
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    let gap = fitness[3] - fitness[0];
    let delta = 0.5 * gap;
    let activity_fraction = 0.25_f64;
    let companion_mass = 1_f64 / 3.;
    let r = activity_fraction;
    let forced = r * companion_mass * (delta / (pmax * (upper + epsilon))).min(1.);
    let active = json!({"native_fitness":fitness,"recipient":0,"higher_donor":3,"gap":gap,"Delta":delta,"a":companion_mass,"r":r,"I":[0],"N":4,"Vpotmax":upper,"epsilon":epsilon,"pmax":pmax,"forced_mean_acceptance_floor":forced,"companion_reference":"uniform nonself donor mass 1/3"});
    emit(
        suite,
        src,
        &["0226", "0271"],
        &active,
        &[
            le("inline_forced_activity_gap", delta, gap),
            truth("inline_forced_activity_positive_gap", delta > 0.),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0228", "0272"],
        &active,
        &[truth(
            "inline_forced_activity_positive_donor_mass",
            companion_mass > 0.,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0231", "0274"],
        &active,
        &[
            eq("inline_forced_activity_fraction", r, 1_f64 / 4.),
            truth("inline_forced_activity_fraction_positive", r > 0.),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0275"],
        &active,
        &[
            truth("inline_forced_activity_lower_bound_positive", forced > 0.),
            le(
                "inline_forced_activity_lower_bound_actual_gate",
                forced,
                r * companion_mass
                    * ((fitness[3] - fitness[0]) / (pmax * (fitness[0] + epsilon))).clamp(0., 1.),
            ),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0273"],
        &active,
        &[eq(
            "inline_constant_fitness_clone_probability_zero",
            cfg.clone_decision.acceptance_probability(
                0,
                fitness[0],
                cfg.fitness.combine(&[z[0]], &[z[0]], &[true])?[0],
            ),
            0.,
        )],
        "The no-activity constant fitness case evaluates the actual clipped probability to zero; a positive activity bound requires the explicit gap and donor-mass hypotheses.",
    )?;
    let parameters = json!({"epsilon_std":0.1,"kappa_anisotropy":2.,"noise_factor_eigenvalues":[1.,2.],"kappa_drift":1.,"kappa_var":100.,"c_min":0.25,"native_standardizer":Standardizer::Global{sigma_min:0.1},"scope":"Declared parameter witness; kappa_var >> 1 is qualitative and has no mathematical numeric threshold in the source."});
    emit(
        suite,
        src,
        &["0254"],
        &parameters,
        &[truth(
            "inline_positive_standardization_regularizer",
            0.1_f64 > 0.,
        )],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0257"],
        &parameters,
        &[truth("inline_anisotropy_eigenvalue_ratio", 2_f64 / 1. > 1.)],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0258"],
        &parameters,
        &[truth("inline_reference_drift_positive", 1_f64 > 0.)],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["0261"],
        &parameters,
        &[eq("inline_variance_regime_witness", 100., 10_f64.powi(2))],
        "Explicit kappa_var=100 regime witness. The printed >>1 has no defined numerical threshold, so this evaluates the chosen regime, not an unprovided global lower bound.",
    )?;
    emit(
        suite,
        src,
        &["0216"],
        &parameters,
        &[truth(
            "inline_relative_collapse_parameter",
            (0_f64..=1.).contains(&parameters["c_min"].as_f64().unwrap())
                && parameters["c_min"].as_f64().unwrap() > 0.,
        )],
        SCOPE,
    )?;
    // Native sphere evaluator and a genuine reward-factorization example.
    let y = 0.3_f64;
    let x1 = [y, 1.];
    let x2 = [y, -2.];
    let reward = crate::Benchmark::Sphere.value(&[y])?;
    let reward_input = json!({"feature_projection":"first coordinate","x1":x1,"x2":x2,"projected1":x1[0],"projected2":x2[0],"reward1":reward,"reward2":crate::Benchmark::Sphere.value(&[x2[0]])?,"feature_valid_region":[-1.,1.],"reward_Lipschitz":2.,"alpha_R":1.,"D":2.});
    emit(
        suite,
        src,
        &["0177", "0190", "0191", "0193"],
        &reward_input,
        &[
            eq(
                "inline_sphere_reward_projection_factorization",
                reward,
                y * y,
            ),
            eq(
                "inline_equal_projected_rewards",
                reward,
                crate::Benchmark::Sphere.value(&[x2[0]])?,
            ),
            eq("inline_equal_features", x1[0], x2[0]),
        ],
        "Feature-factorized Sphere reward on the explicit one-dimensional valid chart [-1,1]; this scope does not assert an unbounded global Sphere Lipschitz constant.",
    )?;
    emit(
        suite,
        src,
        &["0183", "0184", "0199"],
        &reward_input,
        &[
            truth("inline_reward_positive_modulus", 2_f64 > 0.),
            truth(
                "inline_reward_holder_exponent",
                (0_f64..=1.).contains(&reward_input["alpha_R"].as_f64().unwrap())
                    && reward_input["alpha_R"].as_f64().unwrap() > 0.,
            ),
            truth(
                "inline_valid_feature_diameter_finite",
                (1_f64 - (-1.)).is_finite(),
            ),
        ],
        SCOPE,
    )?;
    let rmin = 0.6_f64;
    let h = 0.4_f64;
    let c = 0.2_f64;
    let centered = (-0.4_f64 + 0.4) / 2.;
    emit(
        suite,
        src,
        &["0176", "0179", "0180", "0181"],
        &json!({"r_min":rmin,"h":h,"centre":c,"uniform_support":[-h,h],"law_mean":centered,"law_variance":h*h/3.}),
        &[
            truth("inline_positive_richness_radius", rmin > 0. && rmin <= 2.),
            le("inline_richness_interval_halfwidth", rmin / 2., h),
            eq("inline_shifted_uniform_mean", c + centered, c),
        ],
        "Uniform reference interval with exact mean and halfwidth operands; its analytically uniform law is a reference measure, not a claim that observed native companions are uniform on a continuum interval.",
    )?;
    // The native logistic branch uses precisely the positive maps retained above.
    for t in [-4_f64, -1., 0., 0.1, 1., 4., 12.] {
        let mapped = cfg.fitness.reward_map.map(t)?;
        let pure = mapped - eta;
        let exp = (-t).exp();
        let derivative = amplitude * exp / (1. + exp).powi(2);
        let p = json!({"z":t,"actual_native_positive_map":cfg.fitness.reward_map,"actual_mapped_value":mapped,"g_A":pure,"eta":eta,"derivative":derivative,"L_g":0.5});
        emit(
            suite,
            src,
            &["0697", "0699", "0701"],
            &p,
            &[
                truth("inline_logistic_exponential_positive", exp > 0.),
                truth("inline_native_logistic_positive", pure > 0.),
                truth("inline_logistic_derivative_positive", derivative > 0.),
            ],
            SCOPE,
        )?;
        emit(
            suite,
            src,
            &["0704", "1920"],
            &p,
            &[
                eq("inline_logistic_derivative_global_max", 0.5, amplitude / 4.),
                le("inline_logistic_derivative_cap", derivative, 0.5),
            ],
            SCOPE,
        )?;
        if t == 0. {
            emit(
                suite,
                src,
                &["0705", "0706"],
                &p,
                &[
                    eq("inline_logistic_derivative_at_origin", derivative, 0.5),
                    eq("inline_logistic_origin", t, 0.),
                ],
                SCOPE,
            )?;
        }
        if t > 0. {
            emit(
                suite,
                src,
                &["0702"],
                &p,
                &[le("inline_logistic_asymptotic_defect", 2. - pure, 2. * exp)],
                "Finite upper-tail sequence checks the explicit convergence envelope 0<2-g_A(z)<=2 exp(-z). The infinite-limit statement retains its analytic proof obligation.",
            )?;
        }
    }
    Ok(())
}

fn empirical_law_contracts(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let cfg = GasConfig::euclidean(1, 0.04)?;
    let x = [-0.9_f64, -0.2, 0.35];
    let y = [-0.86_f64, -0.26, 0.4];
    let (out, laws) = enumerate_measurements(&x, &cfg)?;
    let (out2, laws2) = enumerate_measurements(&y, &cfg)?;
    let obs = observations(&x)?;
    let n = x.len();
    let joint=out.iter().map(|o|json!({"probability":o.probability,"distance":o.distance,"native_fitness":o.fitness,"native_z":o.standardized})).collect::<Vec<_>>();
    let input = json!({"native_config":cfg,"positions1":x,"positions2":y,"complete_joint_raw_measurement_law1":joint,"conditional_measurement_marginals1":laws,"conditional_measurement_marginals2":laws2,"complete_joint_mass1":out.iter().map(|o|o.probability).sum::<f64>(),"complete_joint_mass2":out2.iter().map(|o|o.probability).sum::<f64>()});
    let mass = vec![
        eq(
            "inline_native_raw_joint_mass1",
            out.iter().map(|o| o.probability).sum(),
            1.,
        ),
        eq(
            "inline_native_raw_joint_mass2",
            out2.iter().map(|o| o.probability).sum(),
            1.,
        ),
    ];
    emit(
        suite,
        src,
        &["0708", "0711", "0712", "0717", "0869", "0885"],
        &input,
        &mass,
        "Exactly enumerated native conditional raw-distance vector laws at N=3, using actual Gaussian donor weights and production distance and fitness pipeline. The sample-space arrays are explicit conditional outcomes, not fixed means substituted into random standardization.",
    )?;
    emit(
        suite,
        src,
        &["0716", "0897"],
        &input,
        &[
            le(
                "inline_raw_variance_nonnegative_parameter",
                0.,
                n as f64 * 2_f64.powi(2),
            ),
            le(
                "inline_raw_variance_normalized_auxiliary",
                out.iter()
                    .map(|o| o.probability * squared(&o.distance))
                    .sum::<f64>()
                    / n as f64,
                2_f64.powi(2),
            ),
        ],
        "The total-vector variance envelope is the auxiliary ND^2 quantity; the physical 1/N-normalized envelope is D^2, independent of N.",
    )?;
    for i in 0..n {
        for j in 0..n {
            let d = cfg.distance_donors.distance.compare(&obs, i, &obs, j)?;
            emit(
                suite,
                src,
                &["0720", "0739"],
                &json!({"native_metric":cfg.distance_donors.distance,"positions":x,"recipient":i,"donor":j,"actual_native_distance":d}),
                &[eq(
                    "inline_native_raw_distance_definition",
                    d,
                    cfg.distance_donors.distance.compare(&obs, j, &obs, i)?,
                )],
                "Actual production AlgorithmicDistance observable and symmetry identity. Uniform-support proofs retain their declared uniform specialization separately from these native Gaussian draw laws.",
            )?;
        }
    }
    let raw = out[0].distance.clone();
    let raw2 = out2[0].distance.clone();
    let uniform = vec![1. / n as f64; n];
    let mut support = vec![];
    let mut weights = vec![];
    for law in &laws {
        for (v, p) in law {
            support.push(*v);
            weights.push(*p / n as f64);
        }
    }
    let a = raw
        .iter()
        .chain(&support)
        .copied()
        .fold(f64::INFINITY, f64::min);
    let b = raw
        .iter()
        .chain(&support)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let d = b - a;
    let (w1, w2) = scalar_transport(&raw, &uniform, &support, &weights)?;
    let mut knots = raw.iter().chain(&support).copied().collect::<Vec<_>>();
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let cdf = knots
        .windows(2)
        .map(|ab| {
            let midpoint = 0.5 * (ab[0] + ab[1]);
            let f = raw
                .iter()
                .zip(&uniform)
                .filter(|(v, _)| **v <= midpoint)
                .map(|(_, p)| p)
                .sum::<f64>();
            let g = support
                .iter()
                .zip(&weights)
                .filter(|(v, _)| **v <= midpoint)
                .map(|(_, p)| p)
                .sum::<f64>();
            (ab[1] - ab[0]) * (f - g).abs()
        })
        .sum::<f64>();
    let scalar = json!({"native_fixture":input,"empirical_atom_values":raw,"empirical_weights":uniform,"conditional_mixture_atom_values":support,"conditional_mixture_weights":weights,"a":a,"b":b,"D":d,"W1_monotone_transport":w1,"W1_independent_CDF_integral":cdf,"W2_squared":w2});
    emit(
        suite,
        src,
        &["1148"],
        &scalar,
        &[eq("inline_empirical_support_diameter", d, b - a)],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1149"],
        &scalar,
        &raw.iter()
            .enumerate()
            .flat_map(|(i, v)| {
                [
                    le(&format!("inline_empirical_lower_endpoint_{i}"), a, *v),
                    le(&format!("inline_empirical_upper_endpoint_{i}"), *v, b),
                ]
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1150"],
        &scalar,
        &[
            eq(
                "inline_empirical_atom_probability_mass",
                uniform.iter().sum(),
                1.,
            ),
            eq(
                "inline_empirical_first_moment",
                raw.iter().zip(&uniform).map(|(v, p)| v * p).sum(),
                mean(&raw),
            ),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1151"],
        &scalar,
        &[
            eq(
                "inline_conditional_mixture_probability_mass",
                weights.iter().sum(),
                1.,
            ),
            eq(
                "inline_conditional_mixture_first_moment",
                support.iter().zip(&weights).map(|(v, p)| v * p).sum(),
                laws.iter()
                    .map(|law| law.iter().map(|(v, p)| v * p).sum::<f64>() / n as f64)
                    .sum(),
            ),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1156"],
        &scalar,
        &[eq("inline_W1_quantile_CDF_identity", w1, cdf)],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1159"],
        &scalar,
        &raw.iter()
            .flat_map(|x| {
                support.iter().map(move |y| {
                    le(
                        "inline_pointwise_squared_range_transport",
                        (x - y).powi(2),
                        d * (x - y).abs(),
                    )
                })
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    let floor = 0.1_f64;
    let (z1, _) = Standardizer::Global { sigma_min: floor }.apply(&raw, &[true; 3], &obs)?;
    let (z2, _) = Standardizer::Global { sigma_min: floor }.apply(&raw2, &[true; 3], &obs)?;
    emit(
        suite,
        src,
        &["0891", "0892"],
        &input,
        &[
            eq(
                "inline_actual_raw_vector_first_norm",
                squared(&raw),
                out[0].distance.iter().map(|v| v * v).sum(),
            ),
            eq(
                "inline_actual_raw_vector_second_norm",
                squared(&raw2),
                out2[0].distance.iter().map(|v| v * v).sum(),
            ),
        ],
        SCOPE,
    )?;
    emit(
        suite,
        src,
        &["1167"],
        &json!({"native_fixture":input,"native_z1":z1,"native_z2":z2}),
        &z1.iter()
            .zip(&z2)
            .enumerate()
            .map(|(i, (a, b))| {
                eq(
                    &format!("inline_complete_standardized_difference_{i}"),
                    a - b,
                    (raw[i] - mean(&raw)) / (variance(&raw) + floor.powi(2)).sqrt()
                        - (raw2[i] - mean(&raw2)) / (variance(&raw2) + floor.powi(2)).sqrt(),
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    // Uniform balanced binary reference: enumerate every 2^N independent draw pattern.
    let n = 4_usize;
    let site = [0_f64, 0., 1., 1.];
    let p = n as f64 / (2. * (n - 1) as f64);
    let mut law = [0_f64; 5];
    for bits in 0_usize..(1 << n) {
        let count = bits.count_ones() as usize;
        law[count] += p.powi(count as i32) * (1. - p).powi((n - count) as i32);
    }
    let comb = [1_f64, 4., 6., 4., 1.];
    let row_p = (0..n)
        .map(|i| (0..n).filter(|j| *j != i && site[*j] != site[i]).count() as f64 / (n - 1) as f64)
        .collect::<Vec<_>>();
    let binary = json!({"N":n,"reference_positions":site,"kernel":"independent uniform nonself companions","pN":p,"per_row_success_probabilities":row_p,"enumerated_count_law":law,"binomial_coefficients":comb});
    emit(
        suite,
        src,
        &["1161"],
        &binary,
        &row_p
            .iter()
            .enumerate()
            .map(|(i, x)| eq(&format!("inline_balanced_nonself_probability_{i}"), *x, p))
            .collect::<Vec<_>>(),
        "Balanced binary reference on the Euclidean metric chart; this is the declared independent uniform donor specialization, not canonical finite-width Gaussian donor probabilities.",
    )?;
    emit(
        suite,
        src,
        &["1164"],
        &binary,
        &law.iter()
            .enumerate()
            .map(|(k, x)| {
                eq(
                    &format!("inline_binomial_from_all_patterns_{k}"),
                    *x,
                    comb[k] * p.powi(k as i32) * (1. - p).powi((n - k) as i32),
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    Ok(())
}

fn exact_aliases(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let registry: [(&[&str], &[&str]); 12] = [
        (&["0602"], &["piecewise_LP_decimal_approximation"]),
        (&["1036"], &["actual_uniform_raw_measurement_axiom"]),
        (&["0849"], &["native_raw_positional_coefficient_definition"]),
        (
            &["0850"],
            &["native_raw_linear_status_coefficient_definition"],
        ),
        (
            &["0851"],
            &["native_raw_quadratic_status_coefficient_definition"],
        ),
        (&["0465"], &["native_sampled_donor_row"]),
        (
            &["0136"],
            &[
                "native_uniform_threshold_mean",
                "native_uniform_threshold_second_moment",
                "native_uniform_threshold_support_lower",
                "native_uniform_threshold_support_upper",
            ],
        ),
        (&["1450", "1464"], &["native_sampled_donor_row"]),
        (
            &["1280"],
            &[
                "native_intermediate_state_coordinate_count",
                "native_intermediate_state_status_count",
            ],
        ),
        (
            &["1346"],
            &[
                "native_reward_full_standardization_assembly",
                "native_diversity_full_standardization_assembly",
            ],
        ),
        (&["1511"], &["native_gate_expected_absolute_triangle"]),
        (&["1594"], &["native_Fpot_normalized_score_cap_assembly"]),
    ];
    for (indices, ids) in registry {
        let matches = |id: &str, stem: &str| {
            id == stem
                || id.strip_prefix(stem).is_some_and(|tail| {
                    tail.starts_with('_')
                        && tail[1..].split('_').all(|part| {
                            !part.is_empty() && part.chars().all(|c| c.is_ascii_digit())
                        })
                })
        };
        let records = suite
            .evidence
            .iter()
            .filter(|e| {
                e.checks
                    .iter()
                    .any(|c| ids.iter().any(|id| matches(&c.id, id)))
            })
            .take(4)
            .cloned()
            .collect::<Vec<_>>();
        require(
            !records.is_empty(),
            &format!("missing explicit same-operand alias {indices:?}/{ids:?}"),
        )?;
        for original in records {
            let checks = original
                .checks
                .iter()
                .filter(|c| ids.iter().any(|id| matches(&c.id, id)))
                .cloned()
                .collect::<Vec<_>>();
            let start = suite.evidence.len();
            emit(
                suite,
                src,
                indices,
                &original.inputs,
                &checks,
                &format!(
                    "Exact named operand comparison {ids:?}. Its retained finite scope and hypothesis checks apply: {}",
                    original.scope
                ),
            )?;
            for e in &mut suite.evidence[start..] {
                e.hypothesis_checks = original.hypothesis_checks.clone();
            }
        }
    }
    // Assemble the bound parts from native pipeline intermediates rather than
    // confusing measured stable/unstable errors with their upper bounds.
    let records = suite
        .evidence
        .iter()
        .filter(|e| {
            e.checks
                .iter()
                .any(|c| c.id == "native_full_changed_support_potential_assembly")
        })
        .take(4)
        .cloned()
        .collect::<Vec<_>>();
    for e in records {
        let p = &e.inputs;
        let nc = p["status_change_count"].as_f64().unwrap();
        let cdir = p["structural_direct_coefficient"].as_f64().unwrap();
        let cind = p["structural_indirect_coefficient"].as_f64().unwrap();
        let cv = p["value_coefficient"].as_f64().unwrap();
        let da = p["native_companion_distances1"].as_array().unwrap();
        let db = p["native_companion_distances2"].as_array().unwrap();
        let diff = da
            .iter()
            .zip(db)
            .map(|(a, b)| {
                let a = a.as_f64().unwrap();
                let b = b.as_f64().unwrap();
                ((a * a + 1e-6_f64).sqrt() - (b * b + 1e-6_f64).sqrt()).powi(2)
            })
            .sum::<f64>();
        let structural = cdir * nc + cind * nc * nc;
        let unstable = 4.41_f64.powi(2) * nc;
        let stable = 2. * 1.05_f64.powi(2) * (4. * structural + 2. * cv * diff);
        emit(
            suite,
            src,
            &["1582"],
            &json!({"native_fixture":p,"F_unstable":unstable,"F_stable":stable,"actual_source_potential_bound":p["Fpot"]}),
            &[eq(
                "inline_source_potential_parts",
                p["Fpot"].as_f64().unwrap(),
                unstable + stable,
            )],
            "Bound components are independently assembled from retained native separation, value/structure coefficients and bounded positive-map derivatives; observed stable/unstable errors are kept distinct from their upper bounds.",
        )?;
    }
    Ok(())
}

fn branch_contracts(suite: &mut EstimateSuite, src: &Value) -> Result<()> {
    let patch = hermite_patch(2.)?;
    let coeff = patch.coefficients;
    let solved = patch.independently_solved_coefficients;
    for z in [-3_f64, -0.3, 0., 0.2, 0.6, 1., 1.2, 1.6, 2., 2.5] {
        let g = if z <= 0. {
            z.exp()
        } else if z < 1. {
            z.ln_1p() + 1.
        } else if z <= 2. {
            let s = z - 1.;
            ((coeff[0] * s + coeff[1]) * s + coeff[2]) * s + coeff[3]
        } else {
            3_f64.ln() + 1.
        };
        let input = json!({"auxiliary_Hermite_patch":patch,"z":z,"rescale_value":g,"native_map_unchanged":"configured logistic provider"});
        if z <= 0. {
            emit(
                suite,
                src,
                &["0636", "0638"],
                &input,
                &[
                    eq("inline_exponential_branch_value", g, z.exp()),
                    truth("inline_exponential_derivative_positive", z.exp() > 0.),
                ],
                "Auxiliary piecewise rescale: exact negative branch. It is distinct from the configured native logistic map.",
            )?;
        } else if z < 1. {
            emit(
                suite,
                src,
                &["0640", "0642", "0643"],
                &input,
                &[
                    eq("inline_log_branch_value", g, 1. + (1. + z).ln()),
                    truth("inline_log_derivative_positive", 1. / (1. + z) > 0.),
                    truth("inline_log_branch_positive_argument", z > 0.),
                ],
                SCOPE,
            )?;
        } else if z <= 2. {
            let s = z - 1.;
            let derivative = (3. * coeff[0] * s + 2. * coeff[1]) * s + coeff[2];
            let independently_solved =
                ((solved[0] * s + solved[1]) * s + solved[2]) * s + solved[3];
            emit(
                suite,
                src,
                &["0644", "0645"],
                &input,
                &[
                    eq(
                        "inline_polynomial_branch_independent_boundary_solve",
                        g,
                        independently_solved,
                    ),
                    le("inline_patch_derivative_nonnegative", 0., derivative),
                ],
                SCOPE,
            )?;
        } else {
            emit(
                suite,
                src,
                &["0647"],
                &input,
                &[eq(
                    "inline_constant_cap_branch",
                    g,
                    (1_f64 + patch.z_max).ln() + 1.,
                )],
                SCOPE,
            )?;
        }
    }
    let n = 4_usize;
    let mut strata = vec![];
    let mut checks = vec![];
    for i in 0..n {
        let mut rng = RandomStream::new(740001, 0, Stream::Distance, i as u64, 0);
        let u = rng.uniform::<f64>();
        let strat = (i as f64 + u) / n as f64;
        strata.push(strat);
        checks.push(le(
            &format!("inline_stratum_lower_{i}"),
            i as f64 / n as f64,
            strat,
        ));
        checks.push(le(
            &format!("inline_stratum_upper_{i}"),
            strat,
            (i + 1) as f64 / n as f64,
        ));
    }
    emit(
        suite,
        src,
        &["0467"],
        &json!({"stratified_reference_values":strata,"N":n,"innovation_seed":740001,"stream":"Distance","scope":"Reference stratification transform of actual native independently addressed uniform draws; no assertion that configured Independent sampling uses stratification."}),
        &checks,
        SCOPE,
    )?;
    let a = [true, true, true, true];
    let b = [false; 4];
    let nc = a.iter().zip(b).filter(|(a, b)| **a != *b).count();
    let s = pool(&a, 0);
    emit(
        suite,
        src,
        &["0500", "0501"],
        &json!({"alive1":a,"alive2":b,"reference_first_support":s,"reference_second_support":pool(&b,0),"nc":nc}),
        &[le(
            "inline_empty_second_support_status_cost",
            s.len() as f64,
            nc as f64,
        )],
        "Empty second-support branch of the set-difference proof. No expectation is assigned to the empty companion law.",
    )?;
    emit(
        suite,
        src,
        &["1599"],
        &json!({"alpha":0.5,"positive_terms":[0.3,0.8]}),
        &[
            eq(
                "inline_square_root_power",
                0.3_f64.powf(0.5),
                0.3_f64.sqrt(),
            ),
            le(
                "inline_root_subadditivity",
                (0.3_f64 + 0.8).sqrt(),
                0.3_f64.sqrt() + 0.8_f64.sqrt(),
            ),
        ],
        SCOPE,
    )?;
    // Compute all three expectations separately using actual native gate calls.
    let cfg = GasConfig::euclidean(1, 0.04)?;
    let x = [-0.9_f64, -0.2, 0.35];
    let y = [-0.86_f64, -0.26, 0.4];
    let n = x.len();
    let (left, _) = enumerate_measurements(&x, &cfg)?;
    let (right, _) = enumerate_measurements(&y, &cfg)?;
    let donor_rows = |positions: &[f64]| -> Result<Vec<Vec<f64>>> {
        let obs = observations(positions)?;
        let mut rows = vec![];
        for i in 0..n {
            let mut row = vec![0.; n];
            for (j, w) in row.iter_mut().enumerate() {
                if i != j {
                    let d = cfg.cloning_donors.distance.compare(&obs, i, &obs, j)?;
                    *w = cfg
                        .cloning_donors
                        .kernel
                        .log_weight(d, ComparisonKind::Distance)?
                        .exp();
                }
            }
            let mass = row.iter().sum::<f64>();
            for w in &mut row {
                *w /= mass;
            }
            rows.push(row);
        }
        Ok(rows)
    };
    let q1 = donor_rows(&x)?;
    let q2 = donor_rows(&y)?;
    let mut ea = vec![0.; n];
    let mut emid = vec![0.; n];
    let mut eb = vec![0.; n];
    for o in &left {
        let p1 = expected_action(o, &q1, &cfg);
        let p2 = expected_action(o, &q2, &cfg);
        for i in 0..n {
            ea[i] += o.probability * p1[i];
            emid[i] += o.probability * p2[i];
        }
    }
    for o in &right {
        let p2 = expected_action(o, &q2, &cfg);
        for i in 0..n {
            eb[i] += o.probability * p2[i];
        }
    }
    let input = json!({"native_config":cfg,"first_positions":x,"second_positions":y,"actual_cloning_donor_rows1":q1,"actual_cloning_donor_rows2":q2,"complete_expected_actions1":ea,"complete_expected_actions_intermediate":emid,"complete_expected_actions2":eb,"left_measurement_outcomes":left.iter().map(|o|json!({"weight":o.probability,"fitness":o.fitness})).collect::<Vec<_>>(),"right_measurement_outcomes":right.iter().map(|o|json!({"weight":o.probability,"fitness":o.fitness})).collect::<Vec<_>>()});
    emit(
        suite,
        src,
        &["1498"],
        &input,
        &(0..n)
            .map(|i| {
                le(
                    &format!("inline_native_two_law_expected_action_triangle_{i}"),
                    (ea[i] - eb[i]).abs(),
                    (ea[i] - emid[i]).abs() + (emid[i] - eb[i]).abs(),
                )
            })
            .collect::<Vec<_>>(),
        SCOPE,
    )?;
    for o in &left {
        for (i, row) in q2.iter().enumerate() {
            for (j, probability) in row.iter().enumerate() {
                if *probability == 0. {
                    continue;
                }
                let vi = o.fitness[i];
                let vc = o.fitness[j];
                let actual = cfg.clone_decision.acceptance_probability(0, vi, vc);
                let expression = ((vc - vi)
                    / (cfg.clone_decision.saturation * (vi + cfg.clone_decision.epsilon)))
                    .clamp(0., 1.);
                let p = json!({"native_fixture":input,"recipient":i,"donor":j,"complete_native_realized_fitness":o.fitness,"vi":vi,"vc":vc,"native_gate_probability":actual,"actual_donor_probability":q2[i][j]});
                emit(
                    suite,
                    src,
                    &["1451", "1452", "1519"],
                    &p,
                    &[eq(
                        "inline_native_fitness_operands_consumed_by_gate",
                        actual,
                        expression,
                    )],
                    "Actual native acceptance_probability consumes receiver/donor fitness from each fully realized empirical normalization. The conditional f(V) operand is evaluated before averaging; no deterministic expected fitness replaces this random vector.",
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn companion_support_branches_preserve_permutations() {
        assert_eq!(pool(&[false, false, false], 1), Vec::<usize>::new());
        assert_eq!(pool(&[false, true, false], 1), vec![1]);
        assert_eq!(pool(&[false, true, false], 0), vec![1]);
        assert_eq!(pool(&[true, true, false], 0), vec![1]);
        let a = [true, false, true, false];
        let pi = [2, 3, 0, 1];
        let perm = pi.map(|i| a[i]);
        let original = pool(&a, 0).into_iter().collect::<BTreeSet<_>>();
        let mapped = pool(&perm, 2)
            .into_iter()
            .map(|i| pi[i])
            .collect::<BTreeSet<_>>();
        assert_eq!(original, mapped);
    }
    #[test]
    fn identical_companion_law_keeps_generic_function_outside_receiver_mask() {
        let live = [true, true, false];
        let changed = [false, true, false];
        let first = pool(&live, 0);
        let second = pool(&changed, 0);
        let f = [0_f64, 0.5, 1.];
        assert_eq!(first, second);
        let e1 = uniform_expectation(&f, &first).unwrap();
        let e2 = uniform_expectation(&f, &second).unwrap();
        assert!(eq("same_conditional_function_law", e1, e2).passed);
        let raw1 = if live[0] { e1 } else { 0. };
        let raw2 = if changed[0] { e2 } else { 0. };
        assert!(!eq("incorrectly_masked_generic_law", raw1, raw2).passed);
        assert_eq!(uniform_expectation(&f, &[]), None);
    }
    #[test]
    fn a_real_operand_discrepancy_fails() {
        assert!(!eq("corrupt_native_mean", 0.25, 0.4).passed);
        assert!(!le("corrupt_scale_floor", 0.1, 0.05).passed);
        assert!(!truth("corrupt_hypothesis", false).passed);
        assert!(eq("correct_native_mean", 0.3, 0.1 + 0.2).passed);
    }
    #[test]
    fn explicit_source_manifest_retains_whole_formulas() {
        let src = source().unwrap();
        assert_eq!(
            src["0958"]["formula"],
            r"\text{Var}(\mathbf{v}) = m_2(\mathcal{S}, \mathbf{v}) - \mu(\mathcal{S}, \mathbf{v})^2"
        );
        let mut suite = EstimateSuite {
            chapter: 1,
            evidence: vec![],
            scope_notes: vec![],
        };
        numeric_contracts(&mut suite, &src).unwrap();
        assert!(!suite.evidence.is_empty());
        assert!(suite.evidence.iter().all(EstimateEvidence::passed));
    }
}
