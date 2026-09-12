use super::*;
use std::f64::consts::PI;
pub(super) fn run(r: &ExperimentRequest, _a: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    match r.experiment {
        61 => stationarity(r),
        62 => scale(r),
        63 => expansion(r),
        64 => information(r),
        65 => closure(r),
        _ => units(r),
    }
}
fn stationarity(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let initial = p(r, "initial_probability", 0.8, 0., 1.);
    let shift = p(r, "energy_shift", 0., -3., 3.);
    let survival = p(r, "survival", 0.85, 0.5, 1.);
    let k = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
    let pi = stationary(&k)?;
    let centered = [shift - (shift + pi[1]), 1. + shift - (shift + pi[1])];
    let mut mu = vec![1. - initial, initial];
    let mut excess = vec![];
    let mut bounds = vec![];
    let mut alive = vec![];
    for i in 0..41 {
        excess.push([i as f64, mu.iter().zip(centered).map(|(a, b)| a * b).sum()]);
        bounds.push([
            i as f64,
            2. * centered.iter().copied().map(f64::abs).fold(0., f64::max) * tv(&mu, &pi),
        ]);
        alive.push([i as f64, survival.powi(i)]);
        mu = evolve(&mu, &k);
    }
    let mut out = result(
        r,
        "A stationary conditioned law while survival falls",
        "Two-state conservative reference with independent constant killing",
    );
    metric(
        &mut out,
        "Centered QSD expectation",
        pi.iter().zip(centered).map(|(p, e)| p * e).sum(),
        "",
    );
    metric(&mut out, "Energy reference", shift + pi[1], "");
    plot(
        &mut out,
        "Observable relaxation under survival conditioning",
        "step",
        "centered expectation",
        vec![
            line("Conditioned observable", excess),
            line("Total-variation bound", bounds),
        ],
    );
    plot(
        &mut out,
        "Unnormalized surviving mass",
        "step",
        "survival probability",
        vec![line("Constant independent killing", alive)],
    );
    out.details = json!({"stationary_probability":pi,"energy_zero_shift":shift,"centered_observable":centered,"conservative_kernel":k,"survival_per_step":survival});
    Ok(out)
}
fn scale(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let lambda = p(r, "lambda", -0.5, -2., -0.1);
    let d = n(r, "dimension", 3, 2, 3) as f64;
    let ratio = p(r, "range_ratio", 0.1, 0.02, 0.4);
    let radius = (-d * (d - 1.) / (2. * lambda)).sqrt();
    let mut local = vec![];
    let mut refcurve = vec![];
    for i in 1..=30 {
        let e = ratio * radius * i as f64 / 30.;
        let limit = (PI * radius / e).min(12.);
        let measured = e * integral(
            |u| (-2. * radius * radius * (e * u / (2. * radius)).sin().powi(2) / (e * e)).exp(),
            -limit,
            limit,
            4096,
        );
        let tangent = (2. * PI).sqrt() * e;
        local.push([e / radius, measured / tangent - 1.]);
        refcurve.push([e / radius, e * e / (8. * radius * radius)]);
    }
    let mut out = result(
        r,
        "Kernel locality and a declared curvature scale",
        "Gaussian chord-distance integral on a circle, with independent AdS radius algebra",
    );
    metric(&mut out, "Declared AdS radius", radius, "length");
    metric(
        &mut out,
        "Declared sectional curvature",
        -1. / radius.powi(2),
        "inverse length squared",
    );
    plot(
        &mut out,
        "Measured kernel locality error",
        "epsilon / radius",
        "relative correction",
        vec![
            line("Circle quadrature / tangent Gaussian mass - 1", local),
            line("Leading epsilon² / (8 radius²)", refcurve),
        ],
    );
    out.details = json!({"lambda":lambda,"spatial_dimension":d,"range_ratio":ratio,"reference_geometry_for_integral":"positive-curvature circle; independent of the AdS constitutive fixture","curvature_not_inferred_from_range":true});
    Ok(out)
}
fn expansion(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let d = n(r, "dimension", 3, 2, 3) as f64;
    let h = p(r, "expansion", 0.4, 0.1, 1.);
    let forcing = p(r, "forcing", 0.3, 0.05, 1.);
    let mut y = vec![0.];
    let mut measured = vec![];
    let mut bound = vec![];
    for i in 0..=400 {
        let t = i as f64 / 100.;
        measured.push([t, y[0]]);
        bound.push([t, (forcing * d).sqrt() * ((forcing / d).sqrt() * t).tanh()]);
        y = rk4(&y, t, 0.01, |t, v| {
            vec![-v[0] * v[0] / d + forcing + 0.1 * (1. + t.sin())]
        });
    }
    let mut out = result(
        r,
        "Expansion comparison, de Sitter and Milne",
        "Explicit geodesic congruences and Raychaudhuri forcing bounded below",
    );
    metric(&mut out, "de Sitter expansion", d * h, "inverse time");
    metric(
        &mut out,
        "de Sitter Lambda",
        d * (d - 1.) * h * h / 2.,
        "inverse length squared",
    );
    plot(
        &mut out,
        "A lower bound that retains the complete forcing",
        "proper time",
        "expansion",
        vec![
            line("Integrated Riccati equation", measured),
            line("Analytic tanh lower bound", bound),
        ],
    );
    plot(
        &mut out,
        "Positive expansion with two different curvature constants",
        "proper time",
        "expansion",
        vec![
            line("de Sitter", vec![[0.2, d * h], [4., d * h]]),
            line(
                "Milne coordinates in flat spacetime",
                (0..77)
                    .map(|i| {
                        let t = 0.2 + i as f64 / 20.;
                        [t, d / t]
                    })
                    .collect(),
            ),
        ],
    );
    out.details = json!({"dimension":d,"de_sitter":{"theta":d*h,"ricci_uu":-d*h*h,"raychaudhuri_residual":0},"milne":{"ricci":0,"lambda":0,"theta_formula":"d / proper_time"},"forcing_lower_bound":forcing});
    Ok(out)
}
fn information(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let reveal = p(r, "reveal_probability", 1., 0., 1.);
    let samples = n(r, "samples", 4096, 128, 16384);
    let mut random = rng(seed(r), 0);
    let mut macro_counts = [0.; 2];
    let mut joint = [[0.; 2]; 3];
    for _ in 0..samples {
        let bit = random.index(2);
        let state = if random.uniform::<f64>() < reveal {
            bit
        } else {
            2
        };
        macro_counts[bit] += 1.;
        joint[state][bit] += 1.;
    }
    for v in &mut macro_counts {
        *v /= samples as f64;
    }
    let macro_entropy = entropy(&macro_counts);
    let mut conditional = 0.;
    for row in joint {
        let total = row.iter().sum::<f64>();
        if total > 0. {
            conditional += total / samples as f64 * entropy(&row.map(|x| x / total));
        }
    }
    let information = macro_entropy - conditional;
    let expected = reveal * 2_f64.ln();
    let mut out = result(
        r,
        "Does the microscopic past improve prediction?",
        "Independent fair-bit process with a recorded future-bit reveal channel",
    );
    metric(
        &mut out,
        "Measured discarded predictive information",
        information,
        "nats",
    );
    metric(&mut out, "Exact information", expected, "nats");
    metric(&mut out, "Macro causal states", 1., "");
    plot(
        &mut out,
        "A single macro forecast can discard predictive information",
        "reveal probability",
        "conditional mutual information",
        vec![
            line(
                "Exact reveal channel",
                (0..21)
                    .map(|i| {
                        let x = i as f64 / 20.;
                        [x, x * 2_f64.ln()]
                    })
                    .collect(),
            ),
            line("Sampled current experiment", vec![[reveal, information]]),
        ],
    );
    out.details = json!({"samples":samples,"reveal_probability":reveal,"macro_entropy":macro_entropy,"conditional_entropy":conditional,"macro_causal_states":1,"micro_future_symbol_counts":joint,"full_reveal_reference":"X_t=(B_t,B_{t+1}), Y_t=B_t","unit":"natural logarithms"});
    Ok(out)
}
fn closure(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let delta = p(r, "rate_perturbation", 0.15, 0., 0.5);
    let time = p(r, "time", 2., 0.1, 5.);
    let mut q = vec![vec![0.; 4]; 4];
    q[0][1] = 0.6;
    q[0][2] = 0.4 + delta;
    q[1][0] = 0.6;
    q[1][3] = 0.4;
    q[2][3] = 0.5;
    q[2][0] = 0.3;
    q[3][2] = 0.5;
    q[3][1] = 0.3;
    for (i, row) in q.iter_mut().enumerate() {
        row[i] = -row.iter().sum::<f64>();
    }
    let macro_out = 0.4 + delta / 2.;
    let phi = [0., 0., 1., 1.];
    let generator_values = (0..4)
        .map(|i| q[i].iter().zip(phi).map(|(q, v)| q * v).sum::<f64>())
        .collect::<Vec<_>>();
    let approximate = [macro_out, macro_out, -0.3, -0.3];
    let defect = generator_values
        .iter()
        .zip(approximate)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let mut curves = vec![];
    let mut finalres = vec![];
    for initial in 0..2 {
        let mut state = vec![0.; 5];
        state[initial] = 1.;
        let mut points = vec![];
        for step in 0..=200 {
            let t = time * step as f64 / 200.;
            let observable = state[2] + state[3];
            let residual = observable - state[4];
            points.push([t, residual]);
            if step < 200 {
                state = rk4(&state, t, time / 200., |_, s| {
                    let mut out = (0..4)
                        .map(|j| (0..4).map(|i| s[i] * q[i][j]).sum())
                        .collect::<Vec<_>>();
                    out.push((0..4).map(|i| s[i] * approximate[i]).sum());
                    out
                });
            }
        }
        finalres.push(*points.last().unwrap());
        curves.push(line(&format!("Micro initial state {initial}"), points));
    }
    curves.push(line(
        "t epsilon bound",
        vec![[0., 0.], [time, time * defect]],
    ));
    curves.push(line(
        "-t epsilon bound",
        vec![[0., 0.], [time, -time * defect]],
    ));
    let mut out = result(
        r,
        "Exact and approximate Markov closure",
        "Four-state continuous-time rate matrix with deterministic two-block projection",
    );
    metric(&mut out, "Generator residual bound", defect, "inverse time");
    metric(&mut out, "Integrated residual bound", time * defect, "");
    plot(
        &mut out,
        "Expected Dynkin residual",
        "time",
        "residual",
        curves,
    );
    out.details = json!({"generator":q,"projection":[0,0,1,1],"exact_generator_on_macro_indicator":generator_values,"proposed_macro_generator_values":approximate,"sup_residual":defect,"final_residuals":finalres});
    Ok(out)
}
fn units(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let hubble = p(r, "hubble", 70., 30., 100.);
    let omega = p(r, "omega", 0.7, 0., 1.);
    let mpc = 3.085677581491367e22;
    let c = 299_792_458_f64;
    let g = 6.67430e-11;
    let h = hubble * 1000. / mpc;
    let lambda = 3. * h * h * omega / (c * c);
    let energy = lambda * c.powi(4) / (8. * PI * g);
    let density = energy / (c * c);
    let pressure = -energy;
    let mut out = result(
        r,
        "Calibrated units and vacuum stress",
        "Declared flat Friedmann and Einstein unit conventions; user-specified illustrative inputs",
    );
    metric(&mut out, "Lambda", lambda, "m^-2");
    metric(&mut out, "Vacuum energy density", energy, "J m^-3");
    metric(&mut out, "Vacuum mass density", density, "kg m^-3");
    metric(&mut out, "Vacuum pressure", pressure, "Pa");
    metric(
        &mut out,
        "Equation-of-state w",
        if energy != 0. {
            pressure / energy
        } else {
            f64::NAN
        },
        "",
    );
    plot(
        &mut out,
        "The declared density parameter controls the inferred curvature",
        "Omega Lambda",
        "Lambda [m^-2]",
        vec![line(
            "Flat-model inference",
            (0..21)
                .map(|i| {
                    let x = i as f64 / 20.;
                    [x, 3. * h * h * x / (c * c)]
                })
                .collect(),
        )],
    );
    out.details = json!({"hubble_km_s_mpc":hubble,"hubble_s_inverse":h,"omega_lambda":omega,"speed_of_light_m_s":c,"gravitational_constant_si":g,"megaparsec_m":mpc,"reconstructed_omega":if h!=0.{lambda*c*c/(3.*h*h)}else{0.},"stress_signature":"(-,+,+,+)","spatial_dimension":3,"equation":"Einstein + Lambda g = 8 pi G T / c^4"});
    Ok(out)
}
