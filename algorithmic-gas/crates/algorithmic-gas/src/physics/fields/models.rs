use super::*;
use std::f64::consts::{PI, TAU};
pub(super) fn run(r: &ExperimentRequest, a: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    match r.experiment {
        45 => energy(r),
        46 => pressure(r),
        47 => dispersion(r),
        48 => match a {
            Some(archive) => super::observed::thermostat(r, archive),
            None => thermostat(r),
        },
        49 => modes(r),
        50 => crossover(r),
        _ => budgets(r, a),
    }
}
fn energy(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let amplitude = p(r, "amplitude", 0.4, 0., 0.8);
    let e = p(r, "epsilon", 0.15, 0.03, 0.4);
    let grid = n(r, "resolution", 96, 16, 256);
    let dx = 1. / grid as f64;
    let phi = (0..grid)
        .map(|i| amplitude * (TAU * (i as f64 + 0.5) * dx).cos())
        .collect::<Vec<_>>();
    let entropy = phi.iter().map(|f| (1. + f) * (1. + f).ln()).sum::<f64>() * dx;
    let pair = (0..grid)
        .map(|i| {
            (0..grid)
                .map(|j| gaussian_periodic((i as f64 - j as f64) * dx, e) * phi[i] * phi[j])
                .sum::<f64>()
        })
        .sum::<f64>()
        * dx
        * dx
        / 2.;
    let quadratic = phi.iter().map(|f| f * f / 2.).sum::<f64>() * dx + pair;
    let bound =
        phi.iter().map(|f| f.abs().powi(3)).sum::<f64>() * dx / (6. * (1. - amplitude).powi(2));
    let mut out = result(
        r,
        "Fluctuations, correlation energy, and conditional drift",
        "Positive periodic density with Gaussian pair energy; independent finite-state metric replicas",
    );
    metric(&mut out, "Free energy", entropy + pair, "");
    metric(
        &mut out,
        "Quadratic remainder",
        (entropy + pair - quadratic).abs(),
        "",
    );
    metric(&mut out, "Taylor remainder bound", bound, "");
    let mut types = vec![];
    let mut rate = vec![];
    let count = n(r, "population", 64, 8, 512);
    let logfact = |n: usize| (1..=n).map(|i| (i as f64).ln()).sum::<f64>();
    for k in 0..=count {
        let q = k as f64 / count as f64;
        let logp = logfact(count) - logfact(k) - logfact(count - k) - count as f64 * 2_f64.ln();
        types.push([q, -logp / count as f64]);
        let kval = if q == 0. || q == 1. {
            2_f64.ln()
        } else {
            q * (2. * q).ln() + (1. - q) * (2. * (1. - q)).ln()
        };
        rate.push([q, kval]);
    }
    plot(
        &mut out,
        "Independent empirical frequency rate",
        "frequency",
        "negative log probability / N",
        vec![line("Exact binomial", types), line("KL rate", rate)],
    );
    // A finite-state metric observable has a fully enumerable conditional law.
    let probabilities = [0.2, 0.5, 0.3];
    let values = [1., 1.5, 3.];
    let before = 1.5;
    let predicted: f64 = probabilities
        .iter()
        .zip(values)
        .map(|(p, x)| p * (x - before))
        .sum();
    let variance: f64 = probabilities
        .iter()
        .zip(values)
        .map(|(p, x)| p * (x - before - predicted).powi(2))
        .sum();
    let reps = n(r, "replicas", 1024, 16, 16384);
    let groups = (0..2)
        .map(|group| {
            let mut random = rng(seed(r), group);
            (0..reps)
                .map(|_| values[sample(&probabilities, &mut random)] - before)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (a, _, _) = stats(&groups[0]);
    let (b, _, _) = stats(&groups[1]);
    metric(&mut out, "Independent drift discrepancy", b - a, "");
    metric(
        &mut out,
        "Predicted discrepancy SE",
        (2. * variance / reps as f64).sqrt(),
        "",
    );
    out.details = json!({"entropy":entropy,"pair_energy":pair,"quadratic_energy":quadratic,"remainder_bound":bound,"conditional_metric_fixture":{"transition_probabilities":probabilities,"metric_values":values,"exact_drift":predicted,"exact_variance":variance,"prediction_mean":a,"test_mean":b,"replicas_per_group":reps,"independent_random_groups":true},"density":phi.iter().map(|p|1.+p).collect::<Vec<_>>()});
    Ok(out)
}
fn pressure(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let e = p(r, "epsilon", 0.3, 0.05, 0.6);
    let grid = n(r, "resolution", 96, 16, 192);
    let mode = r.text("mode", "positive");
    let sign = if mode == "attractive" { -1. } else { 1. };
    let dx = 2. / grid as f64;
    let u = |x: f64| if mode == "signed" { x } else { 1. };
    let energy = |a: f64| -> f64 {
        sign * (0..grid)
            .map(|i| {
                let x = -1. + (i as f64 + 0.5) * dx;
                (0..grid)
                    .map(|j| {
                        let y = -1. + (j as f64 + 0.5) * dx;
                        (-0.5 * (a * (x - y) / e).powi(2)).exp() * u(x) * u(y)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            * dx
            * dx
            / 2.
    };
    let analytic = sign
        * (0..grid)
            .map(|i| {
                let x = -1. + (i as f64 + 0.5) * dx;
                (0..grid)
                    .map(|j| {
                        let y = -1. + (j as f64 + 0.5) * dx;
                        (x - y).powi(2) * (-0.5 * ((x - y) / e).powi(2)).exp() * u(x) * u(y)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
        * dx
        * dx
        / (4. * e * e);
    let numerical = -derivative(energy, 1., 1e-4) / 2.;
    let mut out = result(
        r,
        "Pressure from a declared dilation",
        "Fixed transported signed or nonnegative density on [-1,1], d=1",
    );
    metric(&mut out, "Analytic pressure", analytic, "");
    metric(&mut out, "Finite-difference pressure", numerical, "");
    metric(
        &mut out,
        "Derivative residual",
        (analytic - numerical).abs(),
        "",
    );
    plot(
        &mut out,
        "Pair energy along a mass-preserving dilation",
        "dilation a",
        "pair energy",
        vec![line(
            "Direct quadrature",
            (0..41)
                .map(|i| {
                    let a = 0.5 + i as f64 / 40.;
                    [a, energy(a)]
                })
                .collect(),
        )],
    );
    let d = n(r, "dimension", 3, 1, 3) as f64;
    plot(
        &mut out,
        "Kernel normalization controls stiffness scaling",
        "epsilon",
        "stiffness",
        vec![
            line(
                "Fixed kernel amplitude",
                (1..41)
                    .map(|i| {
                        let x = i as f64 / 40.;
                        [x, (2. * PI).powf(d / 2.) * x.powf(d + 2.) / 8.]
                    })
                    .collect(),
            ),
            line(
                "Unit kernel mass",
                (1..41)
                    .map(|i| {
                        let x = i as f64 / 40.;
                        [x, x * x / 8.]
                    })
                    .collect(),
            ),
        ],
    );
    out.details = json!({"pressure_dimension":1,"volume":2,"energy_sign":sign,"density_kind":mode,"stiffness_dimension":d,"kernel_mass_normalization":"declared per curve"});
    Ok(out)
}
fn dispersion(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let e = p(r, "epsilon", 0.15, 0.03, 0.5);
    let diffusion = p(r, "diffusion", 0.02, 0., 0.2);
    let lambda = p(r, "rate", 0.7, 0., 2.);
    let length = p(r, "length", 1., 0.5, 5.);
    let grid = n(r, "resolution", 96, 32, 256);
    let mode = n(r, "wave_number", 1, 1, 6);
    let dx = length / grid as f64;
    let k = TAU * mode as f64 / length;
    let kernel = (0..grid)
        .map(|j| gaussian_periodic(j as f64 / grid as f64, e / length))
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f64>();
    let gain = kernel
        .iter()
        .enumerate()
        .map(|(j, w)| w * (k * j as f64 * dx).cos())
        .sum::<f64>()
        / sum;
    let measured = diffusion * 4. * (k * dx / 2.).sin().powi(2) / (dx * dx) + lambda * (1. - gain);
    let expected = diffusion * k * k + lambda * (1. - (-e * e * k * k / 2.).exp());
    let end = 2.;
    // Keep the explicit mode integrator positive for every combined control setting.
    let target_dt = if measured > 0. {
        0.001_f64.min(0.5 / measured)
    } else {
        0.001
    };
    let steps = (end / target_dt).ceil() as usize;
    let dt = end / steps as f64;
    let mut a = 1.;
    let mut times = vec![];
    let mut reference = vec![];
    for step in 0..=steps {
        if step % (steps / 100).max(1) == 0 || step == steps {
            let t = step as f64 * dt;
            times.push([t, a]);
            reference.push([t, (-expected * t).exp()]);
        }
        a *= 1. - dt * measured;
    }
    let mut out = result(
        r,
        "Density modes and exact decay multipliers",
        "Periodized Gaussian redistribution and centered finite-difference Laplacian",
    );
    metric(&mut out, "Continuum Fourier rate", expected, "inverse time");
    metric(&mut out, "Grid operator rate", measured, "inverse time");
    metric(&mut out, "Zero mode decay", 0., "inverse time");
    plot(
        &mut out,
        "A sinusoidal density wave",
        "time",
        "mode amplitude",
        vec![
            line("Discrete grid operator / Euler evolution", times),
            line("Exact Gaussian closure", reference),
        ],
    );
    plot(
        &mut out,
        "Exact versus long-wave truncated multipliers",
        "wave number",
        "decay rate",
        vec![
            line(
                "Exact",
                (0..81)
                    .map(|i| {
                        let q = i as f64 / 2.;
                        [
                            q,
                            diffusion * q * q + lambda * (1. - (-e * e * q * q / 2.).exp()),
                        ]
                    })
                    .collect(),
            ),
            line(
                "Fourth-order series",
                (0..81)
                    .map(|i| {
                        let q = i as f64 / 2.;
                        [
                            q,
                            (diffusion + lambda * e * e / 2.) * q * q
                                - lambda * e.powi(4) * q.powi(4) / 8.,
                        ]
                    })
                    .collect(),
            ),
        ],
    );
    out.details = json!({"kernel_grid_mass":sum*dx,"normalized_gain_multiplier":gain,"time_step":dt,"explicit_euler_multiplier":1.-dt*measured,"time_steps":steps,"end_time":end,"box_length":length,"nonzero_frequency":k});
    Ok(out)
}
fn thermostat(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let gamma = p(r, "friction", 1., 0., 3.);
    let dt = p(r, "time_step", 0.2, 0.001, 1.);
    let v = p(r, "velocity", 1., -3., 3.);
    let sigma = p(r, "noise", 1., 0.05, 3.);
    let a = (-gamma * dt).exp();
    let s2 = if gamma == 0. {
        dt
    } else {
        -(-2. * gamma * dt).exp_m1() / (2. * gamma)
    };
    let mode = r.text("mode", "gaussian");
    let count = n(r, "samples", 8192, 128, 65536);
    let mu4 = if mode == "uniform" { 1.8 } else { 3. };
    let expected = (a * a - 1.) * v * v / 2. + s2 * sigma * sigma / 2.;
    let variance = a * a * s2 * v * v * sigma * sigma + s2 * s2 * sigma.powi(4) * (mu4 - 1.) / 4.;
    let mut random = rng(seed(r), 0);
    let energies = (0..count)
        .map(|_| {
            let xi = if mode == "uniform" {
                (2. * random.uniform::<f64>() - 1.) * 3_f64.sqrt()
            } else {
                random.gaussian::<f64>()
            };
            let w = a * v + s2.sqrt() * sigma * xi;
            (w * w - v * v) / 2.
        })
        .collect::<Vec<_>>();
    let (mean, var, se) = stats(&energies);
    let mut out = result(
        r,
        "Thermostat heating and finite-step diffusion",
        "Frozen scalar BAOAB O-stage reference using the native innovation generator; independently sampled conditional moments",
    );
    metric(&mut out, "Measured mean energy change", mean, "");
    metric(&mut out, "Predicted mean energy change", expected, "");
    metric(&mut out, "Mean standard error", se, "");
    metric(&mut out, "Variance residual", var - variance, "");
    plot(
        &mut out,
        "Mean and fluctuation predictions",
        "moment index",
        "energy units",
        vec![
            line("Measured", vec![[0., mean], [1., var]]),
            line(
                "Exact conditional moments",
                vec![[0., expected], [1., variance]],
            ),
        ],
    );
    let mut discrete = vec![];
    let mut continuous = vec![];
    if gamma > 0. {
        for i in 1..=60 {
            let h = i as f64 / 30.;
            let c = (-gamma * h).exp();
            let thermal = sigma * sigma / (2. * gamma);
            discrete.push([h, thermal * h * (1. + c) / (2. * (1. - c))]);
            continuous.push([h, thermal / gamma]);
        }
        plot(
            &mut out,
            "Long-time free BAOAB diffusion",
            "step size",
            "diffusion coefficient",
            vec![
                line("Exact discrete covariance sum", discrete),
                line("Continuous OU", continuous),
            ],
        );
    }
    out.details = json!({"samples":count,"damping":a,"integrated_variance_time":s2,"factor":sigma,"innovation":mode,"fourth_moment":mu4,"expected_mean":expected,"expected_variance":variance,"measured_variance":var,"measurement":"raw thermostat before boundary handling"});
    Ok(out)
}
fn modes(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let count = n(r, "modes", 8, 1, 32);
    let temperature = p(r, "temperature", 1., 0.1, 3.);
    let volume = p(r, "volume", 2., 0.3, 5.);
    let d = n(r, "dimension", 3, 1, 3) as f64;
    let mobility = p(r, "mobility", 0.7, 0.1, 3.);
    let reps = n(r, "samples", 4096, 128, 16384);
    let f = |v: f64| {
        -(0..count)
            .map(|j| {
                let w = (j + 1).pow(2) as f64 * v.powf(-2. / d);
                0.5 * temperature * (2. * PI * temperature / w).ln()
            })
            .sum::<f64>()
    };
    let measured = -derivative(f, volume, volume * 1e-4);
    let expected = temperature * count as f64 / (d * volume);
    let mut sampled = vec![];
    let mut exact = vec![];
    for j in 0..count {
        let w = (j + 1).pow(2) as f64 * volume.powf(-2. / d);
        let mut random = rng(seed(r), j);
        let xs = (0..reps)
            .map(|_| random.gaussian::<f64>() * (temperature / w).sqrt())
            .collect::<Vec<_>>();
        sampled.push([j as f64, stats(&xs).1]);
        exact.push([j as f64, temperature / w]);
    }
    let mut out = result(
        r,
        "Gaussian mode thermodynamics",
        "Fixed real-mode set, declared quadratic energies, and matched OU mobility/noise",
    );
    metric(&mut out, "Partition-function pressure", measured, "");
    metric(&mut out, "Predicted mode pressure", expected, "");
    metric(
        &mut out,
        "Volume derivative error",
        (measured - expected).abs(),
        "",
    );
    plot(
        &mut out,
        "Equipartition at the specified mode stiffness",
        "mode index",
        "variance",
        vec![
            line("Independent Gaussian draws", sampled),
            line("Theta / stiffness", exact),
        ],
    );
    plot(
        &mut out,
        "Mobility changes relaxation while preserving mode variance",
        "time",
        "normalized mean",
        vec![line(
            "First OU mode",
            (0..61)
                .map(|i| {
                    let t = i as f64 / 20.;
                    [t, (-mobility * volume.powf(-2. / d) * t).exp()]
                })
                .collect(),
        )],
    );
    out.details = json!({"fixed_real_modes":count,"temperature":temperature,"mobility":mobility,"relaxation_rates":(0..count).map(|j|mobility*(j+1).pow(2) as f64*volume.powf(-2./d)).collect::<Vec<_>>(),"reference_measure":"fixed Lebesgue mode coordinates","samples_per_mode":reps});
    Ok(out)
}
fn crossover(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let a = p(r, "attraction", 1., 0.1, 3.);
    let b = p(r, "mode_pressure", 1., 0.1, 3.);
    let d = n(r, "dimension", 3, 1, 3) as f64;
    let root = (b / a).powf(1. / (d + 2.));
    let mut lo = 0.;
    let mut hi = 3. * root;
    for _ in 0..60 {
        let mid = (lo + hi) / 2.;
        if b - a * mid.powf(d + 2.) > 0. {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut out = result(
        r,
        "The prescribed pressure crossover",
        "Declared two-term pressure P=B-A epsilon^(d+2)",
    );
    metric(&mut out, "Analytic crossover", root, "");
    metric(
        &mut out,
        "Numerically bracketed crossover",
        (lo + hi) / 2.,
        "",
    );
    plot(
        &mut out,
        "Pressure across the crossover",
        "epsilon / crossover",
        "pressure",
        vec![line(
            "Declared pressure",
            (0..81)
                .map(|i| {
                    let ratio = i as f64 / 40.;
                    [ratio, b - a * (ratio * root).powf(d + 2.)]
                })
                .collect(),
        )],
    );
    out.details = json!({"attractive_coefficient":a,"mode_pressure":b,"dimension":d});
    Ok(out)
}
fn budgets(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    let mut out = result(
        r,
        "Mechanical balances and constitutive stress",
        "Exact discrete energy/momentum observables, independent of geometric constitutive comparisons",
    );
    if let Some(a) = archive
        && let Some(step) = a.steps.last()
    {
        let velocity_field = match &a.gas_config.kinetic.integrator {
            crate::kinetic::KineticKind::Baoab { velocities, .. } => velocities.as_str(),
            _ => "velocities",
        };
        let budgets =
            step.mechanical_budgets(velocity_field, 1., a.gas_config.include_truncated)?;
        metric(
            &mut out,
            "Recorded stage transitions",
            budgets.len() as f64,
            "",
        );
        plot(
            &mut out,
            "Recorded mechanical stage ledger",
            "stage transition",
            "kinetic increment",
            vec![
                line(
                    "Total change",
                    budgets
                        .iter()
                        .enumerate()
                        .map(|(i, b)| [i as f64, b.energy_after - b.energy_before])
                        .collect(),
                ),
                line(
                    "Common eligible rows",
                    budgets
                        .iter()
                        .enumerate()
                        .map(|(i, b)| [i as f64, b.common_energy_change])
                        .collect(),
                ),
                line(
                    "Eligibility contribution",
                    budgets
                        .iter()
                        .enumerate()
                        .map(|(i, b)| [i as f64, b.eligibility_energy_change])
                        .collect(),
                ),
            ],
        );
        out.details = json!({"source":"recorded engine stages","stage_budgets":budgets});
        super::observed::append_mechanical(&mut out, step, &a.gas_config)?;
        if let crate::kinetic::KineticKind::Baoab { dt, friction, .. } =
            &a.gas_config.kinetic.integrator
        {
            match step.thermostat_moments(
                velocity_field,
                1.,
                *dt,
                *friction,
                a.gas_config.include_truncated,
            ) {
                Ok(m) => {
                    metric(
                        &mut out,
                        "Recorded O-stage energy change",
                        m.measured_energy_change,
                        "",
                    );
                    metric(
                        &mut out,
                        "Conditional O-stage energy prediction",
                        m.predicted_energy_change,
                        "",
                    );
                    out.details["recorded_thermostat"] = json!(m);
                }
                Err(e) => {
                    out.details["recorded_thermostat"] =
                        json!({"status":"unavailable","reason":e.to_string()})
                }
            }
        }
        out.notes.push("Stage budgets are computed from recorded eligibility and velocities; the source stage names identify any combined boundary operation.".into());
        return Ok(out);
    }
    let v = [p(r, "velocity", 1., -3., 3.), -0.6];
    let alpha = p(r, "restitution", 0.5, 0., 1.);
    let mean = (v[0] + v[1]) / 2.;
    let after = [
        mean + alpha * (v[0] - v[1]) / 2.,
        mean - alpha * (v[0] - v[1]) / 2.,
    ];
    let before_energy = (v[0] * v[0] + v[1] * v[1]) / 2.;
    let after_energy = (after[0] * after[0] + after[1] * after[1]) / 2.;
    let expected = -(1. - alpha * alpha) * (v[0] - v[1]).powi(2) / 4.;
    let literal = [v[1], v[1]];
    let literal_delta = literal.iter().map(|v| v * v / 2.).sum::<f64>() - before_energy;
    metric(
        &mut out,
        "Restitution pair energy change",
        after_energy - before_energy,
        "",
    );
    metric(
        &mut out,
        "Restitution identity residual",
        (after_energy - before_energy - expected).abs(),
        "",
    );
    metric(
        &mut out,
        "Pair momentum residual",
        (after.iter().sum::<f64>() - v.iter().sum::<f64>()).abs(),
        "",
    );
    plot(
        &mut out,
        "Do not double-count the literal clone",
        "stage index",
        "kinetic energy increment",
        vec![line(
            "Exact increments",
            vec![
                [0., literal_delta],
                [1., after_energy - before_energy - literal_delta],
                [2., after_energy - before_energy],
            ],
        )],
    );
    let d = n(r, "dimension", 3, 2, 3) as f64;
    let e = p(r, "energy_density", 1., -1., 3.);
    let pressure = p(r, "pressure", 0.2, -2., 2.);
    let lambda = p(r, "lambda", 0.1, -1., 1.);
    let trace = -e + d * pressure;
    let scalar = 2. * ((d + 1.) * lambda - trace) / (d - 1.);
    let ricci_uu = e + trace / (d - 1.) - 2. * lambda / (d - 1.);
    let predicted = ((d - 2.) * e + d * pressure - 2. * lambda) / (d - 1.);
    metric(&mut out, "Ricci contraction", ricci_uu, "");
    metric(
        &mut out,
        "Einstein contraction residual",
        (ricci_uu - predicted).abs(),
        "",
    );
    out.details["pair_fixture"] = json!({"preclone_velocities":v,"literal_velocities":literal,"transformed_velocities":after,"literal_delta":literal_delta,"transform_delta":after_energy-before_energy-literal_delta});
    out.details["constitutive_reference"] = json!({"kappa":1,"dimension":d,"energy":e,"pressure":pressure,"lambda":lambda,"scalar_curvature":scalar,"ricci_uu":ricci_uu,"vacuum_lambda_shift":-pressure});
    Ok(out)
}
