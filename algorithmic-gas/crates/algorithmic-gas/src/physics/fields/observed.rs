//! Readouts from executed, fully recorded operator stages.
use super::*;

pub(super) fn thermostat(
    r: &ExperimentRequest,
    archive: &RunArchive<f64>,
) -> Result<ExperimentResult> {
    let (velocities, dt, friction) = match &archive.gas_config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab {
            velocities,
            dt,
            friction,
            ..
        } => (velocities.as_str(), *dt, *friction),
        _ => {
            return Err(GasError::Capability(
                "recorded thermostat requires BAOAB".into(),
            ));
        }
    };
    if archive.steps.is_empty() {
        return Err(GasError::Capability(
            "no completed recorded thermostat stages".into(),
        ));
    }
    let mut out = result(
        r,
        "Executed thermostat conditional energy and momentum",
        "Raw O-stage observations and predictions from each executed diffusion factor and declared innovation law",
    );
    let mut measured = Vec::new();
    let mut predicted = Vec::new();
    let mut martingale = Vec::new();
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    let mut cumulative = 0.;
    let mut quadratic = 0.;
    let mut reports = Vec::new();
    for step in &archive.steps {
        // This check also verifies that the provider did not transform the sampled noise.
        let checked = step.thermostat_moments(
            velocities,
            1.,
            dt,
            friction,
            archive.gas_config.include_truncated,
        )?;
        let report = crate::physics::balances::analyze_step(step, &archive.gas_config, None)?;
        let moments = report.thermostat.ok_or_else(|| {
            GasError::Capability(
                "executed thermostat variance requires the recorded factor and innovation law"
                    .into(),
            )
        })?;
        let time = step.report.step as f64;
        measured.push([time, checked.measured_energy_change]);
        predicted.push([time, moments.energy_mean_delta]);
        cumulative += checked.measured_energy_change - moments.energy_mean_delta;
        quadratic += moments.energy_variance;
        martingale.push([time, cumulative]);
        positive.push([time, quadratic.sqrt()]);
        negative.push([time, -quadratic.sqrt()]);
        reports.push(json!({"step":step.report.step,"raw_stage":"A1 -> O_before_boundary","measured":checked,"conditional_moments":moments}));
    }
    metric(
        &mut out,
        "Cumulative centered O-stage energy",
        cumulative,
        "energy",
    );
    metric(
        &mut out,
        "Predictable cumulative energy variance",
        quadratic,
        "energy squared",
    );
    plot(
        &mut out,
        "Raw energy increments versus conditional means",
        "recorded step",
        "energy increment",
        vec![
            line("Executed O stage", measured),
            line("Factor/source conditional mean", predicted),
        ],
    );
    plot(
        &mut out,
        "Accumulated thermostat fluctuation",
        "recorded step",
        "energy",
        vec![
            line("Sum of conditionally centered increments", martingale),
            line("Positive square root predictable variation", positive),
            line("Negative square root predictable variation", negative),
        ],
    );
    out.details = json!({"source":"recorded_engine_thermostat","steps":reports,"friction":friction,"time_step":dt,"sampling_unit":"sequential O-stage martingale differences; no independent-walker or independent-step sampling assumption","uncertainty":"Predictable quadratic variation is the sum of conditional energy variances. Its realized square root is a fluctuation scale, not an exact finite-sample confidence band.","parameters_source":"integrator and actual noise records; reference fixture controls do not alter archive predictions"});
    out.notes.push("The metric-dependent factor is a programmed input to the update. Its conditional heating prediction is independently checked by the recorded velocity increment. Boundaries, kicks and clone transfers are excluded from this raw O-stage quantity.".into());
    Ok(out)
}

pub(super) fn append_mechanical(
    out: &mut ExperimentResult,
    step: &crate::tracking::RecordedStep<f64>,
    config: &crate::GasConfig,
) -> Result<()> {
    let report = crate::physics::balances::analyze_step(step, config, None)?;
    metric(
        out,
        "Kinetic telescoping residual",
        report.kinetic_telescoping_residual.abs(),
        "energy",
    );
    metric(
        out,
        "Momentum telescoping residual",
        report
            .momentum_telescoping_residual
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt(),
        "momentum",
    );
    plot(
        out,
        "Measured centered kinetic moment (unnormalized)",
        "flattened tensor component",
        "velocity squared",
        vec![
            line(
                "Initial stage",
                report
                    .initial
                    .kinetic_stress
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
            line(
                "Final stage",
                report
                    .final_state
                    .kinetic_stress
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
        ],
    );
    out.details["full_mechanical_report"] = json!(report);
    let (positions, velocities, dt) = match &config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab {
            positions,
            velocities,
            dt,
            ..
        } => (positions.as_str(), velocities.as_str(), Some(*dt)),
        _ => ("positions", "velocities", None),
    };
    let mut weak = Vec::new();
    let mut weak_points = Vec::new();
    for (index, pair) in step.stages.windows(2).enumerate() {
        let (old, new) = (&pair[0], &pair[1]);
        let ov = &old.fields[velocities];
        let nv = &new.fields[velocities];
        let ox = &old.fields[positions];
        let nx = &new.fields[positions];
        let d = ov.item_shape.iter().product::<usize>();
        let mut impulse = vec![0.; d];
        let mut transport = vec![0.; d];
        let mut eligibility = vec![0.; d];
        let mut direct = vec![0.; d];
        for row in 0..old.validity.len() {
            let a = old.validity[row].eligible(config.include_truncated);
            let b = new.validity[row].eligible(config.include_truncated);
            let pa = if a { ox.values[row * d].tanh() } else { 0. };
            let pb = if b { nx.values[row * d].tanh() } else { 0. };
            for j in 0..d {
                let u = if a { ov.values[row * d + j] } else { 0. };
                let v = if b { nv.values[row * d + j] } else { 0. };
                impulse[j] += f64::from(b) * pb * (v - u);
                transport[j] += f64::from(b) * u * (pb - pa);
                eligibility[j] += (f64::from(b) - f64::from(a)) * u * pa;
                direct[j] += f64::from(b) * v * pb - f64::from(a) * u * pa;
            }
        }
        let residual = direct
            .iter()
            .enumerate()
            .map(|(j, &v)| (v - impulse[j] - transport[j] - eligibility[j]).abs())
            .fold(0., f64::max);
        weak_points.push([index as f64, residual]);
        weak.push(json!({"from":old.stage,"to":new.stage,"impulse":impulse,"transport":transport,"eligibility":eligibility,"measured_increment":direct,"residual":residual}));
    }
    plot(
        out,
        "Weak local momentum identity",
        "operator transition",
        "maximum component residual",
        vec![line(
            "Measured minus impulse, transport and eligibility",
            weak_points,
        )],
    );
    out.details["weak_momentum"] = json!({"test_function":"tanh(first spatial coordinate)","inactive_representative":"zero velocity and zero test-function value","stages":weak});
    if let Some(dt) = dt {
        let mut kicks = Vec::new();
        let mut errors = Vec::new();
        for (index, name) in ["B1", "B2"].iter().enumerate() {
            let input = step
                .stages
                .iter()
                .find(|s| s.stage == format!("{name}_input"));
            let output = step
                .stages
                .iter()
                .find(|s| s.stage == format!("{name}_before_boundary"));
            let force = step
                .field_evaluations
                .iter()
                .find(|f| f.stage == *name && f.field == "total_force");
            if let (Some(input), Some(output), Some(force)) = (input, output, force) {
                if force.version != input.version {
                    return Err(GasError::Capability(
                        "kick force does not match input population version".into(),
                    ));
                }
                let old = &input.fields[velocities];
                let new = &output.fields[velocities];
                let d = old.item_shape.iter().product::<usize>();
                let mut predicted = 0.;
                let mut measured = 0.;
                let mut velocity_residual = 0_f64;
                for row in 0..input.validity.len() {
                    if !input.validity[row].eligible(config.include_truncated) {
                        continue;
                    }
                    for j in 0..d {
                        let k = row * d + j;
                        let v = old.values[k];
                        let w = new.values[k];
                        let dv = dt / 2. * force.values[k];
                        predicted += v * dv + dv * dv / 2.;
                        measured += (w * w - v * v) / 2.;
                        velocity_residual = velocity_residual.max((w - v - dv).abs());
                    }
                }
                errors.push([index as f64, (measured - predicted).abs()]);
                kicks.push(json!({"stage":name,"predicted_energy_increment":predicted,"measured_energy_increment":measured,"velocity_residual":velocity_residual,"force_convention":"executed total_force is acceleration; delta v = dt/2 times total_force"}));
            }
        }
        if !kicks.is_empty() {
            plot(
                out,
                "Exact finite-kick work",
                "kick index",
                "energy residual",
                vec![line(
                    "Measured minus v dot delta-v plus quadratic term",
                    errors,
                )],
            );
        }
        out.details["kick_work"] = json!(kicks);
    }
    Ok(())
}
