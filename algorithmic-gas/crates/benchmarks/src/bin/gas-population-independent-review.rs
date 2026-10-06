//! Independent actual-stage check of the quadratic BAOAB memory observable.
use algorithmic_gas::tracking::RecordedStep;
use algorithmic_gas_benchmarks::convergence_experiments::{ArchiveStore, sha256_file};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

fn stage<'a>(record: &'a RecordedStep<f64>, name: &str, field: &str) -> Option<&'a [f64]> {
    record
        .stages
        .iter()
        .find(|s| s.stage == name)?
        .fields
        .get(field)
        .map(|v| v.values.as_slice())
}
fn gradient<'a>(record: &'a RecordedStep<f64>, name: &str) -> Option<&'a [f64]> {
    record
        .field_evaluations
        .iter()
        .find(|s| s.stage == name && s.field == "potential_gradient")
        .map(|s| s.values.as_slice())
}
fn inverse_cap(velocity: &[f64], radius: f64) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let norm = velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !radius.is_finite() || radius <= norm || velocity.iter().any(|v| !v.is_finite()) {
        return Err("inverse smooth cap requires finite velocity in the open ball".into());
    }
    Ok(velocity
        .iter()
        .map(|v| radius * v / (radius - norm))
        .collect())
}
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
#[derive(Default)]
struct Group {
    count: usize,
    native_frames: usize,
    residual_sum: f64,
    observed_sum: f64,
    expected_sum: f64,
    gaussian_lipschitz_squared_sum: f64,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("gas-population-independent-review COMPLETE_NATIVE_INPUT EMPTY_OUTPUT".into());
    }
    let input = ArchiveStore::open(&args[0])?;
    if !["complete", "completed"].contains(&input.status()) {
        return Err("complete immutable input required".into());
    }
    let mut output = ArchiveStore::new(&args[1])?;
    let source = include_str!(
        "../../../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"
    );
    output.save_json("independent-execution-source",&json!({"runner":include_str!("gas-population-independent-review.rs"),"chapter":source,"native_kinetic":include_str!("../../../algorithmic-gas/src/kinetic.rs"),"binary_sha256":sha256_file(&std::env::current_exe()?)?,"input_index_sha256":sha256_file(&input.root().join("archive-index.json"))?,"new_native_updates":0}))?;
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    let mut raw = vec![];
    let mut batches = vec![];
    let mut consumed = vec![];
    let mut seed_units = BTreeSet::new();
    let mut skipped = 0usize;
    let mut frames = 0usize;
    let mut checks = 0usize;
    let mut failures = 0usize;
    let mut max_kick_error = 0_f64;
    let mut max_inverse_error = 0_f64;
    let mut max_memory_error = 0_f64;
    for entry in input
        .entries()
        .iter()
        .filter(|e| e.kind == "native_run_archive")
    {
        let archive = input.load_archive(&entry.path)?;
        if archive.steps.len() != 1 {
            return Err(
                "this conditional concentration audit requires independent one-update archives"
                    .into(),
            );
        }
        let (law, replica) = entry
            .tag
            .rsplit_once("-rep")
            .ok_or("missing independent replica tag")?;
        let replica = replica
            .parse::<usize>()
            .map_err(|_| "replica tag must end in a numeric independent seed address")?;
        if !seed_units.insert((law.to_owned(), replica)) {
            return Err("duplicate independent seed unit".into());
        }
        let config = serde_json::to_value(&archive.gas_config)?;
        let h = config
            .pointer("/kinetic/integrator/dt")
            .and_then(|v| v.as_f64())
            .ok_or("missing dt")?;
        let radius = config
            .pointer("/kinetic/velocity_cap")
            .and_then(|v| v.as_f64())
            .ok_or("missing terminal smooth cap")?;
        let diffusion = config
            .pointer("/kinetic/position_diffusion")
            .and_then(|v| v.as_f64())
            .ok_or("missing final position diffusion")?;
        if h <= 0. || !h.is_finite() {
            return Err("positive finite timestep required".into());
        }
        let c = h / 2.;
        let k = 1. - c * c;
        let coefficient = k / c;
        let position_sigma = diffusion * h.sqrt();
        consumed.push(json!({"path":entry.path,"tag":entry.tag,"sha256":entry.sha256}));
        for record in &archive.steps {
            let Some(x0) = stage(record, "B1_input", "positions") else {
                skipped += 1;
                continue;
            };
            let Some(v0) = stage(record, "B1_input", "velocities") else {
                skipped += 1;
                continue;
            };
            let Some(g1) = gradient(record, "B1") else {
                skipped += 1;
                continue;
            };
            let Some(x2) = stage(record, "A2", "positions") else {
                skipped += 1;
                continue;
            };
            let Some(g2) = gradient(record, "B2") else {
                skipped += 1;
                continue;
            };
            // The source formula is for grad U=x, not a nonlinear-force approximation.
            if g1.len() != x0.len()
                || g2.len() != x2.len()
                || g1
                    .iter()
                    .zip(x0)
                    .any(|(a, b)| (a - b).abs() > 1e-12 * (1. + b.abs()))
                || g2
                    .iter()
                    .zip(x2)
                    .any(|(a, b)| (a - b).abs() > 1e-12 * (1. + b.abs()))
            {
                skipped += 1;
                continue;
            }
            if config
                .pointer("/kinetic/boundary_schedule")
                .and_then(|v| v.as_str())
                != Some("end_of_step")
            {
                return Err("terminal-only boundary schedule required".into());
            }
            let v1 = stage(record, "B1", "velocities").ok_or("missing B1 velocity")?;
            let v3 = stage(record, "B2", "velocities").ok_or("missing B2 velocity")?;
            let xf = record.final_population.observations.field("positions")?;
            let vf = record.final_population.observations.field("velocities")?;
            let n = record.before.len();
            let d = xf.width();
            if [
                x0.len(),
                v0.len(),
                v1.len(),
                v3.len(),
                x2.len(),
                xf.values().len(),
                vf.values().len(),
            ]
            .iter()
            .any(|&len| len != n * d)
            {
                return Err("complete native phase arrays required".into());
            }
            for row in 0..n {
                let range = row * d..(row + 1) * d;
                let x = &x0[range.clone()];
                let v = &v0[range.clone()];
                let raw_velocity = &v3[range.clone()];
                let before_noise = &x2[range.clone()];
                let position = &xf.values()[range.clone()];
                let inverse = inverse_cap(&vf.values()[range.clone()], radius)?;
                for a in 0..d {
                    let kick_defect = (v1[row * d + a] - v[a] + c * x[a]).abs();
                    let inverse_defect = (inverse[a] - raw_velocity[a]).abs();
                    let memory_defect = (inverse[a] - coefficient * position[a]
                        + coefficient * x[a]
                        + v[a]
                        + coefficient * (position[a] - before_noise[a]))
                        .abs();
                    max_kick_error = max_kick_error.max(kick_defect);
                    max_inverse_error = max_inverse_error.max(inverse_defect);
                    max_memory_error = max_memory_error.max(memory_defect);
                    let scale = 1.
                        + raw_velocity[a].abs()
                        + coefficient.abs() * (position[a].abs() + x[a].abs());
                    checks += 3;
                    failures += usize::from(kick_defect > 1e-10 * scale)
                        + usize::from(inverse_defect > 1e-10 * scale)
                        + usize::from(memory_defect > 1e-10 * scale);
                }
                for (f, frequency) in [0.1_f64, 0.3, 1.].iter().enumerate() {
                    let mut t = vec![0.; d];
                    t[0] = *frequency;
                    if f == 1 {
                        t.fill(*frequency / (d as f64).sqrt());
                    }
                    let input_phase: Vec<_> =
                        x.iter().zip(v).map(|(x, v)| -coefficient * x - v).collect();
                    let output_phase: Vec<_> = inverse
                        .iter()
                        .zip(position)
                        .map(|(v, x)| v - coefficient * x)
                        .collect();
                    let norm_t_squared = dot(&t, &t);
                    let beta_squared = (coefficient * position_sigma).powi(2) * norm_t_squared;
                    let observed = dot(&t, &output_phase).cos();
                    let expected = dot(&t, &input_phase).cos() * (-0.5 * beta_squared).exp();
                    let group = groups
                        .entry(format!("{law}-step{}-fourier{f}", record.report.step))
                        .or_default();
                    group.count += 1;
                    group.residual_sum += observed - expected;
                    group.observed_sum += observed;
                    group.expected_sum += expected;
                    group.gaussian_lipschitz_squared_sum += beta_squared;
                    if row == 0 {
                        group.native_frames += 1;
                    }
                    raw.push(json!({"source_archive":entry.path,"step":record.report.step,"storage_row":row,"frequency":t,"N":n,"d":d,"input_X":x,"input_V":v,"physical_output_x":position,"inverse_physical_cap_velocity":inverse,"observed":observed,"conditional_expectation":expected,"beta_squared":beta_squared,"source_label":"lem-chaos-kinetic-memory-observable"}));
                }
            }
            frames += 1;
            if raw.len() >= 4096 {
                batches.push(output.save_json(
                    &format!("memory-batch{}", batches.len()),
                    &json!({"rows":raw}),
                )?);
                raw.clear();
            }
        }
    }
    if !raw.is_empty() {
        batches.push(output.save_json(
            &format!("memory-batch{}", batches.len()),
            &json!({"rows":raw}),
        )?);
    }
    if frames == 0 {
        return Err("no actual native quadratic rows met the source hypotheses".into());
    }
    let family_alpha = 1e-6_f64;
    let log_factor = (2. * groups.len() as f64 / family_alpha).ln();
    let mut comparisons = vec![];
    for (id, group) in groups {
        let denominator = group.count as f64;
        let allowance =
            (2. * log_factor * group.gaussian_lipschitz_squared_sum).sqrt() / denominator + 1e-10;
        let residual = group.residual_sum / denominator;
        let passed = residual.abs() <= allowance;
        checks += 1;
        failures += usize::from(!passed);
        comparisons.push(json!({"id":id,"source_label":"lem-chaos-kinetic-memory-observable","observed_mean":group.observed_sum/denominator,"predicted_mean":group.expected_sum/denominator,"residual":residual,"gaussian_concentration_allowance":allowance,"family_error_probability":family_alpha,"candidate_rows":group.count,"independent_native_replicas":group.native_frames,"passed":passed}));
    }
    let report = json!({"chapter":9,"scope":"Independent review of actual quadratic native BAOAB recorded stages. The inverse smooth cap and both force kicks reconstruct the exact kinetic memory identity. Complete prepared (X,V), including copying/collision/jitter, is conditioned before OU and final noise. Fourier residuals use Gaussian Lipschitz concentration conditional on the full preparation and a simultaneous family allowance. Native row identities are data addresses; all observable averages use probability normalization. Terminal status does not discard physical coordinates; no QSD/global mixing assertion.","source_label":"lem-chaos-kinetic-memory-observable","comparisons":comparisons,"raw_batches":batches,"consumed_archives":consumed,"summary":{"retained_quadratic_native_frames":frames,"excluded_nonquadratic_or_unavailable_frames":skipped,"comparisons":checks,"comparisons_failed":failures,"new_native_steps":0,"maximum_first_kick_identity_error":max_kick_error,"maximum_inverse_cap_error":max_inverse_error,"maximum_ou_cancellation_error":max_memory_error}});
    output.save_json("completion-report", &report)?;
    output.finish("complete")?;
    let verification = output.verify(true)?;
    fs::write(
        PathBuf::from(&args[1]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        PathBuf::from(&args[1]).join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", report["summary"]);
    if failures > 0 {
        return Err(
            "independent native kinetic memory comparison failed; evidence retained".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radial_inverse_is_vectorial_and_rejects_boundary() {
        let input = [3., 4.];
        let radius = 2.;
        let capped: Vec<_> = input.iter().map(|v| v * radius / (radius + 5.)).collect();
        let recovered = inverse_cap(&capped, radius).unwrap();
        for (a, b) in recovered.iter().zip(input) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(inverse_cap(&[radius, 0.], radius).is_err());
    }
    #[test]
    fn quadratic_cancellation_survives_ou_amplitude_and_resonance() {
        for h in [0.04_f64, 0.2, 2.] {
            for q in [0., 0.2, 5.] {
                let c = h / 2.;
                let k = 1. - c * c;
                let x = 0.3;
                let v = 0.2;
                let v1 = v - c * x;
                let x1 = x + c * v1;
                let v2 = (-h).exp() * v1 + q * 1.7;
                let x2 = x1 + c * v2;
                let v3 = v2 - c * x2;
                assert!((v3 - k / c * x2 + k / c * x + v).abs() < 1e-12);
                if h == 2. {
                    assert!((v3 + v).abs() < 1e-12);
                }
            }
        }
    }
}
