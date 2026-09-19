//! Measurements of the fixed-step population transition, without resimulation.
use crate::{GasError, Population, Result, tracking::RecordedStep};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeanFieldStep {
    pub step: u64,
    pub walkers: usize,
    pub alive_before: usize,
    pub alive_after: usize,
    pub clones: usize,
    pub revivals: usize,
    pub expected_clones: f64,
    pub clone_count_variance: f64,
    pub component_sizes: Vec<usize>,
    pub accepted_edges: usize,
    pub forest_edge_residual: isize,
    pub shared_component_probability: f64,
    pub largest_component_fraction: f64,
    pub mean_cos_position: f64,
    pub mean_sin_position: f64,
    pub mean_square_sin_position: f64,
    pub distinct_pair_sin_product: Option<f64>,
    pub mean_velocity: f64,
    pub mean_velocity_squared: f64,
}
pub fn population_observables(p: &Population<f64>) -> Result<[f64; 6]> {
    let x = p.observations.field("positions")?;
    let v = p.observations.field("velocities")?;
    let n = p.len() as f64;
    let mut out = [0.; 6];
    for i in 0..p.len() {
        let x = x.row(i)?[0];
        let v = v.row(i)?[0];
        out[0] += x.cos() / n;
        out[1] += x.sin() / n;
        out[2] += x.sin().powi(2) / n;
        out[4] += v / n;
        out[5] += v * v / n;
    }
    out[3] = if n > 1. {
        (n * out[1] * out[1] - out[2]) / (n - 1.)
    } else {
        0.
    };
    Ok(out)
}
pub fn step_diagnostics(step: &RecordedStep<f64>) -> Result<MeanFieldStep> {
    let plan = &step.report.clone_plan;
    let n = plan.choices.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut edges = 0usize;
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for (i, c) in plan.choices.iter().enumerate() {
        if !c.accepted {
            continue;
        }
        let d = c
            .donors
            .first()
            .ok_or_else(|| GasError::Configuration("accepted clone has no donor".into()))?;
        let s = plan
            .sources
            .get(d.pool_index as usize)
            .ok_or_else(|| GasError::Configuration("missing donor source".into()))?;
        if s.frame + 1 != step.report.step {
            return Err(GasError::Capability(
                "current-population component diagnostics require current donors".into(),
            ));
        }
        let j = s.slot as usize;
        if j >= n {
            return Err(GasError::Configuration("invalid donor slot".into()));
        }
        if i != j {
            edges += 1;
        }
        let a = root(&mut parent, i);
        let b = root(&mut parent, j);
        parent[a] = b;
    }
    let mut counts = vec![0usize; n];
    for i in 0..n {
        let j = root(&mut parent, i);
        counts[j] += 1;
    }
    let component_sizes: Vec<_> = counts.into_iter().filter(|&s| s > 0).collect();
    let shared = component_sizes
        .iter()
        .map(|&s| s as f64 * (s - 1) as f64)
        .sum::<f64>();
    let observable = population_observables(&step.final_population)?;
    let mut expected = 0.;
    let mut variance = 0.;
    for c in &plan.choices {
        if c.revival {
            continue;
        }
        let p = c
            .probability
            .ok_or_else(|| GasError::Capability("clone probability unavailable".into()))?;
        expected += p;
        variance += p * (1. - p);
    }
    Ok(MeanFieldStep {
        step: step.report.step,
        walkers: n,
        alive_before: step
            .report
            .pre_clone_eligible
            .iter()
            .filter(|&&a| a)
            .count(),
        alive_after: step.report.eligible,
        clones: step.report.clones,
        revivals: step.report.revivals,
        expected_clones: expected,
        clone_count_variance: variance,
        shared_component_probability: if n > 1 {
            shared / (n as f64 * (n - 1) as f64)
        } else {
            0.
        },
        largest_component_fraction: component_sizes.iter().copied().max().unwrap_or(0) as f64
            / n as f64,
        accepted_edges: edges,
        forest_edge_residual: edges as isize - (n - component_sizes.len()) as isize,
        component_sizes,
        mean_cos_position: observable[0],
        mean_sin_position: observable[1],
        mean_square_sin_position: observable[2],
        distinct_pair_sin_product: (n > 1).then_some(observable[3]),
        mean_velocity: observable[4],
        mean_velocity_squared: observable[5],
    })
}
