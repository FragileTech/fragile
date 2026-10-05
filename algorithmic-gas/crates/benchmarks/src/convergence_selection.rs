//! Finite, source-mapped checks of the chapter 3 selection and composition lemmas.
//!
//! Geometric constants belong to the displayed finite cloud. Conditional
//! selection uses retained fitness, and measurement averaging enumerates the
//! actual joint measurement law before evaluating acceptance. The finite
//! transition examples verify algebraic composition under supplied component
//! bounds; they do not certify kinetic contraction of an engine configuration.
use algorithmic_gas::{
    GasError, ObservationBatch, Result, TensorBatch,
    cloning::CloneDecision,
    fitness::{FitnessPipeline, PositiveMap, PositiveMapping, Standardizer},
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::convergence_framework::BoundCheck;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionFixture {
    pub name: String,
    pub scope: String,
    /// Inputs and checked hypotheses of this fixture, not global certificates.
    pub hypotheses: Value,
    pub constants: BTreeMap<String, f64>,
    pub observations: Value,
    pub checks: Vec<BoundCheck>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionReport {
    pub seed: u64,
    pub constants: BTreeMap<String, f64>,
    pub fixtures: Vec<SelectionFixture>,
    pub checks: Vec<BoundCheck>,
    pub scope_notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalTransport1D {
    /// Storage indices of the optimal uniform transport; indices track its
    /// implementation but do not define the empirical measure or its cost.
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
    pub centered_discrepancies: Vec<f64>,
    pub squared_wasserstein: f64,
}

/// Exact monotone centered positional W2 transport for two equal-size uniform
/// one-dimensional empirical measures. This is a positional cost only; it
/// does not optimize a joint position/velocity hypocoercive cost.
pub fn optimal_centered_transport_1d(left: &[f64], right: &[f64]) -> Result<PositionalTransport1D> {
    validate_values(left, 1)?;
    validate_values(right, 1)?;
    if left.len() != right.len() {
        return Err(error(
            "uniform one-dimensional transport requires equal alive counts",
        ));
    }
    let ml = mean(left);
    let mr = mean(right);
    let mut left_indices: Vec<usize> = (0..left.len()).collect();
    let mut right_indices: Vec<usize> = (0..right.len()).collect();
    left_indices.sort_by(|&a, &b| left[a].total_cmp(&left[b]));
    right_indices.sort_by(|&a, &b| right[a].total_cmp(&right[b]));
    let centered_discrepancies: Vec<f64> = left_indices
        .iter()
        .zip(&right_indices)
        .map(|(&i, &j)| (left[i] - ml) - (right[j] - mr))
        .collect();
    let squared_wasserstein =
        centered_discrepancies.iter().map(|d| d * d).sum::<f64>() / left.len() as f64;
    if !squared_wasserstein.is_finite() {
        return Err(error("centered positional transport cost overflow"));
    }
    Ok(PositionalTransport1D {
        left_indices,
        right_indices,
        centered_discrepancies,
        squared_wasserstein,
    })
}

fn error(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|x| (x - m).powi(2)).sum::<f64>() / x.len() as f64
}
fn range(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - x.iter().copied().fold(f64::INFINITY, f64::min)
}
fn check(
    checks: &mut Vec<BoundCheck>,
    id: &str,
    labels: &[&str],
    scope: &str,
    observed: f64,
    upper: f64,
) {
    checks.push(BoundCheck::upper(id, labels, scope, observed, upper));
}
fn lower(
    checks: &mut Vec<BoundCheck>,
    id: &str,
    labels: &[&str],
    scope: &str,
    observed: f64,
    bound: f64,
) {
    check(checks, id, labels, scope, -observed, -bound);
}
fn identity(
    checks: &mut Vec<BoundCheck>,
    id: &str,
    labels: &[&str],
    scope: &str,
    left: f64,
    right: f64,
) {
    check(checks, id, labels, scope, (left - right).abs(), 0.);
}
fn logistic_derivative(z: f64) -> f64 {
    let e = (-z.abs()).exp();
    2. * e / (1. + e).powi(2)
}
fn validate_values(x: &[f64], minimum_length: usize) -> Result<()> {
    if x.len() < minimum_length || x.iter().any(|v| !v.is_finite()) {
        Err(error(
            "finite selection vectors have invalid length or values",
        ))
    } else {
        Ok(())
    }
}

/// Complete-linkage agglomeration on an explicitly supplied symmetric distance
/// matrix. Equal distances are resolved by the current cluster order. All
/// singleton and statistically invalid clusters remain in the result.
pub fn complete_linkage(distances: &[Vec<f64>], maximum_diameter: f64) -> Result<Vec<Vec<usize>>> {
    let n = distances.len();
    if n == 0 || !maximum_diameter.is_finite() || maximum_diameter <= 0. {
        return Err(error("invalid complete-linkage cloud or diameter"));
    }
    for row in distances {
        if row.len() != n {
            return Err(error("complete-linkage distance matrix must be square"));
        }
        for &distance in row {
            if !distance.is_finite() || distance < 0. {
                return Err(error("invalid complete-linkage distance"));
            }
        }
    }
    for (i, row) in distances.iter().enumerate() {
        if row[i] != 0.
            || row
                .iter()
                .enumerate()
                .any(|(j, &distance)| (distance - distances[j][i]).abs() > 1e-12)
        {
            return Err(error(
                "complete-linkage distances must be symmetric with zero diagonal",
            ));
        }
    }
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..groups.len() {
            for b in a + 1..groups.len() {
                let d = groups[a]
                    .iter()
                    .flat_map(|&i| groups[b].iter().map(move |&j| distances[i][j]))
                    .fold(0., f64::max);
                if d <= maximum_diameter && best.is_none_or(|(old, _, _)| d < old) {
                    best = Some((d, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { break };
        let other = groups.remove(b);
        groups[a].extend(other);
        groups[a].sort_unstable();
    }
    groups.sort_by_key(|g| g[0]);
    Ok(groups)
}

fn observations(x: &[f64], v: &[f64]) -> Result<ObservationBatch<f64>> {
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(x.len(), 1, x.to_vec())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(v.len(), 1, v.to_vec())?,
    );
    Ok(obs)
}
fn distances(obs: &ObservationBatch<f64>, lambda: f64) -> Result<Vec<Vec<f64>>> {
    let distance = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 1.,
        velocity_scale: 1.,
        lambda,
        periodic: None,
    };
    distance_matrix(obs, &distance)
}
fn distance_matrix(obs: &ObservationBatch<f64>, distance: &Distance) -> Result<Vec<Vec<f64>>> {
    let n = obs.field("positions")?.rows();
    (0..n)
        .map(|i| (0..n).map(|j| distance.compare(obs, i, obs, j)).collect())
        .collect()
}
fn gaussian_rows(d: &[Vec<f64>], width: f64) -> Result<Vec<Vec<f64>>> {
    let kernel = Kernel::Gaussian { width };
    let mut output = vec![vec![0.; d.len()]; d.len()];
    for i in 0..d.len() {
        for j in 0..d.len() {
            if i != j {
                output[i][j] = <Kernel as InteractionKernel<f64>>::log_weight(
                    &kernel,
                    d[i][j],
                    algorithmic_gas::geometry::ComparisonKind::Distance,
                )?
                .exp();
            }
        }
        let sum = output[i].iter().sum::<f64>();
        if sum <= 0. || !sum.is_finite() {
            return Err(error("unrepresentable Gaussian donor normalization"));
        }
        output[i].iter_mut().for_each(|p| *p /= sum);
    }
    Ok(output)
}

fn validate_geometry(seed: u64) -> Result<SelectionFixture> {
    let scope = "Finite entering alive cloud; actual unsquashed phase-space metric, complete linkage, all invalid clusters retained. Thresholds and diameters are fixture inputs.";
    let mut rng = RandomStream::new(seed, 0, Stream::Initialize, 0, 0);
    let mut x = Vec::new();
    for (center, count) in [(-2., 5), (0.5, 5), (1.2, 5), (2.5, 3)] {
        for i in 0..count {
            x.push(
                center
                    + 0.02 * (i as f64 - (count - 1) as f64 / 2.)
                    + 0.001 * (rng.uniform::<f64>() - 0.5),
            );
        }
    }
    let v: Vec<f64> = x.iter().map(|x| 0.05 * x).collect();
    let lambda_alg: f64 = 2.;
    let lambda_v: f64 = 1.;
    let d_cluster: f64 = 0.12;
    let eps_o: f64 = 0.25;
    let r_var_sq: f64 = 0.8;
    let k = x.len();
    let n = 24;
    let d = distances(&observations(&x, &v)?, lambda_alg)?;
    let groups = complete_linkage(&d, d_cluster)?;
    let k_min = 5_usize.max((0.05 * k as f64).ceil() as usize);
    let mx = mean(&x);
    let mv = mean(&v);
    let vx = variance(&x);
    let vv = variance(&v);
    let vh = vx + lambda_v * vv;
    let dx = range(&x);
    let dv = range(&v);
    let dh_sq = dx * dx + lambda_v * dv * dv;
    let dvalid_sq = dx * dx + lambda_alg * dv * dv;
    let c_lambda = (lambda_v / lambda_alg).max(1.);
    let gmeans: Vec<(f64, f64)> = groups
        .iter()
        .map(|g| {
            (
                g.iter().map(|&i| x[i]).sum::<f64>() / g.len() as f64,
                g.iter().map(|&i| v[i]).sum::<f64>() / g.len() as f64,
            )
        })
        .collect();
    let contributions: Vec<f64> = groups
        .iter()
        .zip(&gmeans)
        .map(|(g, &(gx, gv))| g.len() as f64 * ((gx - mx).powi(2) + lambda_v * (gv - mv).powi(2)))
        .collect();
    let mut valid: Vec<usize> = (0..groups.len())
        .filter(|&g| groups[g].len() >= k_min)
        .collect();
    valid.sort_by(|&a, &b| {
        contributions[b]
            .total_cmp(&contributions[a])
            .then(a.cmp(&b))
    });
    let valid_total = valid.iter().map(|&g| contributions[g]).sum::<f64>();
    let mut chosen = vec![false; groups.len()];
    for (g, members) in groups.iter().enumerate() {
        if members.len() < k_min {
            chosen[g] = true;
        }
    }
    let mut selected_valid = 0.;
    for &g in &valid {
        if selected_valid >= (1. - eps_o) * valid_total {
            break;
        }
        chosen[g] = true;
        selected_valid += contributions[g];
    }
    let mut high = vec![false; k];
    for (g, members) in groups.iter().enumerate() {
        if chosen[g] {
            for &i in members {
                high[i] = true;
            }
        }
    }
    if high.iter().all(|&h| h) || high.iter().all(|&h| !h) {
        return Err(error("geometry fixture needs nonempty high and low groups"));
    }
    let high_count = high.iter().filter(|&&h| h).count();
    let between = contributions.iter().sum::<f64>() / k as f64;
    let within = groups
        .iter()
        .zip(&gmeans)
        .map(|(g, &(gx, gv))| {
            g.iter()
                .map(|&i| (x[i] - gx).powi(2) + lambda_v * (v[i] - gv).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / k as f64;
    let bh = (0..groups.len())
        .filter(|&g| chosen[g])
        .map(|g| contributions[g])
        .sum::<f64>()
        / k as f64;
    let f_h_cl = (1. - eps_o) * (r_var_sq - c_lambda * d_cluster.powi(2) / 2.) / dh_sq;
    let radii: Vec<f64> = groups
        .iter()
        .zip(&gmeans)
        .map(|(g, &(gx, _))| g.iter().map(|&i| (x[i] - gx).abs()).fold(0., f64::max))
        .collect();
    let rho_h = (0..groups.len())
        .filter(|&g| chosen[g])
        .map(|g| radii[g])
        .fold(0., f64::max);
    let rho_l = (0..groups.len())
        .filter(|&g| !chosen[g])
        .map(|g| radii[g])
        .fold(0., f64::max);
    let mut s_hl = f64::INFINITY;
    let mut rl: f64 = 0.;
    for a in 0..groups.len() {
        if chosen[a] {
            for b in 0..groups.len() {
                if !chosen[b] {
                    s_hl = s_hl.min((gmeans[a].0 - gmeans[b].0).abs());
                }
            }
        } else {
            for &i in &groups[a] {
                for &j in &groups[a] {
                    rl = rl.max(d[i][j]);
                }
            }
        }
    }
    let dh = s_hl - rho_h - rho_l;
    if !(vx > r_var_sq && r_var_sq > c_lambda * d_cluster.powi(2) / 2. && dh > rl) {
        return Err(error(
            "geometry fixture does not meet its declared strict thresholds",
        ));
    }
    let mut checks = Vec::new();
    let pair = (0..k)
        .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
        .map(|(i, j)| (x[i] - x[j]).powi(2) + lambda_v * (v[i] - v[j]).powi(2))
        .sum::<f64>()
        / (k * k) as f64;
    identity(
        &mut checks,
        "phase_pairwise_variance",
        &["lem-phase-space-packing"],
        scope,
        pair,
        vh,
    );
    lower(
        &mut checks,
        "hypocoercive_variance",
        &["lem-var-x-implies-var-h"],
        scope,
        vh,
        vx,
    );
    let close = (0..k)
        .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
        .filter(|&(i, j)| d[i][j] < d_cluster)
        .count() as f64
        / (k * (k - 1) / 2) as f64;
    let pack_upper = (dvalid_sq - 2. * vh) / (dvalid_sq - d_cluster.powi(2));
    check(
        &mut checks,
        "close_pair_fraction",
        &["lem-phase-space-packing"],
        scope,
        close,
        pack_upper,
    );
    check(
        &mut checks,
        "packing_threshold",
        &["lem-phase-space-packing"],
        scope,
        pack_upper,
        1.,
    );
    identity(
        &mut checks,
        "within_between_phase_variance",
        &["lem-outlier-cluster-fraction-lower-bound"],
        scope,
        within + between,
        vh,
    );
    check(
        &mut checks,
        "cluster_within_variance",
        &["lem-outlier-cluster-fraction-lower-bound"],
        scope,
        within,
        c_lambda * d_cluster.powi(2) / 2.,
    );
    lower(
        &mut checks,
        "cluster_between_capture",
        &[
            "def-unified-high-low-error-sets",
            "lem-outlier-cluster-fraction-lower-bound",
        ],
        scope,
        bh,
        (1. - eps_o) * between,
    );
    lower(
        &mut checks,
        "cluster_high_fraction",
        &["lem-outlier-cluster-fraction-lower-bound"],
        scope,
        high_count as f64 / k as f64,
        f_h_cl,
    );
    let vvarx = 2. * k as f64 / n as f64 * vx;
    lower(
        &mut checks,
        "two_swarm_variance_threshold",
        &["cor-vvarx-to-high-error-fraction"],
        "Two identical nonextinct fixture clouds in N=24 slots, normalized by alive k/N.",
        vvarx,
        2. * r_var_sq,
    );
    lower(
        &mut checks,
        "fixed_n_high_mass",
        &["cor-vvarx-to-high-error-fraction"],
        scope,
        high_count as f64 / n as f64,
        k as f64 / n as f64 * f_h_cl,
    );
    let mut min_cross = f64::INFINITY;
    for i in 0..k {
        if high[i] {
            for j in 0..k {
                if !high[j] {
                    min_cross = min_cross.min(d[i][j]);
                }
            }
        }
    }
    lower(
        &mut checks,
        "cross_group_separation",
        &["lem-geometric-separation-of-partition"],
        scope,
        min_cross,
        dh,
    );
    for (g, members) in groups.iter().enumerate() {
        let values: Vec<f64> = members.iter().map(|&i| x[i]).collect();
        let physical_diameter = range(&values);
        check(
            &mut checks,
            &format!("cluster_{g}_radius"),
            &["rem-cluster-energy-and-separation"],
            scope,
            radii[g],
            physical_diameter,
        );
        check(
            &mut checks,
            &format!("cluster_{g}_physical_variance"),
            &["rem-cluster-energy-and-separation"],
            scope,
            variance(&values),
            physical_diameter.powi(2) / 2.,
        );
        if !chosen[g] {
            lower(
                &mut checks,
                &format!("cluster_{g}_low_nonself_count"),
                &["lem-geometric-separation-of-partition"],
                scope,
                (members.len() - 1) as f64,
                0.04 * k as f64,
            );
            lower(
                &mut checks,
                &format!("cluster_{g}_valid_mass"),
                &["thm-keystone-averaged-cluster-pressure"],
                "AP6 only for actually valid clusters of this complete-linkage partition.",
                (members.len() - 1) as f64 / (k - 1) as f64,
                0.04,
            );
        }
        if members.len() < k_min {
            identity(
                &mut checks,
                &format!("cluster_{g}_invalid_retained"),
                &["def-unified-high-low-error-sets"],
                scope,
                members.iter().filter(|&&i| high[i]).count() as f64,
                members.len() as f64,
            );
        }
    }
    // Auxiliary global-energy subset is selected separately from the actual H.
    let energy: Vec<f64> = x
        .iter()
        .zip(&v)
        .map(|(&x, &v)| (x - mx).powi(2) + lambda_v * (v - mv).powi(2))
        .collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| energy[b].total_cmp(&energy[a]));
    let mut o = Vec::new();
    let mut captured = 0.;
    for i in order {
        if captured >= (1. - eps_o) * k as f64 * vh {
            break;
        }
        o.push(i);
        captured += energy[i];
    }
    lower(
        &mut checks,
        "auxiliary_global_outlier_fraction",
        &["lem-outlier-fraction-lower-bound"],
        "Separately selected auxiliary global-energy subset O; no identification with the configured cluster H.",
        o.len() as f64 / k as f64,
        (1. - eps_o) * r_var_sq / dh_sq,
    );
    let c_h = (1. - eps_o) * (1. - d_cluster.powi(2) / (2. * r_var_sq));
    let h_position_energy = (0..k)
        .filter(|&i| high[i])
        .map(|i| (x[i] - mx).powi(2))
        .sum::<f64>();
    lower(
        &mut checks,
        "high_position_energy",
        &["lem-variance-concentration-Hk"],
        "Same partition; v_i=.05*x_i makes phase and positional between-cluster contribution rankings identical.",
        h_position_energy,
        c_h * k as f64 * vx,
    );
    let mut common = vec![true; k];
    let omitted = (0..k)
        .find(|&i| high[i])
        .ok_or_else(|| error("missing high row"))?;
    common[omitted] = false;
    let fitness: Vec<f64> = (0..k).map(|i| if high[i] { 0.8 } else { 1.4 }).collect();
    let errors: Vec<f64> = x.iter().map(|x| (x - mx).powi(2)).collect();
    let kernel = gaussian_rows(&d, 2.)?;
    let selection =
        analyze_conditional_selection(&fitness, &kernel, &high, &common, &errors, 1.4, 1., 1e-6)?;
    let target: Vec<usize> = (0..k)
        .filter(|&i| high[i] && common[i] && fitness[i] <= mean(&fitness))
        .collect();
    let b_t = (0..k)
        .filter(|&i| high[i] && !target.contains(&i))
        .map(|i| errors[i])
        .sum::<f64>()
        / n as f64;
    // The centered positional transport to a point mass has alive-normalized
    // cost Var_x. The fixed-slot energy S_k/N has the distinct factor k/N.
    let structural = vx;
    let variance_to_transport = k as f64 / n as f64;
    let target_energy = target.iter().map(|&i| errors[i]).sum::<f64>() / n as f64;
    let c_error = c_h * variance_to_transport / 2.;
    let g_error = b_t;
    lower(
        &mut checks,
        "target_complement_error",
        &["lem-error-concentration-target-set"],
        "Purely positional centered transport to the second cloud's point mass: V_xstruct=Var_x, S_k/N=(k/N)V_xstruct, a=k/N, b=M_j=0; N=24 and one omitted common label charged its actual B_T. No full phase-space transport claim.",
        target_energy,
        c_error * structural - g_error,
    );
    let pu = selection.constants["p_u"];
    let probabilities = selection.observations["acceptance_probabilities"]
        .as_array()
        .ok_or_else(|| error("selection probabilities missing"))?;
    let pressure_error = (0..k)
        .filter(|&i| common[i])
        .map(|i| probabilities[i].as_f64().unwrap_or(0.) * errors[i])
        .sum::<f64>()
        / n as f64;
    lower(
        &mut checks,
        "target_weighted_pressure_error",
        &[
            "cor-cloning-pressure-target-set",
            "lem-error-concentration-target-set",
        ],
        "Retained conditional fitness and independent Gaussian cloning donors; fixed errors of a supplied comparison coupling, with actual common alive storage rows.",
        pressure_error,
        pu * (c_error * structural - g_error),
    );
    let mut constants = BTreeMap::from([
        ("D_x".into(), dx),
        ("D_v".into(), dv),
        ("D_h_squared".into(), dh_sq),
        ("D_valid_squared".into(), dvalid_sq),
        ("R_pack_squared".into(), d_cluster.powi(2) / 2.),
        ("C_lambda".into(), c_lambda),
        ("f_H_cl".into(), f_h_cl),
        ("f_c".into(), 0.04),
        ("D_H".into(), dh),
        ("R_L".into(), rl),
        ("c_H".into(), c_h),
        ("c_error".into(), c_error),
        ("g_error".into(), g_error),
        (
            "variance_to_positional_transport_a".into(),
            variance_to_transport,
        ),
        ("p_u".into(), pu),
    ]);
    for (key, value) in &selection.constants {
        constants.insert(format!("selection_{key}"), *value);
    }
    checks.extend(selection.checks);
    Ok(SelectionFixture {
        name: "finite_cluster_geometry_and_target".into(),
        scope: scope.into(),
        hypotheses: json!({"positions":x,"velocities":v,"lambda_alg":lambda_alg,"lambda_v":lambda_v,"maximum_cluster_diameter":d_cluster,"epsilon_outlier":eps_o,"R_var_squared":r_var_sq,"alive":k,"slots":n,"k_min":k_min,"selection_fitness_input":"prescribed retained positive vector; no measurement-generation claim","other_centered_positions":"zero; positional comparison only"}),
        constants,
        observations: json!({"clusters":groups,"high":high,"common":common,"auxiliary_global_outliers":o,"within_phase_variance":within,"between_phase_variance":between,"position_variance":vx,"phase_variance":vh,"positional_transport_to_point_mass":structural,"fixed_n_position_observable":vvarx,"target":target,"target_error":target_energy,"omitted_error":b_t,"conditional_selection":selection.observations}),
        checks,
    })
}

/// Uses the actual Rust global regularizer and logistic map on one fixed input.
/// Deterministic support bounds are taken from this finite vector's interval.
pub fn analyze_fixed_rescaling(
    raw: &[f64],
    sigma_min: f64,
    floor: f64,
) -> Result<SelectionFixture> {
    validate_values(raw, 2)?;
    if !sigma_min.is_finite() || sigma_min <= 0. || !floor.is_finite() || floor <= 0. {
        return Err(error(
            "fixed-rescale regularizer and floor must be positive",
        ));
    }
    let scope = "One complete realized raw vector with actual common mean and global regularized scale; logistic amplitude two and constant positive floor.";
    let obs = observations(raw, &vec![0.; raw.len()])?;
    let (z, stats) = Standardizer::Global { sigma_min }.apply(raw, &vec![true; raw.len()], &obs)?;
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor,
    };
    let output: Vec<f64> = z.iter().map(|&z| map.map(z)).collect::<Result<_>>()?;
    let a = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let b = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let r = b - a;
    let s_max = (r * r / 4. + sigma_min * sigma_min).sqrt();
    let z_support = r / sigma_min;
    let m_g = logistic_derivative(z_support);
    if !s_max.is_finite() || !z_support.is_finite() || m_g == 0. {
        return Err(error(
            "rescale support or positive derivative is not representable",
        ));
    }
    let r_g = map.map(z_support)? - map.map(-z_support)?;
    let v_raw = variance(raw);
    let v_output = variance(&output);
    let min_var = m_g.powi(2) / s_max.powi(2) * v_raw;
    let max_var = (0.25 / sigma_min.powi(2) * v_raw).min(r_g.powi(2) / 4.);
    let v_max = a.abs().max(b.abs());
    let patch_max = (v_max * v_max + sigma_min * sigma_min).sqrt();
    let derivative_family = logistic_derivative(2. * v_max / sigma_min);
    let mut checks = Vec::new();
    lower(
        &mut checks,
        "fixed_rescale_variance_lower",
        &["prop-fixed-rescale-variance-bound"],
        scope,
        v_output,
        min_var,
    );
    check(
        &mut checks,
        "fixed_rescale_variance_upper",
        &["prop-fixed-rescale-variance-bound"],
        scope,
        v_output,
        max_var,
    );
    check(
        &mut checks,
        "raw_support_variance",
        &["def-max-patched-std"],
        scope,
        v_raw,
        v_max.powi(2),
    );
    check(
        &mut checks,
        "maximum_patched_scale",
        &["def-max-patched-std"],
        scope,
        stats.scale[0],
        patch_max,
    );
    check(
        &mut checks,
        "attained_score_support",
        &["lem-rescale-derivative-lower-bound"],
        scope,
        z.iter().map(|z| z.abs()).fold(0., f64::max),
        2. * v_max / sigma_min,
    );
    let attained_derivative = z
        .iter()
        .map(|&z| logistic_derivative(z))
        .fold(f64::INFINITY, f64::min);
    lower(
        &mut checks,
        "logistic_derivative_lower",
        &["lem-rescale-derivative-lower-bound"],
        scope,
        attained_derivative,
        derivative_family,
    );
    if r > 0. {
        lower(
            &mut checks,
            "variance_to_raw_gap",
            &["lem-variance-to-gap"],
            scope,
            r,
            (2. * v_raw).sqrt(),
        );
        let min_i = raw.iter().position(|&x| x == a).unwrap_or(0);
        let max_i = raw.iter().position(|&x| x == b).unwrap_or(0);
        lower(
            &mut checks,
            "raw_to_rescaled_gap",
            &["lem-raw-gap-to-rescaled-gap"],
            scope,
            (output[max_i] - output[min_i]).abs(),
            derivative_family / patch_max * r,
        );
    }
    Ok(SelectionFixture {
        name: "fixed_shared_rescale".into(),
        scope: scope.into(),
        hypotheses: json!({"raw":raw,"a":a,"b":b,"sigma_min":sigma_min,"floor":floor,"population_normalization":raw.len()}),
        constants: BTreeMap::from([
            ("s_max".into(), s_max),
            ("Z".into(), z_support),
            ("m_g".into(), m_g),
            ("M_g".into(), 0.5),
            ("R_g".into(), r_g),
            ("V_max".into(), v_max),
            ("sigma_patch_max".into(), patch_max),
            ("g_prime_family_min".into(), derivative_family),
            ("variance_lower".into(), min_var),
            ("variance_upper".into(), max_var),
        ]),
        observations: json!({"mean":stats.mean[0],"scale":stats.scale[0],"scores":z,"rescaled":output,"raw_variance":v_raw,"rescaled_variance":v_output}),
        checks,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogGapBounds {
    pub a: f64,
    pub b: f64,
    pub kappa: f64,
    pub chord_slope: f64,
    pub lower_minimizer: f64,
    pub upper_maximizer: f64,
    pub lower: f64,
    pub upper: f64,
}
/// Sharp support-and-mean bounds in the two log-gap lemmas. A lower bound
/// may be negative, and the upper bound need not vanish for equal means.
pub fn log_gap_bounds(a: f64, b: f64, kappa: f64) -> Result<LogGapBounds> {
    if !(a.is_finite()
        && b.is_finite()
        && kappa.is_finite()
        && 0. < a
        && a < b
        && 0. <= kappa
        && kappa <= b - a)
    {
        return Err(error("log gap requires 0<a<b and 0<=kappa<=b-a"));
    }
    let relative_width = (b - a) / a;
    let log_ratio = if relative_width.is_finite() {
        relative_width.ln_1p()
    } else {
        b.ln() - a.ln()
    };
    let c = log_ratio / (b - a);
    let ell = |u: f64| a.ln() + (u - a) / (b - a) * log_ratio;
    // At kappa=b-a, cancellation can put b-kappa one ulp below a.
    // The already-validated feasible interval then consists of the endpoint.
    let feasible_upper = (b - kappa).max(a);
    let lo = (1. / c).clamp(a, feasible_upper);
    let hi = (1. / c - kappa).clamp(a, feasible_upper);
    let lower = ell(lo + kappa) - lo.ln();
    let upper = (hi + kappa).ln() - ell(hi);
    if !c.is_finite() || !lower.is_finite() || !upper.is_finite() {
        return Err(error("log-gap bounds are not representable"));
    }
    Ok(LogGapBounds {
        a,
        b,
        kappa,
        chord_slope: c,
        lower_minimizer: lo,
        upper_maximizer: hi,
        lower,
        upper,
    })
}

fn validate_separation_and_log_fitness() -> Result<SelectionFixture> {
    let scope = "Specified finite H/L populations; arithmetic and logarithmic means, supports, within-group bounds and orientation are evaluated separately.";
    let mut checks = Vec::new();
    let h = [0., 0.2];
    let l = [1., 1.2];
    let within = 0.5 * variance(&h) + 0.5 * variance(&l);
    let all = [0., 0.2, 1., 1.2];
    let gap = mean(&l) - mean(&h);
    identity(
        &mut checks,
        "group_variance_decomposition",
        &["lem-variance-to-mean-separation"],
        scope,
        variance(&all),
        within + 0.25 * gap.powi(2),
    );
    lower(
        &mut checks,
        "group_mean_separation",
        &["lem-variance-to-mean-separation"],
        scope,
        gap,
        ((variance(&all) - within) / 0.25).sqrt(),
    );
    check(
        &mut checks,
        "group_width_within_bound",
        &["lem-variance-to-mean-separation"],
        scope,
        within,
        0.5 * range(&h).powi(2) / 4. + 0.5 * range(&l).powi(2) / 4.,
    );
    // Correlated two-group law: with probability .3 the groups separate; the
    // other outcome is an exact tie. No independence between entries is used.
    let separation_law = [(0.3, vec![0., 0., 1., 1.]), (0.7, vec![0.5; 4])];
    let group_gap = |v: &[f64]| mean(&v[2..]) - mean(&v[..2]);
    let expected_total = separation_law
        .iter()
        .map(|(p, v)| p * variance(v))
        .sum::<f64>();
    let expected_within = separation_law
        .iter()
        .map(|(p, v)| p * (variance(&v[..2]) + variance(&v[2..])) / 2.)
        .sum::<f64>();
    let kappa = 0.07; // Strictly below the independently evaluated total .075.
    let q = (kappa - expected_within) / 0.25;
    let t: f64 = 0.4;
    let event_lower = (q - t.powi(2)) / (1. - t.powi(2));
    lower(
        &mut checks,
        "averaged_separation_second_moment",
        &["cor-averaged-group-separation"],
        "Fixed partition, exactly enumerated correlated separation/tie law; signed orientation holds in both outcomes.",
        separation_law
            .iter()
            .map(|(p, v)| p * group_gap(v).powi(2))
            .sum(),
        q,
    );
    lower(
        &mut checks,
        "averaged_separation_event_probability",
        &["cor-averaged-group-separation"],
        "Two joint outcomes; magnitude threshold .4; equal-fitness outcome retained.",
        separation_law
            .iter()
            .filter(|(_, v)| group_gap(v).abs() > t)
            .map(|(p, _)| p)
            .sum(),
        event_lower,
    );
    lower(
        &mut checks,
        "averaged_separation_variance_hypothesis",
        &["cor-averaged-group-separation"],
        "Specified conditional variance lower input kappa=.07 is checked against the complete joint law.",
        expected_total,
        kappa,
    );
    let (a, b) = (0.1, 2.1);
    let mut sharp = Vec::new();
    for kappa in [0., 0.2, 0.8, 2.] {
        let bounds = log_gap_bounds(a, b, kappa)?;
        // Endpoint mixtures realize the chord. A constant realizes Jensen.
        let mixture_log = |u: f64| {
            let probability_b = (u - a) / (b - a);
            (1. - probability_b) * a.ln() + probability_b * b.ln()
        };
        identity(
            &mut checks,
            &format!("sharp_log_lower_{kappa}"),
            &["lem-log-gap-lower-bound"],
            "Endpoint-mixture X and constant Y with exactly the prescribed mean difference.",
            mixture_log(bounds.lower_minimizer + kappa) - bounds.lower_minimizer.ln(),
            bounds.lower,
        );
        identity(
            &mut checks,
            &format!("sharp_log_upper_{kappa}"),
            &["lem-log-gap-upper-bound"],
            "Constant X and endpoint-mixture Y with exactly the prescribed mean difference.",
            (bounds.upper_maximizer + kappa).ln() - mixture_log(bounds.upper_maximizer),
            bounds.upper,
        );
        // Independent grid optimization is compared with the analytic extremum.
        let grid: Vec<f64> = (0..=1000)
            .map(|j| a + (b - a - kappa) * j as f64 / 1000.)
            .collect();
        let grid_lower = grid
            .iter()
            .map(|&u| mixture_log(u + kappa) - u.ln())
            .fold(f64::INFINITY, f64::min);
        let grid_upper = grid
            .iter()
            .map(|&u| (u + kappa).ln() - mixture_log(u))
            .fold(f64::NEG_INFINITY, f64::max);
        lower(
            &mut checks,
            &format!("log_lower_full_grid_{kappa}"),
            &["lem-log-gap-lower-bound"],
            "1001 endpoint-mixture mean choices over the full feasible interval, evaluated independently of the closed-form extremizer.",
            grid_lower,
            bounds.lower,
        );
        check(
            &mut checks,
            &format!("log_upper_full_grid_{kappa}"),
            &["lem-log-gap-upper-bound"],
            "1001 endpoint-mixture mean choices over the full feasible interval, evaluated independently of the closed-form extremizer.",
            grid_upper,
            bounds.upper,
        );
        sharp.push(bounds);
    }
    let sigma = 0.1;
    let raw_d = [-1., -0.8, 0.8, 1.];
    let raw_r = [0.3, 0.25, 0.1, 0.05];
    let obs = observations(&raw_d, &[0.; 4])?;
    let st = Standardizer::Global { sigma_min: sigma };
    let (zd, _) = st.apply(&raw_d, &[true; 4], &obs)?;
    let (zr, _) = st.apply(&raw_r, &[true; 4], &obs)?;
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let dp: Vec<f64> = zd.iter().map(|&z| map.map(z)).collect::<Result<_>>()?;
    let rp: Vec<f64> = zr.iter().map(|&z| map.map(z)).collect::<Result<_>>()?;
    let alpha: f64 = 0.05;
    let beta: f64 = 1.;
    let pipeline = FitnessPipeline {
        reward_map: map.clone(),
        diversity_map: map,
        reward_exponent: alpha,
        diversity_exponent: beta,
        ..Default::default()
    };
    let fitness = pipeline.combine(&zr, &zd, &[true; 4])?;
    let log_mean = |x: &[f64]| x.iter().map(|v| v.ln()).sum::<f64>() / x.len() as f64;
    let dlog = log_mean(&dp[2..]) - log_mean(&dp[..2]);
    let alog = log_mean(&rp[..2]) - log_mean(&rp[2..]);
    let log_fitness_gap = log_mean(&fitness[2..]) - log_mean(&fitness[..2]);
    let diversity_mean_gap = mean(&dp[2..]) - mean(&dp[..2]);
    let d_star = log_gap_bounds(a, b, diversity_mean_gap)?.lower;
    let a_naive = (b / a).ln();
    let raw_cross_bound = 0.25;
    let kr = 0.5 * raw_cross_bound / sigma;
    let a_regularized = (1. + kr / a).ln();
    let a_star = a_naive.min(a_regularized);
    let delta_log = beta * d_star - alpha * a_star;
    let v_star = a.powf(alpha + beta);
    let v_upper = b.powf(alpha + beta);
    let h_fitness_var = variance(&fitness[..2]);
    let delta = delta_log - h_fitness_var / (2. * v_star.powi(2));
    if delta <= 0. {
        return Err(error(
            "log fitness fixture must discharge its arithmetic-gap condition",
        ));
    }
    identity(
        &mut checks,
        "exact_powered_log_fitness_gap",
        &["thm-derivation-of-stability-condition"],
        scope,
        log_fitness_gap,
        beta * dlog - alpha * alog,
    );
    lower(
        &mut checks,
        "corrective_support_mean_log_gap",
        &["prop-corrective-signal-bound", "lem-log-gap-lower-bound"],
        scope,
        dlog,
        d_star,
    );
    lower(
        &mut checks,
        "corrective_ordered_log_gap",
        &["prop-corrective-signal-bound"],
        "Explicit pairing d'_L[i]>=d'_H[i] for i=0,1; stronger ordered-coupling hypothesis checked.",
        dlog,
        diversity_mean_gap / b,
    );
    check(
        &mut checks,
        "adversarial_naive_log_bound",
        &["prop-adversarial-signal-bound-naive"],
        scope,
        alog.abs(),
        a_naive,
    );
    check(
        &mut checks,
        "raw_reward_population_mean_gap",
        &["prop-raw-reward-mean-gap-bound"],
        "Complete prescribed raw reward channel with cross-pair bound .25; no objective-only regularity claim.",
        (mean(&raw_r[..2]) - mean(&raw_r[2..])).abs(),
        raw_cross_bound,
    );
    check(
        &mut checks,
        "reward_coupled_log_bound",
        &["prop-log-reward-gap-axiom-bound"],
        scope,
        alog.abs(),
        a_regularized,
    );
    lower(
        &mut checks,
        "sufficient_log_fitness_gap",
        &["thm-stability-condition-final-corrected"],
        scope,
        log_fitness_gap,
        delta_log,
    );
    let jensen_h = mean(&fitness[..2]).ln() - log_mean(&fitness[..2]);
    check(
        &mut checks,
        "fitness_jensen_defect",
        &["thm-stability-condition-final-corrected"],
        scope,
        jensen_h,
        h_fitness_var / (2. * v_star.powi(2)),
    );
    lower(
        &mut checks,
        "sufficient_arithmetic_fitness_gap",
        &["thm-stability-condition-final-corrected"],
        scope,
        mean(&fitness[2..]) - mean(&fitness[..2]),
        v_star * delta.exp_m1(),
    );
    // Equal group multisets have positive total variance and no separation.
    let equal = [0.5, 1.5, 0.5, 1.5];
    identity(
        &mut checks,
        "equal_group_variance_negative_control",
        &["rem-group-separation-feasibility"],
        "Equal H/L multisets; strict within<total hypothesis fails and no positive mean-gap bound is emitted.",
        variance(&equal),
        0.5 * variance(&equal[..2]) + 0.5 * variance(&equal[2..]),
    );
    let beta_min_for_log = alpha * a_star / d_star;
    let beta_min_for_arithmetic = (alpha * a_star + h_fitness_var / (2. * v_star.powi(2))) / d_star;
    Ok(SelectionFixture {
        name: "group_and_log_fitness_separation".into(),
        scope: scope.into(),
        hypotheses: json!({"fixed_partition":{"H":[0,1],"L":[2,3]},"raw_diversity":raw_d,"raw_reward":raw_r,"sigma_min":sigma,"positive_factor_support":[a,b],"alpha":alpha,"beta":beta,"powered_fitness_support":[v_star,v_upper],"measurement_laws":"finite correlated joint laws, explicitly enumerated","ordered_diversity_pairing":true}),
        constants: BTreeMap::from([
            ("D_star".into(), d_star),
            ("A_star_naive".into(), a_naive),
            ("A_star_regularized".into(), a_regularized),
            ("K_r".into(), kr),
            ("delta_log".into(), delta_log),
            ("delta_arithmetic".into(), delta),
            ("beta_min_log_given_alpha".into(), beta_min_for_log),
            (
                "beta_min_arithmetic_given_fitness_support_and_variance".into(),
                beta_min_for_arithmetic,
            ),
            ("v_star".into(), v_star),
            ("v_upper".into(), v_upper),
            ("separation_event_lower".into(), event_lower),
        ]),
        observations: json!({"diversity_factors":dp,"reward_factors":rp,"fitness":fitness,"diversity_log_gap":dlog,"reward_log_gap":alog,"fitness_log_gap":log_fitness_gap,"sharp_log_bounds":sharp,"equal_group_control":{"total_variance":variance(&equal),"mean_gap":0.,"strict_within_hypothesis":false}}),
        checks,
    })
}

/// Exact conditional pressure for a supplied retained positive fitness vector
/// and a fresh nonself companion law. H is a supplied subset; its geometry is
/// not certified here. Error weights are supplied coupling contributions,
/// not an observable defined by storage row identity. No positive-variance lower bound
/// is emitted on ties.
#[allow(clippy::too_many_arguments)]
pub fn analyze_conditional_selection(
    fitness: &[f64],
    kernel: &[Vec<f64>],
    high: &[bool],
    common: &[bool],
    errors: &[f64],
    fitness_upper: f64,
    p_max: f64,
    epsilon_clone: f64,
) -> Result<SelectionFixture> {
    validate_values(fitness, 2)?;
    let n = fitness.len();
    if fitness.iter().any(|&v| v <= 0.)
        || kernel.len() != n
        || high.len() != n
        || common.len() != n
        || errors.len() != n
        || errors.iter().any(|&e| !e.is_finite() || e < 0.)
        || !fitness_upper.is_finite()
        || fitness_upper < fitness.iter().copied().fold(0., f64::max)
        || !p_max.is_finite()
        || p_max <= 0.
        || !epsilon_clone.is_finite()
        || epsilon_clone < 0.
    {
        return Err(error("invalid conditional selection inputs"));
    }
    for (i, row) in kernel.iter().enumerate() {
        if row.len() != n
            || row.iter().any(|&p| !p.is_finite() || p < 0.)
            || row[i] != 0.
            || (row.iter().sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err(error(
                "conditional companion rows must be normalized and nonself",
            ));
        }
    }
    let scope = "Retained realized fitness and supplied normalized nonself current companion law; independent uniform cloning threshold on an active cloning step; fixed entering errors of a supplied comparison coupling and subsets. Storage row IDs do not define a physical swarm observable.";
    let decision = CloneDecision {
        epsilon: epsilon_clone,
        saturation: p_max,
        every: 1,
        ..Default::default()
    };
    decision.validate()?;
    let mu = mean(fitness);
    let s2 = variance(fitness);
    let rv = range(fitness);
    let a = (n - 1) as f64
        * kernel
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |&(j, _)| i != j)
                    .map(|(_, &p)| p)
            })
            .fold(f64::INFINITY, f64::min);
    let unfit: Vec<bool> = fitness.iter().map(|&v| v <= mu).collect();
    let target: Vec<usize> = (0..n)
        .filter(|&i| common[i] && high[i] && unfit[i])
        .collect();
    let pressure: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| kernel[i][j] * decision.acceptance_probability(0, fitness[i], fitness[j]))
                .sum()
        })
        .collect();
    let positive: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| kernel[i][j] * (fitness[j] - fitness[i]).max(0.))
                .sum()
        })
        .collect();
    let av = rv.max(p_max * (fitness_upper + epsilon_clone));
    if !s2.is_finite() || !rv.is_finite() || !av.is_finite() {
        return Err(error(
            "conditional selection variance, range or denominator overflow",
        ));
    }
    let pu = if rv > 0. && a > 0. {
        a * s2 / (2. * rv * av)
    } else {
        0.
    };
    let mut checks = Vec::new();
    for i in 0..n {
        let uniform_mean =
            (0..n).filter(|&j| j != i).map(|j| fitness[j]).sum::<f64>() / (n - 1) as f64;
        identity(
            &mut checks,
            &format!("uniform_nonself_mean_{i}"),
            &["lem-mean-companion-fitness-gap"],
            "Auxiliary uniform nonself mean of this retained vector; not substitution for the supplied companion law.",
            uniform_mean - fitness[i],
            n as f64 / (n - 1) as f64 * (mu - fitness[i]),
        );
        if rv > 0. {
            lower(
                &mut checks,
                &format!("clipped_positive_signal_{i}"),
                &["lem-unfit-cloning-pressure"],
                scope,
                pressure[i],
                positive[i] / av,
            );
            if unfit[i] && a > 0. {
                lower(
                    &mut checks,
                    &format!("unfit_positive_signal_{i}"),
                    &["lem-mean-companion-fitness-gap"],
                    scope,
                    positive[i],
                    a * n as f64 / (n - 1) as f64 * s2 / (2. * rv),
                );
                lower(
                    &mut checks,
                    &format!("unfit_pressure_{i}"),
                    &["lem-unfit-cloning-pressure"],
                    scope,
                    pressure[i],
                    pu,
                );
                if target.contains(&i) {
                    lower(
                        &mut checks,
                        &format!("target_pressure_{i}"),
                        &["cor-cloning-pressure-target-set"],
                        scope,
                        pressure[i],
                        pu,
                    );
                }
            }
            let delta = rv / 4.;
            let favorable_mass = (0..n)
                .filter(|&j| fitness[j] - fitness[i] >= delta)
                .map(|j| kernel[i][j])
                .sum::<f64>();
            lower(
                &mut checks,
                &format!("favorable_set_pressure_{i}"),
                &["lem-unfit-cloning-pressure"],
                scope,
                pressure[i],
                favorable_mass * (delta / (p_max * (fitness_upper + epsilon_clone))).min(1.),
            );
        }
    }
    let unfit_count = unfit.iter().filter(|&&u| u).count();
    if rv > 0. {
        lower(
            &mut checks,
            "unfit_fraction",
            &["lem-unfit-fraction-lower-bound"],
            scope,
            unfit_count as f64 / n as f64,
            s2 / (2. * rv.powi(2)),
        );
        lower(
            &mut checks,
            "fit_fraction",
            &["lem-unfit-fraction-lower-bound"],
            scope,
            (n - unfit_count) as f64 / n as f64,
            s2 / (2. * rv.powi(2)),
        );
    }
    let h: Vec<f64> = (0..n).filter(|&i| high[i]).map(|i| fitness[i]).collect();
    let l: Vec<f64> = (0..n).filter(|&i| !high[i]).map(|i| fitness[i]).collect();
    let mut signed_gap = None;
    if !h.is_empty() && !l.is_empty() {
        let fh = h.len() as f64 / n as f64;
        let fl = 1. - fh;
        let gap = mean(&l) - mean(&h);
        signed_gap = Some(gap);
        lower(
            &mut checks,
            "fitness_partition_variance",
            &["lem-mean-companion-fitness-gap"],
            scope,
            s2,
            fh * fl * gap.powi(2),
        );
        if gap > 0. && rv > 0. {
            let overlap = (0..n).filter(|&i| high[i] && unfit[i]).count();
            lower(
                &mut checks,
                "unfit_high_overlap",
                &["thm-unfit-high-error-overlap-fraction"],
                scope,
                overlap as f64 / n as f64,
                fh * fl * gap / rv,
            );
            let not_common = common.iter().filter(|&&c| !c).count();
            lower(
                &mut checks,
                "common_label_target_count",
                &[
                    "thm-unfit-high-error-overlap-fraction",
                    "def-critical-target-set",
                ],
                scope,
                target.len() as f64,
                (overlap as f64 - not_common as f64).max(0.),
            );
        }
    }
    let pressure_error = (0..n)
        .filter(|&i| common[i])
        .map(|i| pressure[i] * errors[i])
        .sum::<f64>()
        / n as f64;
    let target_error = target.iter().map(|&i| errors[i]).sum::<f64>() / n as f64;
    if pu > 0. {
        lower(
            &mut checks,
            "conditional_target_error_pressure",
            &["cor-cloning-pressure-target-set"],
            scope,
            pressure_error,
            pu * target_error,
        );
    }
    Ok(SelectionFixture {
        name: "conditional_retained_selection".into(),
        scope: scope.into(),
        hypotheses: json!({"fitness":fitness,"companion_kernel":kernel,"high_subset":high,"common_labels":common,"entering_error_weights":errors,"fitness_upper":fitness_upper,"p_max":p_max,"epsilon_clone":epsilon_clone,"high_subset_geometry":"supplied, not certified by this function","positive_variance":s2>0.,"all_off_diagonal_probabilities_positive":a>0.}),
        constants: BTreeMap::from([
            ("a".into(), a),
            ("fitness_variance".into(), s2),
            ("fitness_range".into(), rv),
            ("A_V".into(), av),
            ("p_u".into(), pu),
        ]),
        observations: json!({"mean":mu,"unfit":unfit,"target":target,"acceptance_probabilities":pressure,"positive_part_signals":positive,"signed_low_minus_high_mean_gap":signed_gap,"target_error":target_error,"weighted_pressure_error":pressure_error}),
        checks,
    })
}

fn validate_averaged_cluster_pressure() -> Result<SelectionFixture> {
    let scope = "All 81 independent canonical squashed Gaussian measurement patterns for four entering alive rows, actual retained global standardization and fitness pipeline, independent Gaussian current cloning donor and clipped uniform threshold. Small clusters use exact nonself mass. Two empirical swarms are compared by exact one-dimensional optimal transport, independently of storage row IDs.";
    // A singleton and a three-row cloud retain small clusters and exact ties.
    // Their reflected empirical measure differs because their masses differ.
    let x = [0., 1., 1., 1.];
    let obs = observations(&x, &[0.; 4])?;
    let d = distance_matrix(
        &obs,
        &Distance::SquashedPhaseSpace {
            positions: "positions".into(),
            velocities: "velocities".into(),
            position_radius: 2.,
            velocity_radius: 2.,
            lambda: 1.,
        },
    )?;
    let groups = complete_linkage(&d, 0.12)?;
    let kernel = gaussian_rows(&d, 2.)?;
    let diameter = d.iter().flatten().copied().fold(0., f64::max);
    let kappa = (-diameter.powi(2) / 8.).exp();
    let h: f64 = 0.2;
    let vz = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
        .map(|(i, j)| d[i][j].powi(2))
        .sum::<f64>()
        / 16.;
    if h.powi(2) >= vz {
        return Err(error("cluster pressure fixture violates AP1 threshold"));
    }
    let rho_h = (vz - h.powi(2)) / (diameter.powi(2) - h.powi(2));
    let distance_floor: f64 = 0.001;
    let eligible: Vec<f64> = (0..4)
        .flat_map(|i| (0..4).filter(move |&j| j != i).map(move |j| (i, j)))
        .map(|(i, j)| (d[i][j].powi(2) + distance_floor.powi(2)).sqrt())
        .collect();
    let y_min = eligible.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = eligible.iter().copied().fold(0., f64::max);
    let dm = y_max - y_min;
    let eps_s: f64 = 0.1;
    let z_m = dm / eps_s;
    let s_star = (dm.powi(2) / 4. + eps_s.powi(2)).sqrt();
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let f_min = map.map(-z_m)?;
    let f_max = map.map(z_m)?;
    let m_f = logistic_derivative(z_m);
    let pipeline = FitnessPipeline {
        reward_map: map.clone(),
        diversity_map: map,
        reward_standardizer: Standardizer::Global { sigma_min: eps_s },
        diversity_standardizer: Standardizer::Global { sigma_min: eps_s },
        ..Default::default()
    };
    let reward_z = [0.; 4]; // Exact standardized constant rewards; A_i=1.1.
    let a_min = 1.1;
    let f_upper: f64 = 4.41;
    let decision = CloneDecision::default();
    let mut cluster_values = Vec::new();
    let mut row_pi = vec![0.; 4];
    for g in &groups {
        let mut r_g: f64 = 0.;
        for &i in g {
            for &j in g {
                r_g = r_g.max(d[i][j]);
            }
        }
        let ell = (r_g.powi(2) + distance_floor.powi(2)).sqrt();
        let far = (h.powi(2) + distance_floor.powi(2)).sqrt();
        let delta = far - ell;
        let rho_g = (g.len() - 1) as f64 / 3.;
        let gamma = a_min * m_f * delta / s_star;
        let a_g = (gamma.max(0.) / (f_upper + decision.epsilon)).min(1.);
        let pi = if g.len() >= 2 {
            kappa.powi(3) * rho_g.powi(2) * rho_h * a_g
        } else {
            0.
        };
        for &i in g {
            row_pi[i] = pi;
        }
        cluster_values.push(json!({"members":g,"r_G":r_g,"rho_G":rho_g,"ell_G":ell,"delta_G":delta,"A_G_min":a_min,"reward_oscillation":0.,"gamma_G":gamma,"acceptance_lower":a_g,"pi_G":pi,"statistically_valid":false,"eligible_for_AP5":g.len()>=2}));
    }
    let mut pbar = vec![0.; 4];
    let mut prob_sum = 0.;
    let mut zero_signal_probability = 0.;
    let mut near_tie_probability = 0.;
    let mut expected_variance = 0.;
    let mut measurement_law = Vec::with_capacity(81);
    let choices: Vec<Vec<usize>> = (0..4)
        .map(|i| (0..4).filter(|&j| j != i).collect())
        .collect();
    for pattern in 0_usize..81 {
        let mut digit = pattern;
        let mut measurement = vec![0.; 4];
        let mut probability = 1.;
        for i in 0..4 {
            let donor = choices[i][digit % 3];
            digit /= 3;
            probability *= kernel[i][donor];
            measurement[i] = (d[i][donor].powi(2) + distance_floor.powi(2)).sqrt();
        }
        prob_sum += probability;
        let (z, _) = pipeline
            .diversity_standardizer
            .apply(&measurement, &[true; 4], &obs)?;
        let fitness = pipeline.combine(&reward_z, &z, &[true; 4])?;
        expected_variance += probability * variance(&fitness);
        if range(&fitness) == 0. {
            zero_signal_probability += probability;
        }
        if range(&fitness) <= 1e-14 {
            near_tie_probability += probability;
        }
        for i in 0..4 {
            for j in 0..4 {
                pbar[i] += probability
                    * kernel[i][j]
                    * decision.acceptance_probability(0, fitness[i], fitness[j]);
            }
        }
        measurement_law.push((probability, measurement));
    }
    let mut checks = Vec::new();
    identity(
        &mut checks,
        "measurement_joint_mass",
        &["lem-keystone-geometric-measurement-events"],
        scope,
        prob_sum,
        1.,
    );
    let marginal_measurements: Vec<f64> = (0..4)
        .map(|i| measurement_law.iter().map(|(p, y)| p * y[i]).sum())
        .collect();
    let expected_raw_variance = measurement_law
        .iter()
        .map(|(p, y)| p * variance(y))
        .sum::<f64>();
    let centered_noise_variance = measurement_law
        .iter()
        .map(|(p, y)| {
            p * variance(
                &y.iter()
                    .zip(&marginal_measurements)
                    .map(|(y, m)| y - m)
                    .collect::<Vec<_>>(),
            )
        })
        .sum::<f64>();
    let marginal_gap = marginal_measurements[0] - mean(&marginal_measurements[1..]);
    if marginal_gap <= 0. {
        return Err(error(
            "actual marginal measurement fixture needs a positive fixed-partition gap",
        ));
    }
    identity(
        &mut checks,
        "actual_measurement_variance_identity",
        &["thm-geometry-guarantees-variance"],
        "The same full 81-pattern actual pairing law; centering-projection fluctuations evaluated before rescaling.",
        expected_raw_variance,
        variance(&marginal_measurements) + centered_noise_variance,
    );
    lower(
        &mut checks,
        "actual_marginal_group_gap_variance",
        &["thm-geometry-guarantees-variance"],
        "Deterministic position-defined partition: H is the three-row cluster at x=1 (fraction .75), L is the singleton at x=0 (fraction .25), fixed before actual Gaussian measurements; no identification with geometric high-error groups.",
        expected_raw_variance,
        0.75 * 0.25 * marginal_gap.powi(2),
    );
    for j in 0..4 {
        let far_count = (0..4).filter(|&l| l != j && d[j][l] >= h).count() as f64 / 3.;
        let far_mass = (0..4)
            .filter(|&l| l != j && d[j][l] >= h)
            .map(|l| kernel[j][l])
            .sum::<f64>();
        lower(
            &mut checks,
            &format!("far_nonself_fraction_{j}"),
            &["lem-keystone-geometric-measurement-events"],
            scope,
            far_count,
            rho_h,
        );
        lower(
            &mut checks,
            &format!("weighted_far_mass_{j}"),
            &["lem-keystone-geometric-measurement-events"],
            scope,
            far_mass,
            kappa * rho_h,
        );
    }
    for (g_index, g) in groups.iter().enumerate() {
        let rho_g = (g.len() - 1) as f64 / 3.;
        for &i in g.iter().filter(|_| g.len() >= 2) {
            for &j in g {
                if i != j {
                    let near_mass = g
                        .iter()
                        .filter(|&&l| l != i)
                        .map(|&l| kernel[i][l])
                        .sum::<f64>();
                    let far_mass = (0..4)
                        .filter(|&l| l != j && d[j][l] >= h)
                        .map(|l| kernel[j][l])
                        .sum::<f64>();
                    lower(
                        &mut checks,
                        &format!("cluster_{g_index}_measurement_event_{i}_{j}"),
                        &["lem-keystone-geometric-measurement-events"],
                        scope,
                        near_mass * far_mass,
                        kappa.powi(2) * rho_g * rho_h,
                    );
                }
            }
        }
        for &i in g.iter().filter(|_| g.len() >= 2) {
            lower(
                &mut checks,
                &format!("averaged_small_cluster_pressure_{i}"),
                &["thm-keystone-averaged-cluster-pressure"],
                scope,
                pbar[i],
                row_pi[i],
            );
        }
    }
    // Reflect and sort the second cloud. Monotone matching is the exact
    // one-dimensional centered W2 optimizer for uniform alive weights. The
    // reflection preserves the canonical metric and permutes its pressure law.
    let reflected: Vec<f64> = x.iter().map(|x| -x).collect();
    let transport = optimal_centered_transport_1d(&x, &reflected)?;
    let optimal_partner = &transport.right_indices;
    let discrepancy = &transport.centered_discrepancies;
    let errors: Vec<f64> = discrepancy.iter().map(|e| e * e).collect();
    let pi_sum: Vec<f64> = (0..4)
        .map(|i| row_pi[i] + row_pi[optimal_partner[i]])
        .collect();
    let pressure_error = (0..4)
        .map(|i| (pbar[i] + pbar[optimal_partner[i]]) * errors[i])
        .sum::<f64>()
        / 4.;
    let certified = (0..4).map(|i| pi_sum[i] * errors[i]).sum::<f64>() / 4.;
    lower(
        &mut checks,
        "averaged_common_error_pressure",
        &["thm-keystone-averaged-error-capture"],
        scope,
        pressure_error,
        certified,
    );
    let mut refinement = Vec::new();
    for first in &groups {
        for second in &groups {
            let block: Vec<usize> = first
                .iter()
                .copied()
                .filter(|&i| second.contains(&optimal_partner[i]))
                .collect();
            if !block.is_empty() {
                refinement.push(block);
            }
        }
    }
    let refined: f64 = refinement
        .iter()
        .map(|g| {
            let e: Vec<f64> = g.iter().map(|&i| discrepancy[i]).collect();
            let pi = g.iter().map(|&i| pi_sum[i]).fold(f64::INFINITY, f64::min);
            pi * g.len() as f64 / 4. * (mean(&e).powi(2) + variance(&e))
        })
        .sum();
    identity(
        &mut checks,
        "refined_block_error_identity",
        &["thm-keystone-averaged-error-capture"],
        scope,
        refined,
        certified,
    );
    let p_star = pi_sum.iter().copied().fold(f64::INFINITY, f64::min);
    let total_error = mean(&errors);
    lower(
        &mut checks,
        "covered_error_coefficient",
        &["thm-keystone-averaged-error-capture"],
        scope,
        certified,
        p_star * total_error,
    );
    // Artificially leave one original cluster uncertified. Its error is charged
    // exactly, rather than retaining the positive coefficient on those rows.
    let uncovered = groups
        .last()
        .ok_or_else(|| error("missing pressure cluster"))?;
    let uncovered_error = uncovered.iter().map(|&i| errors[i]).sum::<f64>() / 4.;
    let retained_certificate = (0..4)
        .filter(|i| !uncovered.contains(i))
        .map(|i| pi_sum[i] * errors[i])
        .sum::<f64>()
        / 4.;
    lower(
        &mut checks,
        "explicit_uncovered_error",
        &["thm-keystone-averaged-error-capture"],
        "Same original clusters; rows in one chosen cluster deliberately assigned zero combined certificate, while their actual activity stays in the enumerated law. Their exact optimal-coupling error is retained as uncovered error.",
        retained_certificate,
        p_star * (total_error - uncovered_error),
    );
    check(
        &mut checks,
        "uncovered_error_diameter_bound",
        &["thm-keystone-averaged-error-capture"],
        scope,
        uncovered_error,
        4. * range(&x).powi(2) * uncovered.len() as f64 / 4.,
    );
    Ok(SelectionFixture {
        name: "exact_measurement_averaged_small_clusters".into(),
        scope: scope.into(),
        hypotheses: json!({"positions":x,"velocities":[0.,0.,0.,0.],"independent_measurement_draws":true,"independent_cloning_draw":true,"actual_kernel":"Canonical Gaussian width 2; squashed phase-space position/velocity radii 2, lambda 1, nonself","sigma_min":eps_s,"reward":"constant, globally standardized to zero","floor":0.1,"alpha":1.,"beta":1.,"distance_floor":distance_floor,"measurement_patterns":81,"statistically_valid_clusters":false,"second_swarm":"reflected empirical positions, zero velocities; canonical donor/measurement laws reflected and then reordered by exact optimal transport","comparison":"Permutation-invariant centered W2: sorted one-dimensional uniform transport; storage row identity is not used as physical distance"}),
        constants: BTreeMap::from([
            ("rho_h".into(), rho_h),
            ("kappa_D".into(), kappa),
            ("kappa_C".into(), kappa),
            ("D_m".into(), dm),
            ("Z_m".into(), z_m),
            ("s_star".into(), s_star),
            ("f_min".into(), f_min),
            ("f_max".into(), f_max),
            ("m_f".into(), m_f),
            ("F_upper".into(), f_upper),
            ("p_star".into(), p_star),
            ("uncovered_error".into(), uncovered_error),
            ("marginal_measurement_gap".into(), marginal_gap),
            ("optimal_positional_transport_cost".into(), total_error),
        ]),
        observations: json!({"clusters":cluster_values,"common_refinement_under_optimal_coupling":refinement,"optimal_transport":transport,"measurement_probability_sum":prob_sum,"equal_fitness_outcome_probability":zero_signal_probability,"near_tie_fitness_outcome_probability_1e_14":near_tie_probability,"expected_fitness_variance":expected_variance,"marginal_measurements":marginal_measurements,"expected_raw_measurement_variance":expected_raw_variance,"centered_measurement_noise_variance":centered_noise_variance,"averaged_acceptance_probabilities":pbar,"row_pressure_certificates":row_pi,"two_swarm_coupled_pressure_certificates":pi_sum,"optimal_coupling_error_weights":errors,"weighted_pressure_error":pressure_error,"certified_weighted_pressure_error":certified}),
        checks,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftAssemblyInputs {
    /// Same four nonnegative observables on every finite state: (V_W,X,Y,W_b).
    pub observables: Vec<[f64; 4]>,
    /// Row stochastic/substochastic backward transition matrices.
    pub cloning: Vec<Vec<f64>>,
    pub kinetic: Vec<Vec<f64>>,
    /// kappa_x, kappa_b; both strictly positive and at most one.
    pub cloning_rates: [f64; 2],
    /// kappa_W, kappa_v; both strictly positive and at most one.
    pub kinetic_rates: [f64; 2],
    pub cloning_offsets: [f64; 4],
    pub kinetic_offsets: [f64; 4],
    pub variance_weight: f64,
    pub boundary_weight: f64,
    pub steps: usize,
    /// If supplied, independently established clone velocity dissipation
    /// lambda_v*(1-alpha^2)*Ebar_C. It must fit inside the computed defect.
    pub clone_velocity_dissipation: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftAssemblyReport {
    pub kappa_star: f64,
    pub c_star: f64,
    pub weights: [f64; 4],
    pub composed_transition: Vec<Vec<f64>>,
    pub cloning_defects: Vec<[f64; 4]>,
    pub kinetic_defects: Vec<[f64; 4]>,
    pub full_step_moments: Vec<f64>,
    pub full_step_upper: Vec<f64>,
    pub iterated_unnormalized_moments: Vec<f64>,
    pub iterated_upper: Vec<f64>,
    pub surviving_mass: Vec<f64>,
    pub conditioned_moments: Vec<Option<f64>>,
    pub conditioned_upper_from_survival: Vec<Option<f64>>,
    pub checks: Vec<BoundCheck>,
    pub scope: String,
}

fn apply_matrix(matrix: &[Vec<f64>], values: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(values).map(|(p, x)| p * x).sum())
        .collect()
}
fn compose(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = left.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| left[i][k] * right[k][j]).sum())
                .collect()
        })
        .collect()
}
fn finite_transition(matrix: &[Vec<f64>], n: usize) -> Result<()> {
    if matrix.len() != n
        || matrix.iter().any(|row| {
            row.len() != n
                || row.iter().any(|&p| !p.is_finite() || p < 0.)
                || row.iter().sum::<f64>() > 1. + 1e-12
        })
    {
        return Err(error(
            "finite transition must be a nonnegative square sub-Markov matrix",
        ));
    }
    Ok(())
}

/// Checks the common statewise component hypotheses on a finite state space and
/// composes the actual matrices in physical clone-then-kinetic order PC*PK.
/// Rejected component hypotheses remain failed checks in the report. Killed
/// moments use cemetery observables zero and are not conditioned on survival.
pub fn assemble_diagonal_drift(input: &DriftAssemblyInputs) -> Result<DriftAssemblyReport> {
    let n = input.observables.len();
    if n == 0
        || n > 256
        || input.steps > 10000
        || input
            .observables
            .iter()
            .flatten()
            .any(|&x| !x.is_finite() || x < 0.)
        || input
            .cloning_rates
            .iter()
            .chain(&input.kinetic_rates)
            .any(|&r| !r.is_finite() || r <= 0. || r > 1.)
        || input
            .cloning_offsets
            .iter()
            .chain(&input.kinetic_offsets)
            .any(|&x| !x.is_finite() || x < 0.)
        || !input.variance_weight.is_finite()
        || input.variance_weight <= 0.
        || !input.boundary_weight.is_finite()
        || input.boundary_weight <= 0.
    {
        return Err(error(
            "invalid diagonal drift inputs, weights, rates or offsets",
        ));
    }
    finite_transition(&input.cloning, n)?;
    finite_transition(&input.kinetic, n)?;
    if input
        .clone_velocity_dissipation
        .as_ref()
        .is_some_and(|d| d.len() != n || d.iter().any(|&x| !x.is_finite() || x < 0.))
    {
        return Err(error("invalid supplied clone velocity dissipation"));
    }
    let scope = "Exact finite transition operators and common nonnegative observables; rates and offsets are supplied hypotheses, checked statewise. Cemetery observables are zero. No actual engine contraction is inferred.";
    let ac = [
        1.,
        1. - input.cloning_rates[0],
        1.,
        1. - input.cloning_rates[1],
    ];
    let ak = [
        1. - input.kinetic_rates[0],
        1.,
        1. - input.kinetic_rates[1],
        1.,
    ];
    let weights = [
        1.,
        input.variance_weight,
        input.variance_weight,
        input.boundary_weight,
    ];
    let rates = [
        input.kinetic_rates[0],
        input.cloning_rates[0],
        input.kinetic_rates[1],
        input.cloning_rates[1],
    ];
    let kappa_star = rates.iter().copied().fold(f64::INFINITY, f64::min);
    let c_star = (0..4)
        .map(|a| weights[a] * (ak[a] * input.cloning_offsets[a] + input.kinetic_offsets[a]))
        .sum::<f64>();
    if !c_star.is_finite() {
        return Err(error("weighted drift offset overflow"));
    }
    let mut dc = vec![[0.; 4]; n];
    let mut dk = vec![[0.; 4]; n];
    let mut checks = Vec::new();
    for a in 0..4 {
        let f: Vec<f64> = input.observables.iter().map(|f| f[a]).collect();
        let pc = apply_matrix(&input.cloning, &f);
        let pk = apply_matrix(&input.kinetic, &f);
        for s in 0..n {
            dc[s][a] = ac[a] * f[s] + input.cloning_offsets[a] - pc[s];
            dk[s][a] = ak[a] * f[s] + input.kinetic_offsets[a] - pk[s];
            if !dc[s][a].is_finite() || !dk[s][a].is_finite() {
                return Err(error("finite component drift arithmetic overflow"));
            }
            check(
                &mut checks,
                &format!("clone_component_state_{s}_{a}"),
                &["thm-synergistic-foster-lyapunov-preview"],
                scope,
                pc[s],
                ac[a] * f[s] + input.cloning_offsets[a],
            );
            check(
                &mut checks,
                &format!("kinetic_component_state_{s}_{a}"),
                &["thm-synergistic-foster-lyapunov-preview"],
                scope,
                pk[s],
                ak[a] * f[s] + input.kinetic_offsets[a],
            );
        }
    }
    let q = compose(&input.cloning, &input.kinetic);
    for (i, row) in q.iter().enumerate() {
        for (j, &probability) in row.iter().enumerate() {
            if probability == 0.
                && (0..n).any(|k| input.cloning[i][k] > 0. && input.kinetic[k][j] > 0.)
            {
                return Err(GasError::Numerical(
                    "positive finite transition probability underflowed during composition".into(),
                ));
            }
        }
    }
    let phi: Vec<f64> = input
        .observables
        .iter()
        .map(|f| (0..4).map(|a| weights[a] * f[a]).sum())
        .collect();
    let full = apply_matrix(&q, &phi);
    let upper: Vec<f64> = phi
        .iter()
        .map(|&x| (1. - kappa_star) * x + c_star)
        .collect();
    if phi
        .iter()
        .chain(&full)
        .chain(&upper)
        .any(|x| !x.is_finite())
    {
        return Err(error("finite weighted observable or drift bound overflow"));
    }
    let clone_mass: Vec<f64> = input.cloning.iter().map(|row| row.iter().sum()).collect();
    for a in 0..4 {
        let f: Vec<f64> = input.observables.iter().map(|f| f[a]).collect();
        let q_f = apply_matrix(&q, &f);
        let dk_a: Vec<f64> = dk.iter().map(|d| d[a]).collect();
        let pc_dk = apply_matrix(&input.cloning, &dk_a);
        for s in 0..n {
            let rhs =
                ak[a] * ac[a] * f[s] + ak[a] * input.cloning_offsets[a] + input.kinetic_offsets[a]
                    - ak[a] * dc[s][a]
                    - pc_dk[s]
                    - (1. - clone_mass[s]) * input.kinetic_offsets[a];
            identity(
                &mut checks,
                &format!("composed_component_defect_state_{s}_{a}"),
                &["cor-cloning-weighted-assembly-input"],
                "Exact AC7, including -(1-PC1)b_K when cloning itself is sub-Markov.",
                q_f[s],
                rhs,
            );
        }
    }
    let pc_phi = apply_matrix(&input.cloning, &phi);
    let pk_phi = apply_matrix(&input.kinetic, &phi);
    let kinetic_increment: Vec<f64> = pk_phi.iter().zip(&phi).map(|(a, b)| a - b).collect();
    let transported_increment = apply_matrix(&input.cloning, &kinetic_increment);
    for s in 0..n {
        let weighted_rate = (0..4)
            .map(|a| weights[a] * rates[a] * input.observables[s][a])
            .sum::<f64>();
        lower(
            &mut checks,
            &format!("weighted_minimum_rate_state_{s}"),
            &["lemma-weighted-min-coefficient"],
            scope,
            weighted_rate,
            kappa_star * phi[s],
        );
        identity(
            &mut checks,
            &format!("backward_drift_identity_state_{s}"),
            &["cor-cloning-weighted-assembly-input"],
            "Exact AC8 retains clone-then-kinetic order and any killing constant term.",
            full[s] - phi[s],
            pc_phi[s] - phi[s] + transported_increment[s],
        );
        check(
            &mut checks,
            &format!("full_weighted_drift_state_{s}"),
            &["thm-synergistic-foster-lyapunov-preview"],
            scope,
            full[s],
            upper[s],
        );
        if let Some(dissipation) = &input.clone_velocity_dissipation {
            check(
                &mut checks,
                &format!("velocity_defect_hypothesis_state_{s}"),
                &["cor-cloning-weighted-assembly-input"],
                "Independently supplied lambda_v*(1-alpha^2)*Ebar_C; must be contained in computed clone Y defect.",
                dissipation[s],
                dc[s][2],
            );
            let sharp = (0..4)
                .map(|a| {
                    weights[a]
                        * (ak[a] * ac[a] * input.observables[s][a]
                            + ak[a] * input.cloning_offsets[a]
                            + input.kinetic_offsets[a])
                })
                .sum::<f64>()
                - input.variance_weight * ak[2] * dissipation[s];
            check(
                &mut checks,
                &format!("retained_velocity_dissipation_state_{s}"),
                &["cor-cloning-weighted-assembly-input"],
                scope,
                full[s],
                sharp,
            );
        }
    }
    let mut iterated = phi.clone();
    let mut mass = vec![1.; n];
    for _ in 0..input.steps {
        iterated = apply_matrix(&q, &iterated);
        let next_mass = apply_matrix(&q, &mass);
        for s in 0..n {
            if next_mass[s] == 0. && (0..n).any(|j| q[s][j] > 0. && mass[j] > 0.) {
                return Err(GasError::Numerical("positive surviving mass underflowed; zero must not be reported as terminal extinction".into()));
            }
        }
        mass = next_mass;
    }
    let factor = (1. - kappa_star).powi(input.steps as i32);
    let iterated_upper: Vec<f64> = phi
        .iter()
        .map(|&x| factor * x + c_star / kappa_star * (1. - factor))
        .collect();
    let conditioned: Vec<Option<f64>> = (0..n)
        .map(|s| {
            if mass[s] > 0. {
                Some(iterated[s] / mass[s])
            } else {
                None
            }
        })
        .collect();
    let conditioned_upper: Vec<Option<f64>> = (0..n)
        .map(|s| {
            if mass[s] > 0. {
                Some(iterated_upper[s] / mass[s])
            } else {
                None
            }
        })
        .collect();
    if iterated_upper
        .iter()
        .chain(conditioned.iter().flatten())
        .chain(conditioned_upper.iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err(error(
            "iterated or survival-conditioned finite bound overflow",
        ));
    }
    for s in 0..n {
        check(
            &mut checks,
            &format!("iterated_surviving_moment_state_{s}"),
            &["thm-synergistic-foster-lyapunov-preview"],
            "Q^n Phi is the unnormalized surviving moment for a killed finite operator; no conditioning inserted.",
            iterated[s],
            iterated_upper[s],
        );
        if let (Some(observed), Some(bound)) = (conditioned[s], conditioned_upper[s]) {
            check(
                &mut checks,
                &format!("conditioned_moment_with_survival_state_{s}"),
                &["thm-synergistic-foster-lyapunov-preview"],
                "Conditioned bound divides by independently computed exact finite survival Q^n1; no unconditioned rate claimed for this ratio.",
                observed,
                bound,
            );
        }
    }
    Ok(DriftAssemblyReport {
        kappa_star,
        c_star,
        weights,
        composed_transition: q,
        cloning_defects: dc,
        kinetic_defects: dk,
        full_step_moments: full,
        full_step_upper: upper,
        iterated_unnormalized_moments: iterated,
        iterated_upper,
        surviving_mass: mass,
        conditioned_moments: conditioned,
        conditioned_upper_from_survival: conditioned_upper,
        checks,
        scope: scope.into(),
    })
}

/// AC6 offsets with supplied inter-swarm and boundary constants. The uniform
/// revival velocity offset is retained; no all-alive invariance is assumed.
pub fn cloning_assembly_offsets(
    dimensions: usize,
    position_diameter: f64,
    position_jitter: f64,
    lambda_v: f64,
    velocity_cap: f64,
    inter_swarm_offset: f64,
    boundary_offset: f64,
) -> Result<[f64; 4]> {
    if dimensions == 0
        || dimensions > 256
        || [
            position_diameter,
            position_jitter,
            lambda_v,
            velocity_cap,
            inter_swarm_offset,
            boundary_offset,
        ]
        .iter()
        .any(|&x| !x.is_finite() || x < 0.)
        || lambda_v == 0.
    {
        return Err(error("invalid supplied clone assembly constants"));
    }
    let offsets = [
        inter_swarm_offset,
        position_diameter.powi(2) + 2. * dimensions as f64 * position_jitter.powi(2),
        8. * lambda_v * velocity_cap.powi(2),
        boundary_offset,
    ];
    if offsets.iter().any(|x| !x.is_finite()) {
        return Err(error("clone assembly offset overflow"));
    }
    Ok(offsets)
}

fn validate_drift_fixture(killed: bool) -> Result<SelectionFixture> {
    let cloning = vec![vec![1., 0., 0.], vec![0.4, 0.6, 0.], vec![0.3, 0.2, 0.5]];
    let mut kinetic = vec![vec![0.9, 0.1, 0.], vec![0.2, 0.8, 0.], vec![0.1, 0.5, 0.4]];
    if killed {
        for (row, survival) in kinetic.iter_mut().zip([0.95, 0.85, 0.6]) {
            for p in row {
                *p *= survival;
            }
        }
    }
    let input = DriftAssemblyInputs {
        observables: vec![
            [0.1, 0.005, 0.002, 0.02],
            [0.5, 0.01, 0.006, 0.2],
            [1.2, 0.02, 0.015, 0.4],
        ],
        cloning,
        kinetic,
        cloning_rates: [0.3, 0.2],
        kinetic_rates: [0.4, 0.5],
        cloning_offsets: cloning_assembly_offsets(1, 0.2, 0.01, 0.1, 0.2, 0.2, 0.1)?,
        kinetic_offsets: [0.15, 0.003, 0.005, 0.04],
        variance_weight: 1.,
        boundary_weight: 1.,
        steps: 16,
        clone_velocity_dissipation: Some(vec![0.001; 3]),
    };
    let report = assemble_diagonal_drift(&input)?;
    let explicit = (1. - input.kinetic_rates[0]) * input.cloning_offsets[0]
        + input.kinetic_offsets[0]
        + input.variance_weight
            * (input.cloning_offsets[1]
                + input.kinetic_offsets[1]
                + (1. - input.kinetic_rates[1]) * input.cloning_offsets[2]
                + input.kinetic_offsets[2])
        + input.boundary_weight * (input.cloning_offsets[3] + input.kinetic_offsets[3]);
    let mut checks = report.checks.clone();
    identity(
        &mut checks,
        "explicit_offset_stage_order",
        &["cor-cloning-weighted-assembly-input"],
        "AC9 uses A_K b_C+b_K, so the clone W/Y offsets are multiplied by their kinetic retention coefficients.",
        report.c_star,
        explicit,
    );
    lower(
        &mut checks,
        "positive_unit_weight_coupling",
        &["prop-coupling-constant-existence"],
        "Actual common finite-state component hypotheses checked in this fixture; c_V=c_B=1.",
        report.kappa_star,
        0.2,
    );
    Ok(SelectionFixture {
        name: if killed {
            "finite_killed_weighted_composition".into()
        } else {
            "finite_conservative_weighted_composition".into()
        },
        scope: report.scope.clone(),
        hypotheses: json!({"inputs":input,"operator_scope":"auxiliary finite operators for composition algebra, not a calibrated gas chain","kinetic_terminal_killing":killed,"clone_mandatory_revival_analogue":"PC rows sum to one","cemetery_observables":0.}),
        constants: BTreeMap::from([
            ("kappa_star".into(), report.kappa_star),
            ("C_star".into(), report.c_star),
            ("C_x".into(), input.cloning_offsets[1]),
            ("C_v_uniform_revival".into(), input.cloning_offsets[2]),
        ]),
        observations: serde_json::to_value(&report).map_err(|e| error(&e.to_string()))?,
        checks,
    })
}

/// Deterministic finite diagnostics plus complete enumeration of the small
/// measurement fixture. The seed perturbs positions inside their declared
/// cluster margins. All source coverage comes from explicit checks, not metadata.
pub fn validate_selection(seed: u64) -> Result<SelectionReport> {
    let fixtures = vec![
        validate_geometry(seed)?,
        analyze_fixed_rescaling(&[-0.8, -0.2, 0.1, 0.7], 0.1, 0.1)?,
        validate_separation_and_log_fitness()?,
        validate_averaged_cluster_pressure()?,
        validate_drift_fixture(false)?,
        validate_drift_fixture(true)?,
    ];
    let mut constants = BTreeMap::new();
    let mut checks = Vec::new();
    for fixture in &fixtures {
        for (key, value) in &fixture.constants {
            constants.insert(format!("{}.{key}", fixture.name), *value);
        }
        for check in &fixture.checks {
            let mut item = check.clone();
            item.id = format!("{}.{}", fixture.name, item.id);
            checks.push(item);
        }
    }
    Ok(SelectionReport{seed,constants,fixtures,checks,scope_notes:vec![
        "Finite cloud and support constants are evaluated on explicitly supplied fixture inputs. They are not uniform analytic constants for arbitrary simulated landscapes or parameter profiles.".into(),
        "Conditional pressure uses actual clipped acceptance after retaining each sampled fitness vector. Independent measurement patterns are completely enumerated; shared normalization correlations and equal-fitness outcomes remain in that law.".into(),
        "Positive log gaps do not imply arithmetic fitness gaps without the displayed Jensen-variance condition. Equal group multisets and zero-variance retained fitness remain explicit negative controls.".into(),
        "Weighted drift rates are supplied common statewise hypotheses verified only for the auxiliary finite operators. The report does not discharge chapter 5 kinetic contraction for the actual gas engine.".into(),
        "Killed iteration reports unnormalized surviving moments. Conditioned bounds require the separately computed survival probability; zero survival has no conditional moment.".into(),
    ]})
}
