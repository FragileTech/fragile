//! Chapter 9 population-independent constants and exact finite-law interfaces.
//! Bounds remain conditional on the chapter's canonical operator and hypotheses.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};

fn bad(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.
}
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
fn log_nonnegative(x: f64) -> f64 {
    if x == 0. { f64::NEG_INFINITY } else { x.ln() }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantsInput {
    pub minimum_alive_fraction: f64,
    pub measurement_weight_lower: f64,
    pub clone_weight_lower: f64,
    pub separation_upper: f64,
    pub separation_scale_floor: f64,
    pub reward_map_floor: f64,
    pub reward_map_amplitude: f64,
    pub separation_map_floor: f64,
    pub separation_map_amplitude: f64,
    pub reward_exponent: f64,
    pub separation_exponent: f64,
    pub acceptance_epsilon: f64,
    pub acceptance_saturation: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantitativeConstants {
    pub input: ConstantsInput,
    pub c: f64,
    pub d_d: f64,
    pub l_q: f64,
    pub h_s: f64,
    pub l_0: f64,
    pub fitness_lower: f64,
    pub fitness_upper: f64,
    pub l_a: f64,
    pub b: f64,
    pub a_t: f64,
    pub l_t: f64,
    pub a: f64,
    pub n_0: f64,
    pub log_m1_c: f64,
    pub log_m2_c: f64,
    pub log_m3_c: f64,
    pub log_a_d: f64,
    pub log_a_phi_unit: f64,
    pub log_b_star: f64,
    pub log_c_update: f64,
    pub log_k_ordered: f64,
    pub log_a_ordered_unit: f64,
    pub log_b_ordered: f64,
    pub scope: String,
}
/// log M_p(C), integer p=1..=128: factorial series via Poisson/Touchard polynomials.
pub fn log_component_moment(c: f64, p: usize) -> Result<f64> {
    if !c.is_finite() || c < 0. {
        return Err(bad("C must be finite nonnegative"));
    }
    let result = match p {
        1 => 2. * c,
        2 => (1. + 2. * c).ln() + 4. * c,
        3 => (1. + 6. * c + 3. * c * c).ln() + 8. * c,
        4..=128 => {
            // sum [(k+1)^p-k^p] C^k/k! = exp(C) *
            // sum_{j=0}^{p-1} binom(p,j) T_j(C). Stirling recursion
            // evaluates the Touchard polynomial with positive coefficients.
            let mut stirling = vec![0.];
            let mut log_polynomial = 0.;
            let mut log_binomial = 0.;
            for j in 1..p {
                let mut next = vec![f64::NEG_INFINITY; j + 1];
                for k in 1..=j {
                    let previous = stirling.get(k).copied().unwrap_or(f64::NEG_INFINITY);
                    next[k] = log_add((k as f64).ln() + previous, stirling[k - 1]);
                }
                stirling = next;
                let log_c = log_nonnegative(c);
                let touchard = stirling
                    .iter()
                    .enumerate()
                    .skip(1)
                    .fold(f64::NEG_INFINITY, |a, (k, b)| {
                        log_add(a, *b + k as f64 * log_c)
                    });
                log_binomial += ((p - j + 1) as f64).ln() - (j as f64).ln();
                log_polynomial = log_add(log_polynomial, log_binomial + touchard);
            }
            2_f64.powi(p as i32) * c + log_polynomial
        }
        _ => return Err(bad("integer component moment order must be 1..=128")),
    };
    if !result.is_finite() {
        return Err(bad("log moment outside finite arithmetic range"));
    }
    Ok(result)
}
pub fn quantitative_constants(p: ConstantsInput) -> Result<QuantitativeConstants> {
    if ![
        p.minimum_alive_fraction,
        p.measurement_weight_lower,
        p.clone_weight_lower,
        p.separation_scale_floor,
        p.reward_map_floor,
        p.separation_map_floor,
        p.acceptance_epsilon,
        p.acceptance_saturation,
    ]
    .into_iter()
    .all(positive)
        || ![
            p.separation_upper,
            p.reward_map_amplitude,
            p.separation_map_amplitude,
            p.reward_exponent,
            p.separation_exponent,
        ]
        .into_iter()
        .all(|x| x.is_finite() && x >= 0.)
        || p.minimum_alive_fraction > 1.
        || p.measurement_weight_lower > 1.
        || p.clone_weight_lower > 1.
    {
        return Err(bad("invalid canonical fixed-parameter constants"));
    }
    let m = p.minimum_alive_fraction;
    let s = p.separation_upper;
    let sigma = p.separation_scale_floor;
    let c = 2. / (p.clone_weight_lower * m);
    let d_d = 2. / (p.measurement_weight_lower * m);
    let l_q = s / (m * sigma) + 3. * s.powi(3) / (2. * m * sigma.powi(3));
    let h_s = if p.separation_exponent == 0. {
        0.
    } else {
        (p.reward_map_floor + p.reward_map_amplitude).powf(p.reward_exponent)
            * p.separation_map_amplitude
            / 4.
            * p.separation_exponent
            * p.separation_map_floor.powf(p.separation_exponent - 1.).max(
                (p.separation_map_floor + p.separation_map_amplitude)
                    .powf(p.separation_exponent - 1.),
            )
    };
    let l_0 = h_s * l_q;
    let fmin = p.reward_map_floor.powf(p.reward_exponent)
        * p.separation_map_floor.powf(p.separation_exponent);
    let fmax = (p.reward_map_floor + p.reward_map_amplitude).powf(p.reward_exponent)
        * (p.separation_map_floor + p.separation_map_amplitude).powf(p.separation_exponent);
    let l_a = (1. / (p.acceptance_saturation * (fmin + p.acceptance_epsilon))).max(
        (fmax + p.acceptance_epsilon)
            / (p.acceptance_saturation * (fmin + p.acceptance_epsilon).powi(2)),
    );
    let b = c + 2. * l_a * l_0;
    let a_t =
        (1. / m + d_d * d_d) * (2. * s * s / (sigma * sigma) + 5. * s.powi(6) / sigma.powi(6));
    let l_t = 2. * l_a * h_s;
    let a = 1. + c + d_d;
    let n_0 = (8. * a).powf(6. / 5.).ceil();
    if ![c, d_d, l_q, h_s, l_0, fmin, fmax, l_a, b, a_t, l_t, a, n_0]
        .into_iter()
        .all(|x| x.is_finite() && x >= 0.)
    {
        return Err(bad("fixed coefficients outside finite arithmetic range"));
    }
    let m1 = log_component_moment(c, 1)?;
    let m2 = log_component_moment(c, 2)?;
    let m3 = log_component_moment(c, 3)?;
    let ad = log_add(
        9_f64.ln() + log_component_moment(2. * c, 2)? + ((1. + b).powi(2) + b).ln(),
        (1_f64.max(4. * b * b)).ln(),
    );
    let aphi = std::f64::consts::LN_2 + log_add(log_add(ad, 10_f64.ln() + m2), 0.);
    let common = log_add(
        log_nonnegative(3. * l_t * a_t.sqrt()) + log_component_moment(2. * c, 1)?,
        log_nonnegative(4. * l_t * l_t * a_t),
    );
    let small = 0.5 * n_0.ln();
    let bs = log_add(
        log_add(common, 64_f64.ln() + 2. * a.ln()),
        log_add(16_f64.ln() + m3, small),
    );
    let cu = 0.5 * log_add(aphi, 4_f64.ln() + 2. * bs);
    let kord = (27. + 30. * c + 6. * c * c).ln();
    let aord = std::f64::consts::LN_2 + log_add(log_add(ad, kord), 0.);
    let bord = log_add(
        log_add(common, 128_f64.ln() + 2. * a.ln()),
        log_add(16_f64.ln() + m3, small),
    );
    Ok(QuantitativeConstants{input:p,c,d_d,l_q,h_s,l_0,fitness_lower:fmin,fitness_upper:fmax,l_a,b,a_t,l_t,a,n_0,
        log_m1_c:m1,log_m2_c:m2,log_m3_c:m3,log_a_d:ad,log_a_phi_unit:aphi,
        log_b_star:bs,log_c_update:cu,log_k_ordered:kord,log_a_ordered_unit:aord,log_b_ordered:bord,
        scope:"Fixed canonical current-donor, global regularized logistic fitness and connected-component Haar operator; positive alive fraction supplied. All constants independent of N. Logarithms retain finite analytic bounds when exponentials overflow. Ordered-star constants describe the separate priority-decorated operator, not this native Haar experiment.".into()})
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundValue {
    pub log_upper: f64,
    pub upper_float: Option<f64>,
}
pub fn from_log(log_upper: f64) -> Result<BoundValue> {
    if !log_upper.is_finite() {
        return Err(bad("finite log required"));
    }
    Ok(BoundValue {
        log_upper,
        upper_float: (log_upper <= f64::MAX.ln()).then(|| log_upper.exp()),
    })
}
impl QuantitativeConstants {
    pub fn variance_upper(&self, n: usize, test_bound: f64) -> Result<BoundValue> {
        if n < 2 || !positive(test_bound) {
            return Err(bad("N>=2 and positive test bound"));
        }
        from_log(self.log_a_phi_unit + 2. * test_bound.ln() - (n as f64).ln())
    }
    pub fn bias_upper(&self, n: usize, test_bound: f64) -> Result<BoundValue> {
        if n < 2 || !positive(test_bound) {
            return Err(bad("N>=2 and positive test bound"));
        }
        from_log(std::f64::consts::LN_2 + test_bound.ln() + self.log_b_star - 0.5 * (n as f64).ln())
    }
    pub fn mean_square_upper(&self, n: usize, test_bound: f64) -> Result<BoundValue> {
        if n < 2 || !positive(test_bound) {
            return Err(bad("N>=2 and positive test bound"));
        }
        from_log(2. * self.log_c_update + 2. * test_bound.ln() - (n as f64).ln())
    }
}
/// Exact chance of any repeated address under l independent uniform draws;
/// addresses are a sampling device, never intrinsic physical walker labels.
pub fn repeated_address_probability(n: usize, l: usize) -> Result<f64> {
    if n == 0 {
        return Err(bad("nonempty swarm"));
    }
    if l > n {
        return Ok(1.);
    }
    Ok(1. - (0..l).map(|i| 1. - i as f64 / n as f64).product::<f64>())
}
pub fn normalized_alive_ratio_bound(u: f64, v: f64, pop_u: f64, pop_mass: f64) -> Result<f64> {
    if !positive(pop_mass)
        || pop_mass > 1.
        || !v.is_finite()
        || !(0. ..=1.).contains(&v)
        || !u.is_finite()
        || u.abs() > v + 1e-12
        || !pop_u.is_finite()
        || pop_u.abs() > pop_mass + 1e-12
    {
        return Err(bad(
            "marked probability masses and bounded alive test required",
        ));
    }
    Ok(((u - pop_u).abs() + (v - pop_mass).abs()) / pop_mass)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalConstants {
    pub a0: f64,
    pub p: f64,
    pub n: usize,
    pub m_star: f64,
    pub delta: f64,
    pub n_surv: f64,
    pub decay: f64,
}
pub fn survival_constants(a0: f64, p: f64, n: usize) -> Result<SurvivalConstants> {
    if n < 2 || !positive(a0) || a0 > 1. || !positive(p) || p > 1. {
        return Err(bad("a0,p in (0,1], N>=2"));
    }
    Ok(SurvivalConstants {
        a0,
        p,
        n,
        m_star: a0 / 4.,
        delta: ((-p * n as f64 / 8.).exp() + (-a0 * n as f64 / 16.).exp()).min(1.),
        n_surv: ((8. / p).max(16. / a0) * 4_f64.ln()).ceil(),
        decay: (p / 8.).min(a0 / 16.),
    })
}
pub fn full_path_extinction_upper(one_step_upper: f64, steps: usize) -> Result<f64> {
    if !one_step_upper.is_finite() || !(0. ..=1.).contains(&one_step_upper) {
        return Err(bad("hazard in [0,1]"));
    }
    if one_step_upper == 1. {
        return Ok(f64::from(steps > 0));
    }
    Ok(-((steps as f64) * (-one_step_upper).ln_1p()).exp_m1())
}

/// Bridge coefficients to the native current-donor configuration.
pub fn constants_from_native(
    cfg: &algorithmic_gas::GasConfig,
    m: f64,
) -> Result<QuantitativeConstants> {
    use algorithmic_gas::{
        donor::{DonorModule, SamplingLaw},
        fitness::{PositiveMap, Standardizer},
        geometry::{Distance, Kernel},
    };
    fn donor(d: &DonorModule) -> Result<(f64, f64)> {
        if d.law != SamplingLaw::Independent
            || d.count != 1
            || d.allow_self
            || d.history_window != 0
        {
            return Err(bad(
                "independent single current self-excluded companion required",
            ));
        }
        let sq = match d.distance {
            Distance::SquashedPhaseSpace {
                position_radius,
                velocity_radius,
                lambda,
                ..
            } => {
                4. * (position_radius * position_radius
                    + lambda * velocity_radius * velocity_radius)
            }
            _ => {
                return Err(bad(
                    "global positive kernel certificate requires bounded algorithmic features",
                ));
            }
        };
        let k = match d.kernel {
            Kernel::Gaussian { width } => (-sq / (2. * width * width)).exp(),
            _ => return Err(bad("canonical Gaussian kernel required")),
        };
        Ok((sq, k))
    }
    if cfg.n_elite != 0
        || cfg.qft.viscosity.is_some()
        || cfg.qft.graph_viscosity.is_some()
        || cfg.clone_decision.every != 1
        || !cfg.clone_decision.revival_from_companion
    {
        return Err(bad(
            "canonical no elite/no viscosity/every-step current companion revival required",
        ));
    }
    let (sq, kd) = donor(&cfg.distance_donors)?;
    let (_, kc) = donor(&cfg.cloning_donors)?;
    let sigma = match cfg.fitness.diversity_standardizer {
        Standardizer::Global { sigma_min } => sigma_min,
        _ => return Err(bad("global regularized diversity standardizer required")),
    };
    if !matches!(cfg.fitness.reward_standardizer, Standardizer::Global { .. }) {
        return Err(bad("global reward standardizer required"));
    }
    let (ar, er) = match cfg.fitness.reward_map {
        PositiveMap::Logistic { amplitude, floor } => (amplitude, floor),
        _ => return Err(bad("canonical logistic reward map required")),
    };
    let (as_, es) = match cfg.fitness.diversity_map {
        PositiveMap::Logistic { amplitude, floor } => (amplitude, floor),
        _ => return Err(bad("canonical logistic diversity map required")),
    };
    quantitative_constants(ConstantsInput {
        minimum_alive_fraction: m,
        measurement_weight_lower: kd,
        clone_weight_lower: kc,
        separation_upper: (sq + cfg.fitness.distance_floor.powi(2)).sqrt(),
        separation_scale_floor: sigma,
        reward_map_floor: er,
        reward_map_amplitude: ar,
        separation_map_floor: es,
        separation_map_amplitude: as_,
        reward_exponent: cfg.fitness.reward_exponent,
        separation_exponent: cfg.fitness.diversity_exponent,
        acceptance_epsilon: cfg.clone_decision.epsilon,
        acceptance_saturation: cfg.clone_decision.saturation,
    })
}
