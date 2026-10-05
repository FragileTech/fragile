//! Global, N-uniform selected-source moment gain for weak native selection.
//! Conservative all-alive current/historical frames only; this is not law mixing.
use crate::convergence_chapter06_completion::{
    MomentClosure, NativeTailInterface, NativeTailParameters, compose_proved_source_moment,
};
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};

fn invalid(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn nn(x: f64) -> bool {
    x.is_finite() && x >= 0.
}
fn log_add(a: f64, b: f64) -> f64 {
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessChannelRange {
    pub amplitude: f64,
    pub positive_floor: f64,
    pub exponent: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceMomentParameters {
    pub position_feature_radius: f64,
    pub velocity_feature_radius: f64,
    pub algorithmic_velocity_weight: f64,
    pub cloning_bandwidth: f64,
    pub gate_saturation: f64,
    pub gate_epsilon: f64,
    pub fitness_channels: Vec<FitnessChannelRange>,
    pub historical_window: usize,
    /// Native `Some(restitution)`, including `Some(0)`, enables component
    /// collision and rejects accepted historical donors. History needs None.
    pub component_collision_enabled: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceMomentEnvelope {
    pub parameters: SourceMomentParameters,
    pub algorithmic_feature_diameter_squared: f64,
    pub log_kernel_lower: f64,
    pub kernel_lower_float: f64,
    pub log_fitness_lower: f64,
    pub log_fitness_upper: f64,
    pub acceptance_upper: f64,
    pub current_frame_source_moment_coefficient: f64,
    pub history_additive_moment_gain: f64,
    pub hypotheses: Vec<String>,
    pub scope: String,
}

pub fn source_moment_envelope(p: &SourceMomentParameters) -> Result<SourceMomentEnvelope> {
    if p.historical_window > 0 && p.component_collision_enabled {
        return Err(invalid(
            "native accepted historical donors require component collision disabled (restitution None, not Some(0))",
        ));
    }
    if ![
        p.position_feature_radius,
        p.velocity_feature_radius,
        p.cloning_bandwidth,
        p.gate_saturation,
    ]
    .iter()
    .all(|x| x.is_finite() && *x > 0.)
        || !nn(p.algorithmic_velocity_weight)
        || !nn(p.gate_epsilon)
        || p.fitness_channels.is_empty()
        || p.fitness_channels.iter().any(|f| {
            !nn(f.amplitude)
                || !f.positive_floor.is_finite()
                || f.positive_floor <= 0.
                || !nn(f.exponent)
        })
    {
        return Err(invalid(
            "finite actual feature, positive fitness-floor and cloning-gate parameters required",
        ));
    }
    let diameter = 4.
        * (p.position_feature_radius.powi(2)
            + p.algorithmic_velocity_weight * p.velocity_feature_radius.powi(2));
    let log_kernel = -diameter / (2. * p.cloning_bandwidth.powi(2));
    let log_min: f64 = p
        .fitness_channels
        .iter()
        .map(|f| f.exponent * f.positive_floor.ln())
        .sum();
    let log_span: f64 = p
        .fitness_channels
        .iter()
        .map(|f| f.exponent * (f.amplitude / f.positive_floor).ln_1p())
        .sum();
    let log_max = log_min + log_span;
    let acceptance = if log_span == 0. {
        0.
    } else {
        let log_expm1 = if log_span > 1. {
            log_span + (-(-log_span).exp()).ln_1p()
        } else {
            log_span.exp_m1().ln()
        };
        let denominator = p.gate_saturation.ln()
            + log_add(
                log_min,
                if p.gate_epsilon == 0. {
                    f64::NEG_INFINITY
                } else {
                    p.gate_epsilon.ln()
                },
            );
        (log_min + log_expm1 - denominator).min(0.).exp()
    };
    let gain = if acceptance == 0. {
        0.
    } else {
        (acceptance.ln() - log_kernel).exp()
    };
    if ![diameter, log_kernel, log_min, log_max, acceptance, gain]
        .iter()
        .all(|x| x.is_finite())
    {
        return Err(invalid(
            "finite source-gain arithmetic required; kernel positivity may not be replaced by underflow zero",
        ));
    }
    Ok(SourceMomentEnvelope {parameters:p.clone(),algorithmic_feature_diameter_squared:diameter,
        log_kernel_lower:log_kernel,kernel_lower_float:log_kernel.exp(),log_fitness_lower:log_min,
        log_fitness_upper:log_max,acceptance_upper:acceptance,
        current_frame_source_moment_coefficient:1.+gain,
        history_additive_moment_gain:if p.historical_window==0 {gain}else{4.*gain/3.},
        hypotheses:vec!["N>=2, conservative death-disabled kernel; all current and historical source frames have all N eligible rows".into(),
            "actual squashed phase-space distance and Gaussian independent one-donor kernel, no self current donor; no elite/external source injection".into(),
            "actual logistic channel ranges [floor,amplitude+floor], nonnegative exponents, and canonical competitive gate saturation/epsilon".into(),
            "zero-jitter source moment uses 1/N probability normalization; source pressure formula is averaged after actual sampled nonlinear global fitness".into(),
            "all copied historical velocities and frozen current collision inputs satisfy the same configured cap".into()],
        scope:"Exact uniform upper source-moment gain after discarding only negative copying loss. Physical positions remain unbounded. Finite historical windows use an N-uniform delayed moment interface; no swarm/QSD/alive-law contraction is inferred.".into()})
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedMomentClosure {
    pub source: SourceMomentEnvelope,
    pub moment: MomentClosure,
    pub history_block_length: usize,
    pub block_rate_per_step: Option<f64>,
    pub scope: String,
}
pub fn selected_native_moment_closure(
    interface: &NativeTailInterface,
    p: &SourceMomentParameters,
) -> Result<SelectedMomentClosure> {
    if !p.component_collision_enabled && interface.parameters.restitution != 0. {
        return Err(invalid(
            "collision-disabled native source requires zero effective restitution in the kinetic interface",
        ));
    }
    let source = source_moment_envelope(p)?;
    let coefficient = if p.historical_window == 0 {
        source.current_frame_source_moment_coefficient
    } else {
        1. + source.history_additive_moment_gain
    };
    let moment = compose_proved_source_moment(
        interface,
        coefficient,
        0.,
        "Exact finite conservative all-alive selected-source identity; bounded actual squashed-feature Gaussian weights, positive global fitness ranges, and incoming accepted-source load bound. Finite historical windows act on the recent maximum of EXPECTED frame moments, not expectation of a sampled maximum.",
    )?;
    let block = p
        .historical_window
        .checked_add(1)
        .ok_or_else(|| invalid("history block overflow"))?;
    let rate = moment.closes.then(|| {
        if moment.coefficient == 0. {
            0.
        } else {
            moment.coefficient.powf(1. / block as f64)
        }
    });
    Ok(SelectedMomentClosure {source,moment,history_block_length:block,block_rate_per_step:rate,
        scope:"Current-frame global p-moment drift or delayed expected-moment block decay, with N-uniform floor and normalized tail consequences. Requires the full force/noise/collision and source hypotheses. A moment theorem does not certify global law mixing; strong default selection may fail this sufficient inequality.".into()})
}

/// Conservative gate-weighted jitter and component-energy interpolation.
/// Keeps every actual native parameter intact; the refinement requires a
/// verified doubly stochastic first viscous matrix (including identity at nu0).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefinedSelectedMomentClosure {
    pub native_parameters: NativeTailParameters,
    pub original: SelectedMomentClosure,
    pub refined_additive_lp_budget: f64,
    pub collision_normalized_lp_upper: f64,
    pub gate_weighted_jitter_lp_upper: f64,
    pub refined_moment: MomentClosure,
    pub first_viscosity_doubly_stochastic: bool,
    pub hypotheses: Vec<String>,
}
pub fn selected_native_refined_moment_closure(
    interface: &NativeTailInterface,
    source: &SourceMomentParameters,
    first_viscosity_doubly_stochastic: bool,
) -> Result<RefinedSelectedMomentClosure> {
    if !first_viscosity_doubly_stochastic {
        return Err(invalid(
            "normalized component-energy interpolation requires identity or proved doubly stochastic first viscosity; general row normalization is insufficient",
        ));
    }
    let original = selected_native_moment_closure(interface, source)?;
    let p = &interface.parameters;
    let half = p.timestep / 2.;
    let b = half * (1. + (-p.friction * p.timestep).exp());
    let eta = half * b;
    let a = interface.source_lp_multiplier;
    let collision = p.frozen_velocity_norm_bound
        * (1. + 2. * p.restitution).powf((1. - 2. / p.moment_order).max(0.));
    let jitter = p.clone_jitter_standard_deviation
        * interface.gaussian_norm_lp_upper
        * original.source.acceptance_upper.powf(1. / p.moment_order);
    let additive = a * jitter
        + b * collision
        + eta * p.residual_per_coordinate * (p.dimension as f64).sqrt()
        + (half * p.ou_standard_deviation).hypot(p.final_position_standard_deviation)
            * interface.gaussian_norm_lp_upper;
    let budget = if a == 0. {
        additive.powf(p.moment_order)
    } else if p.moment_order == 1. {
        additive
    } else if a < 1. {
        let lambda = interface.source_moment_coefficient;
        let epsilon = (lambda / a.powf(p.moment_order)).powf(1. / (p.moment_order - 1.)) - 1.;
        (1. + 1. / epsilon).powf(p.moment_order - 1.) * additive.powf(p.moment_order)
    } else {
        2_f64.powf(p.moment_order - 1.) * additive.powf(p.moment_order)
    };
    let mut refined_moment = original.moment.clone();
    if !budget.is_finite() || !additive.is_finite() {
        return Err(invalid("refined native moment exceeds finite arithmetic"));
    }
    refined_moment.additive = budget;
    refined_moment.invariant_moment_upper = refined_moment
        .closes
        .then(|| budget / (1. - refined_moment.coefficient));
    Ok(RefinedSelectedMomentClosure {
        native_parameters: interface.parameters.clone(),
        original, refined_additive_lp_budget:additive,
        collision_normalized_lp_upper:collision, gate_weighted_jitter_lp_upper:jitter,
        refined_moment, first_viscosity_doubly_stochastic,
        hypotheses:vec!["All-alive conservative native kernel: no revival, no elite/external source; original recipient jitter is independent of its accepted gate.".into(),
            "Actual component operation preserves momentum and contracts full-slot relative energy, with restitution in [0,1]; historical None mode instead copies capped source velocities.".into(),
            "First viscosity is identity or proved doubly stochastic with nonnegative entries; count normalization satisfies this under its actual symmetric graph and h*nu/2<=1; no general row claim.".into(),
            "Same complete global force/noise/source hypotheses as the original native moment closure. The N-normalized energy and row-norm interpolation give N-independent coefficients.".into()],
    })
}

/// Root-moment Minkowski closure avoids Young's slack. For current frames,
/// u[n+1]<=r*u[n]+C for u=(E M_p)^(1/p), not a linear recurrence for M_p.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootSelectedMomentClosure {
    pub refined: RefinedSelectedMomentClosure,
    pub source_moment_coefficient: f64,
    pub root_moment_coefficient: f64,
    pub additive_root_budget: f64,
    pub closes: bool,
    pub invariant_root_moment_upper: Option<f64>,
    pub invariant_moment_upper: Option<f64>,
    pub log_invariant_moment_upper: Option<f64>,
    pub history_block_length: usize,
    pub scope: String,
}
pub fn selected_native_root_moment_closure(
    interface: &NativeTailInterface,
    source: &SourceMomentParameters,
    first_viscosity_doubly_stochastic: bool,
) -> Result<RootSelectedMomentClosure> {
    let refined = selected_native_refined_moment_closure(
        interface,
        source,
        first_viscosity_doubly_stochastic,
    )?;
    let s = if source.historical_window == 0 {
        refined
            .original
            .source
            .current_frame_source_moment_coefficient
    } else {
        1. + refined.original.source.history_additive_moment_gain
    };
    let p = interface.parameters.moment_order;
    let r = interface.source_lp_multiplier * s.powf(1. / p);
    let c = refined.refined_additive_lp_budget;
    if !r.is_finite() {
        return Err(invalid("finite native root-moment coefficient required"));
    }
    let closes = r < 1.;
    let log_root = closes.then(|| {
        if c == 0. {
            f64::NEG_INFINITY
        } else {
            c.ln() - (-r).ln_1p()
        }
    });
    let evaluate_upper = |log: f64| {
        let value = log.exp();
        if log.is_finite() && value == 0. {
            f64::MIN_POSITIVE
        } else {
            value
        }
    };
    let root = log_root.map(evaluate_upper).filter(|v| v.is_finite());
    let log_moment = log_root.map(|v| p * v);
    let moment = log_moment.map(evaluate_upper).filter(|v| v.is_finite());
    Ok(RootSelectedMomentClosure {history_block_length:refined.original.history_block_length,
        refined,source_moment_coefficient:s,root_moment_coefficient:r,additive_root_budget:c,
        closes,invariant_root_moment_upper:root,invariant_moment_upper:moment,log_invariant_moment_upper:log_moment,
        scope:"Full conservative N-normalized root-moment drift: u_n=(E M_p)^(1/p), u[n+1]<=r*u[n]+C. Current-frame bound u_n<=r^n*u0+C*(1-r^n)/(1-r); finite-history positive root excess contracts in H+1-step blocks. Closure condition a^p*S<1 is weaker than Young lambda*S<1. This is not a linear M_p recurrence, QSD eigenvalue or phase-law distance rate. Binary64 arithmetic; moment overflow retains its log bound.".into()})
}
