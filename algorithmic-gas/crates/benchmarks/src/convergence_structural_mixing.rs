//! Chapter 6a §§15–16 primitive full-law rate registers with explicit applicability.
//! These profiles belong to the configured force and reward; regional samples
//! never substitute for a global bound. No numerical rate is inferred by fitting.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KineticParameters {
    pub dimension: usize,
    pub h: f64,
    pub friction: f64,
    pub velocity_noise: f64,
    pub position_noise: f64,
    pub velocity_cap: f64,
    pub collision_alpha: f64,
    pub position_minorization_radius: f64,
    pub velocity_minorization_radius: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LandscapeProfiles {
    /// None means no proved global finite bound (including an infinite supremum).
    pub discrete_center_bound: Option<f64>,
    pub force_lipschitz: f64,
    pub bounded_reward_oscillation: Option<f64>,
    pub raw_reward_quadratic_growth: Option<f64>,
    pub derivation: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessChannel {
    pub floor: f64,
    pub amplitude: f64,
    pub regularizer: f64,
    pub exponent: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionParameters {
    pub reward: FitnessChannel,
    pub diversity: FitnessChannel,
    pub comparison_feature_diameter: f64,
    pub measurement_scale: f64,
    pub cloning_scale: f64,
    pub distance_regularizer: f64,
    pub cloning_gate_scale: f64,
    pub cloning_gate_floor: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelHypotheses {
    pub conservative_all_alive: bool,
    pub current_frame: bool,
    pub nonviscous: bool,
    pub actual_sampled_global_normalization: bool,
    pub normalized_gaussian_companions: bool,
    pub simultaneous_copying_and_component_haar_collision: bool,
    pub actual_radial_velocity_cap: bool,
    pub profiles_proved_for_configured_force_and_reward: bool,
}
impl KernelHypotheses {
    fn failures(&self) -> Vec<&'static str> {
        [
            (self.conservative_all_alive, "conservative all-alive kernel"),
            (
                self.current_frame,
                "current-frame companions without historical donor terms",
            ),
            (self.nonviscous, "no viscosity"),
            (
                self.actual_sampled_global_normalization,
                "actual sampled global normalization",
            ),
            (
                self.normalized_gaussian_companions,
                "normalized Gaussian companions",
            ),
            (
                self.simultaneous_copying_and_component_haar_collision,
                "simultaneous copying and component Haar collision",
            ),
            (
                self.actual_radial_velocity_cap,
                "actual radial velocity cap",
            ),
            (
                self.profiles_proved_for_configured_force_and_reward,
                "proved profiles for the configured force and reward",
            ),
        ]
        .into_iter()
        .filter_map(|(valid, name)| (!valid).then_some(name))
        .collect()
    }
}
fn validate(k: &KineticParameters, s: &SelectionParameters, p: &LandscapeProfiles) -> Result<()> {
    if k.dimension == 0
        || !k.friction.is_finite()
        || k.friction < 0.
        || !k.collision_alpha.is_finite()
        || ![
            k.h,
            k.velocity_noise,
            k.position_noise,
            k.velocity_cap,
            k.position_minorization_radius,
            k.velocity_minorization_radius,
            s.measurement_scale,
            s.cloning_scale,
            s.cloning_gate_scale,
            s.cloning_gate_floor,
        ]
        .iter()
        .all(|x| x.is_finite() && *x > 0.)
        || ![
            s.comparison_feature_diameter,
            s.distance_regularizer,
            p.force_lipschitz,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.)
    {
        return Err(GasError::Configuration("Structural mixing requires finite primitive profiles, positive noises/scales and nonnegative friction".into()));
    }
    for b in [&s.reward, &s.diversity] {
        if ![b.floor, b.amplitude, b.regularizer]
            .iter()
            .all(|x| x.is_finite() && *x > 0.)
            || !b.exponent.is_finite()
            || b.exponent < 0.
        {
            return Err(GasError::Configuration(
                "Invalid logistic fitness channel".into(),
            ));
        }
    }
    for x in [
        p.discrete_center_bound,
        p.bounded_reward_oscillation,
        p.raw_reward_quadratic_growth,
    ]
    .into_iter()
    .flatten()
    {
        if !x.is_finite() || x < 0. {
            return Err(GasError::Configuration("Profile constants must be finite and nonnegative; use None for an unproved/infinite bound".into()));
        }
    }
    Ok(())
}
fn log_ball_volume(d: usize, r: f64) -> f64 {
    let log_gamma = if d.is_multiple_of(2) {
        (1..=d / 2).map(|j| (j as f64).ln()).sum::<f64>()
    } else {
        std::f64::consts::PI.ln() / 2. - 2_f64.ln()
            + (1..=d / 2).map(|j| (j as f64 + 0.5).ln()).sum::<f64>()
    };
    d as f64 / 2. * std::f64::consts::PI.ln() + d as f64 * r.ln() - log_gamma
}
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
fn number(log: f64) -> Value {
    let v = log.exp();
    if v.is_finite() && v > 0. {
        json!(v)
    } else {
        Value::Null
    }
}

/// Complete native-parameter register. A failed premise returns no positive rate.
pub fn evaluate(
    k: &KineticParameters,
    s: &SelectionParameters,
    p: &LandscapeProfiles,
    hypotheses: &KernelHypotheses,
) -> Result<Value> {
    validate(k, s, p)?;
    let mut failed = hypotheses
        .failures()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let Some(hc) = p.discrete_center_bound else {
        failed.push("The configured force has no proved finite global H_c=sup|x+eta F(x)|; regional values cannot replace it".into());
        return Ok(
            json!({"applicable":false,"failed_hypotheses":failed,"population_independent":true,"dimension":k.dimension,"q2":null,"qw":null,"rate":null,"source_labels":["def-slcc-regime","def-slcw-regime"],"profiles":p}),
        );
    };
    let d = k.dimension as f64;
    let c = k.h / 2.;
    let a = (-k.friction * k.h).exp();
    let b = c * (1. + a);
    let eta = c * c * (1. + a);
    let q2 = if k.friction == 0. {
        k.velocity_noise.powi(2) * k.h
    } else {
        k.velocity_noise.powi(2) * (-(-2. * k.friction * k.h).exp_m1()) / (2. * k.friction)
    };
    let s2 = k.position_noise.powi(2) * k.h;
    let tau2 = c * c * q2 + s2;
    let lambda = 1. / eta;
    let g0 = hc / eta;
    let bk = 1. + (b * k.velocity_cap + hc).powi(2) + d * tau2;
    let r0 = (4. * bk - 1.).sqrt();
    let alpha0 = a / (1. + a);
    if alpha0 == 0. || ![eta, q2, s2, bk].iter().all(|x| x.is_finite() && *x > 0.) {
        return Err(GasError::Configuration(
            "Primitive kinetic register is not representable at these parameters".into(),
        ));
    }
    let r1 = alpha0 * r0 + c * k.velocity_cap + c * c * g0;
    let mv = a * (k.velocity_cap + c * lambda * r0 + c * g0);
    let q = (k.velocity_minorization_radius + c * lambda * r1 + c * g0) / alpha0;
    let log_kv = -d / 2. * (2. * std::f64::consts::PI * q2).ln()
        - (q + mv).powi(2) / (2. * q2)
        - d * (c * c * p.force_lipschitz).ln_1p();
    let log_kx = -d / 2. * (2. * std::f64::consts::PI * s2).ln()
        - (k.position_minorization_radius + r1 + c * q).powi(2) / (2. * s2);
    let log_epsilon = (3_f64 / 4.).ln()
        + log_ball_volume(k.dimension, k.velocity_minorization_radius)
        + log_ball_volume(k.dimension, k.position_minorization_radius)
        + log_kv
        + log_kx;
    if !log_epsilon.is_finite() || log_epsilon >= 0. {
        return Err(GasError::Numerical(
            "Nonfinite or invalid minorization probability register".into(),
        ));
    }
    let epsilon = log_epsilon.exp();
    let vc = (1. + 2. * k.collision_alpha.abs()) * k.velocity_cap;
    let m2 = (b * vc + hc).powi(2) + d * tau2;
    let m4 = (b * vc + hc + tau2.sqrt() * (d * (d + 2.)).powf(0.25)).powi(4);
    let log_beta = log_epsilon - (2. * m4).ln();
    let bw = (1. + epsilon / 2.).next_up();
    let kd = 1.
        + 2. * (s.comparison_feature_diameter.powi(2) / (2. * s.measurement_scale.powi(2))).exp();
    let kappa_c = (-s.comparison_feature_diameter.powi(2) / (2. * s.cloning_scale.powi(2))).exp();
    // Rationalized sqrt(D²+delta²)-delta, including delta=0.
    let sb = if s.comparison_feature_diameter == 0. {
        0.
    } else {
        s.comparison_feature_diameter.powi(2)
            / ((s.comparison_feature_diameter.powi(2) + s.distance_regularizer.powi(2)).sqrt()
                + s.distance_regularizer)
    };
    let ts =
        kd * (sb / s.diversity.regularizer + sb.powi(3) / (2. * s.diversity.regularizer.powi(3)));
    let log_fmin =
        s.reward.exponent * s.reward.floor.ln() + s.diversity.exponent * s.diversity.floor.ln();
    let log_range = s.reward.exponent * (s.reward.amplitude / s.reward.floor).ln_1p()
        + s.diversity.exponent * (s.diversity.amplitude / s.diversity.floor).ln_1p();
    let fmin = log_fmin.exp();
    let fmax = (log_fmin + log_range).exp();
    let range = fmin * log_range.exp_m1();
    let lg = 1. / (s.cloning_gate_scale * (fmin + s.cloning_gate_floor))
        + range / (s.cloning_gate_scale * (fmin + s.cloning_gate_floor).powi(2));
    let acceptance = (range / (s.cloning_gate_scale * (fmin + s.cloning_gate_floor))).min(1.);
    let component = acceptance / kappa_c;
    let roots = acceptance + component;
    let derivative = |channel: &FitnessChannel, other: &FitnessChannel| {
        if channel.exponent == 0. {
            0.
        } else {
            channel.amplitude * channel.exponent / 4.
                * channel
                    .floor
                    .powf(channel.exponent - 1.)
                    .max((channel.floor + channel.amplitude).powf(channel.exponent - 1.))
                * (other.floor + other.amplitude).powf(other.exponent)
        }
    };
    let hr = derivative(&s.reward, &s.diversity);
    let hs = derivative(&s.diversity, &s.reward);
    if ![kd, kappa_c, fmin, fmax, lg, acceptance, hr, hs]
        .iter()
        .all(|x| x.is_finite())
        || kappa_c <= 0.
        || fmin <= 0.
        || (s.reward.exponent > 0. && hr == 0.)
        || (s.diversity.exponent > 0. && hs == 0.)
    {
        return Err(GasError::Numerical("Selection primitive range/derivative underflow or overflow; no contraction coefficient is assigned".into()));
    }
    let subcritical = component.is_finite() && 2. * component < 1.;
    let base_selection = 2. * component * kd + acceptance / kappa_c.powi(2);
    let bounded = if let Some(osc) = p.bounded_reward_oscillation {
        let tr = osc / s.reward.regularizer + osc.powi(3) / (2. * s.reward.regularizer.powi(3));
        let cf = hr * tr + hs * ts;
        let l = base_selection + 2. * lg * cf / kappa_c;
        let lr = if subcritical {
            2. * roots * kd + 2. * l / (1. - 2. * component)
        } else {
            f64::INFINITY
        };
        rate_register(
            "q2",
            log_epsilon,
            if lr == 0. {
                f64::NEG_INFINITY
            } else {
                (2. * lr + lr * lr).ln()
            },
            2. * k.h,
            1.,
            subcritical && failed.is_empty(),
            json!({"T_r":tr,"C_F":cf,"L":l,"L_R":lr}),
            "thm-slcc-active-contraction",
        )
    } else {
        json!({"applicable":false,"rate":null,"reason":"Raw reward is not a configured bounded channel","source_label":"def-slcc-regime"})
    };
    let weighted = if let Some(cr) = p.raw_reward_quadratic_growth {
        // Logs preserve enormous normalization constants and tiny minorization gains.
        let log_cr = if cr == 0. { f64::NEG_INFINITY } else { cr.ln() };
        let log_br = log_cr + log_add(0., -2_f64.ln() - log_beta / 2.);
        let log_br2 = 2_f64.ln() + 2. * log_cr + 0_f64.max(-log_beta);
        let mr = cr * (1. + m4.sqrt());
        let log_mr = if mr == 0. { f64::NEG_INFINITY } else { mr.ln() };
        let log_cvar = log_add(2_f64.ln() + log_br2, 4_f64.ln() + log_mr + log_br);
        let log_hr = if hr == 0. { f64::NEG_INFINITY } else { hr.ln() };
        let log_cf_reward = log_hr
            + log_add(
                2_f64.ln() + log_br - s.reward.regularizer.ln(),
                log_mr + log_cvar - 3. * s.reward.regularizer.ln(),
            );
        let log_cf = log_add(
            log_cf_reward,
            if hs * ts == 0. {
                f64::NEG_INFINITY
            } else {
                (hs * ts).ln()
            },
        );
        let log_l = log_add(
            if base_selection == 0. {
                f64::NEG_INFINITY
            } else {
                base_selection.ln()
            },
            (2. * lg / kappa_c).ln() + log_cf,
        );
        let log_lrem = if subcritical {
            bw.ln()
                + log_add(
                    if roots == 0. {
                        f64::NEG_INFINITY
                    } else {
                        (2. * roots * kd).ln()
                    },
                    2_f64.ln() + log_l - (1. - 2. * component).ln(),
                )
        } else {
            f64::INFINITY
        };
        let log_feedback = log_add((2. * bw).ln() + log_lrem, 2. * log_lrem);
        let mut register = rate_register(
            "qw",
            log_epsilon - 2_f64.ln(),
            log_feedback,
            2. * k.h,
            bw,
            subcritical && failed.is_empty(),
            json!({"log_B_r":log_br,"log_B_r_squared":log_br2,"M_r":mr,"log_C_var":log_cvar,"log_C_F_bar":log_cf,"log_L_rem":log_lrem}),
            "thm-slcw-active-contraction",
        );
        register["log_feedback"] = json!(if log_feedback.is_finite() {
            Some(log_feedback)
        } else {
            None
        });
        register
    } else {
        json!({"applicable":false,"rate":null,"reason":"No proved raw reward quadratic-growth profile","source_label":"def-slcw-regime"})
    };
    if !subcritical {
        failed.push("The accepted-component exploration is not subcritical: 2c_*>=1".into());
    }
    Ok(
        json!({"dimension":k.dimension,"population_independent":true,"profiles":p,"hypotheses":hypotheses,"failed_hypotheses":failed,"kinetic":{"c":c,"a":a,"B":b,"eta":eta,"q_squared":q2,"s_squared":s2,"tau_squared":tau2,"lambda0":lambda,"g0":g0,"b_K":bk,"R0":r0,"alpha0":alpha0,"R1":r1,"m_v":mv,"Q":q,"log_k_v":log_kv,"log_k_x":log_kx,"log_epsilon2":log_epsilon,"epsilon2":number(log_epsilon)},"moments":{"M2":m2,"M4":m4,"log_beta_w":log_beta,"beta_w":number(log_beta),"B_w":bw,"source_labels":["def-slcw-regime","lem-slcw-weighted-kernel"]},"selection":{"K_D":kd,"kappa_C":kappa_c,"S_b":sb,"T_s":ts,"F_min":fmin,"F_max":fmax,"H_r":hr,"H_s":hs,"a_star":acceptance,"c_star":component,"L_g":lg,"subcritical":subcritical},"bounded_reward":bounded,"raw_reward":weighted,"source_labels":["lem-slcc-base-mixing","lem-slcc-selection-perturbation","lem-slcw-normalization","lem-slcw-selection-perturbation"],"scope":"Structural sufficient regime for the configured conservative all-alive full population map, with active components retained; rates do not replace finite empirical-law sampling error or canonical killed-chain rates."}),
    )
}
// Keep the source formula inputs explicit at each q2/qw specialization.
#[allow(clippy::too_many_arguments)]
fn rate_register(
    name: &str,
    log_gain: f64,
    log_feedback: f64,
    period: f64,
    prefactor: f64,
    hypotheses: bool,
    constants: Value,
    label: &str,
) -> Value {
    let valid = hypotheses && log_gain.is_finite() && log_feedback < log_gain;
    let log_margin = if valid {
        Some(log_gain + (-((log_feedback - log_gain).exp())).ln_1p())
    } else {
        None
    };
    let margin = log_margin
        .map(f64::exp)
        .filter(|v| *v > 0. && v.is_finite());
    let coefficient = margin.map(|v| 1. - v).filter(|v| *v < 1. && *v >= 0.);
    json!({"name":name,"applicable":valid,"coefficient":coefficient,"log_gain":log_gain,"gain":number(log_gain),"log_feedback":if log_feedback.is_finite(){Some(log_feedback)}else{None},"feedback":number(log_feedback),"log_contraction_margin":log_margin,"contraction_margin":margin,"rate_per_physical_time":margin.map(|v|-(-v).ln_1p()/period),"log_rate_lower":log_margin.map(|v|v-period.ln()),"prefactor":prefactor,"source_label":label,"constants":constants,"status":if coefficient.is_some(){"computed_sufficient_rate"}else if valid{"positive_analytic_margin_below_coefficient_precision_no_representable_q"}else{"hypotheses_or_feedback_gate_not_closed"},"precision_scope":"Analytic source formula evaluated with binary64/logarithmic range retention; this register is not an interval certificate. Below-range gains are kept logarithmically and are not turned into an invented f64 coefficient."})
}

/// Exact standard Rastrigin structural profile, including the native timestep gate.
pub fn rastrigin_profiles(k: &KineticParameters) -> LandscapeProfiles {
    let eta = (k.h / 2.).powi(2) * (1. + (-k.friction * k.h).exp());
    let coefficient = 1. - 2. * eta;
    LandscapeProfiles {
        discrete_center_bound: if k.h == 1. && k.friction == 0. {
            Some(10. * std::f64::consts::PI * (k.dimension as f64).sqrt())
        } else {
            None
        },
        force_lipschitz: 2. + 40. * std::f64::consts::PI.powi(2),
        bounded_reward_oscillation: None,
        raw_reward_quadratic_growth: Some(
            1. + 20. * std::f64::consts::PI * (k.dimension as f64).sqrt(),
        ),
        derivation: format!(
            "Actual standard Rastrigin U=|x|²+10Σ(1−cos2πx_j), F_j=−2x_j−20πsin2πx_j. x+etaF=(1−2eta)x−20πη sin(2πx); linear coefficient {coefficient}; finite global center certified only for exact h=1,gamma=0,eta=1/2. Hessian entries2+40π²cos2πx_j; reward local Lipschitz on radius L is2L+20πsqrt(d). Source cor-slcw-rastrigin-uniform-law; no clipping or force replacement."
        ),
    }
}

/// The source's actual nonquadratic Rastrigin profile and positive-exponent
/// register, retaining logarithms when binary64 cannot represent its parameters.
/// This creates analysis records, not native trajectories or simulated cloning.
pub fn rastrigin_source_example(dimension: usize) -> Result<Value> {
    let kinetic = KineticParameters {
        dimension,
        h: 1.,
        friction: 0.,
        velocity_noise: 1.,
        position_noise: 1.,
        velocity_cap: 1.,
        collision_alpha: 0.5,
        position_minorization_radius: 1.,
        velocity_minorization_radius: 1.,
    };
    let channel = FitnessChannel {
        floor: 1.,
        amplitude: 1.,
        regularizer: 1.,
        exponent: 0.,
    };
    let mut selection = SelectionParameters {
        reward: channel.clone(),
        diversity: channel,
        comparison_feature_diameter: 8_f64.sqrt(),
        measurement_scale: 4.,
        cloning_scale: 4.,
        distance_regularizer: 0.001,
        cloning_gate_scale: 1.,
        cloning_gate_floor: 1.,
    };
    let hypotheses = KernelHypotheses {
        conservative_all_alive: true,
        current_frame: true,
        nonviscous: true,
        actual_sampled_global_normalization: true,
        normalized_gaussian_companions: true,
        simultaneous_copying_and_component_haar_collision: true,
        actual_radial_velocity_cap: true,
        profiles_proved_for_configured_force_and_reward: true,
    };
    let profiles = rastrigin_profiles(&kinetic);
    let control_profiles=LandscapeProfiles{discrete_center_bound:Some(0.),force_lipschitz:2.,bounded_reward_oscillation:Some(1.),raw_reward_quadratic_growth:Some(1.),derivation:"Declared F=-2x at h=1,gamma=0, hence center x+etaF=0; actual configured bounded reward R=-tanh(|x|²), oscillation1 and |R|<=1<=1+|x|²".into()};
    let closed_bounded_center_control =
        evaluate(&kinetic, &selection, &control_profiles, &hypotheses)?;
    let mut weak_selection = selection.clone();
    weak_selection.reward.exponent = 1e-100;
    weak_selection.diversity.exponent = 1e-100;
    let closed_positive_feedback_control =
        evaluate(&kinetic, &weak_selection, &control_profiles, &hypotheses)?;
    let mut large_selection = selection.clone();
    large_selection.reward.exponent = 1.;
    large_selection.diversity.exponent = 1.;
    let failed_component_control =
        evaluate(&kinetic, &large_selection, &control_profiles, &hypotheses)?;
    let no_feedback = evaluate(&kinetic, &selection, &profiles, &hypotheses)?;
    let registry = &no_feedback["raw_reward"]["constants"];
    let log_br = registry["log_B_r"]
        .as_f64()
        .ok_or_else(|| GasError::Numerical("Missing weighted register".into()))?;
    let log_cvar = registry["log_C_var"].as_f64().unwrap();
    let mr = registry["M_r"].as_f64().unwrap();
    let kd = no_feedback["selection"]["K_D"].as_f64().unwrap();
    let kappa = no_feedback["selection"]["kappa_C"].as_f64().unwrap();
    let ts = no_feedback["selection"]["T_s"].as_f64().unwrap();
    let bw = no_feedback["moments"]["B_w"].as_f64().unwrap();
    let log_epsilon = no_feedback["kinetic"]["log_epsilon2"].as_f64().unwrap();
    let a0 = 16. * 4_f64.ln() / 5.;
    let g0 = 4. / 5. + 64. * 4_f64.ln() / 25.;
    let log_cf0 = log_add(log_add(2_f64.ln() + log_br, mr.ln() + log_cvar), ts.ln());
    let log_l0 = log_add(
        (2. * a0 * kd / kappa + a0 / kappa.powi(2)).ln(),
        (2. * g0 / kappa).ln() + log_cf0,
    );
    let log_h0 = log_add((2. * a0 * (1. + 1. / kappa) * kd).ln(), 4_f64.ln() + log_l0);
    let log_theta = -2_f64.ln()
        + 0_f64
            .min((kappa / (4. * a0)).ln())
            .min(log_epsilon - 8_f64.ln() - 2. * bw.ln() - log_h0);
    selection.reward.exponent = 0.01;
    selection.diversity.exponent = 0.01;
    let specified_feedback = evaluate(&kinetic, &selection, &profiles, &hypotheses)?;
    let mut small_timestep = kinetic.clone();
    small_timestep.h = 0.04;
    small_timestep.friction = 1.;
    let small_timestep_gate = evaluate(
        &small_timestep,
        &selection,
        &rastrigin_profiles(&small_timestep),
        &hypotheses,
    )?;
    Ok(
        json!({"dimension":dimension,"source_label":"cor-slcw-rastrigin-uniform-law","configured_force":"Actual standard Rastrigin; no replacement trap or reward clipping","source_kinetic_parameters":kinetic,"source_selection_bases":selection,"zero_feedback_register":no_feedback,"specified_p_0_01_register":specified_feedback,"native_h_0_04_gate":small_timestep_gate,"closed_bounded_center_control":closed_bounded_center_control,"closed_positive_feedback_control":closed_positive_feedback_control,"failed_component_control":failed_component_control,"positive_exponent_certificate":{"log_theta_w":log_theta,"theta_w":number(log_theta),"log_C_F0_w":log_cf0,"log_L0_w":log_l0,"log_H0_w":log_h0,"log_qw_margin_lower":log_epsilon+(15_f64/64.).ln(),"log_TV_rate_lower":log_epsilon+(15_f64/(128.*kinetic.h)).ln(),"log_squared_W2_rate_lower":log_epsilon+(15_f64/(256.*kinetic.h)).ln(),"native_parameter_representable":log_theta.exp()>0.,"scope":"Exact source corollary's mathematical positive exponent interval, evaluated logarithmically. Below-range theta is not configured as zero or claimed simulated; underflow is retained explicitly. This is a formula register, not an interval proof."}}),
    )
}
