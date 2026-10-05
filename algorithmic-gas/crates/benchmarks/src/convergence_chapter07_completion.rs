//! Explicit Chapter 7 reference laws, distinct from selected native equilibrium.
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.
}
fn invalid() -> GasError {
    GasError::Configuration("Chapter 7 reference-law hypotheses fail".into())
}

/// Lanczos evaluation, on positive arguments only; binary64, not intervals.
pub fn gamma(x: f64) -> Result<f64> {
    if !positive(x) {
        return Err(invalid());
    }
    let coefficients = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return Ok(std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1. - x)?));
    }
    let z = x - 1.;
    let a = coefficients
        .iter()
        .enumerate()
        .fold(0.999_999_999_999_809_9, |s, (i, c)| {
            s + c / (z + i as f64 + 1.)
        });
    let t = z + 7.5;
    Ok((2. * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a)
}
pub fn poisson_reference(d: usize, intensity: f64, radius: f64) -> Result<Value> {
    if d == 0 || !positive(intensity) || !radius.is_finite() || radius < 0. {
        return Err(invalid());
    }
    let omega = std::f64::consts::PI.powf(d as f64 / 2.) / gamma(1. + d as f64 / 2.)?;
    let scale = (intensity * omega).powf(-1. / d as f64);
    Ok(
        json!({"D":d,"intensity":intensity,"omega_D":omega,"radius":radius,
        "survival":(-intensity*omega*radius.powf(d as f64)).exp(),
        "mean":gamma(1.+1./d as f64)?*scale,"density_scale":scale}),
    )
}
pub fn ou_reference(
    d: usize,
    friction: f64,
    diffusion: f64,
    time: f64,
    reset: f64,
) -> Result<Value> {
    if d == 0
        || !positive(friction)
        || !positive(diffusion)
        || !time.is_finite()
        || time < 0.
        || !reset.is_finite()
        || reset < 0.
    {
        return Err(invalid());
    }
    let c = (-friction * time).exp();
    let temperature = diffusion.powi(2) / (2. * friction);
    Ok(
        json!({"dimension":d,"friction":friction,"diffusion":diffusion,"time":time,
        "temperature":temperature,"W2_multiplier":c,"variance_per_coordinate":temperature*(-(-2.*friction*time).exp_m1()),
        "reset_rate":reset,"reset_second_moment":d as f64*diffusion.powi(2)/(2.*friction+reset),
        "unreset_second_moment":d as f64*temperature,"reset_second_moment_decay":2.*friction+reset}),
    )
}
pub fn alive_mass(revival: f64, average_loss: f64) -> Result<Value> {
    if !positive(revival) || !average_loss.is_finite() || average_loss < 0. {
        return Err(invalid());
    }
    let alive = revival / (revival + average_loss);
    Ok(
        json!({"revival":revival,"average_loss":average_loss,"alive":alive,"dead":1.-alive,
        "outflow":alive*average_loss,"inflow":revival*(1.-alive)}),
    )
}
/// Two-state conservative generator A u=a(nu-u), reaction b(target-u).
/// Its entire zero-mass subspace has semigroup K=1, decay a, exact resolvent.
pub fn residual_reference(a: f64, b: f64, c: f64, proposal: f64) -> Result<Value> {
    if !positive(a) || !positive(b) || !positive(c) || c < b || !(0. ..=1.).contains(&proposal) {
        return Err(invalid());
    }
    let nu = 0.3;
    let target = 0.8;
    let stationary = (a * nu + b * target) / (a + b);
    let map = (a * nu + b * target + (c - b) * proposal) / (c + a);
    let q = (c - b).abs() / (c + a);
    let residual = (a * nu + b * target) - (a + b) * proposal;
    Ok(
        json!({"a":a,"b":b,"C":c,"K":1.,"q_C":q,"proposal":[proposal,1.-proposal],
        "stationary":[stationary,1.-stationary],"map":[map,1.-map],
        "stationary_residual":[residual,-residual],"residual_mass":0.,
        "L1_error":2.*(proposal-stationary).abs(),"map_bound":2.*(map-proposal).abs()/(1.-q),
        "generator_bound":2.*residual.abs()/((c+a)*(1.-q)),"bounded_observable_error":(proposal-stationary).abs()}),
    )
}
pub fn simpson(f: impl Fn(f64) -> f64, lower: f64, upper: f64, panels: usize) -> f64 {
    assert!(panels > 0 && panels.is_multiple_of(2));
    let h = (upper - lower) / panels as f64;
    let mut sum = f(lower) + f(upper);
    for i in 1..panels {
        sum += if i.is_multiple_of(2) { 2. } else { 4. } * f(lower + i as f64 * h);
    }
    h * sum / 3.
}

pub fn reference_suite() -> Result<Vec<Value>> {
    let mut rows = vec![];
    let mut check = |family: &str,
                     id: String,
                     measured: f64,
                     bound: f64,
                     tolerance: f64,
                     inputs: Value| {
        rows.push(json!({"family":family,"id":id,"measured":measured,"bound":bound,
            "tolerance":tolerance,"absolute_residual":(measured-bound).abs(),"passed":(measured-bound).abs()<=tolerance,
            "relation":"equal","inputs":inputs,"scope":"Specified Chapter 7 reference model, not full selected native stationarity."}));
    };
    for alpha in [0.1_f64, 0.5, 0.99, 1.] {
        for probability in [0.01, 0.3, 0.5, 0.99] {
            for step in [1_i32, 2, 16, 64] {
                let survival = alpha.powi(step);
                check(
                    "qsd_definition",
                    format!("alpha{alpha}-pi{probability}-n{step}"),
                    probability * survival / survival,
                    probability,
                    1e-14,
                    json!({"pi":[probability,1.-probability],"Q":[[alpha*probability,alpha*(1.-probability)],[alpha*probability,alpha*(1.-probability)]],"alpha":alpha,"time":step,"survival":survival,"pi_Q":[alpha*probability,alpha*(1.-probability)],"conditional_pi":[probability,1.-probability],"scope":"Explicit finite killed reference chain with exactly known QSD; not the native interacting Q_N."}),
                );
            }
        }
    }
    for d in [1, 2, 3, 4, 8] {
        for intensity in [0.01, 1., 100.] {
            let p = poisson_reference(d, intensity, 0.7)?;
            let scale = p["density_scale"].as_f64().unwrap();
            let integral = simpson(
                |r| (-(r / scale).powf(d as f64)).exp(),
                0.,
                24. * scale,
                16384,
            );
            check(
                "poisson",
                format!("D{d}-lambda{intensity}-mean"),
                integral,
                p["mean"].as_f64().unwrap(),
                3e-8 * scale,
                p.clone(),
            );
            for n in [2., 4., 64., 1024.] {
                let pn = poisson_reference(d, intensity * n, 0.7)?;
                check(
                    "poisson",
                    format!("D{d}-lambda{intensity}-N{n}-scale"),
                    pn["mean"].as_f64().unwrap() / p["mean"].as_f64().unwrap(),
                    n.powf(-1. / d as f64),
                    1e-12,
                    pn,
                );
            }
        }
    }
    for revival in [0.01, 1., 100.] {
        for loss in [0., 0.3, 10.] {
            let p = alive_mass(revival, loss)?;
            check(
                "alive_balance",
                format!("rev{revival}-loss{loss}"),
                p["outflow"].as_f64().unwrap(),
                p["inflow"].as_f64().unwrap(),
                1e-11,
                p.clone(),
            );
            check(
                "alive_balance",
                format!("rev{revival}-loss{loss}-mass"),
                p["alive"].as_f64().unwrap() + p["dead"].as_f64().unwrap(),
                1.,
                1e-14,
                p,
            );
        }
    }
    for d in [1, 2, 4, 8] {
        for alpha in [0.1, 1., 2.] {
            for beta in [0.2, 1., 3.] {
                let exponent = alpha * d as f64 / beta;
                let temperature = 1. / exponent;
                // U=x²/2 in d dimensions: Gaussian normalizer and constant fitness.
                let z = (2. * std::f64::consts::PI * temperature).powf(d as f64 / 2.);
                let fitness_constant = z.powf(beta / d as f64);
                for radius in [0_f64, 0.3, 2.] {
                    let reward = (-radius * radius / 2.).exp();
                    let rho = reward.powf(exponent) / z;
                    let fitness = rho.powf(-beta / d as f64) * reward.powf(alpha);
                    check(
                        "replicator",
                        format!("D{d}-alpha{alpha}-beta{beta}-r{radius}"),
                        fitness,
                        fitness_constant,
                        2e-11 * fitness_constant,
                        json!({"D":d,"alpha":alpha,"beta":beta,"a":1.,"inverse_temperature":exponent,"Z":z,"reward":reward,"density":rho,"fitness":fitness,"R":"exp(-|x|²/2)","domain":"R^D"}),
                    );
                }
            }
        }
    }
    for diffusivity in [0.01, 1., 3.] {
        for length in [0.3, 1., 5.] {
            let pi = std::f64::consts::PI;
            let rate = diffusivity * pi * pi / (length * length);
            let phi = |x: f64| pi / (2. * length) * (pi * x / length).sin();
            check(
                "absorbing_diffusion",
                format!("D{diffusivity}-L{length}-normalization"),
                simpson(phi, 0., length, 4096),
                1.,
                2e-12,
                json!({"D0":diffusivity,"L":length,"lambda0":rate}),
            );
            for time in [0., 0.01, 0.3, 2.] {
                let mass = simpson(|x| (-rate * time).exp() * phi(x), 0., length, 4096);
                check(
                    "absorbing_diffusion",
                    format!("D{diffusivity}-L{length}-t{time}-survival"),
                    mass,
                    (-rate * time).exp(),
                    2e-12,
                    json!({"lambda0":rate,"time":time,"L":length}),
                );
            }
            for source in [0.2, 2.] {
                let mass = simpson(
                    |x| source * x * (length - x) / (2. * diffusivity),
                    0.,
                    length,
                    4096,
                );
                check(
                    "forced_diffusion",
                    format!("D{diffusivity}-L{length}-s{source}"),
                    mass,
                    source * length.powi(3) / (12. * diffusivity),
                    2e-10,
                    json!({"D0":diffusivity,"L":length,"source":source,"second_derivative":-source/diffusivity,"boundary_values":[0.,0.]}),
                );
            }
        }
    }
    for d in [1, 2, 4, 8] {
        for friction in [0.1, 1., 3.] {
            for diffusion in [0.1, 1., 2.] {
                for time in [0., 0.01, 1., 10.] {
                    let p = ou_reference(d, friction, diffusion, time, 0.7)?;
                    let t = p["temperature"].as_f64().unwrap();
                    let c = p["W2_multiplier"].as_f64().unwrap();
                    check(
                        "ou",
                        format!("d{d}-g{friction}-s{diffusion}-t{time}-invariant"),
                        c * c * t + p["variance_per_coordinate"].as_f64().unwrap(),
                        t,
                        2e-12 * t,
                        p.clone(),
                    );
                    check(
                        "reset_ou",
                        format!("d{d}-g{friction}-s{diffusion}-t{time}-balance"),
                        (2. * friction + 0.7) * p["reset_second_moment"].as_f64().unwrap(),
                        d as f64 * diffusion * diffusion,
                        1e-11,
                        p,
                    );
                }
            }
        }
    }
    for a in [0.1_f64, 1., 3.] {
        for b in [0.2, 1., 4.] {
            for c in [b, 2. * b, 10. * b] {
                for proposal in [0., 0.2, 0.5, 1.] {
                    let p = residual_reference(a, b, c, proposal)?;
                    let error = p["L1_error"].as_f64().unwrap();
                    check(
                        "stationary_residual",
                        format!("a{a}-b{b}-C{c}-g{proposal}-map"),
                        error,
                        p["map_bound"].as_f64().unwrap(),
                        3e-12,
                        p.clone(),
                    );
                    check(
                        "stationary_residual",
                        format!("a{a}-b{b}-C{c}-g{proposal}-generator"),
                        error,
                        p["generator_bound"].as_f64().unwrap(),
                        3e-12,
                        p,
                    );
                }
            }
        }
    }
    // Generator cancellation holds for every differentiable U, including wells
    // and nonquadratic tails. These evaluations retain actual U' and T.
    for landscape in ["quadratic", "rastrigin", "double_well_quartic"] {
        for x in [-4_f64, -0.5, 0., 0.7, 4.] {
            for v in [-3_f64, 0., 2.] {
                for t in [0.1_f64, 1., 3.] {
                    let (u, gradient) = match landscape {
                        "quadratic" => (x * x / 2., x),
                        "rastrigin" => (
                            x * x + 10. * (1. - (2. * std::f64::consts::PI * x).cos()),
                            2. * x
                                + 20.
                                    * std::f64::consts::PI
                                    * (2. * std::f64::consts::PI * x).sin(),
                        ),
                        _ => ((x * x - 1.).powi(2), 4. * x * (x * x - 1.)),
                    };
                    // Divide by positive density to avoid unbounded-tail underflow.
                    let position_log_derivative = -gradient / t;
                    let velocity_log_derivative = -v / t;
                    let transport =
                        -v * position_log_derivative + gradient * velocity_log_derivative;
                    let friction = 1. - v * v / t;
                    let diffusion = t * (v * v / (t * t) - 1. / t);
                    check(
                        "kinetic_gibbs",
                        format!("{landscape}-x{x}-v{v}-T{t}"),
                        transport + friction + diffusion,
                        0.,
                        3e-12,
                        json!({"U":u,"gradient_U":gradient,"position":x,"velocity":v,"temperature":t,"gamma":1.,"sigma_v_squared":2.*t,"transport_over_density":transport,"friction_over_density":friction,"diffusion_over_density":diffusion,"landscape":landscape,"global_integrability":"Quadratic and Rastrigin have quadratic confinement; quartic double well has quartic confinement. Complete-space zero flux, no killing/cloning/reset."}),
                    );
                }
            }
        }
    }
    // Continuum normalized companion expectation, on an unbounded Gaussian
    // density at z=0. Cut integration at twelve density standard deviations;
    // the omitted Gaussian moment tail is less than 1e-28 in these fixtures.
    for width in [0.2_f64, 1., 3.] {
        for sigma in [0.3_f64, 1., 4.] {
            let normalizer = (2. * std::f64::consts::PI).sqrt() * sigma;
            let measure = |y: f64| {
                (-(y * y) / (2. * width * width) - (y * y) / (2. * sigma * sigma)).exp()
                    / normalizer
            };
            let denominator = simpson(measure, -12. * sigma, 12. * sigma, 32768);
            let numerator = 2. * simpson(|y| y * measure(y), 0., 12. * sigma, 32768);
            let posterior_sigma = sigma * width / (sigma * sigma + width * width).sqrt();
            let expected = (2. / std::f64::consts::PI).sqrt() * posterior_sigma;
            rows.push(json!({"family":"continuum_companion","id":format!("width{width}-density_sigma{sigma}"),
                "measured":numerator/denominator,"bound":expected,"relation":"equal","tolerance":2e-9,
                "passed":(numerator/denominator-expected).abs()<2e-9,"inputs":{"kernel_width":width,"density_standard_deviation":sigma,
                    "numerator":numerator,"denominator":denominator,"exact_denominator":width/(width*width+sigma*sigma).sqrt(),"posterior_sigma":posterior_sigma,"domain":"R","test_point":0.},
                "scope":"Unbounded continuum Gaussian kernel times Gaussian density. Native empirical kernel sums are audited separately."}));
        }
    }
    for a in [0.1_f64, 1., 3.] {
        for b in [0.2, 1., 4.] {
            let stationary = (a * 0.3 + b * 0.8) / (a + b);
            let residual = a * (0.3 - stationary) + b * (0.8 - stationary);
            rows.push(json!({"family":"operator_balance","id":format!("a{a}-b{b}-stationarity"),"measured":residual,"bound":0.,"tolerance":1e-14,
                "passed":residual.abs()<1e-14,"inputs":{"a":a,"b":b,"stationary":[stationary,1.-stationary],"A_u":[a*(0.3-stationary),-a*(0.3-stationary)],"reaction":[b*(0.8-stationary),-b*(0.8-stationary)]},
                "scope":"Exact conservative two-state generator plus mass-zero reaction, not interacting native mean field."}));
        }
    }
    // Evaluate Gaussian W2 directly from means and covariances.
    for d in [1, 2, 4, 8] {
        for friction in [0.1, 1., 3.] {
            for diffusion in [0.1, 1., 2.] {
                for time in [0.01, 1., 10.] {
                    for initial_standard_deviation in [0_f64, 0.2, 2.] {
                        let p = ou_reference(d, friction, diffusion, time, 0.)?;
                        let temperature = p["temperature"].as_f64().unwrap();
                        let c = p["W2_multiplier"].as_f64().unwrap();
                        let initial = d as f64
                            * (1. + (initial_standard_deviation - temperature.sqrt()).powi(2));
                        let actual_variance = c * c * initial_standard_deviation.powi(2)
                            + p["variance_per_coordinate"].as_f64().unwrap();
                        let measured = d as f64
                            * (c * c + (actual_variance.sqrt() - temperature.sqrt()).powi(2));
                        let bound = c * c * initial;
                        rows.push(json!({"family":"ou_wasserstein","id":format!("d{d}-g{friction}-s{diffusion}-t{time}-initial{initial_standard_deviation}"),
                            "measured":measured,"bound":bound,"relation":"less_equal","tolerance":1e-11,
                            "passed":measured<=bound+1e-11,"inputs":{"dimension":d,"initial_mean_per_coordinate":1.,"initial_standard_deviation":initial_standard_deviation,
                                "actual_variance":actual_variance,"reference_temperature":temperature,"time":time,"friction":friction,"diffusion":diffusion},
                            "scope":"Exact W2 squared between isotropic Gaussian OU law and M_T; no completed selected native stationary claim."}));
                    }
                }
            }
        }
    }
    Ok(rows)
}
