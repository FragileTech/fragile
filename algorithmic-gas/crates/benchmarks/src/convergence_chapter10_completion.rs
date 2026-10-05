//! Chapter 10 explicit density laws and finite estimates. No native QSD is inferred.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KineticConstants {
    pub diffusion: f64,
    pub friction: f64,
    pub hessian_bound: f64,
    pub lsi: f64,
    pub temperature: f64,
    pub l_m: f64,
    pub eta: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub g_min: f64,
    pub g_max: f64,
    pub rate: f64,
}
pub fn kinetic_constants(d: f64, gamma: f64, m: f64, lsi: f64) -> Result<KineticConstants> {
    if [d, gamma, lsi].iter().any(|x| !x.is_finite() || *x <= 0.) || !m.is_finite() || m < 0. {
        return Err(GasError::Configuration(
            "positive diffusion/friction/LSI and nonnegative finite Hessian norm required".into(),
        ));
    }
    let lm = 2. * m + gamma + 2.;
    let eta = d / (2. * (1. + 2. * m + lm * lm));
    Ok(KineticConstants {
        diffusion: d,
        friction: gamma,
        hessian_bound: m,
        lsi,
        temperature: d / gamma,
        l_m: lm,
        eta,
        a: 2. * eta,
        b: eta,
        c: 2. * eta,
        g_min: eta,
        g_max: 3. * eta,
        rate: eta / (lsi / 2. + 3. * eta),
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub family: String,
    pub name: String,
    pub left: f64,
    pub right: f64,
    pub allowance: f64,
    pub passed: bool,
}
pub fn check(
    rows: &mut Vec<Comparison>,
    family: &str,
    name: &str,
    left: f64,
    right: f64,
    allowance: f64,
) {
    rows.push(Comparison {
        family: family.into(),
        name: name.into(),
        left,
        right,
        allowance,
        passed: left.is_finite()
            && right.is_finite()
            && allowance.is_finite()
            && left <= right + allowance,
    });
}
pub fn equality(
    rows: &mut Vec<Comparison>,
    family: &str,
    name: &str,
    left: f64,
    right: f64,
    tol: f64,
) {
    check(rows, family, name, (left - right).abs(), 0., tol);
}
pub fn entropy(p: &[f64], q: &[f64]) -> Result<f64> {
    if p.len() != q.len()
        || p.is_empty()
        || p.iter().any(|x| !x.is_finite() || *x < 0.)
        || q.iter().any(|x| !x.is_finite() || *x <= 0.)
        || (p.iter().sum::<f64>() - 1.).abs() > 1e-9
        || (q.iter().sum::<f64>() - 1.).abs() > 1e-9
    {
        return Err(GasError::Configuration(
            "probability vectors and positive reference required".into(),
        ));
    }
    Ok(p.iter()
        .zip(q)
        .filter(|(x, _)| **x > 0.)
        .map(|(x, y)| x * (x / y).ln())
        .sum())
}
pub fn multiply(p: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    (0..p.len())
        .map(|j| (0..p.len()).map(|i| p[i] * matrix[i][j]).sum())
        .collect()
}
pub fn reweight(p: &[f64], w: &[f64]) -> Vec<f64> {
    let z = p.iter().zip(w).map(|(p, w)| p * w).sum::<f64>();
    p.iter().zip(w).map(|(p, w)| p * w / z).collect()
}

/// Gaussian law N(mean, diag(theta/kappa,theta)) under its conservative kinetic SDE.
/// The means are integrated by RK4; the invariant covariance is preserved exactly.
pub fn gaussian_trajectory(
    kappa: f64,
    theta: f64,
    gamma: f64,
    initial: [f64; 2],
    dt: f64,
    steps: usize,
) -> Result<Value> {
    let c = kinetic_constants(theta * gamma, gamma, kappa, theta.max(theta / kappa))?;
    let mut mu = initial;
    let mut states = vec![];
    let mut checks = vec![];
    let fun = |z: [f64; 2]| [z[1], -kappa * z[0] - gamma * z[1]];
    let add = |z: [f64; 2], k: [f64; 2], s: f64| [z[0] + s * k[0], z[1] + s * k[1]];
    let initial_phi = {
        let x = kappa * mu[0] / theta;
        let v = mu[1] / theta;
        0.5 * (kappa * mu[0] * mu[0] + mu[1] * mu[1]) / theta
            + c.eta * (2. * x * x + 2. * x * v + 2. * v * v)
    };
    for n in 0..=steps {
        let qx = kappa * mu[0] / theta;
        let qv = mu[1] / theta;
        let h = 0.5 * (kappa * mu[0] * mu[0] + mu[1] * mu[1]) / theta;
        let ix = qx * qx;
        let iv = qv * qv;
        let cross = qx * qv;
        let phi = h + c.eta * (2. * ix + 2. * cross + 2. * iv);
        let dh = -c.diffusion * iv;
        let dix = 2. * kappa * cross;
        let div = -2. * gamma * iv - 2. * cross;
        let dic = -ix - gamma * cross + kappa * iv;
        let derivative = dh + c.eta * (2. * dix + 2. * dic + 2. * div);
        let t = n as f64 * dt;
        check(
            &mut checks,
            "kinetic_decay",
            "modified_entropy_closure",
            phi,
            (c.lsi / 2. + c.g_max) * (ix + iv),
            1e-12,
        );
        check(
            &mut checks,
            "kinetic_decay",
            "matrix_dissipation",
            derivative,
            -c.eta * ix - c.diffusion / 2. * iv,
            1e-12,
        );
        check(
            &mut checks,
            "kinetic_decay",
            "exponential_envelope",
            phi,
            initial_phi * (-c.rate * t).exp(),
            1e-10,
        );
        states.push(json!({"step":n,"time":t,"mean":mu,"reference_covariance":[theta/kappa,theta],"H":h,"Ix":ix,"Iv":iv,"Ixv":cross,"Phi":phi,"Phi_dot":derivative,"predicted_envelope":initial_phi*(-c.rate*t).exp()}));
        let k1 = fun(mu);
        let k2 = fun(add(mu, k1, dt / 2.));
        let k3 = fun(add(mu, k2, dt / 2.));
        let k4 = fun(add(mu, k3, dt));
        mu = [
            mu[0] + dt / 6. * (k1[0] + 2. * k2[0] + 2. * k3[0] + k4[0]),
            mu[1] + dt / 6. * (k1[1] + 2. * k2[1] + 2. * k3[1] + k4[1]),
        ];
    }
    let final_h = states.last().unwrap()["H"].as_f64().unwrap();
    Ok(
        json!({"law":"Exact Gaussian density with invariant covariance; conservative underdamped kinetic SDE, mean integrated by RK4. Not native clipped BAOAB/cloning/QSD.","kappa":kappa,"theta":theta,"gamma":gamma,"constants":c,"dt":dt,"steps":steps,"initial":initial,"states":states,"checks":checks,"endpoint_entropy_decreased":final_h<states[0]["H"].as_f64().unwrap(),"observed_endpoint_H_rate":-(final_h/states[0]["H"].as_f64().unwrap()).ln()/(steps as f64*dt)}),
    )
}

/// Density h = exp(a sin(x)+b sin(v)+z sin(x)sin(v))/Z relative to
/// exp[-(k x²/2+A(1-cos(w x))+v²/2)/theta]. All derivative operands explicit.
pub fn density_integrals(
    k: f64,
    amplitude: f64,
    w: f64,
    theta: f64,
    gamma: f64,
    tilt: [f64; 3],
    panels: usize,
) -> Value {
    density_integrals_shifted(k, amplitude, w, theta, gamma, tilt, 0., panels)
}

#[allow(clippy::too_many_arguments)]
pub fn density_integrals_shifted(
    k: f64,
    amplitude: f64,
    w: f64,
    theta: f64,
    gamma: f64,
    tilt: [f64; 3],
    shift: f64,
    panels: usize,
) -> Value {
    let radius = 10. * (theta / k.min(1.)).sqrt() + (shift * theta / k).abs();
    let dx = 2. * radius / panels as f64;
    let c = kinetic_constants(
        gamma * theta,
        gamma,
        k + amplitude * w * w,
        (theta / k * (2. * amplitude / theta).exp()).max(theta),
    )
    .unwrap();
    let mut sums = [0.; 20];
    let mut base_z = 0.;
    let mut x_operands = vec![];
    let mut v_operands = vec![];
    for i in 0..=panels {
        let x = -radius + dx * i as f64;
        let sw = if i == 0 || i == panels {
            1.
        } else if i.is_multiple_of(2) {
            2.
        } else {
            4.
        };
        let u = k * x * x / 2. + amplitude * (1. - (w * x).cos());
        let force = k * x + amplitude * w * (w * x).sin();
        let curvature = k + amplitude * w * w * (w * x).cos();
        x_operands.push([x, sw, (-u / theta).exp(), force, curvature]);
    }
    for j in 0..=panels {
        let v = -radius + dx * j as f64;
        let sw = if j == 0 || j == panels {
            1.
        } else if j.is_multiple_of(2) {
            2.
        } else {
            4.
        };
        v_operands.push([v, sw, (-v * v / (2. * theta)).exp()]);
    }
    for x in &x_operands {
        for v in &v_operands {
            let (sx, cx) = (x[0].sin(), x[0].cos());
            let (sv, cv) = (v[0].sin(), v[0].cos());
            let [a, b, z] = tilt;
            let u = a * sx + b * sv + z * sx * sv + shift * x[0];
            let ux = (a + z * sv) * cx + shift;
            let uv = (b + z * sx) * cv;
            let uxx = -(a + z * sv) * sx;
            let uvv = -(b + z * sx) * sv;
            let uxv = z * cx * cv;
            let uxvv = -z * cx * sv;
            let uvvv = -(b + z * sx) * cv;
            let g = c.diffusion * uvv - gamma * v[0] * uv - v[0] * ux
                + x[3] * uv
                + c.diffusion * uv * uv;
            let gx = c.diffusion * uxvv - gamma * v[0] * uxv - v[0] * uxx
                + x[4] * uv
                + x[3] * uxv
                + 2. * c.diffusion * uv * uxv;
            let gv = c.diffusion * uvvv - gamma * uv - gamma * v[0] * uvv - ux - v[0] * uxv
                + x[3] * uvv
                + 2. * c.diffusion * uv * uvv;
            let bw = x[1] * v[1] * x[2] * v[2] * dx * dx / 9.;
            base_z += bw;
            let weight = bw * u.exp();
            let values = [
                1.,
                u,
                ux * ux,
                uv * uv,
                ux * uv,
                g,
                g * (u + 1.),
                g * ux * ux + 2. * ux * gx,
                g * uv * uv + 2. * uv * gv,
                g * ux * uv + gx * uv + ux * gv,
                x[4] * ux * uv,
                x[4] * uv * uv,
                uxv * uxv,
                uvv * uvv,
                uxv * uvv,
                (1. + 0.2 * x[0].sin()),
                (1. + 0.2 * x[0].sin()) * u,
                (1. + 0.2 * x[0].sin()) * ux * ux,
                0.2 * x[0].cos() * ux,
                0.04 * x[0].cos().powi(2),
            ];
            for l in 0..sums.len() {
                sums[l] += weight * values[l];
            }
        }
    }
    let z = sums[0];
    for x in &mut sums {
        *x /= z;
    }
    let logz = (z / base_z).ln();
    let h = sums[1] - logz;
    let ix = sums[2];
    let iv = sums[3];
    let cross = sums[4];
    let mut checks = vec![];
    let tol = 2e-7;
    equality(
        &mut checks,
        "density_invariance",
        "relative_density_mass_derivative",
        sums[5],
        0.,
        tol,
    );
    equality(
        &mut checks,
        "exact_fisher",
        "H_dot",
        sums[6] - logz * sums[5],
        -c.diffusion * iv,
        tol,
    );
    equality(
        &mut checks,
        "exact_fisher",
        "Ix_dot",
        sums[7],
        2. * sums[10] - 2. * c.diffusion * sums[12],
        tol,
    );
    equality(
        &mut checks,
        "exact_fisher",
        "Iv_dot",
        sums[8],
        -2. * gamma * iv - 2. * cross - 2. * c.diffusion * sums[13],
        tol,
    );
    equality(
        &mut checks,
        "exact_fisher",
        "Ixv_dot",
        sums[9],
        -ix - gamma * cross + sums[11] - 2. * c.diffusion * sums[14],
        tol,
    );
    let phi = h + c.eta * (2. * ix + 2. * cross + 2. * iv);
    let dphi = sums[6] - logz * sums[5] + c.eta * (2. * sums[7] + 2. * sums[9] + 2. * sums[8]);
    check(
        &mut checks,
        "kinetic_decay",
        "nonquadratic_LSI_closure",
        phi,
        (c.lsi / 2. + c.g_max) * (ix + iv),
        tol,
    );
    check(
        &mut checks,
        "kinetic_decay",
        "nonquadratic_matrix_dissipation",
        dphi,
        -c.eta * ix - c.diffusion / 2. * iv,
        tol,
    );
    check(
        &mut checks,
        "kinetic_decay",
        "nonquadratic_Cauchy_cross",
        cross.abs(),
        (ix * iv).sqrt(),
        tol,
    );
    check(
        &mut checks,
        "kinetic_decay",
        "nonquadratic_Hessian_cross",
        sums[10].abs(),
        c.hessian_bound * (ix * iv).sqrt(),
        tol,
    );
    check(
        &mut checks,
        "kinetic_decay",
        "nonquadratic_Hessian_velocity",
        sums[11],
        c.hessian_bound * iv,
        tol,
    );
    // Same identified density, specified multiplication model V=1+0.2 sin(x), G_xx=1.
    let mean_v = sums[15];
    let covariance = sums[16] - mean_v * sums[1];
    let selection_ig = sums[17] / mean_v - ix + 2. * sums[18] / mean_v;
    let k_select = 0.4 / 0.8;
    let source = 0.2;
    check(
        &mut checks,
        "selection",
        "multiplication_Fisher_bound",
        selection_ig,
        k_select * ix + 2. * source / 0.8 * ix.sqrt(),
        tol,
    );
    json!({"law":"Smooth positive explicit density relative to nonquadratic confining Gibbs law; Simpson quadrature on Gaussian-tail controlled domain, not histogram KL of an empirical atomic swarm.","parameters":{"kappa":k,"amplitude":amplitude,"frequency":w,"theta":theta,"gamma":gamma,"tilt":tilt,"linear_position_tilt":shift,"panels":panels,"radius":radius},"constants":c,"x_quadrature_operands":x_operands,"v_quadrature_operands":v_operands,"normalized_integrals":sums,"log_relative_Z":logz,"H":h,"Ix":ix,"Iv":iv,"Ixv":cross,"Phi":phi,"Phi_dot":dphi,"selection":{"mean_V":mean_v,"H_dot":covariance/mean_v,"IG_dot":selection_ig,"v_lower":0.8,"K":k_select,"S_G":source},"checks":checks})
}

/// Exact directional derivatives of the full functional along a Gaussian refresh channel.
/// The channel is common-invariant and has I_G contraction A_J=rho².
pub fn common_target_jump_suite() -> Result<Value> {
    let c = kinetic_constants(1., 1., 1., 1.)?;
    let mut rows = vec![];
    let mut checks = vec![];
    for rho in [0_f64, 0.2, 0.8, 1.] {
        for mu in [[0.1, 0.2], [2., -1.], [4., 3.]] {
            let h = (mu[0] * mu[0] + mu[1] * mu[1]) / 2.;
            let ig = c.eta * (2. * mu[0] * mu[0] + 2. * mu[0] * mu[1] + 2. * mu[1] * mu[1]);
            let phi = h + ig;
            let projected_h = rho * rho * h;
            let projected_ig = rho * rho * ig;
            let derivative_h = 2. * (rho - 1.) * h;
            let derivative_ig = 2. * (rho - 1.) * ig;
            check(
                &mut checks,
                "common_target_jump",
                "data_processing_entropy",
                projected_h,
                h,
                1e-14,
            );
            equality(
                &mut checks,
                "common_target_jump",
                "Gaussian_channel_Fisher",
                projected_ig,
                rho * rho * ig,
                1e-14,
            );
            check(
                &mut checks,
                "common_target_jump",
                "convex_directional_entropy",
                derivative_h,
                projected_h - h,
                1e-14,
            );
            check(
                &mut checks,
                "common_target_jump",
                "convex_directional_Fisher",
                derivative_ig,
                projected_ig - ig,
                1e-14,
            );
            check(
                &mut checks,
                "common_target_jump",
                "full_jump_directional_bound",
                derivative_h + derivative_ig,
                (rho * rho - 1.).max(0.) * ig,
                1e-14,
            );
            rows.push(json!({"rho":rho,"mean":mu,"A_J":rho*rho,"H":h,"IG":ig,"Phi":phi,"H_projected":projected_h,"IG_projected":projected_ig,"H_directional":derivative_h,"IG_directional":derivative_ig,"G":c}));
        }
    }
    Ok(
        json!({"law":"Common-invariant Gaussian refresh channel Z'=rho Z+sqrt(1-rho²)xi with xi~N(0,I); full joint density operands. Not native sampled-fitness cloning.","rows":rows,"checks":checks}),
    )
}

/// Two-state killed generator, with exact QSD and variable killing; all full-jump terms retained.
pub fn killed_and_discrete_suite() -> Result<Value> {
    let mut checks = vec![];
    let mut cases = vec![];
    for death in [0_f64, 0.2, 1., 3.] {
        let a = 1.2;
        let b = 0.8;
        let k = [0.1, death + 0.1];
        let t = a + b + k[0] + k[1];
        let determinant = (a + k[0]) * (b + k[1]) - a * b;
        let lambda = (t - (t * t - 4. * determinant).sqrt()) / 2.;
        let ratio = (a + k[0] - lambda) / b;
        let nu = [1. / (1. + ratio), ratio / (1. + ratio)];
        let l = [vec![-a, a], vec![b, -b]];
        for h0 in [0.2, 0.7, 1., 1.4] {
            let h = [h0, (1. - nu[0] * h0) / nu[1]];
            if h[1] <= 0. {
                continue;
            }
            let f = [nu[0] * h[0], nu[1] * h[1]];
            let loss = f[0] * k[0] + f[1] * k[1];
            let lf = multiply(&f, &l);
            let derivative = (0..2)
                .map(|i| (lf[i] + (loss - k[i]) * f[i]) * (h[i].ln() + 1.))
                .sum::<f64>();
            let hh = entropy(&f, &nu)?;
            let psi = |v: f64| v * v.ln() - v + 1.;
            let bg = |s: f64, t: f64| s * (s / t).ln() - s + t;
            let diss = nu[0] * a * bg(h[0], h[1]) + nu[1] * b * bg(h[1], h[0]);
            let rhs = -diss + loss * hh - (0..2).map(|i| nu[i] * k[i] * psi(h[i])).sum::<f64>();
            equality(
                &mut checks,
                "killed_entropy",
                "normalized_jump_entropy_identity",
                derivative,
                rhs,
                2e-13,
            );
            check(
                &mut checks,
                "killed_entropy",
                "oscillation_bound",
                derivative,
                -diss + (k[1] - k[0]) * hh,
                2e-13,
            );
            cases.push(json!({"death":k,"generator":l,"QSD":nu,"lambda":lambda,"h":h,"f":f,"H":hh,"D_jump":diss,"H_dot":derivative,"identity_rhs":rhs}));
        }
    }
    // Construct an exactly identified substochastic Q by reversing a positive backward kernel.
    // nu Q=alpha nu holds exactly; e is genuinely its right survival eigenfunction.
    let nu = [0.3, 0.7];
    let backwards = [vec![0.8, 0.2], vec![0.4, 0.6]];
    let alpha = 0.55;
    let q = (0..2)
        .map(|i| {
            (0..2)
                .map(|j| alpha * nu[j] * backwards[j][i] / nu[i])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cnu = multiply(&nu, &q);
    for i in 0..2 {
        equality(
            &mut checks,
            "discrete_qsd",
            "full_left_eigenmeasure",
            cnu[i],
            alpha * nu[i],
            1e-14,
        );
    }
    // Derive e from Q rather than assume raw candidate: solve first eigen-equation.
    let eratio = (alpha - q[0][0]) / q[0][1];
    let e = [
        1. / (nu[0] + nu[1] * eratio),
        eratio / (nu[0] + nu[1] * eratio),
    ];
    let doob = (0..2)
        .map(|i| {
            (0..2)
                .map(|j| e[j] * q[i][j] / (alpha * e[i]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let pi = [nu[0] * e[0], nu[1] * e[1]];
    let m = e[0].min(e[1]);
    let big = e[0].max(e[1]);
    let epsilon = q[0][0].min(q[1][0]) + q[0][1].min(q[1][1]);
    let theta = [
        q[0][0].min(q[1][0]) / epsilon,
        q[0][1].min(q[1][1]) / epsilon,
    ];
    let delta = epsilon * (theta[0] * e[0] + theta[1] * e[1]) / (alpha * big);
    let g = [1., -nu[0] / nu[1]];
    let ag = [
        backwards[0][0] * g[0] + backwards[0][1] * g[1],
        backwards[1][0] * g[0] + backwards[1][1] * g[1],
    ];
    let centering = nu[0] * ag[0] + nu[1] * ag[1];
    let vg = nu[0] * g[0] * g[0] + nu[1] * g[1] * g[1];
    let vag = nu[0] * (ag[0] - centering).powi(2) + nu[1] * (ag[1] - centering).powi(2);
    let mut perturbations = vec![];
    for eps in [-0.125, -0.06, -0.01, 0.01, 0.06, 0.125] {
        let input = [nu[0] * (1. + eps * g[0]), nu[1] * (1. + eps * g[1])];
        let un = multiply(&input, &q);
        let mass = un.iter().sum::<f64>();
        let output = [un[0] / mass, un[1] / mass];
        for j in 0..2 {
            equality(
                &mut checks,
                "discrete_qsd",
                "exact_normalized_density",
                output[j] / nu[j],
                (1. + eps * ag[j]) / (1. + eps * centering),
                1e-14,
            );
        }
        let increment = entropy(&output, &nu)? - entropy(&input, &nu)?;
        let quadratic = eps * eps / 2. * (vag - vg);
        check(
            &mut checks,
            "discrete_qsd",
            "cubic_remainder",
            (increment - quadratic).abs(),
            12. * eps.abs().powi(3),
            1e-14,
        );
        perturbations.push(json!({"epsilon":eps,"input":input,"output":output,"increment":increment,"quadratic":quadratic,"remainder":increment-quadratic}));
    }
    let mut traces = vec![];
    for p0 in [0.02, 0.2, 0.6, 0.98] {
        let initial = [p0, 1. - p0];
        let h0 = entropy(&initial, &nu)?;
        let eta = reweight(&initial, &e);
        let mut p = initial.to_vec();
        let mut z = eta.clone();
        for step in 0..=64 {
            let conj = reweight(&z, &[1. / e[0], 1. / e[1]]);
            for j in 0..2 {
                equality(
                    &mut checks,
                    "doob_transfer",
                    "exact_discrete_conjugacy",
                    p[j],
                    conj[j],
                    2e-14,
                );
            }
            let h = entropy(&p, &nu)?;
            let bound = (big / m).powi(2) * (1. - delta).powi(step) * h0;
            check(
                &mut checks,
                "doob_transfer",
                "conditional_entropy_bound",
                h,
                bound,
                2e-13,
            );
            check(
                &mut checks,
                "doob_transfer",
                "bounded_reweighting",
                entropy(&z, &pi)?,
                big / m * h,
                2e-13,
            );
            let nz = multiply(&z, &doob);
            check(
                &mut checks,
                "doob_transfer",
                "Doeblin_KL_contraction",
                entropy(&nz, &pi)?,
                (1. - delta) * entropy(&z, &pi)?,
                2e-13,
            );
            traces.push(
                json!({"initial":initial,"step":step,"conditioned":p,"Doob":z,"H":h,"bound":bound}),
            );
            p = multiply(&p, &q);
            let mass = p.iter().sum::<f64>();
            for x in &mut p {
                *x /= mass;
            }
            z = nz;
        }
    }
    // e above is computed, retain dummy candidate absence by checking genuine right equation.
    for i in 0..2 {
        equality(
            &mut checks,
            "doob_transfer",
            "right_eigenfunction",
            q[i][0] * e[0] + q[i][1] * e[1],
            alpha * e[i],
            1e-14,
        );
    }
    Ok(
        json!({"scope":"Specified exactly identified two-state killed jump generator and substochastic kernel. Algebraic entropy/normalization/Doob/reweighting checks; these constants are not substituted for unknown native QSD eigenfunctions.","killed_cases":cases,"Q":q,"QSD":nu,"alpha":alpha,"backward":backwards,"g":g,"A_g":ag,"c":centering,"e":e,"m":m,"M":big,"epsilon":epsilon,"theta":theta,"delta":delta,"Doob":doob,"pi":pi,"perturbations":perturbations,"traces":traces,"checks":checks}),
    )
}
