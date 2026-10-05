//! Chapter 11 finite-measure identities and explicitly declared reference models.
//! Every number uses probability normalization. Native grid laws are pushforwards,
//! never entropy estimates of an atomic swarm relative to a continuous density.
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};

fn invalid() -> GasError {
    GasError::Configuration("Chapter 11 measure hypotheses fail".into())
}
fn measure(x: &[f64]) -> bool {
    !x.is_empty() && x.iter().all(|v| v.is_finite() && *v >= 0.)
}

pub fn hellinger_squared(p: &[f64], q: &[f64]) -> Result<f64> {
    if !measure(p) || !measure(q) || p.len() != q.len() {
        return Err(invalid());
    }
    Ok(p.iter()
        .zip(q)
        .map(|(a, b)| (a.sqrt() - b.sqrt()).powi(2))
        .sum())
}
pub fn relative_entropy(p: &[f64], q: &[f64]) -> Result<f64> {
    if !measure(p)
        || !measure(q)
        || p.len() != q.len()
        || (p.iter().sum::<f64>() - 1.).abs() > 1e-10
        || (q.iter().sum::<f64>() - 1.).abs() > 1e-10
    {
        return Err(invalid());
    }
    let mut answer = 0.;
    for (&a, &b) in p.iter().zip(q) {
        if a > 0. {
            if b == 0. {
                return Ok(f64::INFINITY);
            }
            answer += a * (a / b).ln();
        }
    }
    Ok(answer.max(0.))
}
/// Exact monotone coupling for two finite, normalized one-dimensional laws.
/// Duplicate coordinates are allowed and no walker identity participates.
pub fn wasserstein_1d_squared(x: &[f64], px: &[f64], y: &[f64], py: &[f64]) -> Result<f64> {
    if x.len() != px.len()
        || y.len() != py.len()
        || !measure(px)
        || !measure(py)
        || x.iter().chain(y).any(|v| !v.is_finite())
        || (px.iter().sum::<f64>() - 1.).abs() > 1e-9
        || (py.iter().sum::<f64>() - 1.).abs() > 1e-9
    {
        return Err(invalid());
    }
    let mut a: Vec<_> = x
        .iter()
        .copied()
        .zip(px.iter().copied())
        .filter(|v| v.1 > 0.)
        .collect();
    let mut b: Vec<_> = y
        .iter()
        .copied()
        .zip(py.iter().copied())
        .filter(|v| v.1 > 0.)
        .collect();
    a.sort_by(|u, v| u.0.total_cmp(&v.0));
    b.sort_by(|u, v| u.0.total_cmp(&v.0));
    let (mut i, mut j, mut result) = (0, 0, 0.);
    while i < a.len() && j < b.len() {
        let amount = a[i].1.min(b[j].1);
        result += amount * (a[i].0 - b[j].0).powi(2);
        a[i].1 -= amount;
        b[j].1 -= amount;
        if a[i].1 <= 0. {
            i += 1;
        }
        if b[j].1 <= 0. {
            j += 1;
        }
    }
    Ok(result)
}
pub fn poisson_binomial(probabilities: &[f64]) -> Result<Vec<f64>> {
    if !measure(probabilities) || probabilities.iter().any(|p| *p > 1.) {
        return Err(invalid());
    }
    let mut law = vec![1.];
    for &p in probabilities {
        let mut next = vec![0.; law.len() + 1];
        for (i, &w) in law.iter().enumerate() {
            next[i] += w * (1. - p);
            next[i + 1] += w * p;
        }
        law = next;
    }
    Ok(law)
}
fn check(
    out: &mut Vec<Value>,
    name: &str,
    ids: &[usize],
    lhs: f64,
    rhs: f64,
    identity: bool,
    operands: Value,
) {
    let tolerance = 2e-10 * (1. + lhs.abs() + rhs.abs());
    out.push(json!({"id":name,"expression_ids":ids.iter().map(|i|format!("chapter11-expression-{i:04}")).collect::<Vec<_>>(),
        "lhs":lhs,"rhs":rhs,"relation":if identity {"="} else {"<="},"tolerance":tolerance,
        "passed":lhs.is_finite() && rhs.is_finite() && if identity {(lhs-rhs).abs()<=tolerance} else {lhs<=rhs+tolerance},
        "operands":operands,"scope":"explicit finite/reference law; not full native QSD or an LSI certificate"}));
}
/// Exact Gaussian OU laws: genuine continuous-law KL, Hellinger and W2 values.
pub fn gaussian_curve(
    d: usize,
    theta: f64,
    drift: f64,
    initial_variance: f64,
    initial_mean_norm_squared: f64,
    t: f64,
) -> Result<Value> {
    if d == 0
        || [theta, drift, initial_variance]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        || initial_mean_norm_squared < 0.
        || t < 0.
    {
        return Err(invalid());
    }
    let decay = (-2. * drift * t).exp();
    let variance = theta + (initial_variance - theta) * decay;
    let mean_norm_squared = initial_mean_norm_squared * decay;
    let ratio = variance / theta;
    let entropy = 0.5 * (d as f64 * (ratio - 1. - ratio.ln()) + mean_norm_squared / theta);
    let log_affinity = d as f64 / 2. * (2. * (theta * variance).sqrt() / (theta + variance)).ln()
        - mean_norm_squared / (4. * (theta + variance));
    let h2 = -2. * log_affinity.exp_m1();
    let w2 = d as f64 * (variance.sqrt() - theta.sqrt()).powi(2) + mean_norm_squared;
    Ok(
        json!({"d":d,"theta":theta,"drift":drift,"t":t,"variance":variance,"mean_norm_squared":mean_norm_squared,"entropy":entropy,"hellinger_squared":h2,"wasserstein_squared":w2,"log_affinity":log_affinity,"LSI_C":theta,"entropy_rate":2.*drift}),
    )
}
fn row_mul(v: [f64; 2], a: [[f64; 2]; 2]) -> [f64; 2] {
    [
        v[0] * a[0][0] + v[1] * a[1][0],
        v[0] * a[0][1] + v[1] * a[1][1],
    ]
}
fn norm1(v: [f64; 2]) -> f64 {
    v[0].abs() + v[1].abs()
}
fn killed_flow(f: [f64; 2], q: [[f64; 2]; 2], c: [f64; 2]) -> [f64; 2] {
    let k = row_mul(f, q);
    let rate = f[0] * c[0] + f[1] * c[1];
    [k[0] - f[0] + rate * f[0], k[1] - f[1] + rate * f[1]]
}

pub fn reference_suite() -> Result<Vec<Value>> {
    let mut out = vec![];
    let shapes = [
        vec![0.1, 0.2, 0.7],
        vec![0.5, 0.3, 0.2],
        vec![0.2, 0.4, 0.4],
        vec![1., 0., 0.],
    ];
    let locations = [-2., 0.3, 4.];
    for (i, p) in shapes.iter().enumerate() {
        for (j, q) in shapes.iter().enumerate() {
            for m in [0.01_f64, 0.3, 1., 2.] {
                for n in [0.02_f64, 0.4, 1., 3.] {
                    let mu: Vec<_> = p.iter().map(|v| m * v).collect();
                    let nu: Vec<_> = q.iter().map(|v| n * v).collect();
                    let h = hellinger_squared(&mu, &nu)?;
                    let shape = hellinger_squared(p, q)?;
                    let root = (m.sqrt() - n.sqrt()).powi(2);
                    let affinity: f64 = p.iter().zip(q).map(|(a, b)| (a * b).sqrt()).sum();
                    check(
                        &mut out,
                        &format!("mass-m-{i}-{j}-{m}-{n}"),
                        &[3, 17],
                        mu.iter().sum(),
                        m,
                        true,
                        json!({"mu":mu,"p":p,"m":m}),
                    );
                    check(
                        &mut out,
                        &format!("mass-n-{i}-{j}-{m}-{n}"),
                        &[4, 18],
                        nu.iter().sum(),
                        n,
                        true,
                        json!({"nu":nu,"q":q,"n":n}),
                    );
                    for axis in 0..p.len() {
                        check(
                            &mut out,
                            &format!("normalize-p-{i}-{j}-{m}-{n}-{axis}"),
                            &[5],
                            mu[axis] / m,
                            p[axis],
                            true,
                            json!({"mu":mu,"m":m,"axis":axis,"p":p}),
                        );
                        check(
                            &mut out,
                            &format!("normalize-q-{i}-{j}-{m}-{n}-{axis}"),
                            &[6],
                            nu[axis] / n,
                            q[axis],
                            true,
                            json!({"nu":nu,"n":n,"axis":axis,"q":q}),
                        );
                    }
                    check(
                        &mut out,
                        &format!("affinity-{i}-{j}-{m}-{n}"),
                        &[23],
                        affinity,
                        1. - shape / 2.,
                        true,
                        json!({"p":p,"q":q,"affinity":affinity,"shape_H_squared":shape}),
                    );
                    let lambda = [0.2_f64, 0.7, 2.];
                    let dominated_h: f64 = mu
                        .iter()
                        .zip(&nu)
                        .zip(lambda)
                        .map(|((&a, &b), l)| l * ((a / l).sqrt() - (b / l).sqrt()).powi(2))
                        .sum();
                    check(
                        &mut out,
                        &format!("dominating-measure-{i}-{j}-{m}-{n}"),
                        &[1],
                        dominated_h,
                        h,
                        true,
                        json!({"mu":mu,"nu":nu,"lambda":lambda}),
                    );
                    let name = format!("mass-shape-{i}-{j}-{m}-{n}");
                    let raw = json!({"p":p,"q":q,"mu":mu,"nu":nu,"m":m,"n":n,"m0":m.min(n),"affinity":affinity});
                    check(
                        &mut out,
                        &name,
                        &[1, 19, 24],
                        h,
                        root + (m * n).sqrt() * shape,
                        true,
                        raw.clone(),
                    );
                    check(
                        &mut out,
                        &format!("shape-{name}"),
                        &[25],
                        shape,
                        2. - 2. * affinity,
                        true,
                        raw.clone(),
                    );
                    check(
                        &mut out,
                        &format!("root-{name}"),
                        &[22],
                        root,
                        (m - n).powi(2) / (4. * m.min(n)),
                        false,
                        raw.clone(),
                    );
                    let kl = relative_entropy(p, q)?;
                    if kl.is_finite() {
                        check(
                            &mut out,
                            &format!("entropy-{name}"),
                            &[20, 27],
                            shape,
                            -2. * (-kl / 2.).exp_m1(),
                            false,
                            json!({"p":p,"q":q,"KL":kl}),
                        );
                        check(
                            &mut out,
                            &format!("jensen-{name}"),
                            &[26],
                            -kl / 2.,
                            affinity.ln(),
                            false,
                            raw.clone(),
                        );
                    }
                    let w = wasserstein_1d_squared(&locations, p, &locations, q)?;
                    check(
                        &mut out,
                        &format!("additive-{name}"),
                        &[7],
                        h + w,
                        hellinger_squared(&mu, &nu)? + w,
                        true,
                        json!({"locations":locations,"p":p,"q":q,"m":m,"n":n,"W2_squared":w}),
                    );
                    for t in [0.01_f64, 0.2, 0.7, 0.99] {
                        let integrand: f64 = mu
                            .iter()
                            .zip(&nu)
                            .map(|(a, b)| {
                                let z = (1. - t) * a.sqrt() + t * b.sqrt();
                                let dz = b.sqrt() - a.sqrt();
                                if z == 0. {
                                    0.
                                } else {
                                    let rho = z * z;
                                    let rate = 2. * dz / z;
                                    rho * rate * rate / 4.
                                }
                            })
                            .sum();
                        check(
                            &mut out,
                            &format!("reaction-{name}-{t}"),
                            &[11, 12, 13, 14],
                            integrand,
                            h,
                            true,
                            json!({"t":t,"mu":mu,"nu":nu,"velocity":0.,"reaction_action_integrand":integrand}),
                        );
                    }
                }
            }
        }
    }
    for m in [0.1_f64, 1., 3.] {
        for n in [0.2_f64, 1., 2.] {
            for distance in [0_f64, 0.01, 0.5, 1., 2., 4.] {
                let hk =
                    m + n - 2. * (m * n).sqrt() * distance.min(std::f64::consts::FRAC_PI_2).cos();
                let h = if distance == 0. {
                    (m.sqrt() - n.sqrt()).powi(2)
                } else {
                    m + n
                };
                check(
                    &mut out,
                    &format!("dirac-reaction-{m}-{n}-{distance}"),
                    &[9, 73],
                    hk,
                    h,
                    false,
                    json!({"m":m,"n":n,"distance":distance,"exact_Dirac_HK_squared":hk}),
                );
                if m == n {
                    check(
                        &mut out,
                        &format!("dirac-transport-{m}-{distance}"),
                        &[10, 15],
                        hk,
                        m * distance.powi(2),
                        false,
                        json!({"mass":m,"distance":distance,"r":0.}),
                    );
                }
            }
        }
    }
    for n in [4_usize, 8, 32, 128] {
        for pattern in [0_usize, 1, 2] {
            let probabilities: Vec<_> = (0..n)
                .map(|i| match pattern {
                    0 => 0.5,
                    1 => 0.2 + 0.6 * i as f64 / (n - 1) as f64,
                    _ => 0.8,
                })
                .collect();
            let law = poisson_binomial(&probabilities)?;
            let mean = probabilities.iter().sum::<f64>() / n as f64;
            let var = probabilities.iter().map(|p| p * (1. - p)).sum::<f64>() / (n * n) as f64;
            let exact_mean = law
                .iter()
                .enumerate()
                .map(|(i, p)| i as f64 / n as f64 * p)
                .sum::<f64>();
            let exact_var = law
                .iter()
                .enumerate()
                .map(|(i, p)| (i as f64 / n as f64 - mean).powi(2) * p)
                .sum::<f64>();
            let raw = json!({"N":n,"probabilities":probabilities,"poisson_binomial_law":law,"mean":mean,"variance":var});
            check(
                &mut out,
                &format!("alive-mean-{n}-{pattern}"),
                &[29, 30],
                exact_mean,
                mean,
                true,
                raw.clone(),
            );
            check(
                &mut out,
                &format!("alive-var-{n}-{pattern}"),
                &[35],
                exact_var,
                1. / (4. * n as f64),
                false,
                raw.clone(),
            );
            for (r, eta, m0, target) in [(0.3_f64, 1., 0.4, 0.7), (0.7, 0.1, 0.8, 0.6)] {
                let delta =
                    (mean - target).abs().max(r * (m0 - target).abs()) - r * (m0 - target).abs();
                let bound = (1. + eta) * r * r * (m0 - target).powi(2)
                    + (1. + 1. / eta) * delta * delta
                    + 1. / (4. * n as f64);
                check(
                    &mut out,
                    &format!("mass-drift-hypothesis-{n}-{pattern}-{r}"),
                    &[32],
                    (mean - target).abs(),
                    r * (m0 - target).abs() + delta,
                    false,
                    json!({"conditional_mean":mean,"m_star":target,"r":r,"m":m0,"delta":delta}),
                );
                check(
                    &mut out,
                    &format!("mass-drift-{n}-{pattern}-{r}"),
                    &[33],
                    var + (mean - target).powi(2),
                    bound,
                    false,
                    json!({"N":n,"r":r,"eta":eta,"entering_mass":m0,"target":target,"delta":delta,"probabilities":probabilities,"R":(1.+eta)*r*r}),
                );
            }
            for b in [0.1_f64, 0.3, 0.45] {
                if b < mean {
                    let probability = law
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i as f64 / n as f64) < b)
                        .map(|(_, p)| p)
                        .sum::<f64>();
                    check(
                        &mut out,
                        &format!("hoeffding-{n}-{pattern}-{b}"),
                        &[53, 54],
                        probability,
                        (-2. * n as f64 * (mean - b).powi(2)).exp(),
                        false,
                        json!({"N":n,"b":b,"p0":mean,"optimizing_t":4.*(mean-b),"law":law}),
                    );
                }
            }
            let extinction = law[0];
            let q = probabilities.iter().map(|p| 1. - p).fold(0_f64, f64::max);
            for horizon in [1_usize, 4, 16, 64] {
                check(
                    &mut out,
                    &format!("survival-{n}-{pattern}-{horizon}"),
                    &[50],
                    -((horizon as f64) * (1. - extinction).ln()).exp_m1(),
                    horizon as f64 * q.powi(n as i32),
                    false,
                    json!({"N":n,"safe_fraction":1.,"q":q,"T":horizon,"exact_one_step_extinction":extinction,"iid_reference_steps":true}),
                );
            }
        }
    }
    for (r, b, u0) in [(0.2_f64, 0.1, 3.), (0.7, 0.01, 0.5), (0.95, 0., 1.)] {
        let mut u = u0;
        for j in 0..65 {
            let bound = r.powi(j) * u0 + b * (1. - r.powi(j)) / (1. - r);
            check(
                &mut out,
                &format!("affine-{r}-{j}"),
                &[38, 39, 40, 42],
                u,
                bound,
                true,
                json!({"r":r,"b":b,"initial":u0,"step":j,"floor":b/(1.-r)}),
            );
            u = r * u + b;
        }
    }
    for (a, b, eta) in [(0.2_f64, -0.3, 0.1), (2., 3., 1.), (-1., 0.5, 4.)] {
        check(
            &mut out,
            &format!("young-{a}-{eta}"),
            &[36],
            (a + b).powi(2),
            (1. + eta) * a * a + (1. + 1. / eta) * b * b,
            false,
            json!({"a":a,"b":b,"eta":eta}),
        );
    }
    for cmax in [0.1_f64, 1., 3.] {
        for rate in [0.1_f64, 1., 2.] {
            for c in [0_f64, cmax / 2., cmax] {
                for initial in [0_f64, 0.4, 1.] {
                    for t in [0_f64, 0.1, 1., 5.] {
                        let equilibrium = rate / (c + rate);
                        let mass = equilibrium + (initial - equilibrium) * (-(c + rate) * t).exp();
                        let lower = rate / (cmax + rate)
                            + (initial - rate / (cmax + rate)) * (-(cmax + rate) * t).exp();
                        let derivative = -(c + rate) * (mass - equilibrium);
                        let raw = json!({"C":cmax,"c":c,"lambda":rate,"m_initial":initial,"t":t,"mass":mass,"integrating_factor":((cmax+rate)*t).exp()});
                        check(
                            &mut out,
                            &format!("continuous-mass-{cmax}-{rate}-{c}-{initial}-{t}"),
                            &[46],
                            lower,
                            mass,
                            false,
                            raw.clone(),
                        );
                        check(
                            &mut out,
                            &format!("continuous-derivative-{cmax}-{rate}-{c}-{initial}-{t}"),
                            &[46, 47],
                            derivative,
                            -c * mass + rate * (1. - mass),
                            true,
                            raw,
                        );
                    }
                }
            }
        }
    }
    for s in [0.1_f64, 0.5, 0.9] {
        for eps in [0.01_f64, 0.1, 0.4] {
            let joint = eps.min(s);
            check(
                &mut out,
                &format!("condition-prob-{s}-{eps}"),
                &[57],
                joint / s,
                eps / s,
                false,
                json!({"P_survival":s,"P_bad":eps,"P_bad_and_survival":joint}),
            );
            check(
                &mut out,
                &format!("condition-mean-{s}-{eps}"),
                &[58],
                2. * joint / s,
                (2. * joint + 3. * (1. - s)) / s,
                false,
                json!({"surviving_Y_integral":2.*joint,"outside_Y_integral":3.*(1.-s),"s":s}),
            );
        }
    }
    for n in [4_usize, 8, 32, 128] {
        for c in [0_f64, 0.1, 0.5] {
            let covariance = c / n as f64;
            let variance = 0.25 / n as f64 + (n - 1) as f64 * covariance / n as f64;
            check(
                &mut out,
                &format!("exchangeable-{n}-{c}"),
                &[59, 60],
                variance,
                (0.25 + c) / n as f64,
                false,
                json!({"N":n,"C":c,"latent_success_probabilities":[0.5-covariance.sqrt(),0.5+covariance.sqrt()],"diagonal_variance":0.25,"off_diagonal_covariance":covariance}),
            );
        }
    }
    for d in [1_usize, 2, 4, 8, 16] {
        for theta in [0.2_f64, 1., 3.] {
            for drift in [0.1_f64, 1., 3.] {
                for initial_variance in [theta, theta / 2., theta * 3.] {
                    let initial = gaussian_curve(d, theta, drift, initial_variance, d as f64, 0.)?;
                    let h0 = initial["entropy"].as_f64().unwrap();
                    for t in [0_f64, 0.1, 0.5, 1., 3., 8.] {
                        let curve = gaussian_curve(d, theta, drift, initial_variance, d as f64, t)?;
                        let kl = curve["entropy"].as_f64().unwrap();
                        let h = curve["hellinger_squared"].as_f64().unwrap();
                        let w = curve["wasserstein_squared"].as_f64().unwrap();
                        let bound = h0 * (-2. * drift * t).exp();
                        let tag = format!("Gaussian-d{d}-{theta}-{drift}-{initial_variance}-{t}");
                        check(
                            &mut out,
                            &format!("entropy-{tag}"),
                            &[62, 75],
                            kl,
                            bound,
                            false,
                            curve.clone(),
                        );
                        check(
                            &mut out,
                            &format!("hellinger-{tag}"),
                            &[63, 76],
                            h,
                            bound,
                            false,
                            curve.clone(),
                        );
                        check(
                            &mut out,
                            &format!("transport-{tag}"),
                            &[64],
                            w,
                            2. * theta * bound,
                            false,
                            curve.clone(),
                        );
                        check(
                            &mut out,
                            &format!("normalized-D-{tag}"),
                            &[77],
                            h + w,
                            (1. + 2. * theta) * bound,
                            false,
                            curve.clone(),
                        );
                        let target = 0.7_f64;
                        let m = target + (0.2 - target) * (-0.4 * t).exp();
                        let root = (m.sqrt() - target.sqrt()).powi(2);
                        let mass_shape = root + (m * target).sqrt() * h;
                        let joint = root + (m * target).sqrt() * h + w;
                        let combined = (0.2_f64 - target).powi(2) * (-0.8 * t).exp() / (4. * 0.2)
                            + (0.7 + 2. * theta) * bound;
                        let operands = json!({"gaussian":curve,"mass":m,"target_mass":target,"m0":0.2,"m1":0.7,"A_m":0.25,"lambda_m":0.8,"A_h":h0,"lambda_h":2.*drift,"B_m":0.,"B_h":0.,"C":theta,"mass_shape_H_squared":mass_shape});
                        check(
                            &mut out,
                            &format!("joint-{tag}"),
                            &[69, 70, 71, 72],
                            joint,
                            combined,
                            false,
                            operands,
                        );
                    }
                }
            }
        }
    }
    let conservative = [[0.8_f64, 0.2], [0.2, 0.8]];
    let q = [0.5_f64, 0.5];
    let killed = [[0.8_f64, 0.1], [0.2, 0.5]];
    let alpha = (1.3_f64 + (0.3_f64.powi(2) + 0.08).sqrt()) / 2.;
    let ratio = 0.1 / (alpha - 0.5);
    let pi = [1. / (1. + ratio), ratio / (1. + ratio)];
    check(
        &mut out,
        "conservative-invariant",
        &[83, 92],
        norm1(row_mul(q, conservative)),
        1.,
        true,
        json!({"P":conservative,"pi":q,"alpha":1.}),
    );
    check(
        &mut out,
        "killed-eigenmeasure",
        &[86, 104],
        norm1([
            row_mul(pi, killed)[0] - alpha * pi[0],
            row_mul(pi, killed)[1] - alpha * pi[1],
        ]),
        0.,
        true,
        json!({"Q":killed,"pi":pi,"alpha":alpha}),
    );
    for mu in [[0.1_f64, 0.9], [0.7, 0.3], [0.4, 0.6]] {
        let maximum_ratio = (mu[0] / pi[0]).max(mu[1] / pi[1]);
        for mask in 0..4 {
            let a = (0..2)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| mu[i])
                .sum::<f64>();
            let b = (0..2)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| pi[i])
                .sum::<f64>();
            check(
                &mut out,
                &format!("set-order-{mu:?}-{mask}"),
                &[79, 80],
                a,
                maximum_ratio * b,
                false,
                json!({"mu":mu,"pi":pi,"M":maximum_ratio,"subset_mask":mask}),
            );
        }
        for i in 0..2 {
            check(
                &mut out,
                &format!("RN-order-{mu:?}-{i}"),
                &[81],
                mu[i] / pi[i],
                maximum_ratio,
                false,
                json!({"mu":mu,"pi":pi,"M":maximum_ratio,"axis":i}),
            );
        }
    }
    for initial in [[0.1_f64, 0.9], [0.7, 0.3], [0.4, 0.6]] {
        let mut cp = initial;
        let mut kp = initial;
        let upper = (initial[0] / pi[0]).max(initial[1] / pi[1]);
        let lower = (initial[0] / pi[0]).min(initial[1] / pi[1]);
        let conservative_upper = 2. * initial[0].max(initial[1]);
        for j in 0..33 {
            let survival = kp[0] + kp[1];
            for i in 0..2 {
                let data = json!({"initial":initial,"step":j,"coordinate":i,"P":conservative,"Q":killed,"pi":pi,"alpha":alpha,"M":upper,"m":lower,"survival":survival});
                check(
                    &mut out,
                    &format!("domination-{initial:?}-{j}-{i}"),
                    &[84],
                    cp[i],
                    conservative_upper * q[i],
                    false,
                    data.clone(),
                );
                check(
                    &mut out,
                    &format!("killed-domination-{initial:?}-{j}-{i}"),
                    &[87, 90],
                    kp[i] / survival,
                    upper * alpha.powi(j) / survival * pi[i],
                    false,
                    data.clone(),
                );
                check(
                    &mut out,
                    &format!("killed-lower-{initial:?}-{j}-{i}"),
                    &[91],
                    lower * alpha.powi(j),
                    survival,
                    false,
                    data,
                );
            }
            cp = row_mul(cp, conservative);
            kp = row_mul(kp, killed);
        }
    }
    for d in [1_usize, 2, 4, 8] {
        for sigma in [0.1_f64, 0.5, 2.] {
            let peak = (2. * std::f64::consts::PI * sigma * sigma).powf(-(d as f64) / 2.);
            let points: Vec<Vec<f64>> = (0..8)
                .map(|i| (0..d).map(|k| 0.1 * ((i + k) as f64).sin()).collect())
                .collect();
            for t in [-0.2_f64, 0., 0.3] {
                let y = vec![t; d];
                let squared: Vec<f64> = points
                    .iter()
                    .map(|x| x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum())
                    .collect();
                let radius2 = squared.iter().copied().fold(0_f64, f64::max);
                let density = squared
                    .iter()
                    .map(|r| peak * (-r / (2. * sigma * sigma)).exp() / 8.)
                    .sum::<f64>();
                let lower = peak * (-radius2 / (2. * sigma * sigma)).exp();
                let derivative = squared
                    .iter()
                    .zip(&points)
                    .map(|(r, x)| {
                        (-(y[0] - x[0]) / (sigma * sigma))
                            * peak
                            * (-r / (2. * sigma * sigma)).exp()
                            / 8.
                    })
                    .sum::<f64>();
                let data = json!({"D":d,"sigma":sigma,"sources":points,"source_weights":vec![0.125;8],"observation":y,"R":radius2.sqrt(),"mu_C":1.,"density":density,"Gaussian_peak":peak,"lower":lower,"first_derivative":derivative,"first_derivative_supremum":peak*(-0.5_f64).exp()/sigma});
                check(
                    &mut out,
                    &format!("Gaussian-lower-{d}-{sigma}-{t}"),
                    &[100, 101, 102, 105, 107],
                    lower,
                    density,
                    false,
                    data.clone(),
                );
                check(
                    &mut out,
                    &format!("Gaussian-upper-{d}-{sigma}-{t}"),
                    &[98, 106],
                    density,
                    peak,
                    false,
                    data.clone(),
                );
                for (source, squared_distance) in squared.iter().enumerate() {
                    let value = peak * (-squared_distance / (2. * sigma * sigma)).exp();
                    check(
                        &mut out,
                        &format!("kernel-point-{d}-{sigma}-{t}-{source}"),
                        &[96],
                        value,
                        peak / lower * lower,
                        false,
                        json!({"source":points[source],"target":y,"sigma":sigma,"kernel_s":value,"kernel_t":lower,"C":peak/lower,"bounded_argument_set":true}),
                    );
                }
                check(
                    &mut out,
                    &format!("Gaussian-derivative-{d}-{sigma}-{t}"),
                    &[],
                    derivative.abs(),
                    peak * (-0.5_f64).exp() / sigma,
                    false,
                    data.clone(),
                );
                check(
                    &mut out,
                    &format!("kernel-comparison-{d}-{sigma}-{t}"),
                    &[97, 108],
                    density / lower,
                    peak / lower,
                    false,
                    json!({"gaussian":data,"comparison_density":lower,"comparison_C":peak/lower,"nonnegative_datum":vec![0.125;8]}),
                );
            }
        }
        for gamma in [0_f64, 0.5, 2.] {
            for i in 0..d {
                let mut bracket = vec![0.; 2 * d];
                bracket[i] = 1.;
                bracket[d + i] = -gamma;
                check(
                    &mut out,
                    &format!("bracket-{d}-{gamma}-{i}"),
                    &[93, 94, 95],
                    bracket[i],
                    1.,
                    true,
                    json!({"d":d,"gamma":gamma,"noise_axis":i,"bracket":bracket,"noise_and_bracket_rank":2*d,"coefficient_derivative":"d(v_j)/dv_i=delta_ij; d(-gamma*v_j-U_xj)/dv_i=-gamma*delta_ij"}),
                );
            }
        }
    }
    let c = [0.1_f64, 0.3];
    for epsilon in [-0.1_f64, -0.03, 0.02, 0.1] {
        let h = [epsilon, -epsilon];
        let f = [pi[0] + h[0], pi[1] + h[1]];
        let fq = killed_flow(pi, killed, c);
        let fh = killed_flow(f, killed, c);
        let linear = row_mul(h, killed);
        let cq = c[0] * pi[0] + c[1] * pi[1];
        let ch = c[0] * h[0] + c[1] * h[1];
        let derivative = [
            linear[0] - h[0] + cq * h[0] + ch * pi[0],
            linear[1] - h[1] + cq * h[1] + ch * pi[1],
        ];
        let remainder = [ch * h[0], ch * h[1]];
        let residual = [
            fh[0] - fq[0] - derivative[0] - remainder[0],
            fh[1] - fq[1] - derivative[1] - remainder[1],
        ];
        let data = json!({"Q":killed,"c":c,"q":pi,"h":h,"F_q":fq,"F_f":fh,"DF_q_h":derivative,"R":remainder,"M":0.6,"signed_increment_mass":h[0]+h[1]});
        check(
            &mut out,
            &format!("equilibrium-{epsilon}"),
            &[109],
            norm1(fq),
            0.,
            true,
            data.clone(),
        );
        check(
            &mut out,
            &format!("linearization-{epsilon}"),
            &[111, 112, 113, 115, 116, 117],
            norm1(residual),
            0.,
            true,
            data.clone(),
        );
        check(
            &mut out,
            &format!("remainder-{epsilon}"),
            &[110, 111],
            norm1(remainder),
            0.3 * norm1(h).powi(2),
            false,
            data,
        );
    }
    for a in [0.5_f64, 1., 3.] {
        for b in [0_f64, a / 10., a / 3.] {
            for k in [1_f64, 1.5, 2.] {
                for t in [0_f64, 0.2, 1., 4.] {
                    let actual = (-(a - b) * t).exp();
                    let data = json!({"a":a,"b":b,"K":k,"t":t,"base_zero_mass_norm":(-a*t).exp(),"perturbed_zero_mass_norm":actual,"base_generator":[[-a/2.,a/2.],[a/2.,-a/2.]],"perturbation":[[b/2.,-b/2.],[-b/2.,b/2.]]});
                    check(
                        &mut out,
                        &format!("perturbation-{a}-{b}-{k}-{t}"),
                        &[118, 119, 120],
                        actual,
                        k * (-(a - k * b) * t).exp(),
                        false,
                        data.clone(),
                    );
                    let convolution = if b == 0. {
                        t * (-a * t).exp()
                    } else {
                        (-a * t).exp() * ((b * t).exp() - 1.) / b
                    };
                    check(
                        &mut out,
                        &format!("duhamel-{a}-{b}-{k}-{t}"),
                        &[121],
                        actual,
                        k * (-a * t).exp() + k * b * convolution,
                        false,
                        data,
                    );
                }
            }
        }
    }
    let kernel = [[0.2_f64, -0.1], [0.3, 0.4]];
    let multiplier = [0.4_f64, 0.7];
    let k1 = 0.7;
    let k2 = 0.7;
    for h in [[0.1_f64, -0.1], [0.4, 0.2], [-0.2, -0.7]] {
        let integral = row_mul(h, kernel);
        let bh = [
            integral[0] - multiplier[0] * h[0],
            integral[1] - multiplier[1] * h[1],
        ];
        check(
            &mut out,
            &format!("nonlocal-{h:?}"),
            &[122, 123, 124, 125],
            norm1(bh),
            (k1 + k2) * norm1(h),
            false,
            json!({"h":h,"kernel":kernel,"a":multiplier,"K1":k1,"K2":k2,"Bh":bh}),
        );
    }
    for a in [0.1_f64, 1., 3.] {
        for delta in [0.1_f64, 0.5, 1.] {
            for t in [delta, delta + 0.5, delta + 2.] {
                let error1 = 0.8 * (-a * t).exp();
                let errorinf = error1 / 2.;
                let cdelta = 0.5 * (-a * delta).exp();
                let data = json!({"a":a,"delta":delta,"t":t,"K":1.,"C_delta":cdelta,"initial_L1":0.8,"reset_reference":[0.5,0.5],"P_t":"exp(-a*t)*I+(1-exp(-a*t))*q*1^T"});
                check(
                    &mut out,
                    &format!("fixed-time-smoothing-{a}-{delta}-{t}"),
                    &[132],
                    0.4 * (-a * delta).exp(),
                    cdelta * 0.8,
                    true,
                    data.clone(),
                );
                check(
                    &mut out,
                    &format!("L1-Linfty-{a}-{delta}-{t}"),
                    &[131, 132, 134],
                    errorinf,
                    cdelta * (-a * (t - delta)).exp() * 0.8,
                    true,
                    data,
                );
            }
        }
    }
    Ok(out)
}
