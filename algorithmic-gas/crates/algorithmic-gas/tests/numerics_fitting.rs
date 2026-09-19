use algorithmic_gas::{
    GasError,
    physics::numerics::{
        LevenbergConfig, LinearFit, Model, NonlinearFit, Prior, Whitener, chi2_q,
        generalized_symmetric, linear_fit, minimize, submatrix,
    },
};
// Reference correlator: 0.7 exp(-0.3 t) + 0.3 exp(-1.2 t) at t = 1..=12 plus correlated noise.
// The fitted values quoted below are those of scipy.optimize.least_squares(method = "lm") on the
// same augmented residuals, cross-checked with lsqfit.
const Y: [f64; 12] = [
    0.607756001510337,
    0.40909227610832,
    0.291479674032826,
    0.211699984223383,
    0.152148316299675,
    0.11198794564233,
    0.082550602767283,
    0.0617476130191936,
    0.0455554951887863,
    0.0338499637317231,
    0.0262995778816251,
    0.0196382570436904,
];
const SIGMA: [f64; 12] = [
    0.00297500009990642,
    0.00245484844850228,
    0.00213403549995025,
    0.00189887489556627,
    0.00170637138359428,
    0.00153964708367527,
    0.00139153310897461,
    0.0012585236546817,
    0.00113854313903858,
    0.00103011691519575,
    0.000932059076395571,
    0.00084335116640899,
];
const Y_A: [f64; 8] = [
    0.605999127018749,
    0.410643989549884,
    0.291386809118012,
    0.211050911440536,
    0.154993435686855,
    0.113931164154812,
    0.0838450346523933,
    0.0618521482447156,
];
const Y_B: [f64; 8] = [
    0.41865115145118,
    0.191445883593272,
    0.106629086581926,
    0.0666074800399191,
    0.0469096373603237,
    0.033215006493661,
    0.023659777908054,
    0.0173402695256227,
];
const SIGMA_B: [f64; 8] = [
    0.00204823592264596,
    0.00114219162211419,
    0.000771888022669685,
    0.000602192336827955,
    0.000509481136966814,
    0.00044797281671122,
    0.000400551018109124,
    0.000360670836754226,
];
/// `sigma_i sigma_j rho^|i-j|`.
fn covariance(sigma: &[f64], rho: f64) -> Vec<f64> {
    let n = sigma.len();
    (0..n * n)
        .map(|c| sigma[c / n] * sigma[c % n] * rho.powi((c / n).abs_diff(c % n) as i32))
        .collect()
}
/// Closed-form tridiagonal inverse of `covariance(sigma, rho)`.
fn precision(sigma: &[f64], rho: f64) -> Vec<f64> {
    let n = sigma.len();
    (0..n * n)
        .map(|c| {
            let (i, j) = (c / n, c % n);
            let entry = match i.abs_diff(j) {
                0 if i == 0 || i == n - 1 => 1.,
                0 => 1. + rho * rho,
                1 => -rho,
                _ => 0.,
            };
            entry / ((1. - rho * rho) * sigma[i] * sigma[j])
        })
        .collect()
}
fn bilinear(x: &[f64], m: &[f64], y: &[f64]) -> f64 {
    let n = y.len();
    (0..n * n).map(|c| x[c / n] * m[c] * y[c % n]).sum()
}
/// `[[1, t_0], [1, t_1], ..]`.
fn line(t: &[f64]) -> Vec<f64> {
    t.iter().flat_map(|&t| [1., t]).collect()
}
/// Independent straight-line fit from the 2x2 normal equations with an explicit precision matrix.
fn line_oracle(t: &[f64], y: &[f64], p: &[f64]) -> LinearFit {
    let one = vec![1.; t.len()];
    let (a, b, c) = (
        bilinear(&one, p, &one),
        bilinear(&one, p, t),
        bilinear(t, p, t),
    );
    let (u, v) = (bilinear(&one, p, y), bilinear(t, p, y));
    let det = a * c - b * b;
    let beta = [(c * u - b * v) / det, (a * v - b * u) / det];
    let r: Vec<f64> = (0..t.len())
        .map(|i| y[i] - beta[0] - beta[1] * t[i])
        .collect();
    LinearFit {
        beta: beta.to_vec(),
        covariance: vec![c / det, -b / det, -b / det, a / det],
        chi2: bilinear(&r, p, &r),
        dof: t.len() - 2,
    }
}
/// `C_c(t) = sum_n A_{c,n} exp(-E_n t)` with `E_n = sum_{m <= n} dE_m` and parameters
/// `[ln dE_0.., then per channel ln A_{c,0}..]`.
struct Exponentials {
    nexp: usize,
    channels: usize,
    /// `[channels, points]`.
    times: Vec<f64>,
}
impl Exponentials {
    /// `(channel, time)` of every output in order.
    fn points(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.times
            .chunks_exact(self.times.len() / self.channels)
            .enumerate()
            .flat_map(|(c, t)| t.iter().map(move |&t| (c, t)))
    }
    fn energies(&self, p: &[f64]) -> Vec<f64> {
        p[..self.nexp]
            .iter()
            .scan(0., |e, x| {
                *e += x.exp();
                Some(*e)
            })
            .collect()
    }
}
impl Model for Exponentials {
    fn parameters(&self) -> usize {
        self.nexp * (1 + self.channels)
    }
    fn outputs(&self) -> usize {
        self.times.len()
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        let e = self.energies(p);
        for ((c, t), out) in self.points().zip(out) {
            let amplitude = &p[self.nexp * (1 + c)..];
            *out = (0..self.nexp)
                .map(|n| (amplitude[n] - e[n] * t).exp())
                .sum();
        }
    }
    fn jacobian(&self, p: &[f64], out: &mut [f64]) {
        let (e, k) = (self.energies(p), self.parameters());
        out.fill(0.);
        for ((c, t), row) in self.points().zip(out.chunks_exact_mut(k)) {
            let first = self.nexp * (1 + c);
            for n in 0..self.nexp {
                let term = (p[first + n] - e[n] * t).exp();
                row[first + n] = term;
                for m in 0..=n {
                    row[m] -= t * p[m].exp() * term;
                }
            }
        }
    }
}
/// The same model differentiated by the provided central differences.
struct Differenced(Exponentials);
impl Model for Differenced {
    fn parameters(&self) -> usize {
        self.0.parameters()
    }
    fn outputs(&self) -> usize {
        self.0.outputs()
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        self.0.evaluate(p, out);
    }
}
struct Polynomial {
    t: Vec<f64>,
    degree: usize,
}
impl Model for Polynomial {
    fn parameters(&self) -> usize {
        self.degree + 1
    }
    fn outputs(&self) -> usize {
        self.t.len()
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        for (out, t) in out.iter_mut().zip(&self.t) {
            *out = p.iter().rev().fold(0., |s, c| s * t + c);
        }
    }
}
/// A model that ignores its parameters, optionally returning a nonfinite value.
struct Constant {
    parameters: usize,
    values: Vec<f64>,
}
impl Model for Constant {
    fn parameters(&self) -> usize {
        self.parameters
    }
    fn outputs(&self) -> usize {
        self.values.len()
    }
    fn evaluate(&self, _: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&self.values);
    }
}
/// `f(p) = sqrt(p_0)`, undefined for a negative parameter.
struct Root;
impl Model for Root {
    fn parameters(&self) -> usize {
        1
    }
    fn outputs(&self) -> usize {
        1
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        out[0] = p[0].sqrt();
    }
}
/// `f(p) = (p_0, c p_0² + p_0)`: at the minimum `p_0 = 0` of the data `(-1, 1)` an undamped
/// Gauss-Newton step multiplies the distance to the minimum by `c`.
struct Curved(f64);
impl Model for Curved {
    fn parameters(&self) -> usize {
        1
    }
    fn outputs(&self) -> usize {
        2
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&[p[0], self.0 * p[0] * p[0] + p[0]]);
    }
    fn jacobian(&self, p: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&[1., 2. * self.0 * p[0] + 1.]);
    }
}
/// Rosenbrock residuals `f(p) = (10 (p_1 − p_0²), p_0)`, fitted to the data `(0, 1)`.
struct Valley;
impl Model for Valley {
    fn parameters(&self) -> usize {
        2
    }
    fn outputs(&self) -> usize {
        2
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&[10. * (p[1] - p[0] * p[0]), p[0]]);
    }
    fn jacobian(&self, p: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&[-20. * p[0], 10., 1., 0.]);
    }
}
/// `f_i(p) = exp(p_0)` for every output.
struct Level(usize);
impl Model for Level {
    fn parameters(&self) -> usize {
        1
    }
    fn outputs(&self) -> usize {
        self.0
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        out.fill(p[0].exp());
    }
}
/// `f(p) = (p_0, p_0)` up to `p_0 = 1` and undefined beyond.
struct Ledge;
impl Model for Ledge {
    fn parameters(&self) -> usize {
        1
    }
    fn outputs(&self) -> usize {
        2
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        out.fill(if p[0] <= 1. { p[0] } else { f64::NAN });
    }
    fn jacobian(&self, _: &[f64], out: &mut [f64]) {
        out.fill(1.);
    }
}
fn times(first: usize, last: usize) -> Vec<f64> {
    (first..=last).map(|t| t as f64).collect()
}
/// Gap priors `N(ln 0.1, 3)` and amplitude priors `N(ln |y(t_min)|, 5)` per channel.
fn priors(nexp: usize, first_values: &[f64]) -> Vec<Prior> {
    let gap = Prior::Gaussian {
        mean: 0.1_f64.ln(),
        sigma: 3.,
    };
    let amplitude = |y: &f64| Prior::Gaussian {
        mean: y.abs().ln(),
        sigma: 5.,
    };
    std::iter::repeat_n(gap, nexp)
        .chain(
            first_values
                .iter()
                .flat_map(|y| std::iter::repeat_n(amplitude(y), nexp)),
        )
        .collect()
}
fn errors(fit: &NonlinearFit) -> Vec<f64> {
    let k = fit.p.len();
    (0..k).map(|j| fit.covariance[j * k + j].sqrt()).collect()
}
fn two_state_fit(model: &dyn Model, config: &LevenbergConfig) -> NonlinearFit {
    let whitener = Whitener::new(&covariance(&SIGMA, 0.6), 12, 1e-6).unwrap();
    let start = [
        -0.926763521695001,
        -0.233616341135055,
        -0.795296227868524,
        -1.48844340842847,
    ];
    minimize(model, &Y, &whitener, &priors(2, &Y[..1]), &start, config).unwrap()
}
const TWO_STATE: [f64; 4] = [
    -1.20073030413953,
    -0.214895721987624,
    -0.372878844263785,
    -1.21596028317044,
];
/// `C(t) = Z diag(exp(-E t)) Zᵀ` for three operators and the two states `E = (0.25, 0.7)`.
fn two_state_matrix(t: f64, rows: &[usize]) -> Vec<f64> {
    let z = [[1., 0.5], [0.4, 1.], [0.7, -0.3]];
    let decay = [(-0.25 * t).exp(), (-0.7 * t).exp()];
    rows.iter()
        .flat_map(|&i| rows.iter().map(move |&j| (i, j)))
        .map(|(i, j)| (0..2).map(|s| z[i][s] * decay[s] * z[j][s]).sum())
        .collect()
}
fn congruence(m: &[f64], c: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|e| {
            (0..n * n)
                .map(|f| m[e / n * n + f / n] * c[f] * m[e % n * n + f % n])
                .sum()
        })
        .collect()
}
#[test]
fn exact_linear_data_is_recovered_with_zero_chi2_and_a_covariance_independent_of_the_data() {
    let sigma = [0.5, 0.2, 0.1, 0.05, 0.08, 0.3];
    let t = times(1, 6);
    let whitener = Whitener::new(&covariance(&sigma, 0.95), 6, 0.).unwrap();
    let exact: Vec<f64> = t.iter().map(|t| 1.5 - 0.25 * t).collect();
    let fit = linear_fit(&line(&t), &exact, &whitener, 2).unwrap();
    assert!((fit.beta[0] - 1.5).abs() < 1e-11 && (fit.beta[1] + 0.25).abs() < 1e-11);
    assert!(fit.chi2 < 1e-18);
    assert_eq!(fit.dof, 4);
    let oracle = line_oracle(&t, &exact, &precision(&sigma, 0.95));
    for (got, want) in fit.covariance.iter().zip(&oracle.covariance) {
        assert!((got - want).abs() < 1e-10 * want.abs());
    }
    let other = linear_fit(&line(&t), &[3., -1., 0.5, 2., 0., 7.], &whitener, 2).unwrap();
    assert_eq!(other.covariance, fit.covariance);
}
#[test]
fn correlated_straight_line_fit_matches_the_closed_form_normal_equations() {
    let sigma = [0.05, 0.04, 0.05, 0.03, 0.06, 0.04, 0.05];
    let t = times(3, 9);
    let y = [2.31, 2.02, 1.78, 1.49, 1.27, 0.95, 0.74];
    let whitener = Whitener::new(&covariance(&sigma, 0.7), 7, 0.).unwrap();
    let fit = linear_fit(&line(&t), &y, &whitener, 2).unwrap();
    let oracle = line_oracle(&t, &y, &precision(&sigma, 0.7));
    assert!((fit.beta[0] - oracle.beta[0]).abs() < 1e-10);
    assert!((fit.beta[1] - oracle.beta[1]).abs() < 1e-11);
    assert!((fit.chi2 - oracle.chi2).abs() < 1e-9 * oracle.chi2);
    for (got, want) in fit.covariance.iter().zip(&oracle.covariance) {
        assert!((got - want).abs() < 1e-10 * want.abs());
    }
    assert_eq!(fit.dof, oracle.dof);
    assert_eq!(fit.covariance[1].to_bits(), fit.covariance[2].to_bits());
}
#[test]
fn uncorrelated_constant_fit_is_the_inverse_variance_weighted_mean() {
    let (y, sigma) = ([1., 2., 4.], [1., 0.5, 2.]);
    let whitener = Whitener::diagonal(&sigma).unwrap();
    let fit = linear_fit(&[1.; 3], &y, &whitener, 1).unwrap();
    // Weights 1, 4 and 1/4: mean 10/5.25, variance 1/5.25.
    let mean = 10. / 5.25;
    assert!((fit.beta[0] - mean).abs() < 1e-14);
    assert!((fit.covariance[0] - 1. / 5.25).abs() < 1e-15);
    let chi2 = (1. - mean).powi(2) + 4. * (2. - mean).powi(2) + 0.25 * (4. - mean).powi(2);
    assert!((fit.chi2 - chi2).abs() < 1e-13);
    assert_eq!(fit.dof, 2);
}
#[test]
fn linear_fit_is_covariant_under_a_change_of_units_of_a_design_column() {
    let sigma = [0.05, 0.04, 0.05, 0.03, 0.06];
    let t = times(1, 5);
    let y = [2.3, 2., 1.8, 1.5, 1.3];
    let whitener = Whitener::new(&covariance(&sigma, 0.5), 5, 0.).unwrap();
    let fit = linear_fit(&line(&t), &y, &whitener, 2).unwrap();
    let scaled: Vec<f64> = t.iter().map(|t| 1e9 * t).collect();
    let other = linear_fit(&line(&scaled), &y, &whitener, 2).unwrap();
    assert!((other.beta[0] - fit.beta[0]).abs() < 1e-11);
    assert!((other.beta[1] * 1e9 - fit.beta[1]).abs() < 1e-11);
    assert!((other.covariance[3] * 1e18 - fit.covariance[3]).abs() < 1e-9 * fit.covariance[3]);
    assert!((other.chi2 - fit.chi2).abs() < 1e-9 * fit.chi2);
}
#[test]
fn collinear_designs_fail_numerically_and_malformed_inputs_are_rejected_explicitly() {
    let whitener = Whitener::diagonal(&[1., 2., 1., 0.5]).unwrap();
    let y = [1., 2., 3., 5.];
    let duplicated: Vec<f64> = (1..=4).flat_map(|t| [t as f64, t as f64]).collect();
    let tripled: Vec<f64> = (1..=4)
        .flat_map(|t| [0.1 * t as f64, 0.3 * t as f64])
        .collect();
    for design in [duplicated, tripled, line(&[2.; 4]), vec![0.; 8]] {
        assert!(matches!(
            linear_fit(&design, &y, &whitener, 2),
            Err(GasError::Numerical(_))
        ));
    }
    let design = line(&times(1, 4));
    for (design, y, p) in [
        (&design[..6], &y[..], 2),
        (&design[..], &y[..3], 2),
        (&design[..], &y[..], 0),
        (&[1.; 20][..], &y[..], 5),
    ] {
        assert!(matches!(
            linear_fit(design, y, &whitener, p),
            Err(GasError::Configuration(_))
        ));
    }
    let nonfinite = [1., f64::NAN, 3., 5.];
    assert!(matches!(
        linear_fit(&design, &nonfinite, &whitener, 2),
        Err(GasError::Configuration(_))
    ));
}
#[test]
fn one_datum_with_a_gaussian_prior_gives_the_closed_form_posterior() {
    let model = Polynomial {
        t: vec![0.],
        degree: 0,
    };
    let prior = [Prior::Gaussian {
        mean: 0.,
        sigma: 1.,
    }];
    let whitener = Whitener::diagonal(&[1.]).unwrap();
    let fit = minimize(
        &model,
        &[2.],
        &whitener,
        &prior,
        &[0.3],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert!((fit.p[0] - 1.).abs() < 1e-9);
    // Posterior precision 1 + 1: variance 1/2, mean (2 + 0)/2.
    assert!((fit.covariance[0] - 0.5).abs() < 1e-9);
    assert!((fit.chi2 - 2.).abs() < 1e-9 && (fit.chi2_prior - 1.).abs() < 1e-9);
    assert_eq!(fit.dof, 1);
    assert!((fit.q.unwrap() - 0.157299207050281).abs() < 1e-9);
}
#[test]
fn model_without_parameter_sensitivity_returns_the_prior() {
    let model = Constant {
        parameters: 2,
        values: vec![1., 2., 3.],
    };
    let prior = [
        Prior::Gaussian {
            mean: -1.5,
            sigma: 0.5,
        },
        Prior::Gaussian {
            mean: 4.,
            sigma: 3.,
        },
    ];
    let whitener = Whitener::diagonal(&[1., 1., 2.]).unwrap();
    let data = [1.5, 2., 2.];
    let fit = minimize(
        &model,
        &data,
        &whitener,
        &prior,
        &[0.7, -2.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert!((fit.p[0] + 1.5).abs() < 1e-7 && (fit.p[1] - 4.).abs() < 1e-7);
    for (got, want) in fit.covariance.iter().zip([0.25, 0., 0., 9.]) {
        assert!((got - want).abs() < 1e-12);
    }
    assert!(fit.chi2_prior < 1e-12 && (fit.chi2 - 0.5).abs() < 1e-12);
    assert_eq!(fit.dof, 3);
}
#[test]
fn flat_prior_polynomial_fit_agrees_with_the_linear_solution() {
    let sigma = [0.05, 0.04, 0.05, 0.03, 0.06, 0.04, 0.05];
    let t = times(3, 9);
    let y = [2.31, 2.02, 1.78, 1.49, 1.27, 0.95, 0.74];
    let whitener = Whitener::new(&covariance(&sigma, 0.7), 7, 0.).unwrap();
    let linear = linear_fit(&line(&t), &y, &whitener, 2).unwrap();
    let model = Polynomial { t, degree: 1 };
    let fit = minimize(
        &model,
        &y,
        &whitener,
        &[Prior::Flat; 2],
        &[0., 0.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    for j in 0..2 {
        assert!((fit.p[j] - linear.beta[j]).abs() < 1e-8);
    }
    for (got, want) in fit.covariance.iter().zip(&linear.covariance) {
        assert!((got - want).abs() < 1e-7 * want.abs());
    }
    assert!((fit.chi2 - linear.chi2).abs() < 1e-8);
    assert_eq!(fit.chi2_prior, 0.);
    assert_eq!(fit.dof, linear.dof);
    assert_eq!(fit.q, chi2_q(fit.chi2, 5));
}
#[test]
fn single_exponential_fit_reproduces_the_reference_posterior() {
    let window: Vec<usize> = (4..12).collect();
    let whitener =
        Whitener::new(&submatrix(&covariance(&SIGMA, 0.6), 12, &window), 8, 1e-6).unwrap();
    let model = Exponentials {
        nexp: 1,
        channels: 1,
        times: times(5, 12),
    };
    let start = [-1.1826531162013, -0.350576604170283];
    let fit = minimize(
        &model,
        &Y[4..],
        &whitener,
        &priors(1, &Y[4..5]),
        &start,
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged && fit.iterations <= 6);
    for (got, want) in fit.p.iter().zip([-1.21087999036475, -0.395360218045603]) {
        assert!((got - want).abs() < 1e-10);
    }
    let reference = [
        0.000249740028384817,
        0.000443311171206794,
        0.000443311171206794,
        0.000890374131190169,
    ];
    for (got, want) in fit.covariance.iter().zip(reference) {
        assert!((got - want).abs() < 1e-8 * want);
    }
    assert!((fit.chi2_prior - 0.220935368796893).abs() < 1e-10);
    assert!((fit.chi2 - fit.chi2_prior - 3.55633731547114).abs() < 1e-8);
    assert_eq!(fit.dof, 8);
    assert!((fit.q.unwrap() - 0.876638780592568).abs() < 1e-9);
    let rate = fit.p[0].exp();
    assert!((rate - 0.297934984122792).abs() < 1e-10);
    assert!((rate * errors(&fit)[0] - 0.00470831575453567).abs() < 1e-10);
}
#[test]
fn two_exponential_fit_reproduces_the_reference_posterior_and_level_errors() {
    let model = Exponentials {
        nexp: 2,
        channels: 1,
        times: times(1, 12),
    };
    let fit = two_state_fit(&model, &LevenbergConfig::default());
    assert!(fit.converged && fit.iterations <= 20);
    for (got, want) in fit.p.iter().zip(TWO_STATE) {
        assert!((got - want).abs() < 1e-10);
    }
    let reference = [
        0.017156218512104,
        0.188344219232067,
        0.0348309516114681,
        0.0669080343513022,
    ];
    for (got, want) in errors(&fit).iter().zip(reference) {
        assert!((got - want).abs() < 1e-8 * want);
    }
    assert!((fit.covariance[1] - 0.00253230328495833).abs() < 1e-10);
    assert!((fit.covariance[3] + 0.000341763107703681).abs() < 1e-10);
    assert!((fit.covariance[11] + 0.000473351735417762).abs() < 1e-10);
    assert!((fit.chi2_prior - 0.640415852899023).abs() < 1e-10);
    assert!((fit.chi2 - fit.chi2_prior - 9.27951392746472).abs() < 1e-8);
    assert_eq!(fit.dof, 12);
    assert!((fit.q.unwrap() - 0.622985133442628).abs() < 1e-9);
    // Level E_1 = dE_0 + dE_1 with gradient (dE_0, dE_1, 0, 0).
    let g = [fit.p[0].exp(), fit.p[1].exp()];
    let variance: f64 = (0..4)
        .map(|c| g[c / 2] * fit.covariance[c / 2 * 4 + c % 2] * g[c % 2])
        .sum();
    assert!((g[0] + g[1] - 1.10759987793386).abs() < 1e-10);
    assert!((variance.sqrt() - 0.156002858822353).abs() < 1e-9);
    // A Gaussian prior bounds the posterior width of its parameter.
    assert!(
        errors(&fit)[..2].iter().all(|e| *e <= 3.) && errors(&fit)[2..].iter().all(|e| *e <= 5.)
    );
}
#[test]
fn provided_finite_difference_jacobian_reaches_the_same_two_exponential_optimum() {
    let model = Differenced(Exponentials {
        nexp: 2,
        channels: 1,
        times: times(1, 12),
    });
    let fit = two_state_fit(&model, &LevenbergConfig::default());
    assert!(fit.converged);
    for (got, want) in fit.p.iter().zip(TWO_STATE) {
        assert!((got - want).abs() < 5e-6);
    }
    assert!((fit.chi2 - 9.27951392746472 - 0.640415852899023).abs() < 1e-8);
    assert!((errors(&fit)[1] - 0.188344219232067).abs() < 1e-5 * 0.188344219232067);
}
#[test]
fn shared_energies_over_two_correlated_channels_reproduce_the_reference_and_sharpen_the_gap() {
    let sigma: Vec<f64> = SIGMA[..8].iter().chain(&SIGMA_B).copied().collect();
    let joint: Vec<f64> = (0..256)
        .map(|c| (c / 16, c % 16))
        .map(|(i, j)| {
            let block = if i / 8 == j / 8 { 1. } else { 0.5 };
            block * sigma[i] * sigma[j] * 0.6_f64.powi((i % 8).abs_diff(j % 8) as i32)
        })
        .collect();
    let data: Vec<f64> = Y_A.iter().chain(&Y_B).copied().collect();
    let model = Exponentials {
        nexp: 2,
        channels: 2,
        times: [times(1, 8), times(1, 8)].concat(),
    };
    let start = [
        -0.9437854926853,
        -0.250638312125354,
        -0.804872002180526,
        -1.49801918274047,
        -1.17471254864805,
        -1.867859729208,
    ];
    let whitener = Whitener::new(&joint, 16, 1e-6).unwrap();
    let prior = priors(2, &[Y_A[0], Y_B[0]]);
    let fit = minimize(
        &model,
        &data,
        &whitener,
        &prior,
        &start,
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    let reference = [
        (-1.18187070144309, 0.0115699172543818),
        (-0.12068831151568, 0.0153629773303157),
        (-0.337506811391301, 0.0157464700987515),
        (-1.32029990833769, 0.0781468329095229),
        (-1.59054402663297, 0.0208034409177154),
        (-0.121525123628682, 0.0131218878894053),
    ];
    for ((got, error), (p, sigma)) in fit.p.iter().zip(errors(&fit)).zip(reference) {
        assert!((got - p).abs() < 1e-10);
        assert!((error - sigma).abs() < 1e-8 * sigma);
    }
    assert!((fit.chi2_prior - 0.739622708181537).abs() < 1e-10);
    assert!((fit.chi2 - fit.chi2_prior - 18.6557945855162).abs() < 1e-8);
    assert_eq!(fit.dof, 16);
    assert!((fit.q.unwrap() - 0.248692193024625).abs() < 1e-9);
    let alone = Exponentials {
        nexp: 2,
        channels: 1,
        times: times(1, 8),
    };
    let whitener = Whitener::new(&covariance(&SIGMA[..8], 0.6), 8, 1e-6).unwrap();
    let single = minimize(
        &alone,
        &Y_A,
        &whitener,
        &priors(2, &Y_A[..1]),
        &start[..4],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!((single.p[0] + 1.19316865196962).abs() < 1e-10);
    assert!((errors(&single)[0] - 0.0235718011878448).abs() < 1e-9);
    assert!(errors(&fit)[0] < errors(&single)[0]);
}
#[test]
fn noise_free_data_with_flat_priors_converges_at_the_exact_parameters() {
    let t = times(1, 10);
    let y: Vec<f64> = t.iter().map(|t| 0.8 * (-0.35 * t).exp()).collect();
    let sigma: Vec<f64> = y.iter().map(|y| 0.01 * y).collect();
    let model = Exponentials {
        nexp: 1,
        channels: 1,
        times: t,
    };
    let whitener = Whitener::diagonal(&sigma).unwrap();
    let fit = minimize(
        &model,
        &y,
        &whitener,
        &[Prior::Flat; 2],
        &[-1.5, 0.3],
        &LevenbergConfig::default(),
    )
    .unwrap();
    // The residual is roundoff, so neither its direction nor its decrease can signal the end.
    assert!(fit.converged && fit.iterations <= 20);
    assert!((fit.p[0] - 0.35_f64.ln()).abs() < 1e-10 && (fit.p[1] - 0.8_f64.ln()).abs() < 1e-10);
    assert!(fit.chi2 < 1e-18);
    assert_eq!(fit.chi2_prior, 0.);
    assert_eq!(fit.dof, 8);
}
#[test]
fn exhausted_iterations_are_reported_as_unconverged_without_an_error() {
    let model = Exponentials {
        nexp: 2,
        channels: 1,
        times: times(1, 12),
    };
    let config = LevenbergConfig {
        max_iterations: 2,
        ..LevenbergConfig::default()
    };
    let fit = two_state_fit(&model, &config);
    assert!(!fit.converged);
    assert_eq!(fit.iterations, 2);
    let full = two_state_fit(&model, &LevenbergConfig::default());
    assert!(fit.chi2.is_finite() && fit.chi2 > full.chi2);
}
#[test]
fn unidentified_parameters_and_nonfinite_models_fail_numerically() {
    let whitener = Whitener::diagonal(&[1., 1.]).unwrap();
    let config = LevenbergConfig::default();
    let blind = Constant {
        parameters: 1,
        values: vec![1., 2.],
    };
    assert!(matches!(
        minimize(
            &blind,
            &[1., 2.5],
            &whitener,
            &[Prior::Flat],
            &[0.],
            &config
        ),
        Err(GasError::Numerical(_))
    ));
    let broken = Constant {
        parameters: 1,
        values: vec![1., f64::INFINITY],
    };
    let prior = [Prior::Gaussian {
        mean: 0.,
        sigma: 1.,
    }];
    assert!(matches!(
        minimize(&broken, &[1., 2.5], &whitener, &prior, &[0.], &config),
        Err(GasError::Numerical(_))
    ));
    // The model is finite at the start but its differenced Jacobian is not.
    let single = Whitener::diagonal(&[1.]).unwrap();
    assert!(matches!(
        minimize(&Root, &[0.1], &single, &prior, &[1e-7], &config),
        Err(GasError::Numerical(_))
    ));
}
#[test]
fn malformed_fit_requests_are_configuration_errors() {
    let whitener = Whitener::diagonal(&[1., 1.]).unwrap();
    let model = Polynomial {
        t: vec![0., 1.],
        degree: 0,
    };
    let config = LevenbergConfig::default();
    let flat = [Prior::Flat];
    let narrow = [Prior::Gaussian {
        mean: 0.,
        sigma: 0.,
    }];
    let configuration =
        |r: algorithmic_gas::Result<NonlinearFit>| matches!(r, Err(GasError::Configuration(_)));
    assert!(configuration(minimize(
        &model,
        &[1.],
        &whitener,
        &flat,
        &[0.],
        &config
    )));
    assert!(configuration(minimize(
        &model,
        &[1., 2.],
        &whitener,
        &[],
        &[0.],
        &config
    )));
    assert!(configuration(minimize(
        &model,
        &[1., 2.],
        &whitener,
        &flat,
        &[0., 1.],
        &config
    )));
    assert!(configuration(minimize(
        &model,
        &[1., 2.],
        &whitener,
        &flat,
        &[f64::NAN],
        &config
    )));
    assert!(configuration(minimize(
        &model,
        &[1., 2.],
        &whitener,
        &narrow,
        &[0.],
        &config
    )));
    for broken in [
        LevenbergConfig {
            max_iterations: 0,
            ..config
        },
        LevenbergConfig {
            tolerance: -1.,
            ..config
        },
        LevenbergConfig {
            initial_damping: 0.,
            ..config
        },
    ] {
        assert!(broken.validate().is_err());
        assert!(configuration(minimize(
            &model,
            &[1., 2.],
            &whitener,
            &flat,
            &[0.],
            &broken
        )));
    }
}
#[test]
fn two_by_two_pencil_has_the_closed_form_eigenvalues_and_metric_orthonormal_vectors() {
    let (a, b) = ([2., 1., 1., 2.], [2., 0., 0., 1.]);
    let eigen = generalized_symmetric(&a, &b, 2, 0.).unwrap();
    // det(a - x b) = 2 x^2 - 6 x + 3.
    let root = 3_f64.sqrt();
    assert_eq!(eigen.rank, 2);
    assert!((eigen.values[0] - (3. + root) / 2.).abs() < 1e-13);
    assert!((eigen.values[1] - (3. - root) / 2.).abs() < 1e-13);
    for k in 0..2 {
        let v = &eigen.vectors[2 * k..2 * k + 2];
        for i in 0..2 {
            let av = a[2 * i] * v[0] + a[2 * i + 1] * v[1];
            let bv = b[2 * i] * v[0] + b[2 * i + 1] * v[1];
            assert!((av - eigen.values[k] * bv).abs() < 1e-13);
        }
        for l in 0..2 {
            let overlap = bilinear(v, &b, &eigen.vectors[2 * l..2 * l + 2]);
            assert!((overlap - if k == l { 1. } else { 0. }).abs() < 1e-13);
        }
    }
}
#[test]
fn constructed_three_by_three_pencil_returns_its_eigenvalues_in_descending_order() {
    // a = S diag(3, -1, 0.5) Sᵀ and b = S Sᵀ: the pencil has exactly these eigenvalues.
    let s = [1., 0., 0., 0.5, 2., 0., -0.3, 0.4, 1.5];
    let spectrum = [3., -1., 0.5];
    let product = |weights: [f64; 3]| -> Vec<f64> {
        (0..9)
            .map(|c| {
                (0..3)
                    .map(|k| s[c / 3 * 3 + k] * weights[k] * s[c % 3 * 3 + k])
                    .sum()
            })
            .collect()
    };
    let (a, b) = (product(spectrum), product([1.; 3]));
    let eigen = generalized_symmetric(&a, &b, 3, 1e-6).unwrap();
    assert_eq!(eigen.rank, 3);
    for (got, want) in eigen.values.iter().zip([3., 0.5, -1.]) {
        assert!((got - want).abs() < 1e-12);
    }
    for k in 0..3 {
        let v = &eigen.vectors[3 * k..3 * k + 3];
        assert!((bilinear(v, &b, v) - 1.).abs() < 1e-12);
        assert!((bilinear(v, &a, v) - eigen.values[k]).abs() < 1e-12);
        assert!(
            v.iter()
                .fold(0., |s: f64, x| if x.abs() > s.abs() { *x } else { s })
                > 0.
        );
    }
}
#[test]
fn two_state_correlator_over_three_operators_has_rank_two_and_exact_decay_eigenvalues() {
    let all = [0, 1, 2];
    let metric = two_state_matrix(1., &all);
    let reference = [
        0.902947109019257,
        0.559812965124267,
        0.470672752581272,
        0.559812965124267,
        0.621193429082834,
        0.0690886281225705,
        0.470672752581272,
        0.0690886281225705,
        0.426305061046215,
    ];
    for (got, want) in metric.iter().zip(reference) {
        assert!((got - want).abs() < 1e-14);
    }
    for t in [1., 2., 3., 6.] {
        let eigen = generalized_symmetric(&two_state_matrix(t, &all), &metric, 3, 1e-3).unwrap();
        assert_eq!(eigen.rank, 2);
        assert_eq!(eigen.vectors.len(), 6);
        for (got, rate) in eigen.values.iter().zip([0.25_f64, 0.7]) {
            assert!((got - (-rate * (t - 1.)).exp()).abs() < 1e-12);
        }
    }
    // Below the reference time the descending order exchanges the state labels.
    let early = generalized_symmetric(&two_state_matrix(0., &all), &metric, 3, 1e-3).unwrap();
    assert!((early.values[0] - 2.01375270747047).abs() < 1e-12);
    assert!((early.values[1] - 1.28402541668774).abs() < 1e-12);
    let eigen = generalized_symmetric(&two_state_matrix(3., &all), &metric, 3, 1e-3).unwrap();
    let vectors = [
        0.45842142219454,
        0.0511836725531691,
        0.934647945501456,
        0.181633454226136,
        1.06742138874182,
        -0.869431442461234,
    ];
    for (got, want) in eigen.vectors.iter().zip(vectors) {
        assert!((got - want).abs() < 1e-10);
    }
    for c in 0..4 {
        let (v, w) = (
            &eigen.vectors[c / 2 * 3..][..3],
            &eigen.vectors[c % 2 * 3..][..3],
        );
        assert!((bilinear(v, &metric, w) - if c / 2 == c % 2 { 1. } else { 0. }).abs() < 1e-12);
    }
}
#[test]
fn eigenvalues_are_invariant_under_mixing_and_rescaling_of_the_operators() {
    let all = [0, 1, 2];
    let (a, b) = (two_state_matrix(4., &all), two_state_matrix(1., &all));
    let plain = generalized_symmetric(&a, &b, 3, 1e-3).unwrap();
    let mixing = [1., 0.2, 0., -0.3, 1., 0.5, 0.1, 0., 2.];
    let scaling = [1e6, 0., 0., 0., 1., 0., 0., 0., 1e-4];
    for m in [mixing, scaling] {
        let other =
            generalized_symmetric(&congruence(&m, &a, 3), &congruence(&m, &b, 3), 3, 1e-3).unwrap();
        assert_eq!(other.rank, 2);
        for (got, want) in other.values.iter().zip(&plain.values) {
            assert!((got - want).abs() < 1e-10);
        }
    }
}
#[test]
fn small_eigenvalues_keep_their_relative_accuracy() {
    let pair = [0, 1];
    let eigen = generalized_symmetric(
        &two_state_matrix(30., &pair),
        &two_state_matrix(1., &pair),
        2,
        1e-3,
    )
    .unwrap();
    for (got, rate) in eigen.values.iter().zip([0.25_f64, 0.7]) {
        let want = (-rate * 29.).exp();
        assert!((got - want).abs() < 1e-6 * want);
    }
}
#[test]
fn relative_cut_on_the_normalized_metric_selects_the_rank() {
    let a = [2., 1., 1., 2.];
    // The unit-diagonal metric has eigenvalues 1.999 and 0.001: a ratio of 5.0e-4.
    let b = [4., 1.998, 1.998, 1.];
    assert_eq!(generalized_symmetric(&a, &b, 2, 1e-4).unwrap().rank, 2);
    let eigen = generalized_symmetric(&a, &b, 2, 1e-3).unwrap();
    assert_eq!(eigen.rank, 1);
    // The kept direction is (1/2, 1)/sqrt(2 * 1.999) in operator space.
    let v = [0.5 / 3.998_f64.sqrt(), 1. / 3.998_f64.sqrt()];
    assert!((eigen.values[0] - bilinear(&v, &a, &v)).abs() < 1e-13);
    assert!((eigen.vectors[0] - v[0]).abs() < 1e-13 && (eigen.vectors[1] - v[1]).abs() < 1e-13);
    assert!((bilinear(&eigen.vectors, &b, &eigen.vectors) - 1.).abs() < 1e-13);
}
#[test]
fn traceless_and_vanishing_matrices_are_solved_at_their_own_scale() {
    let identity = [1., 0., 0., 1.];
    let swap = generalized_symmetric(&[0., 1e-9, 1e-9, 0.], &identity, 2, 0.).unwrap();
    assert!((swap.values[0] - 1e-9).abs() < 1e-22 && (swap.values[1] + 1e-9).abs() < 1e-22);
    let zero = generalized_symmetric(&[0.; 4], &identity, 2, 0.).unwrap();
    assert_eq!(zero.values, vec![0., 0.]);
    assert_eq!(zero.vectors, identity.to_vec());
}
#[test]
fn indefinite_metrics_fail_numerically_and_malformed_pencils_are_configuration_errors() {
    let a = [2., 1., 1., 2.];
    for b in [[1., 2., 2., 1.], [0., 0., 0., 1.], [-1., 0., 0., 1.]] {
        assert!(matches!(
            generalized_symmetric(&a, &b, 2, 1e-3),
            Err(GasError::Numerical(_))
        ));
    }
    let identity = [1., 0., 0., 1.];
    for (a, b, n, cut) in [
        (&a[..], &identity[..], 2, 1.),
        (&a[..], &identity[..], 2, -1e-3),
        (&a[..], &identity[..], 2, f64::NAN),
        (&a[..3], &identity[..], 2, 1e-3),
        (&a[..], &identity[..], 0, 1e-3),
        (&[2., 1., 1.1, 2.][..], &identity[..], 2, 1e-3),
        (&a[..], &[1., 0.1, 0., 1.][..], 2, 1e-3),
        (&[2., f64::NAN, f64::NAN, 2.][..], &identity[..], 2, 1e-3),
    ] {
        assert!(matches!(
            generalized_symmetric(a, b, n, cut),
            Err(GasError::Configuration(_))
        ));
    }
}
#[test]
fn saturated_fit_has_zero_chi2_no_degrees_of_freedom_and_the_interpolating_covariance() {
    let t = [1., 3.];
    let whitener = Whitener::new(&covariance(&[0.1, 0.2], 0.5), 2, 0.).unwrap();
    let fit = linear_fit(&line(&t), &[2., -1.], &whitener, 2).unwrap();
    assert!((fit.beta[0] - 3.5).abs() < 1e-12 && (fit.beta[1] + 1.5).abs() < 1e-12);
    assert!(fit.chi2 < 1e-20);
    assert_eq!(fit.dof, 0);
    // beta = X⁻¹ y, so its covariance is X⁻¹ C X⁻ᵀ with X⁻¹ = [[3, -1], [-1, 1]] / 2.
    let inverse = [1.5, -0.5, -0.5, 0.5];
    let want = congruence(&inverse, &covariance(&[0.1, 0.2], 0.5), 2);
    for (got, want) in fit.covariance.iter().zip(want) {
        assert!((got - want).abs() < 1e-12 * want.abs());
    }
}
#[test]
fn floored_whitener_enters_the_linear_fit_through_its_conditioned_precision() {
    // Correlation eigenvalues 1.9 on (1, 1)/sqrt 2 and 0.1 on (1, -1)/sqrt 2; the cut 0.2 lifts the
    // second to 0.38.
    let sigma = [0.5, 0.2];
    let whitener = Whitener::new(&covariance(&sigma, 0.9), 2, 0.2).unwrap();
    assert_eq!(whitener.rank(), 1);
    let precision: Vec<f64> = (0..4)
        .map(|c| {
            let sign = if c / 2 == c % 2 { 1. } else { -1. };
            0.5 * (1. / 1.9 + sign / 0.38) / (sigma[c / 2] * sigma[c % 2])
        })
        .collect();
    let (one, y) = ([1., 1.], [1.2, 0.7]);
    let fit = linear_fit(&one, &y, &whitener, 1).unwrap();
    let information = bilinear(&one, &precision, &one);
    let mean = bilinear(&one, &precision, &y) / information;
    let r = [y[0] - mean, y[1] - mean];
    assert!((fit.beta[0] - mean).abs() < 1e-13);
    assert!((fit.covariance[0] - 1. / information).abs() < 1e-13 / information);
    assert!((fit.chi2 - bilinear(&r, &precision, &r)).abs() < 1e-12);
    assert_eq!(fit.dof, 1);
    let sharp = linear_fit(
        &one,
        &y,
        &Whitener::new(&covariance(&sigma, 0.9), 2, 0.).unwrap(),
        1,
    )
    .unwrap();
    assert!(fit.chi2 < sharp.chi2 && fit.covariance[0] > sharp.covariance[0]);
}
#[test]
fn gaussian_prior_acts_as_one_more_datum_and_cancels_its_parameter_in_the_degrees_of_freedom() {
    let t = vec![0., 1., 2.];
    let (y, sigma) = ([1.1, 1.9, 3.2], [0.1, 0.2, 0.1]);
    let prior = [
        Prior::Flat,
        Prior::Gaussian {
            mean: 0.5,
            sigma: 0.3,
        },
    ];
    let fit = minimize(
        &Polynomial {
            t: t.clone(),
            degree: 1,
        },
        &y,
        &Whitener::diagonal(&sigma).unwrap(),
        &prior,
        &[0., 0.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    // The same posterior as a linear fit with the prior appended as the row (0, 1) = 0.5 ± 0.3.
    let design = [line(&t), vec![0., 1.]].concat();
    let augmented = Whitener::diagonal(&[0.1, 0.2, 0.1, 0.3]).unwrap();
    let linear = linear_fit(&design, &[1.1, 1.9, 3.2, 0.5], &augmented, 2).unwrap();
    assert!(fit.converged);
    for j in 0..2 {
        assert!((fit.p[j] - linear.beta[j]).abs() < 1e-10);
    }
    for (got, want) in fit.covariance.iter().zip(&linear.covariance) {
        assert!((got - want).abs() < 1e-10 * want.abs());
    }
    assert!((fit.chi2 - linear.chi2).abs() < 1e-10 * linear.chi2);
    assert!((fit.chi2_prior - ((fit.p[1] - 0.5) / 0.3).powi(2)).abs() < 1e-13);
    assert!(fit.chi2_prior > 0.1 && fit.chi2_prior < fit.chi2);
    assert_eq!(fit.dof, 2);
    assert_eq!(fit.q, chi2_q(fit.chi2, 2));
    assert!(fit.covariance[3] < 0.09);
}
#[test]
fn nonlinear_fit_with_a_residual_reaches_the_closed_form_minimum() {
    // exp(p) fitted to equal-error data is their mean.
    let y = [1., 2., 4., 5.];
    let fit = minimize(
        &Level(4),
        &y,
        &Whitener::diagonal(&[0.5; 4]).unwrap(),
        &[Prior::Flat],
        &[-2.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert!((fit.p[0] - 3_f64.ln()).abs() < 1e-9);
    // Information 4 exp(2p) / 0.25 = 144, chi2 = (4 + 1 + 1 + 4) / 0.25.
    assert!((fit.covariance[0] - 1. / 144.).abs() < 1e-9);
    assert!((fit.chi2 - 40.).abs() < 1e-9);
    assert_eq!(fit.chi2_prior, 0.);
    assert_eq!(fit.dof, 3);
    assert_eq!(fit.q, chi2_q(fit.chi2, 3));
}
#[test]
fn trial_steps_into_an_undefined_region_are_rejected_and_a_saturated_fit_has_no_probability() {
    // The first undamped-like step from 4 would land at -3.6, where the model is NaN.
    let fit = minimize(
        &Root,
        &[0.1],
        &Whitener::diagonal(&[1.]).unwrap(),
        &[Prior::Flat],
        &[4.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert!((fit.p[0] - 0.01).abs() < 1e-9);
    assert!(fit.chi2 < 1e-20);
    assert_eq!(fit.dof, 0);
    assert_eq!(fit.q, None);
}
#[test]
fn curved_valley_is_descended_to_its_exact_minimum() {
    let fit = minimize(
        &Valley,
        &[0., 1.],
        &Whitener::diagonal(&[1., 1.]).unwrap(),
        &[Prior::Flat; 2],
        &[-1.2, 1.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged && fit.iterations < 60);
    assert!((fit.p[0] - 1.).abs() < 1e-9 && (fit.p[1] - 1.).abs() < 1e-9);
    assert!(fit.chi2 < 1e-18);
    // J = [[-20, 10], [1, 0]] at the minimum: (JᵀJ)⁻¹ = [[1, 2], [2, 4.01]].
    for (got, want) in fit.covariance.iter().zip([1., 2., 2., 4.01]) {
        assert!((got - want).abs() < 1e-7);
    }
}
#[test]
fn undamped_refinement_does_not_leave_a_minimum_where_gauss_newton_is_unstable() {
    let fit = minimize(
        &Curved(-2.),
        &[-1., 1.],
        &Whitener::diagonal(&[1., 1.]).unwrap(),
        &[Prior::Flat],
        &[0.4],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.p[0].abs() < 2e-8);
    assert!((fit.chi2 - 2.).abs() < 1e-12);
    assert!((fit.covariance[0] - 0.5).abs() < 1e-7);
}
#[test]
fn fits_and_eigenproblems_are_bitwise_reproducible() {
    let model = Exponentials {
        nexp: 2,
        channels: 1,
        times: times(1, 12),
    };
    let config = LevenbergConfig::default();
    assert_eq!(
        two_state_fit(&model, &config),
        two_state_fit(&model, &config)
    );
    let all = [0, 1, 2];
    let (a, b) = (two_state_matrix(3., &all), two_state_matrix(1., &all));
    assert_eq!(
        generalized_symmetric(&a, &b, 3, 1e-3).unwrap(),
        generalized_symmetric(&a, &b, 3, 1e-3).unwrap()
    );
}
#[test]
fn equal_matrices_have_unit_eigenvalues_and_metric_orthonormal_vectors() {
    let b = two_state_matrix(1., &[0, 1]);
    let eigen = generalized_symmetric(&b, &b, 2, 1e-6).unwrap();
    assert_eq!(eigen.rank, 2);
    for c in 0..4 {
        let (v, w) = (
            &eigen.vectors[c / 2 * 2..][..2],
            &eigen.vectors[c % 2 * 2..][..2],
        );
        assert!((bilinear(v, &b, w) - if c / 2 == c % 2 { 1. } else { 0. }).abs() < 1e-13);
    }
    assert!(eigen.values.iter().all(|x| (x - 1.).abs() < 1e-13));
}
#[test]
fn six_states_seen_by_six_operators_decay_from_the_reference_time_at_their_own_rates() {
    let n = 6;
    let rate = |s: usize| 0.2 * (s + 1) as f64;
    let overlap = |i: usize, s: usize| {
        if i == s {
            1.
        } else {
            ((5 * i + 3 * s) % 7) as f64 / 10. - 0.3
        }
    };
    let matrix = |t: f64| -> Vec<f64> {
        (0..n * n)
            .map(|c| {
                (0..n)
                    .map(|s| overlap(c / n, s) * (-rate(s) * t).exp() * overlap(c % n, s))
                    .sum()
            })
            .collect()
    };
    let metric = matrix(2.);
    for t in [3., 5., 9.] {
        let eigen = generalized_symmetric(&matrix(t), &metric, n, 1e-9).unwrap();
        assert_eq!(eigen.rank, n);
        for (s, got) in eigen.values.iter().enumerate() {
            let want = (-rate(s) * (t - 2.)).exp();
            assert!((got - want).abs() < 1e-10 * want);
        }
        for c in 0..n * n {
            let (v, w) = (
                &eigen.vectors[c / n * n..][..n],
                &eigen.vectors[c % n * n..][..n],
            );
            assert!((bilinear(v, &metric, w) - if c / n == c % n { 1. } else { 0. }).abs() < 1e-9);
        }
    }
}
#[test]
fn provided_jacobian_keeps_its_accuracy_for_parameters_near_zero() {
    let exact = Exponentials {
        nexp: 2,
        channels: 1,
        times: times(1, 6),
    };
    let p = [1e-9, -0.4, 0., -1e-7];
    let (mut want, mut got) = (vec![0.; 24], vec![0.; 24]);
    exact.jacobian(&p, &mut want);
    Differenced(exact).jacobian(&p, &mut got);
    for (got, want) in got.iter().zip(&want) {
        assert!((got - want).abs() < 1e-8 * want.abs());
    }
}
#[test]
fn floored_whitener_conditions_the_nonlinear_fit_without_changing_its_degrees_of_freedom() {
    // The conditioned precision of the floored linear fit above: 1.9 and the lifted 0.38.
    let sigma = [0.5, 0.2];
    let whitener = Whitener::new(&covariance(&sigma, 0.9), 2, 0.2).unwrap();
    assert_eq!((whitener.n(), whitener.rank()), (2, 1));
    let precision: Vec<f64> = (0..4)
        .map(|c| {
            let sign = if c / 2 == c % 2 { 1. } else { -1. };
            0.5 * (1. / 1.9 + sign / 0.38) / (sigma[c / 2] * sigma[c % 2])
        })
        .collect();
    let (one, y) = ([1., 1.], [1.2, 0.7]);
    let model = Polynomial {
        t: vec![0., 0.],
        degree: 0,
    };
    let fit = minimize(
        &model,
        &y,
        &whitener,
        &[Prior::Flat],
        &[0.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    let information = bilinear(&one, &precision, &one);
    let mean = bilinear(&one, &precision, &y) / information;
    let r = [y[0] - mean, y[1] - mean];
    assert!(fit.converged);
    assert!((fit.p[0] - mean).abs() < 1e-9);
    assert!((fit.covariance[0] - 1. / information).abs() < 1e-8 / information);
    assert!((fit.chi2 - bilinear(&r, &precision, &r)).abs() < 1e-9);
    // Two data points and one flat parameter, whatever the rank of the whitener.
    assert_eq!(fit.dof, 1);
    assert_eq!(fit.q, chi2_q(fit.chi2, 1));
}
#[test]
fn exhausted_damping_returns_the_last_accepted_point_as_unconverged() {
    // The data pull the parameter beyond 1, where every trial step is undefined.
    let fit = minimize(
        &Ledge,
        &[3., 3.],
        &Whitener::diagonal(&[1., 1.]).unwrap(),
        &[Prior::Flat],
        &[1.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.iterations, 1);
    assert_eq!(fit.p, vec![1.]);
    assert_eq!(fit.chi2, 8.);
    assert_eq!(fit.covariance, vec![0.5]);
    assert_eq!(fit.dof, 1);
    assert_eq!(fit.q, chi2_q(8., 1));
}
#[test]
fn start_at_the_minimum_converges_in_the_first_iteration_without_moving() {
    // The residual (-1, 1) is orthogonal to the Jacobian column (1, 1) at p = 0.
    let fit = minimize(
        &Curved(-2.),
        &[-1., 1.],
        &Whitener::diagonal(&[1., 1.]).unwrap(),
        &[Prior::Flat],
        &[0.],
        &LevenbergConfig::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert_eq!(fit.iterations, 1);
    assert_eq!(fit.p, vec![0.]);
    assert_eq!(fit.chi2, 2.);
    assert_eq!(fit.covariance, vec![0.5]);
}
#[test]
fn nonfinite_jacobian_is_named_as_the_cause_of_the_failure() {
    let prior = [Prior::Gaussian {
        mean: 0.,
        sigma: 1.,
    }];
    let error = minimize(
        &Root,
        &[0.1],
        &Whitener::diagonal(&[1.]).unwrap(),
        &prior,
        &[1e-7],
        &LevenbergConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(error, GasError::Numerical(_)));
    assert!(error.to_string().contains("Jacobian"), "{error}");
}
#[test]
fn cubic_on_a_distant_window_is_inside_the_conditioning_limit_of_the_linear_fit() {
    // Variance inflation 4.3e6 on t = 20..=30: far from collinear by the 1e13 standard.
    let t = times(20, 30);
    let coefficients = [1., 0.5, -0.02, 0.001];
    let design: Vec<f64> = t.iter().flat_map(|&t| [1., t, t * t, t * t * t]).collect();
    let y: Vec<f64> = design
        .chunks_exact(4)
        .map(|row| row.iter().zip(coefficients).map(|(x, c)| x * c).sum())
        .collect();
    let fit = linear_fit(&design, &y, &Whitener::diagonal(&[0.1; 11]).unwrap(), 4).unwrap();
    for (got, want) in fit.beta.iter().zip(coefficients) {
        assert!((got - want).abs() < 1e-6 * want.abs());
    }
    assert!(fit.chi2 < 1e-10);
    assert_eq!(fit.dof, 7);
}
#[test]
fn nonfinite_mismatched_and_oversized_inputs_are_configuration_errors_in_every_solver() {
    let configuration =
        |r: algorithmic_gas::Result<()>| matches!(r, Err(GasError::Configuration(_)));
    let pair = Whitener::diagonal(&[1., 1.]).unwrap();
    let design = line(&[1., 2.]);
    for broken in [f64::NAN, f64::INFINITY] {
        let mut x = design.clone();
        x[3] = broken;
        assert!(configuration(
            linear_fit(&x, &[1., 2.], &pair, 2).map(|_| ())
        ));
    }
    let model = Polynomial {
        t: vec![0., 1.],
        degree: 0,
    };
    let config = LevenbergConfig::default();
    let flat = [Prior::Flat];
    for data in [[1., f64::NAN], [f64::NEG_INFINITY, 2.]] {
        assert!(configuration(
            minimize(&model, &data, &pair, &flat, &[0.], &config).map(|_| ())
        ));
    }
    let triple = Whitener::diagonal(&[1., 1., 1.]).unwrap();
    assert!(configuration(
        minimize(&model, &[1., 2.], &triple, &flat, &[0.], &config).map(|_| ())
    ));
    let (a, identity) = ([2., 1., 1., 2.], [1., 0., 0., 1.]);
    assert!(configuration(
        generalized_symmetric(&a, &[f64::INFINITY, 0., 0., 1.], 2, 1e-3).map(|_| ())
    ));
    assert!(configuration(
        generalized_symmetric(&a, &identity[..3], 2, 1e-3).map(|_| ())
    ));
    // 65 parameters or operators are over the capacity of 64.
    let many = Whitener::diagonal(&[1.; 65]).unwrap();
    let unit = |rows: usize, columns: usize| -> Vec<f64> {
        (0..rows * columns)
            .map(|c| if c / columns == c % columns { 1. } else { 0. })
            .collect()
    };
    assert!(configuration(
        linear_fit(&unit(65, 65), &[1.; 65], &many, 65).map(|_| ())
    ));
    assert!(linear_fit(&unit(65, 64), &[1.; 65], &many, 64).is_ok());
    let wide = Polynomial {
        t: times(1, 65),
        degree: 64,
    };
    assert!(configuration(
        minimize(
            &wide,
            &[1.; 65],
            &many,
            &[Prior::Flat; 65],
            &[0.; 65],
            &config
        )
        .map(|_| ())
    ));
    assert!(configuration(
        generalized_symmetric(&unit(65, 65), &unit(65, 65), 65, 1e-3).map(|_| ())
    ));
    assert_eq!(
        generalized_symmetric(&unit(64, 64), &unit(64, 64), 64, 1e-3)
            .unwrap()
            .rank,
        64
    );
}
#[test]
fn metric_directions_below_the_roundoff_floor_are_discarded_even_without_a_cut() {
    let a = [2., 1., 1., 2.];
    // Unit-diagonal metric with eigenvalues 2 - e on (1, 1) and e on (1, -1).
    let metric = |e: f64| [1., 1. - e, 1. - e, 1.];
    // A ratio of 1e-13 is under the floor 1e-12: only (1, 1) / sqrt(2 (2 - e)) is kept.
    let eigen = generalized_symmetric(&a, &metric(2e-13), 2, 0.).unwrap();
    assert_eq!(eigen.rank, 1);
    assert!((eigen.values[0] - 1.5).abs() < 1e-12);
    assert!((eigen.vectors[0] - 0.5).abs() < 1e-12 && (eigen.vectors[1] - 0.5).abs() < 1e-12);
    // A ratio of 1e-11 is kept, with (1, -1) a (1, -1)ᵀ / (2 e) = 1 / e.
    let eigen = generalized_symmetric(&a, &metric(2e-11), 2, 0.).unwrap();
    assert_eq!(eigen.rank, 2);
    assert!((eigen.values[0] - 5e10).abs() < 1e-4 * 5e10);
    assert!((eigen.values[1] - 1.5).abs() < 1e-9);
}
#[test]
fn negative_metric_eigenvalues_are_roundoff_up_to_1e_8_and_indefinite_beyond() {
    let a = [2., 1., 1., 2.];
    // Unit-diagonal metric with eigenvalues 2 + e and -e.
    let metric = |e: f64| [1., 1. + e, 1. + e, 1.];
    let eigen = generalized_symmetric(&a, &metric(1e-10), 2, 0.).unwrap();
    assert_eq!(eigen.rank, 1);
    assert!((eigen.values[0] - 1.5).abs() < 1e-9);
    assert!(matches!(
        generalized_symmetric(&a, &metric(1e-6), 2, 0.),
        Err(GasError::Numerical(_))
    ));
}
#[test]
fn single_damped_iteration_takes_the_marquardt_step_and_is_not_refined() {
    // f = (p, p) with errors 1/2: A = 8 and g = -16 at p = 0, so (1 + 1e-3) A d = 16.
    let config = LevenbergConfig {
        max_iterations: 1,
        ..LevenbergConfig::default()
    };
    let fit = minimize(
        &Curved(0.),
        &[2., 2.],
        &Whitener::diagonal(&[0.5, 0.5]).unwrap(),
        &[Prior::Flat],
        &[0.],
        &config,
    )
    .unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.iterations, 1);
    let p = 2. / 1.001;
    assert!((fit.p[0] - p).abs() < 1e-15);
    assert!((fit.chi2 - 8. * (2. - p) * (2. - p)).abs() < 1e-18);
    assert!((fit.covariance[0] - 0.125).abs() < 1e-16);
}
