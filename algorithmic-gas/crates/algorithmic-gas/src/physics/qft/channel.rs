//! Fermionic attenuation through an explicit exterior-algebra vacuum dilation.
use super::{math::*, rate, steps};
use crate::physics::partvi::{ExperimentRequest, ExperimentResult, Series};
use serde_json::json;
/// K_loss is the environment-occupation matrix element of the exterior-power
/// isometry e_j -> eta_j e_j + sqrt(1-eta_j²) f_j. The sign counts crossings
/// needed to put system modes before environment modes.
pub fn attenuation_kraus(eta: &[f64]) -> Vec<Vec<C>> {
    let m = eta.len();
    let n = 1usize << m;
    let mut operators = vec![vec![C::ZERO; n * n]; n];
    for input in 0usize..n {
        for (loss, operator) in operators.iter_mut().enumerate() {
            if loss & !input != 0 {
                continue;
            }
            let output = input & !loss;
            let mut coefficient = 1.;
            let mut crossings = 0;
            for (j, &attenuation) in eta.iter().enumerate() {
                if input & (1 << j) == 0 {
                    continue;
                }
                if loss & (1 << j) != 0 {
                    coefficient *= (1. - attenuation * attenuation).max(0.).sqrt();
                    crossings += (output >> (j + 1)).count_ones();
                } else {
                    coefficient *= attenuation;
                }
            }
            if crossings % 2 != 0 {
                coefficient = -coefficient;
            }
            operator[output * n + input] = C::from(coefficient);
        }
    }
    operators
}
pub fn apply(kraus: &[Vec<C>], x: &[C], n: usize) -> Vec<C> {
    let mut y = vec![C::ZERO; n * n];
    for k in kraus {
        let term = mul(&mul(&adjoint(k, n), x, n), k, n);
        for i in 0..n * n {
            y[i] = y[i] + term[i];
        }
    }
    y
}
pub fn analyze(r: &ExperimentRequest) -> ExperimentResult {
    let m = r.usize("modes", 2).clamp(1, 3);
    let eta: Vec<f64> = (0..m)
        .map(|i| (-rate(r) * (1. + 0.2 * i as f64)).exp())
        .collect();
    analyze_contraction(r, &eta)
}
/// Canonical singular-value form of a measured or supplied finite contraction.
/// The source and target bases are documented by the caller.
pub(super) fn analyze_contraction(r: &ExperimentRequest, eta: &[f64]) -> ExperimentResult {
    let m = eta.len();
    let n = 1usize << m;
    let k = attenuation_kraus(eta);
    let gram: Vec<C> = (0..n * n).map(|x| dot(&k[x / n], &k[x % n])).collect();
    let eigen = hermitian_eigenvalues(&gram, n);
    let mut choi = vec![C::ZERO; n * n * n * n];
    for op in &k {
        for i in 0..n * n {
            for j in 0..n * n {
                choi[i * n * n + j] =
                    choi[i * n * n + j] + op[i % n * n + i / n] * op[j % n * n + j / n].conj();
            }
        }
    }
    let mut residual: f64 = 0.;
    let mut vacuum_defects = vec![];
    let mut predicted = vec![];
    for (j, &attenuation) in eta.iter().enumerate() {
        let a = adjoint(&creator(m, j), n);
        let transformed = apply(&k, &a, n);
        let target: Vec<C> = a.iter().map(|&x| x * attenuation).collect();
        residual = residual.max(difference(&transformed, &target));
        let aad = mul(&a, &adjoint(&a, n), n);
        let actual = apply(&k, &aad, n);
        let product = mul(&transformed, &adjoint(&transformed, n), n);
        vacuum_defects.push([j as f64, (actual[0] - product[0]).re]);
        predicted.push([j as f64, 1. - attenuation * attenuation]);
    }
    let mut o = ExperimentResult::new(
        4,
        "Multimode recorded CAR attenuation",
        "Exterior-power vacuum dilation with explicit fermionic Kraus signs",
    );
    o.metric(
        "Unital residual",
        difference(&apply(&k, &identity(n), n), &identity(n)),
        "",
    )
    .metric(
        "Minimum Choi eigenvalue",
        0f64.min(eigen.iter().copied().fold(f64::INFINITY, f64::min)),
        "",
    )
    .metric("Linear CAR contraction residual", residual, "")
    .metric("Vacuum multiplicative defect", vacuum_defects[0][1], "")
    .metric("Predicted defect", predicted[0][1], "")
    .metric("Fock dimension", n as f64, "states")
    .metric("Choi dimension", (n * n) as f64, "states");
    o.plot(
        "Modewise multiplicative defect",
        "mode",
        "defect",
        vec![
            Series::line("Explicit channel", vacuum_defects),
            Series::line("I minus C* C", predicted),
        ],
    );
    let mut exterior = vec![];
    for order in 0..=m {
        let occupied = (1usize << order) - 1;
        let initial = identity(n);
        let mut word = initial;
        for j in 0..order {
            word = mul(&word, &creator(m, j), n);
        }
        let transformed = apply(&k, &word, n);
        let coefficient = (0..order).map(|j| eta[j]).product::<f64>();
        let direct = transformed[occupied * n];
        let original = word[occupied * n];
        exterior.push([order as f64, (direct - original * coefficient).abs()]);
    }
    o.plot(
        "Exterior-sector contraction",
        "degree",
        "matrix-element residual",
        vec![Series::line("Kraus lift versus exterior product", exterior)],
    );
    let decay: Vec<[f64; 2]> = (0..=steps(r))
        .map(|t| [t as f64, eta.iter().product::<f64>().powi(t as i32)])
        .collect();
    o.plot(
        "Top exterior sector through time",
        "steps",
        "amplitude",
        vec![Series::line("Product of one-particle contractions", decay)],
    );
    o.details = json!({"modes":m,"one_particle_contraction":eta,"nonzero_choi_spectrum":eigen,"choi_real":choi.iter().map(|z|z.re).collect::<Vec<_>>(),"choi_imaginary":choi.iter().map(|z|z.im).collect::<Vec<_>>(),"choi_zero_eigenvalue_multiplicity":n*n-n,"kraus_count":n,"construction":"Exterior-power isometry followed by the environment-vacuum partial trace"});
    o
}
