use algorithmic_gas::{
    GasConfig, GasError,
    physics::{
        qft::math::{C, Rng, adjoint, dot, identity, mean, mul, rmul},
        spectroscopy::{
            config::{
                ChannelSpec, ChiralProjector, DiracGamma, MeasurementConfig, PhaseLink,
                PropagatorConfig, RoleClass, TensorMode,
            },
            contract::{
                Capabilities, Element, ElementKind, ExchangeParity, FrameState, LocalOperator,
                OperatorContext, Record, Requirements, Signature, SpatialParity, WalkerRole,
                channel_id,
            },
            fields::dirac::{Matrix4, Spinor, bilinear, embed, gamma, gamma5, lift, sigma},
        },
    },
};

const I: C = C { re: 0., im: 1. };
/// Slots of the reference colours in `reference()`: a source pair `(A, B)`,
/// the same slots at a sink time `(A2, B2)` and the exact hand case `(P, Q)`.
const A: u32 = 0;
const B: u32 = 1;
const A2: u32 = 2;
const B2: u32 = 3;
const P: u32 = 4;
const Q: u32 = 5;

/// `c_a = F_a e^{i v_a} / |F|`: the colour state at unit phase factor.
fn color(force: [f64; 3], velocity: [f64; 3]) -> [C; 3] {
    let norm = force.iter().map(|f| f * f).sum::<f64>().sqrt();
    std::array::from_fn(|a| C::phase(velocity[a]) * (force[a] / norm))
}
fn state(colors: &[[C; 3]]) -> FrameState {
    FrameState {
        n: colors.len(),
        d: 3,
        color: colors.concat(),
        color_valid: vec![true; colors.len()],
        eligible: vec![true; colors.len()],
        ..FrameState::default()
    }
}
fn reference() -> FrameState {
    let h = 0.5_f64.sqrt();
    state(&[
        color([1., 2., -2.], [0.3, -0.7, 1.1]),
        color([2., -1., 2.], [-0.4, 0.9, 0.2]),
        color([1.5, 1., -2.5], [0.1, -0.5, 0.9]),
        color([1., -2., 2.], [-0.6, 0.4, 0.3]),
        [C::new(0., h), C::new(h, 0.), C::ZERO],
        [C::new(0.5, 0.), C::new(0., 0.5), C::new(0.5, 0.5)],
    ])
}
fn colors(state: &FrameState, walker: u32) -> [C; 3] {
    std::array::from_fn(|a| state.color[walker as usize * 3 + a])
}
fn spinor(state: &FrameState, walker: u32) -> Spinor {
    lift(&colors(state, walker)).unwrap()
}
fn pair(i: u32, j: u32) -> Element {
    Element {
        walkers: [i, j, i],
        kind: ElementKind::DistancePair,
        weight: 1.,
        generation: [0; 3],
    }
}
fn dirac(gamma: DiracGamma) -> ChannelSpec {
    ChannelSpec::Dirac {
        gamma,
        projector: ChiralProjector::None,
        pairs: RoleClass::Any,
        link: PhaseLink::None,
    }
}
fn projected(gamma: DiracGamma, projector: ChiralProjector) -> ChannelSpec {
    ChannelSpec::Dirac {
        gamma,
        projector,
        pairs: RoleClass::Any,
        link: PhaseLink::None,
    }
}
fn tensor(mode: TensorMode) -> ChannelSpec {
    ChannelSpec::Tensor { mode }
}
fn signature_in(spec: &ChannelSpec, kind: ElementKind) -> algorithmic_gas::Result<Signature> {
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let capabilities = Capabilities::nominal();
    spec.signature(
        kind,
        &OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        },
    )
}
fn signature(spec: &ChannelSpec) -> Signature {
    signature_in(spec, ElementKind::DistancePair).unwrap()
}
/// Values of `spec` on the pair `(i, j)`, `None` when the element is masked.
fn measured(
    spec: &ChannelSpec,
    measurement: &MeasurementConfig,
    state: &FrameState,
    i: u32,
    j: u32,
) -> Option<Vec<f64>> {
    measured_on(spec, &GasConfig::default(), measurement, state, &pair(i, j))
}
/// Values of `spec` on `element` under the given gas parameters.
fn measured_on(
    spec: &ChannelSpec,
    gas: &GasConfig,
    measurement: &MeasurementConfig,
    state: &FrameState,
    element: &Element,
) -> Option<Vec<f64>> {
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas,
        measurement,
        capabilities: &capabilities,
    };
    let components = spec.signature(element.kind, &context).unwrap().components;
    let mut out = vec![0.; components];
    spec.evaluate(element, state, &context, &mut out)
        .then_some(out)
}
fn evaluated(spec: &ChannelSpec, state: &FrameState, i: u32, j: u32) -> Option<Vec<f64>> {
    measured(spec, &MeasurementConfig::default(), state, i, j)
}
fn value(spec: &ChannelSpec, state: &FrameState, i: u32, j: u32) -> Vec<f64> {
    evaluated(spec, state, i, j).unwrap()
}
fn contracted(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}
/// Row-major matrix from entries `0`, `1`, `-1`, `i`, `-i`; `|` separates rows.
fn table(text: &str) -> Matrix4 {
    let entries: Vec<C> = text
        .split_whitespace()
        .filter(|&token| token != "|")
        .map(|token| match token {
            "0" => C::ZERO,
            "1" => C::ONE,
            "-1" => -C::ONE,
            "i" => I,
            "-i" => -I,
            other => panic!("unknown matrix entry {other}"),
        })
        .collect();
    entries.try_into().unwrap()
}
fn times(a: &Matrix4, b: &Matrix4) -> Matrix4 {
    mul(a, b, 4).try_into().unwrap()
}
/// `Γ ψ`, written out here so that it is independent of the library.
fn applied(matrix: &Matrix4, psi: &Spinor) -> Spinor {
    std::array::from_fn(|r| (0..4).fold(C::ZERO, |s, j| s + matrix[r * 4 + j] * psi[j]))
}
fn dagger(a: &Matrix4) -> Matrix4 {
    adjoint(a, 4).try_into().unwrap()
}
fn scaled(a: &Matrix4, factor: C) -> Matrix4 {
    a.map(|z| factor * z)
}
fn sum(a: &Matrix4, b: &Matrix4) -> Matrix4 {
    std::array::from_fn(|k| a[k] + b[k])
}
fn unit() -> Matrix4 {
    identity(4).try_into().unwrap()
}
fn projector(sign: f64) -> Matrix4 {
    scaled(
        &sum(&unit(), &scaled(&gamma5(), C::from(sign))),
        C::from(0.5),
    )
}
/// `Γ` of every component of an arm, in kept order.
fn matrices(arm: DiracGamma) -> Vec<Matrix4> {
    match arm {
        DiracGamma::Scalar => vec![unit()],
        DiracGamma::Pseudoscalar => vec![gamma5()],
        DiracGamma::Vector => (1..4).map(gamma).collect(),
        DiracGamma::Axial => (1..4).map(|k| times(&gamma5(), &gamma(k))).collect(),
        DiracGamma::Tensor => vec![sigma(1, 2), sigma(1, 3), sigma(2, 3)],
        DiracGamma::TensorTime => (1..4).map(|k| sigma(0, k)).collect(),
    }
}
fn rotation_z(t: f64) -> Vec<f64> {
    vec![t.cos(), -t.sin(), 0., t.sin(), t.cos(), 0., 0., 0., 1.]
}
fn rotation_x(t: f64) -> Vec<f64> {
    vec![1., 0., 0., 0., t.cos(), -t.sin(), 0., t.sin(), t.cos()]
}
fn rotated(r: &[f64], c: [C; 3]) -> [C; 3] {
    std::array::from_fn(|a| (0..3).fold(C::ZERO, |s, b| s + c[b] * r[a * 3 + b]))
}
/// The inversion `c → −c*`.
fn inverted(c: [C; 3]) -> [C; 3] {
    c.map(|z| -z.conj())
}

#[test]
fn gamma_tables_satisfy_the_clifford_algebra_and_gamma5_is_their_product_exactly() {
    let tables = [
        "1 0 0 0 | 0 1 0 0 | 0 0 -1 0 | 0 0 0 -1",
        "0 0 0 1 | 0 0 1 0 | 0 -1 0 0 | -1 0 0 0",
        "0 0 0 -i | 0 0 i 0 | 0 i 0 0 | -i 0 0 0",
        "0 0 1 0 | 0 0 0 -1 | -1 0 0 0 | 0 1 0 0",
    ];
    for (mu, text) in tables.iter().enumerate() {
        assert_eq!(gamma(mu), table(text));
    }
    assert_eq!(gamma5(), table("0 0 1 0 | 0 0 0 1 | 1 0 0 0 | 0 1 0 0"));
    let metric = [1., -1., -1., -1.];
    for (mu, sign) in metric.into_iter().enumerate() {
        for nu in 0..4 {
            let anticommutator = sum(
                &times(&gamma(mu), &gamma(nu)),
                &times(&gamma(nu), &gamma(mu)),
            );
            let expected = if mu == nu { 2. * sign } else { 0. };
            assert_eq!(anticommutator, scaled(&unit(), C::from(expected)));
        }
        assert_eq!(
            sum(&times(&gamma5(), &gamma(mu)), &times(&gamma(mu), &gamma5())),
            [C::ZERO; 16]
        );
        assert_eq!(
            dagger(&gamma(mu)),
            times(&times(&gamma(0), &gamma(mu)), &gamma(0))
        );
    }
    let ordered = (1..4).fold(gamma(0), |a, mu| times(&a, &gamma(mu)));
    assert_eq!(gamma5(), scaled(&ordered, I));
    assert_eq!(times(&gamma5(), &gamma5()), unit());
    // The geometric signature (−, +, +, +) is reached by γ → iγ with the same γ5.
    let geometric: Vec<Matrix4> = (0..4).map(|mu| scaled(&gamma(mu), I)).collect();
    for (matrix, sign) in geometric.iter().zip(metric) {
        assert_eq!(times(matrix, matrix), scaled(&unit(), C::from(-sign)));
    }
    let ordered = (1..4).fold(geometric[0], |a, mu| times(&a, &geometric[mu]));
    assert_eq!(gamma5(), scaled(&ordered, I));
}

#[test]
fn sigma_matrices_match_their_tables_and_are_antisymmetric() {
    let tables = [
        (0, 1, "0 0 0 i | 0 0 i 0 | 0 i 0 0 | i 0 0 0"),
        (0, 2, "0 0 0 1 | 0 0 -1 0 | 0 1 0 0 | -1 0 0 0"),
        (0, 3, "0 0 i 0 | 0 0 0 -i | i 0 0 0 | 0 -i 0 0"),
        (1, 2, "1 0 0 0 | 0 -1 0 0 | 0 0 1 0 | 0 0 0 -1"),
        (1, 3, "0 i 0 0 | -i 0 0 0 | 0 0 0 i | 0 0 -i 0"),
        (2, 3, "0 1 0 0 | 1 0 0 0 | 0 0 0 1 | 0 0 1 0"),
    ];
    for (mu, nu, text) in tables {
        assert_eq!(sigma(mu, nu), table(text));
        assert_eq!(sigma(mu, nu), scaled(&times(&gamma(mu), &gamma(nu)), I));
        assert_eq!(sigma(nu, mu), scaled(&sigma(mu, nu), -C::ONE));
    }
    for mu in 0..4 {
        assert_eq!(sigma(mu, mu), [C::ZERO; 16]);
    }
}

#[test]
fn gamma0_gamma_is_hermitian_for_every_real_kept_part_and_anti_hermitian_for_gamma5() {
    // Sign η of γ0 Γ γ0 = η Γ per arm: the declared inversion parity.
    let parities = [
        (DiracGamma::Scalar, 1.),
        (DiracGamma::Pseudoscalar, -1.),
        (DiracGamma::Vector, -1.),
        (DiracGamma::Axial, 1.),
        (DiracGamma::Tensor, 1.),
        (DiracGamma::TensorTime, -1.),
    ];
    for (arm, eta) in parities {
        for matrix in matrices(arm) {
            let sandwiched = times(&gamma(0), &matrix);
            let hermitian = if arm == DiracGamma::Pseudoscalar {
                -1.
            } else {
                1.
            };
            assert_eq!(dagger(&sandwiched), scaled(&sandwiched, C::from(hermitian)));
            assert_eq!(times(&dagger(&sandwiched), &sandwiched), unit());
            assert_eq!(times(&sandwiched, &gamma(0)), scaled(&matrix, C::from(eta)));
        }
        let declared = if eta > 0. {
            SpatialParity::Even
        } else {
            SpatialParity::Odd
        };
        assert_eq!(
            signature(&dirac(arm)).descriptor.spatial_parity,
            Some(declared)
        );
    }
    // The two rows outside the arms: γ0 (value ψ†ψ) and γ5 γ0.
    assert_eq!(times(&gamma(0), &gamma(0)), unit());
    let mixed = times(&gamma(0), &times(&gamma5(), &gamma(0)));
    assert_eq!(dagger(&mixed), mixed);
    assert_eq!(mixed, scaled(&gamma5(), -C::ONE));
}

#[test]
fn declared_exchange_parity_follows_the_hermiticity_of_the_projected_matrix() {
    let projectors = [
        (ChiralProjector::None, unit()),
        (ChiralProjector::Left, projector(-1.)),
        (ChiralProjector::Right, projector(1.)),
    ];
    for &arm in DiracGamma::ALL {
        for (choice, chiral) in &projectors {
            let mut parities = matrices(arm).into_iter().map(|matrix| {
                let m = times(&times(&gamma(0), &matrix), chiral);
                let hermitian = dagger(&m) == m;
                let anti_hermitian = dagger(&m) == scaled(&m, -C::ONE);
                let imaginary_part = arm == DiracGamma::Pseudoscalar;
                if hermitian != imaginary_part && (hermitian || anti_hermitian) {
                    ExchangeParity::Even
                } else if hermitian || anti_hermitian {
                    ExchangeParity::Odd
                } else {
                    ExchangeParity::Mixed
                }
            });
            let expected = parities.next().unwrap();
            assert!(parities.all(|p| p == expected));
            let declared = signature(&projected(arm, *choice));
            assert_eq!(declared.exchange, expected, "{arm:?} {choice:?}");
            assert_eq!(
                declared.descriptor.spatial_parity.is_some(),
                *choice == ChiralProjector::None
            );
        }
    }
    let (left, right) = (projector(-1.), projector(1.));
    assert_eq!(times(&left, &left), left);
    assert_eq!(times(&right, &right), right);
    assert_eq!(times(&left, &right), [C::ZERO; 16]);
    assert_eq!(sum(&left, &right), unit());
    // The upper components are a parity block, not a chirality eigenspace.
    let upper = [C::ONE, C::ZERO, C::ZERO, C::ZERO];
    assert_eq!(
        applied(&left, &upper),
        [C::from(0.5), C::ZERO, C::from(-0.5), C::ZERO]
    );
}

#[test]
fn embedding_is_exactly_odd_and_masks_vectors_at_or_below_the_threshold() {
    for w in [[1., 2., -2.], [0.3, -0.7, 1.1], [1e-9, 0., -3e-10]] {
        let (plus, minus) = (embed(w).unwrap(), embed(w.map(|x| -x)).unwrap());
        // Exact equality: only the sign of a zero may differ.
        assert_eq!(plus, minus.map(|z| -z));
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((dot(&plus, &plus).re - norm).abs() < 1e-15 * norm.max(1.));
        assert_eq!(plus[1].im, 0.);
    }
    assert!(embed([0.; 3]).is_none());
    assert!(embed([1e-12, 0., 0.]).is_none());
    assert!(embed([1.0000001e-12, 0., 0.]).is_some());
    assert!(embed([f64::NAN, 1., 0.]).is_none());
    assert!(embed([f64::INFINITY, 1., 0.]).is_none());
    assert!(embed([1e200, 1e200, 0.]).is_none());
}

#[test]
fn spin_vector_of_the_embedding_is_even_so_no_rotation_acts_on_it_equivariantly() {
    let spin = |w: [f64; 3]| {
        let [a, b] = embed(w).unwrap();
        let cross = a.conj() * b;
        [2. * cross.re, 2. * cross.im, a.abs2() - b.abs2()]
    };
    assert!(close(&spin([0., 0., 1.]), &[0., 0., -1.], 1e-15));
    assert!(close(&spin([1., 0., 0.]), &[0., 0., 1.], 1e-15));
    assert!(close(&spin([0., 1., 0.]), &[0., 0., 1.], 1e-15));
    let w = [1., 2., -2.];
    assert!(close(&spin(w), &[-4. / 3., 8. / 3., 1. / 3.], 1e-14));
    assert_eq!(spin(w), spin(w.map(|x| -x)));
    // A quarter turn about the second axis sends z to x; an equivariant spin
    // vector would follow it to −x.
    let turned = [-1., 0., 0.];
    assert!(!close(&spin([1., 0., 0.]), &turned, 0.5));
    assert!(!close(&spin([1., 0., 0.]), &turned.map(|x: f64| -x), 0.5));
    // Rotations about the third axis act as diag(e^{iφ}, 1).
    let phi = 0.7;
    let r = rotation_z(phi);
    let image: [f64; 3] = std::array::from_fn(|a| (0..3).map(|b| r[a * 3 + b] * w[b]).sum());
    let (before, after) = (embed(w).unwrap(), embed(image).unwrap());
    assert!((after[0] - C::phase(phi) * before[0]).abs() < 1e-15);
    assert!((after[1] - before[1]).abs() < 1e-15);
}

#[test]
fn lift_places_the_imaginary_part_above_and_inversion_acts_as_gamma0() {
    let frame = reference();
    let expected = [
        [
            (1.14535000134154e-01, -4.99359916723054e-01),
            (-6.90811882255960e-01, 0.),
            (3.88193137799684e-01, 6.21574684905609e-01),
            (-3.68630118653011e-01, 0.),
        ],
        [
            (-4.15019302226549e-01, -4.17412011875918e-01),
            (2.11730157731068e-01, 0.),
            (6.40090612605981e-01, -2.15993679019648e-01),
            (6.81096496637749e-01, 0.),
        ],
    ];
    for (walker, rows) in [A, B].into_iter().zip(expected) {
        let psi = spinor(&frame, walker);
        for (z, (re, im)) in psi.iter().zip(rows) {
            assert!((*z - C::new(re, im)).abs() < 1e-12);
        }
    }
    let quarter = 0.5_f64.powf(0.25);
    assert!(close(
        &spinor(&frame, P).map(|z| z.re + 10. * z.im),
        &[quarter, 0., 10. * quarter, 0.],
        1e-15
    ));
    for walker in [A, B, A2, B2, P, Q] {
        let c = colors(&frame, walker);
        let psi = spinor(&frame, walker);
        let inverse = lift(&inverted(c)).unwrap();
        assert_eq!(inverse, applied(&gamma(0), &psi));
        let real: f64 = c.iter().map(|z| z.re * z.re).sum::<f64>().sqrt();
        let imaginary: f64 = c.iter().map(|z| z.im * z.im).sum::<f64>().sqrt();
        let norm = dot(&psi, &psi).re;
        assert!((norm - real - imaginary).abs() < 1e-15);
        assert!((1. - 1e-15..=2_f64.sqrt() + 1e-15).contains(&norm));
    }
    assert!((dot(&spinor(&frame, A), &spinor(&frame, A)).re - 1.41263681487981).abs() < 1e-12);
    let real = [C::new(0.6, 0.), C::new(0., 0.), C::new(0.8, 0.)];
    assert!(lift(&real).is_none());
    assert!(lift(&real.map(|z| I * z)).is_none());
    assert!(lift(&[C::new(0.6, 1e-12), C::new(0.8, 0.), C::ZERO]).is_none());
    assert!(lift(&[C::new(0.6, 1.0000001e-12), C::new(0.8, 0.), C::ZERO]).is_some());
    assert!(lift(&[C::new(1e-12, 0.6), C::new(0., 0.8), C::ZERO]).is_none());
    assert!(lift(&[C::new(f64::NAN, 0.8), C::ONE, C::ONE]).is_none());
    assert!(lift(&real[..2]).is_none());
}

#[test]
fn bilinears_reproduce_the_reference_vectors_with_the_kept_part_even_under_exchange() {
    let frame = reference();
    let (a, b) = (spinor(&frame, A), spinor(&frame, B));
    let axial = |k| times(&gamma5(), &gamma(k));
    let rows: [(Matrix4, f64, (f64, f64)); 16] = [
        (unit(), 1., (1.51488984825142e-01, 2.26659095810621e-01)),
        (gamma5(), -1., (2.09272309231564e-01, 1.98967745584818e-01)),
        (gamma(1), 1., (-1.28992004655424e-01, 5.11587823188651e-01)),
        (gamma(2), 1., (-9.45754556592779e-02, -4.49395167915868e-01)),
        (gamma(3), 1., (3.09170645039393e-01, 3.90825772316461e-01)),
        (axial(1), 1., (-3.39391086688369e-01, -5.03521667699854e-02)),
        (axial(2), 1., (6.85597739481141e-01, 2.37903912948256e-01)),
        (axial(3), 1., (-6.72465563221038e-01, 7.36763673879651e-01)),
        (
            sigma(0, 1),
            1.,
            (-4.67058756412035e-01, -5.99353622494441e-01),
        ),
        (
            sigma(0, 2),
            1.,
            (5.90988008569102e-01, 4.76378035424152e-01),
        ),
        (
            sigma(0, 3),
            1.,
            (-1.98967745584818e-01, 9.94191188578443e-01),
        ),
        (
            sigma(1, 2),
            1.,
            (-5.81249623292038e-02, 2.26659095810621e-01),
        ),
        (
            sigma(1, 3),
            1.,
            (-3.20350492145332e-01, -7.62803416347347e-01),
        ),
        (
            sigma(2, 3),
            1.,
            (2.82510471287251e-01, 7.37813296295205e-01),
        ),
        (gamma(0), 1., (-1.22211219103511e-01, -7.36763673879651e-01)),
        (
            times(&gamma5(), &gamma(0)),
            1.,
            (7.87948686974784e-01, -3.90825772316461e-01),
        ),
    ];
    let bound = (dot(&a, &a).re * dot(&b, &b).re).sqrt();
    assert!((bound - 1.36116319428221).abs() < 1e-12);
    for (matrix, hermitian, (re, im)) in rows {
        let forward = bilinear(&a, &matrix, &b);
        assert!((forward - C::new(re, im)).abs() < 1e-12);
        // The sandwiched matrix γ0 Γ gives the same number through the plain
        // Hermitian product.
        let sandwiched = dot(&a, &applied(&times(&gamma(0), &matrix), &b));
        assert!((forward - sandwiched).abs() < 1e-15);
        let backward = bilinear(&b, &matrix, &a);
        assert!((backward - forward.conj() * hermitian).abs() < 1e-15);
        let diagonal = bilinear(&a, &matrix, &a);
        let vanishing = if hermitian > 0. {
            diagonal.im
        } else {
            diagonal.re
        };
        assert!(vanishing.abs() < 1e-15);
        assert!(forward.abs() <= bound && bound <= 2_f64.sqrt());
    }
    assert!((bilinear(&a, &unit(), &a).re - 6.67624838230038e-02).abs() < 1e-12);
    assert!((bilinear(&a, &gamma5(), &a).im - 5.30080299166323e-01).abs() < 1e-12);
    let kept = [
        (DiracGamma::Scalar, vec![1.51488984825142e-01]),
        (DiracGamma::Pseudoscalar, vec![1.98967745584818e-01]),
        (
            DiracGamma::Vector,
            vec![
                -1.28992004655424e-01,
                -9.45754556592779e-02,
                3.09170645039393e-01,
            ],
        ),
        (
            DiracGamma::Axial,
            vec![
                -3.39391086688369e-01,
                6.85597739481141e-01,
                -6.72465563221038e-01,
            ],
        ),
        (
            DiracGamma::TensorTime,
            vec![
                -4.67058756412035e-01,
                5.90988008569102e-01,
                -1.98967745584818e-01,
            ],
        ),
        (
            DiracGamma::Tensor,
            vec![
                -5.81249623292038e-02,
                -3.20350492145332e-01,
                2.82510471287251e-01,
            ],
        ),
    ];
    for (arm, expected) in kept {
        let spec = dirac(arm);
        let forward = value(&spec, &frame, A, B);
        assert!(close(&forward, &expected, 1e-12), "{arm:?}");
        assert!(close(&value(&spec, &frame, B, A), &forward, 1e-15));
        assert_eq!(signature(&spec).exchange, ExchangeParity::Even);
        // The operator and the matrix route agree to the last bit.
        let matrix_route: Vec<f64> = matrices(arm)
            .iter()
            .map(|matrix| {
                let z = bilinear(&a, matrix, &b);
                if arm == DiracGamma::Pseudoscalar {
                    z.im
                } else {
                    z.re
                }
            })
            .collect();
        assert_eq!(forward, matrix_route);
    }
}

#[test]
fn exact_hand_case_has_rational_kept_parts() {
    let frame = reference();
    let expected = [
        (DiracGamma::Scalar, vec![0.]),
        (DiracGamma::Pseudoscalar, vec![0.]),
        (DiracGamma::Vector, vec![0.5, -0.5, 1.]),
        (DiracGamma::Axial, vec![-0.5, 0.5, 0.]),
        (DiracGamma::TensorTime, vec![-0.5, 0.5, 0.]),
        (DiracGamma::Tensor, vec![0., -0.5, 0.5]),
    ];
    for (arm, kept) in expected {
        assert!(
            close(&value(&dirac(arm), &frame, P, Q), &kept, 1e-15),
            "{arm:?}"
        );
    }
    let (p, q) = (spinor(&frame, P), spinor(&frame, Q));
    assert!((bilinear(&p, &unit(), &q) - I).abs() < 1e-15);
    assert!((bilinear(&p, &gamma(0), &q)).abs() < 1e-15);
    assert!((bilinear(&p, &times(&gamma5(), &gamma(0)), &q) + C::ONE).abs() < 1e-15);
    let half = 0.125_f64.sqrt();
    assert!(close(
        &value(&tensor(TensorMode::Components), &frame, P, Q),
        &[0., half, half],
        1e-15
    ));
}

#[test]
fn inversion_multiplies_every_kept_part_by_the_declared_parity() {
    let frame = reference();
    let mut image = frame.clone();
    image.color = (0..frame.n as u32)
        .flat_map(|w| inverted(colors(&frame, w)))
        .collect();
    image.fitness = Some(vec![1., 1.7, 0.4, 2.2, 1.1, 0.9]);
    let mut fitted = frame.clone();
    fitted.fitness = image.fitness.clone();
    for &arm in DiracGamma::ALL {
        for link in [PhaseLink::None, PhaseLink::U1, PhaseLink::Su2] {
            let spec = ChannelSpec::Dirac {
                gamma: arm,
                projector: ChiralProjector::None,
                pairs: RoleClass::Any,
                link,
            };
            let eta = match signature(&spec).descriptor.spatial_parity.unwrap() {
                SpatialParity::Even => 1.,
                SpatialParity::Odd => -1.,
            };
            let expected: Vec<f64> = value(&spec, &fitted, A, B)
                .iter()
                .map(|x| eta * x)
                .collect();
            assert_eq!(value(&spec, &image, A, B), expected, "{arm:?} {link:?}");
        }
    }
    for mode in [TensorMode::Components, TensorMode::Envelope] {
        let spec = tensor(mode);
        assert_eq!(
            signature(&spec).descriptor.spatial_parity,
            Some(SpatialParity::Even)
        );
        assert!(close(
            &value(&spec, &image, A, B),
            &value(&spec, &frame, A, B),
            1e-15
        ));
    }
}

#[test]
fn pseudoscalar_is_minus_the_third_tensor_time_component() {
    let frame = reference();
    for (i, j) in [(A, B), (A2, B2), (B, A2), (P, Q)] {
        let pseudoscalar = value(&dirac(DiracGamma::Pseudoscalar), &frame, i, j)[0];
        let tensor_time = value(&dirac(DiracGamma::TensorTime), &frame, i, j)[2];
        assert!((pseudoscalar + tensor_time).abs() < 1e-15);
    }
    let sink = value(&dirac(DiracGamma::Pseudoscalar), &frame, A2, B2)[0];
    assert!((sink - 1.51943002384814e-01).abs() < 1e-12);
}

#[test]
fn closed_forms_in_the_real_and_imaginary_colour_vectors_agree_with_the_operators() {
    let frame = reference();
    let parts = |walker: u32| {
        let c = colors(&frame, walker);
        (c.map(|z| z.im), c.map(|z| z.re))
    };
    let norm = |p: [f64; 3]| p.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = |p: [f64; 3], q: [f64; 3]| 1. / (norm(p) * norm(q)).sqrt();
    let d = |p: [f64; 3], q: [f64; 3]| scale(p, q) * contracted(&p, &q);
    // The cross product reflected in the first axis.
    let x = |p: [f64; 3], q: [f64; 3]| {
        let s = scale(p, q);
        [
            -s * (p[1] * q[2] - p[2] * q[1]),
            s * (p[2] * q[0] - p[0] * q[2]),
            s * (p[0] * q[1] - p[1] * q[0]),
        ]
    };
    let m = |p: [f64; 3], q: [f64; 3]| {
        let s = scale(p, q);
        [
            s * (p[0] * q[2] + p[2] * q[0]),
            -s * (p[1] * q[2] + p[2] * q[1]),
            s * (p[0] * q[0] + p[1] * q[1] - p[2] * q[2]),
        ]
    };
    for (i, j) in [(A, B), (A2, B2), (Q, A)] {
        let ((ui, wi), (uj, wj)) = (parts(i), parts(j));
        let combine = |a: [f64; 3], b: [f64; 3], sign: f64| -> Vec<f64> {
            (0..3).map(|k| a[k] + sign * b[k]).collect()
        };
        let expected = [
            (DiracGamma::Scalar, vec![d(ui, uj) - d(wi, wj)]),
            (DiracGamma::Pseudoscalar, vec![x(ui, wj)[2] - x(wi, uj)[2]]),
            (DiracGamma::Vector, combine(m(ui, wj), m(wi, uj), 1.)),
            (
                DiracGamma::Axial,
                combine(m(ui, uj), m(wi, wj), 1.)
                    .iter()
                    .map(|v| -v)
                    .collect(),
            ),
            (DiracGamma::TensorTime, combine(x(wi, uj), x(ui, wj), -1.)),
            (DiracGamma::Tensor, {
                let s = combine(m(ui, uj), m(wi, wj), -1.);
                vec![s[2], -s[1], s[0]]
            }),
        ];
        for (arm, closed) in expected {
            assert!(
                close(&value(&dirac(arm), &frame, i, j), &closed, 1e-14),
                "{arm:?}"
            );
        }
    }
}

#[test]
fn only_rotations_about_the_third_colour_axis_preserve_every_contraction() {
    let frame = reference();
    let invariants = |r: &[f64]| -> Vec<f64> {
        let turned = state(&[rotated(r, colors(&frame, A)), rotated(r, colors(&frame, B))]);
        let square = |arm| {
            let v = value(&dirac(arm), &turned, 0, 1);
            contracted(&v, &v)
        };
        vec![
            value(&dirac(DiracGamma::Scalar), &turned, 0, 1)[0],
            value(&dirac(DiracGamma::Pseudoscalar), &turned, 0, 1)[0],
            square(DiracGamma::Vector),
            square(DiracGamma::Axial),
            square(DiracGamma::Tensor),
            square(DiracGamma::TensorTime),
        ]
    };
    let identity = [
        0.151488984825142,
        0.198967745584818,
        0.121169941832259,
        1.03744050382335,
        0.185815115450472,
        0.606998871996734,
    ];
    assert!(close(&invariants(&rotation_z(0.)), &identity, 1e-12));
    assert!(close(&invariants(&rotation_z(0.7)), &identity, 1e-12));
    let about_x = [
        0.151488984825142,
        -0.0468801410984043,
        0.311032909440697,
        0.914104379685583,
        0.0526718312887483,
        0.606998871996735,
    ];
    assert!(close(&invariants(&rotation_x(0.4)), &about_x, 1e-12));
}

#[test]
fn component_kept_propagator_summand_differs_from_the_product_of_component_means() {
    let frame = reference();
    let rows = [
        (
            DiracGamma::Scalar,
            vec![2.44685838001336e-01],
            3.70672091999116e-02,
            3.70672091999116e-02,
        ),
        (
            DiracGamma::Pseudoscalar,
            vec![1.51943002384814e-01],
            3.02317566418950e-02,
            3.02317566418950e-02,
        ),
        (
            DiracGamma::Vector,
            vec![
                1.72505604973672e-01,
                -7.24215528203745e-01,
                5.35441109353216e-01,
            ],
            2.11783842934885e-01,
            -1.54740253283570e-04,
        ),
        (
            DiracGamma::Axial,
            vec![
                -4.62872592973816e-01,
                8.57370417211074e-01,
                -6.49037411962260e-01,
            ],
            1.18136136105229,
            9.22731206133545e-03,
        ),
        (
            DiracGamma::TensorTime,
            vec![
                -4.87204319356980e-01,
                2.62973443031629e-01,
                -1.51943002384814e-01,
            ],
            4.13198951563160e-01,
            3.13639123614271e-03,
        ),
        (
            DiracGamma::Tensor,
            vec![
                -1.94178615069147e-02,
                -3.16748426167081e-01,
                5.70308878997384e-02,
            ],
            1.18710999695972e-01,
            2.97635821625694e-03,
        ),
    ];
    for (arm, sink, summand, mean_product) in rows {
        let spec = dirac(arm);
        let (source, at_sink) = (value(&spec, &frame, A, B), value(&spec, &frame, A2, B2));
        assert!(close(&at_sink, &sink, 1e-12), "{arm:?}");
        assert!((contracted(&source, &at_sink) - summand).abs() < 1e-12);
        let reversed = contracted(&value(&spec, &frame, B, A), &value(&spec, &frame, B2, A2));
        assert!((reversed - summand).abs() < 1e-14);
        assert!((mean(&source) * mean(&at_sink) - mean_product).abs() < 1e-12);
    }
}

#[test]
fn chiral_projections_sum_to_the_unprojected_bilinear_and_split_by_exchange() {
    let frame = reference();
    let of = |arm, projector, i, j| value(&projected(arm, projector), &frame, i, j);
    let (left, right) = (ChiralProjector::Left, ChiralProjector::Right);
    let scalar = DiracGamma::Scalar;
    assert!((of(scalar, left, A, B)[0] + 2.88916622032109e-02).abs() < 1e-12);
    assert!((of(scalar, right, A, B)[0] - 1.80380647028353e-01).abs() < 1e-12);
    assert!((of(scalar, left, B, A)[0] - 1.80380647028353e-01).abs() < 1e-12);
    assert!((of(scalar, right, B, A)[0] + 2.88916622032108e-02).abs() < 1e-12);
    let current_left = [
        -2.34191545671896e-01,
        2.95511141910931e-01,
        -1.81647459090822e-01,
    ];
    let current_right = [
        1.05199541016472e-01,
        -3.90086597570209e-01,
        4.90818104130215e-01,
    ];
    for (i, j) in [(A, B), (B, A)] {
        assert!(close(
            &of(DiracGamma::Vector, left, i, j),
            &current_left,
            1e-12
        ));
        assert!(close(
            &of(DiracGamma::Vector, right, i, j),
            &current_right,
            1e-12
        ));
    }
    assert!(close(
        &of(DiracGamma::Vector, left, P, Q),
        &[0., 0., 0.5],
        1e-15
    ));
    assert!(close(
        &of(DiracGamma::Vector, right, P, Q),
        &[0.5, -0.5, 0.5],
        1e-15
    ));
    assert!(of(scalar, left, P, Q)[0].abs() < 1e-15);
    for &arm in DiracGamma::ALL {
        let whole = value(&dirac(arm), &frame, A, B);
        let parts: Vec<f64> = of(arm, left, A, B)
            .iter()
            .zip(of(arm, right, A, B))
            .map(|(l, r)| l + r)
            .collect();
        assert!(close(&parts, &whole, 1e-15), "{arm:?}");
        let frame_mean = |projector| -> Vec<f64> {
            of(arm, projector, A, B)
                .iter()
                .zip(of(arm, projector, B, A))
                .map(|(x, y)| 0.5 * (x + y))
                .collect()
        };
        if matches!(arm, DiracGamma::Vector | DiracGamma::Axial) {
            // Projected currents are even under exchange on their own.
            assert!(close(&frame_mean(left), &of(arm, left, A, B), 1e-15));
            assert!(close(&frame_mean(right), &of(arm, right, A, B), 1e-15));
        } else {
            // On the mutual pair the part odd under exchange cancels: both
            // projections have half the frame mean of the unprojected bilinear.
            let half: Vec<f64> = whole.iter().map(|x| 0.5 * x).collect();
            assert!(close(&frame_mean(left), &half, 1e-15), "{arm:?}");
            assert!(close(&frame_mean(right), &half, 1e-15), "{arm:?}");
        }
    }
    // γ5 γ^k P_L = γ^k P_L and γ5 γ^k P_R = −γ^k P_R: a projected axial current
    // is the projected vector current, and its descriptor says so.
    for (i, j) in [(A, B), (B, A), (A2, B2), (P, Q)] {
        let vector = |projector| value(&projected(DiracGamma::Vector, projector), &frame, i, j);
        let axial = |projector| value(&projected(DiracGamma::Axial, projector), &frame, i, j);
        assert!(close(&axial(left), &vector(left), 1e-15));
        let mirrored: Vec<f64> = vector(right).iter().map(|x| -x).collect();
        assert!(close(&axial(right), &mirrored, 1e-15));
    }
    let independent = |arm, projector| {
        !signature(&projected(arm, projector))
            .descriptor
            .note
            .contains("not an independent channel")
    };
    assert!(!independent(DiracGamma::Axial, left) && !independent(DiracGamma::Axial, right));
    assert!(
        independent(DiracGamma::Vector, left)
            && independent(DiracGamma::Axial, ChiralProjector::None)
    );
    // The difference of the scalar projections is the part odd under exchange.
    let (a, b) = (spinor(&frame, A), spinor(&frame, B));
    let odd = of(scalar, left, A, B)[0] - of(scalar, right, A, B)[0];
    assert!((odd + bilinear(&a, &gamma5(), &b).re).abs() < 1e-15);
    assert!((odd + 2.09272309231564e-01).abs() < 1e-12);
    assert_eq!(
        signature(&projected(scalar, left)).exchange,
        ExchangeParity::Mixed
    );
    assert_eq!(
        signature(&projected(DiracGamma::Vector, left)).exchange,
        ExchangeParity::Even
    );
}

#[test]
fn role_classes_admit_pairs_by_the_chirality_of_eligible_walkers() {
    let mut frame = reference();
    let classed = |pairs| ChannelSpec::Dirac {
        gamma: DiracGamma::Scalar,
        projector: ChiralProjector::Left,
        pairs,
        link: PhaseLink::None,
    };
    for &pairs in &RoleClass::ALL[1..] {
        assert!(evaluated(&classed(pairs), &frame, A, B).is_none());
    }
    frame.role = Some(vec![
        WalkerRole::Cloner,
        WalkerRole::Persister,
        WalkerRole::StrongResister,
        WalkerRole::WeakResister,
        WalkerRole::Cloner,
        WalkerRole::Cloner,
    ]);
    let unrestricted = value(&classed(RoleClass::Any), &frame, A, B);
    let admitted = |pairs, i, j| evaluated(&classed(pairs), &frame, i, j);
    assert_eq!(admitted(RoleClass::LeftRight, A, B), Some(unrestricted));
    assert!(admitted(RoleClass::RightLeft, A, B).is_none());
    assert!(admitted(RoleClass::RightLeft, B, A).is_some());
    assert!(admitted(RoleClass::LeftLeft, A, A2).is_some());
    assert!(admitted(RoleClass::LeftLeft, A, B).is_none());
    assert!(admitted(RoleClass::RightRight, B, B2).is_some());
    assert!(admitted(RoleClass::RightRight, A2, B2).is_none());
    // The validity mask of the spinor layer holds both walkers alive, with
    // or without a role class; the colour tensor reads colour validity only.
    let mut dead = frame.clone();
    dead.eligible[B as usize] = false;
    for &pairs in RoleClass::ALL {
        assert!(evaluated(&classed(pairs), &dead, A, B).is_none());
        assert!(evaluated(&classed(pairs), &dead, B, A).is_none());
    }
    assert!(evaluated(&dirac(DiracGamma::Vector), &dead, A, B).is_none());
    assert!(evaluated(&dirac(DiracGamma::Vector), &dead, A, A2).is_some());
    assert!(evaluated(&tensor(TensorMode::Components), &dead, A, B).is_some());
    let mut unknown = frame.clone();
    unknown.eligible.clear();
    assert!(evaluated(&dirac(DiracGamma::Scalar), &unknown, A, B).is_none());
    let requires = |pairs| signature(&classed(pairs)).requires;
    assert_eq!(
        requires(RoleClass::Any),
        Requirements::new([Record::Color]).in_dimension(3)
    );
    assert_eq!(
        requires(RoleClass::LeftRight),
        Requirements::new([
            Record::Color,
            Record::Fitness,
            Record::CloningCompanions,
            Record::ClonePlan
        ])
        .in_dimension(3)
    );
    let current = |pairs| {
        signature(&ChannelSpec::Dirac {
            gamma: DiracGamma::Vector,
            projector: ChiralProjector::Left,
            pairs,
            link: PhaseLink::None,
        })
        .exchange
    };
    assert_eq!(current(RoleClass::LeftLeft), ExchangeParity::Even);
    assert_eq!(current(RoleClass::RightRight), ExchangeParity::Even);
    assert_eq!(current(RoleClass::LeftRight), ExchangeParity::Mixed);
    assert_eq!(current(RoleClass::RightLeft), ExchangeParity::Mixed);
}

#[test]
fn phase_links_multiply_the_bilinear_by_the_fitness_phases() {
    let mut frame = reference();
    let linked = |gamma, link| ChannelSpec::Dirac {
        gamma,
        projector: ChiralProjector::None,
        pairs: RoleClass::Any,
        link,
    };
    assert!(evaluated(&linked(DiracGamma::Vector, PhaseLink::U1), &frame, A, B).is_none());
    frame.fitness = Some(vec![1., 1.7, 0.4, 2.2, 1.1, 0.9]);
    let mut measurement = MeasurementConfig::default();
    measurement.electroweak.h_eff = 0.8;
    measurement.electroweak.h_s = Some(0.35);
    let epsilon = GasConfig::default().clone_decision.epsilon;
    let (a, b) = (spinor(&frame, A), spinor(&frame, B));
    let u1 = C::phase(-(1.7 - 1.) / 0.8);
    let su2 = C::phase((1.7 - 1.) / ((1. + epsilon) * 0.35));
    for (link, phase) in [(PhaseLink::U1, u1), (PhaseLink::Su2, su2)] {
        for &arm in DiracGamma::ALL {
            let expected: Vec<f64> = matrices(arm)
                .iter()
                .map(|matrix| {
                    let z = phase * bilinear(&a, matrix, &b);
                    if arm == DiracGamma::Pseudoscalar {
                        z.im
                    } else {
                        z.re
                    }
                })
                .collect();
            let spec = linked(arm, link);
            let forward = measured(&spec, &measurement, &frame, A, B).unwrap();
            assert!(close(&forward, &expected, 1e-15), "{arm:?} {link:?}");
            let backward = measured(&spec, &measurement, &frame, B, A).unwrap();
            let declared = signature(&spec).exchange;
            if link == PhaseLink::U1 {
                assert_eq!(declared, ExchangeParity::Even);
                assert!(close(&backward, &forward, 1e-15));
            } else {
                assert_eq!(declared, ExchangeParity::Mixed);
                assert!(!close(&backward, &forward, 1e-3));
            }
        }
    }
    // Independent values: θ = −0.875, ϑ_AB = 0.7 / (1.000001 · 0.35) and
    // ϑ_BA = −0.7 / (1.700001 · 0.35).
    let of = |gamma, link, i, j| measured(&linked(gamma, link), &measurement, &frame, i, j);
    let rows = [
        (
            DiracGamma::Scalar,
            PhaseLink::U1,
            A,
            B,
            vec![0.271074679531403],
        ),
        (
            DiracGamma::Pseudoscalar,
            PhaseLink::U1,
            B,
            A,
            vec![-0.033087901352907],
        ),
        (
            DiracGamma::Vector,
            PhaseLink::U1,
            A,
            B,
            vec![0.309982439799206, -0.405552911007055, 0.498153194154456],
        ),
        (
            DiracGamma::Scalar,
            PhaseLink::Su2,
            A,
            B,
            vec![-0.269142107543346],
        ),
        (
            DiracGamma::Scalar,
            PhaseLink::Su2,
            B,
            A,
            vec![-0.151064260544822],
        ),
        (
            DiracGamma::Vector,
            PhaseLink::Su2,
            B,
            A,
            vec![-0.521883270873065, 0.378571979140575, -0.242052906474654],
        ),
    ];
    for (gamma, link, i, j, expected) in rows {
        let got = of(gamma, link, i, j).unwrap();
        assert!(close(&got, &expected, 1e-12), "{gamma:?} {link:?}");
    }
    // The conjugate U(1) phase of the reference code is another statistic.
    let plain = value(&dirac(DiracGamma::Scalar), &frame, A, B)[0];
    let imaginary = bilinear(&a, &unit(), &b).im;
    let theta = -(1.7 - 1.) / 0.8_f64;
    let conjugate = theta.cos() * plain + theta.sin() * imaginary;
    assert!((conjugate + 0.076866752892868).abs() < 1e-12);
    // A projector and a link compose: Re[e^{iθ} ψ̄ γ^k P_L ψ].
    let composed = ChannelSpec::Dirac {
        gamma: DiracGamma::Vector,
        projector: ChiralProjector::Left,
        pairs: RoleClass::Any,
        link: PhaseLink::U1,
    };
    assert!(close(
        &measured(&composed, &measurement, &frame, A, B).unwrap(),
        &[0.026893170557759, 0.108257344252097, 0.316301525738304],
        1e-12
    ));
    // A dead walker has no fitness phase, whatever its recorded fitness.
    let mut dead = frame.clone();
    dead.eligible[A as usize] = false;
    assert!(
        measured(
            &linked(DiracGamma::Scalar, PhaseLink::U1),
            &measurement,
            &dead,
            A,
            B
        )
        .is_none()
    );
    let requires = |link| signature(&linked(DiracGamma::Vector, link)).requires;
    assert_eq!(
        requires(PhaseLink::U1),
        Requirements::new([Record::Color, Record::Fitness]).in_dimension(3)
    );
    assert_eq!(
        requires(PhaseLink::Su2),
        Requirements::new([Record::Color, Record::Fitness, Record::ClonePlan]).in_dimension(3)
    );
    frame.fitness.as_mut().unwrap()[B as usize] = f64::NAN;
    assert!(evaluated(&linked(DiracGamma::Vector, PhaseLink::U1), &frame, A, B).is_none());
    assert!(evaluated(&linked(DiracGamma::Vector, PhaseLink::None), &frame, A, B).is_some());
}

#[test]
fn cloning_phase_reads_the_gas_epsilon_the_absolute_fitness_and_falls_back_to_h_eff() {
    let mut frame = reference();
    frame.fitness = Some(vec![1., 1.7, 0.4, 2.2, 1.1, 0.9]);
    let linked = |gamma| ChannelSpec::Dirac {
        gamma,
        projector: ChiralProjector::None,
        pairs: RoleClass::Any,
        link: PhaseLink::Su2,
    };
    let scalar = linked(DiracGamma::Scalar);
    // Without h_S the action scale is h_eff: ϑ_AB = 0.7 / (1.000001 · 0.8).
    let mut fallback = MeasurementConfig::default();
    fallback.electroweak.h_eff = 0.8;
    assert_eq!(fallback.electroweak.h_s, None);
    let got = measured(&scalar, &fallback, &frame, A, B).unwrap();
    assert!(close(&got, &[-0.076866524026182], 1e-12));
    let got = measured(&linked(DiracGamma::Vector), &fallback, &frame, A, B).unwrap();
    assert!(close(
        &got,
        &[-0.475349178919546, 0.284307455563129, -0.101797943099174],
        1e-12
    ));
    // ε_clone of the gas, not a constant: ϑ_AB = 0.7 / (1.5 · 0.35) and
    // ϑ_BA = −0.7 / (2.2 · 0.35).
    let mut measurement = MeasurementConfig::default();
    measurement.electroweak.h_eff = 0.8;
    measurement.electroweak.h_s = Some(0.35);
    let mut gas = GasConfig::default();
    gas.clone_decision.epsilon = 0.5;
    let under = |i, j| measured_on(&scalar, &gas, &measurement, &frame, &pair(i, j)).unwrap();
    assert!(close(&under(A, B), &[-0.184662664734681], 1e-12));
    assert!(close(&under(B, A), &[-0.085737254865112], 1e-12));
    // A negative fitness enters the denominator through its absolute value:
    // ϑ_AB = 2.7 / (1.000001 · 0.35). The plain fitness would give +0.2455.
    frame.fitness.as_mut().unwrap()[A as usize] = -1.;
    let got = measured(&scalar, &measurement, &frame, A, B).unwrap();
    assert!(close(&got, &[-0.203356039255061], 1e-12));
    let got = measured(
        &linked(DiracGamma::Pseudoscalar),
        &measurement,
        &frame,
        A,
        B,
    )
    .unwrap();
    assert!(close(&got, &[0.234939621065544], 1e-12));
    let u1 = ChannelSpec::Dirac {
        gamma: DiracGamma::Scalar,
        projector: ChiralProjector::None,
        pairs: RoleClass::Any,
        link: PhaseLink::U1,
    };
    let got = measured(&u1, &measurement, &frame, A, B).unwrap();
    assert!(close(&got, &[-0.199806047714738], 1e-12));
}

#[test]
fn element_weight_generation_and_pair_kind_do_not_enter_the_value() {
    let frame = reference();
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let rows = [
        (
            tensor(TensorMode::Components),
            vec![
                -5.16297952379122e-01,
                2.52550793025207e-01,
                5.84785241555750e-02,
            ],
        ),
        (tensor(TensorMode::Envelope), vec![3.33765216475950e-01]),
        (
            projected(DiracGamma::Scalar, ChiralProjector::Left),
            vec![-2.88916622032109e-02],
        ),
        (
            dirac(DiracGamma::Axial),
            vec![
                -3.39391086688369e-01,
                6.85597739481141e-01,
                -6.72465563221038e-01,
            ],
        ),
    ];
    for (spec, expected) in rows {
        let plain = value(&spec, &frame, A, B);
        assert!(close(&plain, &expected, 1e-12));
        for kind in [ElementKind::DistancePair, ElementKind::CloningPair] {
            let element = Element {
                walkers: [A, B, A],
                kind,
                weight: 0.25,
                generation: [7, 11, 7],
            };
            let got = measured_on(&spec, &gas, &measurement, &frame, &element);
            assert_eq!(got.as_ref(), Some(&plain), "{kind:?}");
        }
    }
}

#[test]
fn nonfinite_real_parts_and_truncated_records_mask_without_a_panic() {
    let frame = reference();
    let specs = [
        tensor(TensorMode::Components),
        tensor(TensorMode::Envelope),
        dirac(DiracGamma::Vector),
    ];
    let masked = |state: &FrameState| {
        specs.iter().all(|spec| {
            evaluated(spec, state, A, B).is_none() && evaluated(spec, state, B, A).is_none()
        })
    };
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for slot in 0..3 {
            let mut real = frame.clone();
            real.color[slot].re = poison;
            assert!(masked(&real));
            let mut imaginary = frame.clone();
            imaginary.color[3 + slot].im = poison;
            assert!(masked(&imaginary));
            // The other walkers of the frame are untouched.
            assert!(evaluated(&specs[0], &real, A2, B2).is_some());
        }
    }
    // A colour record shorter than the frame masks the walkers it lacks.
    let mut short = frame.clone();
    short.color.truncate(4);
    assert!(masked(&short));
    let mut roles = frame.clone();
    roles.role = Some(vec![WalkerRole::Cloner]);
    let classed = ChannelSpec::Dirac {
        gamma: DiracGamma::Scalar,
        projector: ChiralProjector::None,
        pairs: RoleClass::LeftRight,
        link: PhaseLink::None,
    };
    assert!(evaluated(&classed, &roles, A, B).is_none());
    assert!(evaluated(&classed, &roles, B, A).is_none());
    let mut fitted = frame.clone();
    fitted.fitness = Some(vec![1.]);
    for link in [PhaseLink::U1, PhaseLink::Su2] {
        let spec = ChannelSpec::Dirac {
            gamma: DiracGamma::Scalar,
            projector: ChiralProjector::None,
            pairs: RoleClass::Any,
            link,
        };
        assert!(evaluated(&spec, &fitted, A, B).is_none());
        assert!(evaluated(&spec, &fitted, B, A).is_none());
    }
}

#[test]
fn tensor_components_match_the_reference_and_change_sign_exactly_under_exchange() {
    let frame = reference();
    let spec = tensor(TensorMode::Components);
    let forward = value(&spec, &frame, A, B);
    assert!(close(
        &forward,
        &[
            -5.16297952379122e-01,
            2.52550793025207e-01,
            5.84785241555750e-02
        ],
        1e-12
    ));
    assert!(close(
        &value(&spec, &frame, A2, B2),
        &[
            -4.17559412329678e-01,
            3.37100735089252e-01,
            -3.23847964495926e-01
        ],
        1e-12
    ));
    for (i, j) in [(A, B), (A2, B2), (P, Q), (A, Q)] {
        let (forward, backward) = (value(&spec, &frame, i, j), value(&spec, &frame, j, i));
        for p in 0..3 {
            assert_eq!(forward[p] + backward[p], 0.);
            assert_eq!(forward[p].abs().to_bits(), backward[p].abs().to_bits());
        }
    }
    // Two walkers with one colour: every component vanishes exactly.
    let twins = state(&[colors(&frame, A), colors(&frame, A)]);
    assert_eq!(value(&spec, &twins, 0, 1), vec![0.; 3]);
}

#[test]
fn tensor_is_the_real_part_of_the_conjugate_cross_product() {
    let frame = reference();
    let o = value(&tensor(TensorMode::Components), &frame, A, B);
    let (a, b) = (colors(&frame, A), colors(&frame, B));
    let cross = |p: [f64; 3], q: [f64; 3]| {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    };
    let (real, imaginary) = (
        cross(a.map(|z| z.re), b.map(|z| z.re)),
        cross(a.map(|z| z.im), b.map(|z| z.im)),
    );
    let dual: Vec<f64> = (0..3).map(|k| real[k] + imaginary[k]).collect();
    assert!(close(&[o[2], -o[1], o[0]], &dual, 1e-15));
    assert!(close(
        &dual,
        &[
            5.84785241555750e-02,
            -2.52550793025207e-01,
            -5.16297952379122e-01
        ],
        1e-12
    ));
    // Lagrange identity of the complex components T = i (ā^μ b^ν − ā^ν b^μ).
    let complex = |mu: usize, nu: usize| I * (a[mu].conj() * b[nu] - a[nu].conj() * b[mu]);
    let components = [complex(0, 1), complex(0, 2), complex(1, 2)];
    let expected = [
        (1.94080366671155e-01, -5.16297952379122e-01),
        (4.65516308856653e-01, 2.52550793025207e-01),
        (-3.92294033344451e-01, 5.84785241555750e-02),
    ];
    for ((t, (re, im)), kept) in components.iter().zip(expected).zip(&o) {
        assert!((*t - C::new(re, im)).abs() < 1e-12);
        assert!((t.im - kept).abs() < 1e-15);
    }
    let bilinear_product = (0..3).fold(C::ZERO, |s, k| s + a[k] * b[k]);
    let total: f64 = components.iter().map(|t| t.abs2()).sum();
    assert!((total - (1. - bilinear_product.abs2())).abs() < 1e-15);
    assert!((total - 7.42032447612339e-01).abs() < 1e-12);
    assert!(contracted(&o, &o) <= 1.);
}

#[test]
fn tensor_is_invariant_under_a_common_phase_and_not_under_a_common_special_unitary_frame() {
    let frame = reference();
    let spec = tensor(TensorMode::Components);
    let (a, b) = (colors(&frame, A), colors(&frame, B));
    let reference_value = value(&spec, &frame, A, B);
    let rephased = state(&[a.map(|z| C::phase(0.3) * z), b.map(|z| C::phase(0.3) * z)]);
    assert!(close(
        &value(&spec, &rephased, 0, 1),
        &reference_value,
        1e-15
    ));
    // Per-walker phases are not a symmetry of the kept imaginary part.
    let local = state(&[a.map(|z| C::phase(0.3) * z), b.map(|z| C::phase(1.1) * z)]);
    assert!(!close(&value(&spec, &local, 0, 1), &reference_value, 1e-3));
    let frame_change = [C::phase(0.5), C::phase(-0.5), C::ONE];
    let changed = |c: [C; 3]| -> [C; 3] { std::array::from_fn(|k| frame_change[k] * c[k]) };
    let (ua, ub) = (changed(a), changed(b));
    let moved = value(&spec, &state(&[ua, ub]), 0, 1);
    assert!(close(
        &moved,
        &[
            -2.21228478722360e-01,
            4.23542272587991e-01,
            -9.44239781075570e-02
        ],
        1e-12
    ));
    assert!((contracted(&moved, &moved) - 2.37245984108466e-01).abs() < 1e-12);
    assert!((contracted(&reference_value, &reference_value) - 3.33765216475950e-01).abs() < 1e-12);
    let overlap_before = dot(&a, &b);
    assert!((dot(&ua, &ub) - overlap_before).abs() < 1e-15);
    assert!((overlap_before - C::new(-9.98173837679004e-02, -1.71416602276065e-02)).abs() < 1e-12);
}

#[test]
fn tensor_rotates_as_a_vector_so_the_propagator_summand_is_orientation_even_and_invariant() {
    let frame = reference();
    let spec = tensor(TensorMode::Components);
    let summand = |state: &FrameState, i, j, k, l| {
        contracted(&value(&spec, state, i, j), &value(&spec, state, k, l))
    };
    assert!((summand(&frame, A, B, A2, B2) - 2.81781976544104e-01).abs() < 1e-12);
    assert_eq!(summand(&frame, B, A, B2, A2), summand(&frame, A, B, A2, B2));
    let r = rmul(&rotation_z(0.7), &rotation_x(0.4), 3);
    let turned = state(&[A, B, A2, B2].map(|w| rotated(&r, colors(&frame, w))));
    assert!((summand(&turned, 0, 1, 2, 3) - 2.81781976544104e-01).abs() < 1e-12);
    let o = value(&spec, &turned, 0, 1);
    assert!(close(
        &o,
        &[
            -5.73889816388580e-01,
            -1.35354043071986e-02,
            6.50575741299379e-02
        ],
        1e-12
    ));
    // The dual vector (O12, −O02, O01) rotates with R.
    let before = value(&spec, &frame, A, B);
    let dual = [before[2], -before[1], before[0]];
    let image: Vec<f64> = (0..3)
        .map(|a| (0..3).map(|b| r[a * 3 + b] * dual[b]).sum())
        .collect();
    assert!(close(&[o[2], -o[1], o[0]], &image, 1e-15));
}

#[test]
fn mutual_pairing_cancels_the_tensor_frame_sum_within_roundoff() {
    let n = 32;
    let mut rng = Rng::new(7);
    let mut draw = || [rng.normal(), rng.normal(), rng.normal()];
    let walkers: Vec<[C; 3]> = (0..n).map(|_| color(draw(), draw())).collect();
    let frame = state(&walkers);
    let spec = tensor(TensorMode::Components);
    // Walker i is paired with its mirror in the reversed order of slots.
    let mut total = [0.; 3];
    let mut magnitude = 0.;
    for i in 0..n as u32 {
        let o = value(&spec, &frame, i, n as u32 - 1 - i);
        for p in 0..3 {
            total[p] += o[p];
            magnitude += o[p].abs();
        }
    }
    assert!(magnitude > 1.);
    assert!(total.iter().all(|s: &f64| s.abs() < 1e-14 * n as f64));
    let reference = reference();
    let pair_sum: Vec<f64> = value(&spec, &reference, A, B)
        .iter()
        .zip(value(&spec, &reference, B, A))
        .map(|(x, y)| x + y)
        .collect();
    assert_eq!(pair_sum, vec![0.; 3]);
}

#[test]
fn envelope_is_the_squared_norm_of_the_components_and_never_enters_a_correlator() {
    let frame = reference();
    let (components, envelope) = (tensor(TensorMode::Components), tensor(TensorMode::Envelope));
    let described = signature(&envelope);
    assert!(!described.correlatable);
    assert_eq!(described.components, 1);
    assert_eq!(described.exchange, ExchangeParity::Even);
    assert!(signature(&components).correlatable);
    let o = value(&components, &frame, A, B);
    let forward = value(&envelope, &frame, A, B);
    assert_eq!(forward, vec![contracted(&o, &o)]);
    assert!((forward[0] - 3.33765216475950e-01).abs() < 1e-12);
    let backward = value(&envelope, &frame, B, A);
    assert_eq!(forward, backward);
    let frame_mean = 0.5 * (forward[0] + backward[0]);
    assert!((frame_mean.sqrt() - 5.77724169890744e-01).abs() < 1e-12);
}

#[test]
fn signatures_state_components_requirements_and_the_default_propagators_cover_the_odd_arms() {
    let color_only = Requirements::new([Record::Color]).in_dimension(3);
    for &arm in DiracGamma::ALL {
        let described = signature(&dirac(arm));
        let scalar = matches!(arm, DiracGamma::Scalar | DiracGamma::Pseudoscalar);
        assert_eq!(described.components, if scalar { 1 } else { 3 });
        assert_eq!(described.exchange, ExchangeParity::Even);
        assert_eq!(described.requires, color_only);
        assert!(described.correlatable && !described.degenerate);
        assert_eq!(described.normalization, None);
        assert!(described.descriptor.note.contains("not norm preserving"));
        assert!(described.descriptor.note.contains("continuum analogue"));
        let part = if arm == DiracGamma::Pseudoscalar {
            r"\operatorname{Im}"
        } else {
            r"\operatorname{Re}"
        };
        assert!(described.descriptor.definition.starts_with(part));
    }
    let components = signature(&tensor(TensorMode::Components));
    assert_eq!(components.components, 3);
    assert_eq!(components.exchange, ExchangeParity::Odd);
    assert_eq!(components.requires, color_only);
    assert!(components.descriptor.note.contains("no spin-two part"));
    let propagators = PropagatorConfig::default();
    for spec in ChannelSpec::standard_set() {
        if !matches!(spec, ChannelSpec::Tensor { .. } | ChannelSpec::Dirac { .. }) {
            continue;
        }
        for kind in [ElementKind::DistancePair, ElementKind::CloningPair] {
            let described = signature_in(&spec, kind).unwrap();
            let id = channel_id(&spec.id(), kind);
            if described.exchange == ExchangeParity::Odd {
                assert!(propagators.enabled(&spec.id(), &id), "{id}");
            }
        }
    }
    assert!(propagators.enabled("tensor/components", "tensor/components/distance"));
    for spec in [tensor(TensorMode::Components), dirac(DiracGamma::Vector)] {
        for kind in [ElementKind::Site, ElementKind::Triplet] {
            assert!(matches!(
                signature_in(&spec, kind),
                Err(GasError::Capability(_))
            ));
        }
    }
}

#[test]
fn invalid_colours_self_pairs_and_other_dimensions_mask_the_element() {
    let frame = reference();
    let specs = [
        tensor(TensorMode::Components),
        tensor(TensorMode::Envelope),
        dirac(DiracGamma::Scalar),
        dirac(DiracGamma::Tensor),
    ];
    for spec in &specs {
        assert!(evaluated(spec, &frame, A, B).is_some());
        assert!(evaluated(spec, &frame, A, A).is_none());
        assert!(evaluated(spec, &frame, A, 99).is_none());
        assert!(evaluated(spec, &frame, u32::MAX, B).is_none());
        let mut invalid = frame.clone();
        invalid.color_valid[B as usize] = false;
        assert!(evaluated(spec, &invalid, A, B).is_none());
        assert!(evaluated(spec, &invalid, B, A).is_none());
        let mut nonfinite = frame.clone();
        nonfinite.color[4].im = f64::NAN;
        assert!(evaluated(spec, &nonfinite, A, B).is_none());
        let mut planar = frame.clone();
        planar.d = 2;
        assert!(evaluated(spec, &planar, A, B).is_none());
    }
    // A vanishing phase leaves the colours real: no lift, while the colour
    // tensor is still defined.
    let real = state(&[color([1., 2., -2.], [0.; 3]), color([2., -1., 2.], [0.; 3])]);
    assert!(evaluated(&dirac(DiracGamma::Scalar), &real, 0, 1).is_none());
    assert!(evaluated(&tensor(TensorMode::Components), &real, 0, 1).is_some());
}

#[test]
fn batch_evaluation_agrees_with_the_single_element_oracle() {
    let frame = reference();
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let elements = [pair(A, B), pair(B, A), pair(A, A), pair(A2, B2)];
    for spec in [tensor(TensorMode::Components), dirac(DiracGamma::Axial)] {
        let (mut values, mut valid) = (vec![0.; 12], vec![false; 4]);
        spec.evaluate_all(&elements, &frame, &context, &mut values, &mut valid);
        assert_eq!(valid, vec![true, true, false, true]);
        for (element, row) in elements.iter().zip(values.chunks_exact(3)) {
            if let Some(single) = evaluated(&spec, &frame, element.walkers[0], element.walkers[1]) {
                assert_eq!(row, single);
            }
        }
    }
}

#[test]
fn book_labels_name_the_defining_statements_of_the_fractal_set_chapters() {
    let linked = ChannelSpec::Dirac {
        gamma: DiracGamma::Vector,
        projector: ChiralProjector::Left,
        pairs: RoleClass::Any,
        link: PhaseLink::Su2,
    };
    let labels = [
        (dirac(DiracGamma::Scalar), "def-qft-dirac-lift"),
        (linked, "thm-sm-ew-operator-layers"),
        (
            tensor(TensorMode::Components),
            "prop-qft-color-gamma-parities",
        ),
        (tensor(TensorMode::Envelope), ""),
    ];
    for (spec, label) in &labels {
        assert_eq!(signature(spec).descriptor.book_label, *label);
    }
    // The crate is also built outside the book's repository, where only the
    // declared labels can be checked.
    let chapters = ["09_qft_calibration.md", "04_standard_model.md"].map(|file| {
        let path = format!(
            "{}/../../../docs/source/2_fractal_gas/2_fractal_set/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(path)
    });
    let [Ok(calibration), Ok(standard_model)] = chapters else {
        return;
    };
    for (_, label) in labels {
        let line = format!(":label: {label}");
        assert!(
            label.is_empty()
                || calibration
                    .lines()
                    .chain(standard_model.lines())
                    .any(|l| l.trim() == line),
            "{label} is not defined in the book"
        );
    }
}
