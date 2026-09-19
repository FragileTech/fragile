//! Edge spinors and the twistor arms against reference vectors evaluated from
//! the Pauli-matrix definitions, closed forms and the exact algebraic
//! identities of two unit spinors.
use algorithmic_gas::{
    GasConfig, GasError,
    boundary::{BoundaryPolicy, BoxDomain},
    physics::{
        qft::math::C,
        spectroscopy::{
            config::{
                ChannelSpec, FrameNormalization, MeasurementConfig, PairSelection,
                TwistorObservable,
            },
            contract::{
                Capabilities, Element, ElementKind, ExchangeParity, FrameState, LocalOperator,
                OperatorContext, Record,
            },
            fields::spinor::{EdgeSpinor, contraction, edge, pauli_bilinear, spin_two},
            operators::{catalog, channels},
        },
    },
};

/// Reference edges `(dx, dv)` at `dt = 0.1`, velocity scale `0.8`.
const EDGES: [([f64; 3], [f64; 3]); 4] = [
    ([0.2, 0.3, 0.4], [0.5, 0.6, 0.7]),
    ([-0.7, 0.1, 0.25], [0.3, -0.2, 0.9]),
    ([0.3, -0.4, -0.5], [0.1, 0.2, 0.3]),
    ([0.5, -0.1, 0.6], [-0.4, -0.3, 0.2]),
];
/// Reference triplets: indices of the two edges, `τ`, `W`, the five spin-2
/// components and `(λ_a† λ_b)²`.
type Triplet = ((usize, usize), [f64; 2], [[f64; 2]; 3], [f64; 5], [f64; 2]);
const TRIPLETS: [Triplet; 3] = [
    (
        (0, 1),
        [0.152826313541636, 0.162553123141373],
        [
            [-0.114908242220626, 0.236410931627515],
            [-0.691427787479635, 0.67345791889707],
            [0.0910033711704242, 0.201954945079293],
        ],
        [
            -0.0797620623366019,
            -0.0582013941303238,
            -0.198930416605627,
            -0.0475267972425475,
            -0.0191260098166491,
        ],
        [-0.0506635939665419, -0.948869005183891],
    ),
    (
        (0, 3),
        [0.770906851661833, -0.203278985852135],
        [
            [0.0428844372146531, -0.978575583055625],
            [-0.0756386924100967, -0.184706985483584],
            [0.794206270306015, 0.0746256742081758],
        ],
        [
            -0.183993468769705,
            0.107085951583607,
            -0.0462888404671746,
            -0.655753602408076,
            0.912253574101572,
        ],
        [-0.358971946951998, 0.0625470201831892],
    ),
    (
        (1, 2),
        [-0.717968174035193, 0.656103722675639],
        [
            [-0.912074984722969, 0.36097056022639],
            [-0.129404540458517, 0.127701849136694],
            [-0.388970373074139, -0.894102131881508],
        ],
        [
            0.0719300362369664,
            0.677514694524161,
            0.164513027937966,
            0.495783153438552,
            -0.815786289093159,
        ],
        [0.0538981341199761, 0.00404364499434107],
    ),
];
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{a} vs {b}");
}
fn close_complex(a: C, b: [f64; 2], tol: f64) {
    close(a.re, b[0], tol);
    close(a.im, b[1], tol);
}
fn reference(index: usize) -> EdgeSpinor {
    edge(EDGES[index].0, EDGES[index].1, 0.1, 0.8).unwrap()
}
fn twistor(observable: TwistorObservable, velocity_scale: f64) -> ChannelSpec {
    ChannelSpec::Twistor {
        observable,
        velocity_scale,
    }
}
fn triplet(walkers: [u32; 3]) -> Element {
    Element {
        walkers,
        kind: ElementKind::Triplet,
        weight: 1.,
        generation: [0; 3],
    }
}
fn capabilities(dt: f64) -> Capabilities {
    Capabilities {
        dimension: 3,
        time_step: Some(dt),
        ..Capabilities::default()
    }
}
/// Four walkers whose distance companions are `i + 1` and cloning companions
/// `i + 2` modulo 4; walkers 1, 2, 3 sit at the first, second and fourth
/// reference edge from walker 0.
fn frame() -> FrameState {
    FrameState {
        n: 4,
        d: 3,
        x: vec![
            0.1, -0.2, 0.3, 0.3, 0.1, 0.7, -0.6, -0.1, 0.55, 0.6, -0.3, 0.9,
        ],
        v: Some(vec![
            0.05, 0.1, -0.15, 0.55, 0.7, 0.55, 0.35, -0.1, 0.75, -0.35, -0.2, 0.05,
        ]),
        eligible: vec![true; 4],
        ..FrameState::default()
    }
}
/// Walker 0 at rest at the origin with the two edges `a`, `b` as walkers 1, 2.
fn star(a: ([f64; 3], [f64; 3]), b: ([f64; 3], [f64; 3])) -> FrameState {
    FrameState {
        n: 3,
        d: 3,
        x: [[0.; 3], a.0, b.0].concat(),
        v: Some([[0.; 3], a.1, b.1].concat()),
        eligible: vec![true; 3],
        ..FrameState::default()
    }
}
fn values(
    spec: &ChannelSpec,
    walkers: [u32; 3],
    state: &FrameState,
    gas: &GasConfig,
    capabilities: &Capabilities,
) -> Option<Vec<f64>> {
    let measurement = MeasurementConfig::default();
    let context = OperatorContext {
        gas,
        measurement: &measurement,
        capabilities,
    };
    let components = spec
        .signature(ElementKind::Triplet, &context)
        .unwrap()
        .components;
    let mut out = vec![0.; components];
    spec.evaluate(&triplet(walkers), state, &context, &mut out)
        .then_some(out)
}
fn value(observable: TwistorObservable, walkers: [u32; 3], state: &FrameState) -> Option<Vec<f64>> {
    values(
        &twistor(observable, 0.8),
        walkers,
        state,
        &GasConfig::default(),
        &capabilities(0.1),
    )
}
fn rotated(edge: ([f64; 3], [f64; 3]), angle: f64) -> ([f64; 3], [f64; 3]) {
    let (s, c) = angle.sin_cos();
    let turn = |u: [f64; 3]| [c * u[0] - s * u[1], s * u[0] + c * u[1], u[2]];
    (turn(edge.0), turn(edge.1))
}

#[test]
fn edge_spinors_reproduce_the_reference_vectors_and_the_closed_form_column_rule() {
    let lambda = [
        [
            [0.46994473774807, 0.526338106277839],
            [-0.263169053138919, 0.657922632847298],
        ],
        [
            [-0.754240887896975, 0.122783400355322],
            [-0.131553643237844, -0.631457487541654],
        ],
        [
            [0.49614615298182, 0.517717724850595],
            [0.647147156063244, -0.258858862425298],
        ],
        [
            [0.628719971720022, 0.143707422107434],
            [0.664646827246881, -0.377231983032013],
        ],
    ];
    let mu = [
        [
            [0.379715348100441, 0.473704295650055],
            [0.0150382316079383, 0.0488742527257993],
        ],
        [
            [-0.235042509251615, 0.498149795727305],
            [0.535423327978027, -0.0666538459071745],
        ],
        [
            [0.099229230596364, -0.0258858862425298],
            [0.74421922947273, -0.198458461192728],
        ],
        [
            [0.810150592130657, -0.0215561133161151],
            [-0.00359268555268588, 0.197597705397721],
        ],
    ];
    let gap = [0.256, -0.252, -0.52, 0.848];
    for (index, &(dx, dv)) in EDGES.iter().enumerate() {
        let s = reference(index);
        for r in 0..2 {
            close_complex(s.lambda[r], lambda[index][r], 1e-12);
            close_complex(s.mu[r], mu[index][r], 1e-12);
        }
        close(s.lambda[0].abs2() + s.lambda[1].abs2(), 1., 1e-12);
        // Column norms differ by 4 (dt dz − α (dx × dv)_z); column one wins
        // only when that is negative.
        let difference = 4. * (0.1 * dx[2] - 0.8 * (dx[0] * dv[1] - dx[1] * dv[0]));
        close(difference, gap[index], 1e-12);
        assert_eq!(s.column, usize::from(difference < 0.));
    }
}
#[test]
fn exact_column_ties_select_column_zero_and_null_or_overflowing_edges_have_no_spinor() {
    // Both column norms equal 11.0625 exactly on binary-exact inputs.
    let tie = edge([1., 2., 0.], [2., 4., 0.5], 1., 0.5).unwrap();
    let norm = 11.0625f64.sqrt();
    assert_eq!(tie.column, 0);
    close_complex(tie.lambda[0], [1. / norm, 0.25 / norm], 1e-15);
    close_complex(tie.lambda[1], [-1. / norm, 3. / norm], 1e-15);
    close_complex(tie.mu[0], [1.80395046720679, 1.57845665880594], 1e-12);
    close_complex(tie.mu[1], [-0.150329205600566, 1.57845665880594], 1e-12);
    let zero = edge([0.; 3], [0.; 3], 1., 1.).unwrap();
    assert_eq!(
        (zero.column, zero.lambda, zero.mu),
        (0, [C::ONE, C::ZERO], [C::ONE, C::ZERO])
    );
    // Without a velocity part the sign of dz alone decides the column.
    let down = edge([0., 0., -2.], [9.; 3], 1., 0.).unwrap();
    assert_eq!(
        (down.column, down.lambda, down.mu),
        (1, [C::ZERO, C::ONE], [C::ZERO, C::from(3.)])
    );
    let up = edge([0.6, -0.8, 0.5], [1., 2., 3.], 0.25, 0.).unwrap();
    assert_eq!(up.column, 0);
    close_complex(up.lambda[0], [0.6, 0.], 1e-15);
    close_complex(up.lambda[1], [0.48, -0.64], 1e-15);
    close_complex(up.mu[0], [1.25, 0.], 1e-15);
    close_complex(up.mu[1], [0.24, -0.32], 1e-15);
    assert!(edge([0.; 3], [0.; 3], 0., 1.).is_none());
    assert!(edge([0.; 3], [0.; 3], 1e-12, 1.).is_none());
    assert!(edge([0.; 3], [0.; 3], 2e-12, 1.).is_some());
    assert!(edge([f64::NAN, 0., 0.], [0.; 3], 1., 1.).is_none());
    assert!(edge([0.; 3], [0., f64::INFINITY, 0.], 1., 1.).is_none());
    assert!(edge([0.; 3], [0.; 3], 1., f64::NAN).is_none());
    // The column norm overflows: no unit spinor exists in double precision.
    assert!(edge([1e200, 0., 0.], [0.; 3], 1., 1.).is_none());
}
#[test]
fn spinor_depends_only_on_ratios_to_the_time_step_and_on_the_scaled_velocity() {
    for (index, &(dx, dv)) in EDGES.iter().enumerate() {
        let s = reference(index);
        let scaled = edge(dx.map(|x| 3.5 * x), dv.map(|x| 3.5 * x), 0.35, 0.8).unwrap();
        assert_eq!(scaled.column, s.column);
        for r in 0..2 {
            close((scaled.lambda[r] - s.lambda[r]).abs(), 0., 1e-12);
            close((scaled.mu[r] - s.mu[r] * 3.5).abs(), 0., 1e-12);
        }
        // Only the product of velocity scale and velocity enters.
        assert_eq!(edge(dx, dv.map(|x| x / 4.), 0.1, 3.2).unwrap(), s);
    }
}
#[test]
fn triplet_bilinears_reproduce_the_reference_vectors_and_the_fierz_identities() {
    for ((ia, ib), tau, w, q, square) in TRIPLETS {
        let (a, b) = (reference(ia), reference(ib));
        let (t, bilinear) = (contraction(&a, &b), pauli_bilinear(&a, &b));
        close_complex(t, tau, 1e-12);
        for axis in 0..3 {
            close_complex(bilinear[axis], w[axis], 1e-12);
        }
        let tensor = spin_two(&bilinear);
        for m in 0..5 {
            close(tensor[m], q[m], 1e-12);
        }
        let overlap = a.lambda[0].conj() * b.lambda[0] + a.lambda[1].conj() * b.lambda[1];
        close(t.abs2() + overlap.abs2(), 1., 1e-12);
        close(
            bilinear.iter().map(|c| c.abs2()).sum::<f64>(),
            1. + t.abs2(),
            1e-12,
        );
        let sum = bilinear.iter().fold(C::ZERO, |s, &c| s + c * c);
        close_complex(sum, square, 1e-12);
        close((sum - overlap * overlap).abs(), 0., 1e-12);
    }
}
#[test]
fn spin_two_norm_is_the_traceless_part_of_the_symmetric_square_not_the_plain_component_sum() {
    let expected = [0.10126999974322, 1.3571466791888, 1.89383746024693];
    for (((ia, ib), ..), invariant) in TRIPLETS.into_iter().zip(expected) {
        let w = pauli_bilinear(&reference(ia), &reference(ib));
        let q = spin_two(&w);
        // Frobenius norm of Re(W Wᵀ) without its trace, entry by entry.
        let trace = (0..3).map(|a| (w[a] * w[a]).re).sum::<f64>() / 3.;
        let mut frobenius = 0.;
        for a in 0..3 {
            for b in 0..3 {
                let entry = (w[a] * w[b]).re - if a == b { trace } else { 0. };
                frobenius += entry * entry;
            }
        }
        let weighted = 2. * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) + q[3] * q[3] + q[4] * q[4];
        close(weighted, invariant, 1e-12);
        close(weighted, frobenius, 1e-12);
        assert!((q.iter().map(|x| x * x).sum::<f64>() - frobenius).abs() > 1e-3);
    }
}
#[test]
fn squared_contraction_is_the_two_twistor_mass_and_a_repeated_twistor_is_null() {
    let momentum = |a: &EdgeSpinor, b: &EdgeSpinor| {
        let entry = |r: usize, c: usize| {
            a.lambda[r] * a.lambda[c].conj() + b.lambda[r] * b.lambda[c].conj()
        };
        [entry(0, 0), entry(0, 1), entry(1, 0), entry(1, 1)]
    };
    for ((ia, ib), ..) in TRIPLETS {
        let (a, b) = (reference(ia), reference(ib));
        let p = momentum(&a, &b);
        let determinant = p[0] * p[3] - p[1] * p[2];
        close((p[0] + p[3]).re, 2., 1e-12);
        close(determinant.re, contraction(&a, &b).abs2(), 1e-12);
        close(determinant.im, 0., 1e-12);
    }
    close(
        contraction(&reference(0), &reference(1)).abs2(),
        0.049779399953741,
        1e-12,
    );
    // One twistor twice: a null momentum, a real bilinear of unit length.
    let a = reference(0);
    let p = momentum(&a, &a);
    close((p[0] * p[3] - p[1] * p[2]).abs(), 0., 1e-12);
    assert_eq!(contraction(&a, &a), C::ZERO);
    let w = pauli_bilinear(&a, &a);
    let bloch = [0.445229681978799, 0.895406360424028, -0.00424028268551233];
    for axis in 0..3 {
        assert_eq!(w[axis].im, 0.);
        close(w[axis].re, bloch[axis], 1e-12);
    }
    close(w.iter().map(|c| c.abs2()).sum::<f64>(), 1., 1e-12);
}
#[test]
fn role_exchange_flips_the_contraction_conjugates_the_bilinear_and_keeps_the_spin_two_part() {
    for ((ia, ib), ..) in TRIPLETS {
        let (a, b) = (reference(ia), reference(ib));
        assert_eq!(contraction(&b, &a), -contraction(&a, &b));
        let (forward, backward) = (pauli_bilinear(&a, &b), pauli_bilinear(&b, &a));
        for axis in 0..3 {
            close((backward[axis] - forward[axis].conj()).abs(), 0., 1e-15);
        }
        let (q, p) = (spin_two(&forward), spin_two(&backward));
        for m in 0..5 {
            close(q[m], p[m], 1e-15);
        }
    }
    let state = frame();
    for (observable, sign) in [
        (TwistorObservable::Scalar, -1.),
        (TwistorObservable::Pseudoscalar, -1.),
        (TwistorObservable::Axial, -1.),
        (TwistorObservable::Abs2, 1.),
        (TwistorObservable::Vector, 1.),
        (TwistorObservable::Tensor, 1.),
        (TwistorObservable::TensorMean, 1.),
    ] {
        let forward = value(observable, [0, 1, 2], &state).unwrap();
        let backward = value(observable, [0, 2, 1], &state).unwrap();
        for (f, b) in forward.iter().zip(&backward) {
            close(*b, sign * f, 1e-15);
        }
    }
}
#[test]
fn mirror_of_the_second_axis_with_velocity_reversal_conjugates_every_spinor_exactly() {
    for (index, &(dx, dv)) in EDGES.iter().enumerate() {
        let s = reference(index);
        let m = edge([dx[0], -dx[1], dx[2]], [-dv[0], dv[1], -dv[2]], 0.1, 0.8).unwrap();
        assert_eq!(m.column, s.column);
        assert_eq!(m.lambda, s.lambda.map(C::conj));
    }
    let mirror = |e: ([f64; 3], [f64; 3])| ([e.0[0], -e.0[1], e.0[2]], [-e.1[0], e.1[1], -e.1[2]]);
    for ((ia, ib), ..) in TRIPLETS {
        let plain = star(EDGES[ia], EDGES[ib]);
        let mirrored = star(mirror(EDGES[ia]), mirror(EDGES[ib]));
        let read = |observable, state: &FrameState| value(observable, [0, 1, 2], state).unwrap();
        assert_eq!(
            read(TwistorObservable::Scalar, &mirrored),
            read(TwistorObservable::Scalar, &plain)
        );
        assert_eq!(
            read(TwistorObservable::Pseudoscalar, &mirrored)[0],
            -read(TwistorObservable::Pseudoscalar, &plain)[0]
        );
        let (v, a) = (
            read(TwistorObservable::Vector, &plain),
            read(TwistorObservable::Axial, &plain),
        );
        assert_eq!(
            read(TwistorObservable::Vector, &mirrored),
            [v[0], -v[1], v[2]]
        );
        assert_eq!(
            read(TwistorObservable::Axial, &mirrored),
            [-a[0], a[1], -a[2]]
        );
    }
}
#[test]
fn rotation_about_the_third_axis_mixes_scalar_and_pseudoscalar_but_keeps_the_modulus() {
    // Equal columns rotate the contraction by ±0.7 rad, mixed columns leave it.
    let expected = [
        [0.152826313541636, 0.162553123141373],
        [0.720578000747321, 0.341155484885156],
        [-0.126458725819588, 0.964343602923952],
    ];
    let phase = [0., 0.7, -0.7];
    for (index, ((ia, ib), tau, ..)) in TRIPLETS.into_iter().enumerate() {
        let state = star(rotated(EDGES[ia], 0.7), rotated(EDGES[ib], 0.7));
        let read = |observable| value(observable, [0, 1, 2], &state).unwrap()[0];
        let (s, p) = (
            read(TwistorObservable::Scalar),
            read(TwistorObservable::Pseudoscalar),
        );
        close(s, expected[index][0], 1e-12);
        close(p, expected[index][1], 1e-12);
        let turned = C::new(tau[0], tau[1]) * C::phase(phase[index]);
        close_complex(turned, [s, p], 1e-12);
        close(
            read(TwistorObservable::Abs2),
            tau[0] * tau[0] + tau[1] * tau[1],
            1e-12,
        );
    }
}
#[test]
fn spin_two_components_exceed_one_at_the_extremal_spinor_pairs() {
    let spinor = |lambda: [C; 2]| EdgeSpinor {
        lambda,
        mu: [C::ZERO; 2],
        column: 0,
    };
    let q = spin_two(&pauli_bilinear(
        &spinor([C::ZERO, C::ONE]),
        &spinor([C::ONE, C::ZERO]),
    ));
    close(q[3], 2f64.sqrt(), 1e-15);
    let h = 0.5f64.sqrt();
    let q = spin_two(&pauli_bilinear(
        &spinor([C::from(h), C::from(h)]),
        &spinor([C::from(h), C::from(-h)]),
    ));
    close(q[3], h, 1e-12);
    close(q[4], 3. / 6f64.sqrt(), 1e-12);
}
#[test]
fn every_arm_reproduces_the_one_frame_fixture_and_its_fixed_normalisation_means() {
    let scalars = [
        [0.152826313541636, 0.162553123141373, 0.049779399953741],
        [-0.614136293555917, 0.764613670363437, 0.961797451969247],
        [-0.221748963587469, 0.491252949538483, 0.290502063282376],
        [0.498963425301775, -0.0374604818791425, 0.250367787491498],
    ];
    let vector = [
        [-0.114908242220626, -0.691427787479635, 0.0910033711704242],
        [-0.59000237544871, 0.693650361491915, -0.364486094327721],
        [0.491610856558338, 0.0233150858648572, 0.288677576300403],
        [0.364921156809061, -0.298919355402953, -0.789473284772676],
    ];
    let axial = [
        [0.236410931627515, 0.67345791889707, 0.201954945079293],
        [0.652095270711293, 0.701082329650756, 0.288008744564154],
        [0.508432486491536, 0.742861584496734, -0.39318642609814],
        [-0.622557532181939, -0.0565729726338069, 0.117478109838671],
    ];
    let tensor = [
        [
            -0.0797620623366019,
            -0.0582013941303238,
            -0.198930416605627,
            -0.0475267972425475,
            -0.0191260098166491,
        ],
        [
            -0.8664278325556,
            0.0272385212175965,
            -0.454743752688027,
            -0.0472063285733313,
            0.0764621287258327,
        ],
        [
            -0.366233013191963,
            0.341825782830024,
            0.298813633972575,
            0.377932580819901,
            0.173750657891782,
        ],
        [
            -0.144301927197364,
            -0.214958622202532,
            0.242634931285075,
            -0.240814201654253,
            0.566318792161395,
        ],
    ];
    let mean = [
        -0.0807093360263498,
        -0.252935452774706,
        0.165217928464464,
        0.0417757944784641,
    ];
    let state = frame();
    let all = |v: Vec<f64>, expected: &[f64]| {
        assert_eq!(v.len(), expected.len());
        for (a, b) in v.iter().zip(expected) {
            close(*a, *b, 1e-12);
        }
    };
    for i in 0..4u32 {
        let (walkers, row) = ([i, (i + 1) % 4, (i + 2) % 4], i as usize);
        let read = |observable| value(observable, walkers, &state).unwrap();
        all(read(TwistorObservable::Scalar), &scalars[row][..1]);
        all(read(TwistorObservable::Pseudoscalar), &scalars[row][1..2]);
        all(read(TwistorObservable::Abs2), &scalars[row][2..]);
        all(read(TwistorObservable::Vector), &vector[row]);
        all(read(TwistorObservable::Axial), &axial[row]);
        all(read(TwistorObservable::Tensor), &tensor[row]);
        all(read(TwistorObservable::TensorMean), &mean[row..=row]);
    }
    // The batch entry point of the accumulator agrees with the oracle, and
    // the frame mean over all four valid walkers is the fixed 1/N mean.
    let (gas, measurement, capabilities) = (
        GasConfig::default(),
        MeasurementConfig::default(),
        capabilities(0.1),
    );
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let elements: Vec<Element> = (0..4u32)
        .map(|i| triplet([i, (i + 1) % 4, (i + 2) % 4]))
        .collect();
    for (observable, expected) in [
        (TwistorObservable::Scalar, -0.0460238795749937),
        (TwistorObservable::Pseudoscalar, 0.345239815291038),
        (TwistorObservable::Abs2, 0.388111675674215),
    ] {
        let (mut out, mut valid) = (vec![0.; 4], vec![false; 4]);
        twistor(observable, 0.8).evaluate_all(&elements, &state, &context, &mut out, &mut valid);
        assert_eq!(valid, [true; 4]);
        close(out.iter().sum::<f64>() / 4., expected, 1e-12);
    }
}
#[test]
fn ineligible_repeated_missing_and_nonfinite_walkers_mask_the_element() {
    let scalar = |walkers, state: &FrameState| value(TwistorObservable::Scalar, walkers, state);
    let cycle = |i: u32| [i, (i + 1) % 4, (i + 2) % 4];
    let mut state = frame();
    state.eligible[2] = false;
    for i in 0..3 {
        assert_eq!(scalar(cycle(i), &state), None);
    }
    close(
        scalar(cycle(3), &state).unwrap()[0],
        0.498963425301775,
        1e-12,
    );
    let state = frame();
    for walkers in [[0, 1, 1], [0, 0, 2], [0, 1, 0], [0, 1, 4], [7, 1, 2]] {
        assert_eq!(scalar(walkers, &state), None);
    }
    let mut nonfinite = frame();
    nonfinite.x[3] = f64::NAN;
    for i in [0, 1, 3] {
        assert_eq!(scalar(cycle(i), &nonfinite), None);
    }
    assert!(scalar(cycle(2), &nonfinite).is_some());
    let mut still = frame();
    still.v = None;
    assert_eq!(scalar(cycle(0), &still), None);
    let mut short = frame();
    short.v.as_mut().unwrap().truncate(9);
    assert_eq!(scalar(cycle(1), &short), None);
    assert!(scalar(cycle(0), &short).is_some());
    let mut planar = frame();
    planar.d = 2;
    assert_eq!(scalar(cycle(0), &planar), None);
    // Without a time step, on another element kind or with a buffer of
    // another width nothing is written.
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let spec = twistor(TwistorObservable::Vector, 0.8);
    let run = |capabilities: &Capabilities, kind, width| {
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities,
        };
        let element = Element {
            kind,
            ..triplet([0, 1, 2])
        };
        spec.evaluate(&element, &state, &context, &mut vec![0.; width])
    };
    assert!(run(&capabilities(0.1), ElementKind::Triplet, 3));
    assert!(!run(&capabilities(0.1), ElementKind::Triplet, 1));
    assert!(!run(&capabilities(0.1), ElementKind::CloningPair, 3));
    let stepless = Capabilities {
        time_step: None,
        ..capabilities(0.1)
    };
    assert!(!run(&stepless, ElementKind::Triplet, 3));
}
#[test]
fn full_spatial_inversion_has_no_definite_sign_and_becomes_even_for_a_small_time_step() {
    let inverted = |e: ([f64; 3], [f64; 3])| (e.0.map(|x| -x), e.1.map(|x| -x));
    // Contraction of the first two reference triplets after x → −x, v → −v.
    let expected = [
        [-0.559961484367176, -0.777336805552418],
        [0.693370835885988, 0.145827993261736],
    ];
    for (((ia, ib), tau, ..), after) in TRIPLETS.into_iter().zip(expected) {
        let state = star(inverted(EDGES[ia]), inverted(EDGES[ib]));
        let read = |observable| value(observable, [0, 1, 2], &state).unwrap()[0];
        let (s, p) = (
            read(TwistorObservable::Scalar),
            read(TwistorObservable::Pseudoscalar),
        );
        close(s, after[0], 1e-12);
        close(p, after[1], 1e-12);
        assert!((s.abs() - tau[0].abs()).abs() > 0.05 && (p.abs() - tau[1].abs()).abs() > 0.05);
        let modulus = tau[0] * tau[0] + tau[1] * tau[1];
        assert!((read(TwistorObservable::Abs2) - modulus).abs() > 0.1);
    }
    // The time component alone breaks the inversion symmetry: with a step far
    // below the edge lengths every arm, the imaginary parts included, is even.
    let (gas, capabilities) = (GasConfig::default(), capabilities(1e-6));
    for ((ia, ib), ..) in TRIPLETS {
        let plain = star(EDGES[ia], EDGES[ib]);
        let mirrored = star(inverted(EDGES[ia]), inverted(EDGES[ib]));
        for &observable in TwistorObservable::ALL {
            let spec = twistor(observable, 0.8);
            let read = |state| values(&spec, [0, 1, 2], state, &gas, &capabilities).unwrap();
            for (a, b) in read(&plain).iter().zip(read(&mirrored)) {
                close(*a, b, 1e-4);
            }
        }
    }
}
#[test]
fn overflowing_coordinates_mask_the_element_instead_of_writing_zeros() {
    let mut far = frame();
    far.x[3] = 1e200;
    for &observable in TwistorObservable::ALL {
        for walkers in [[0, 1, 2], [1, 2, 3], [3, 0, 1]] {
            assert_eq!(value(observable, walkers, &far), None);
        }
        assert!(value(observable, [2, 3, 0], &far).is_some());
    }
}
#[test]
fn batch_evaluation_equals_the_single_element_oracle_exactly_and_flags_masked_elements() {
    let (gas, measurement, capabilities) = (
        GasConfig::default(),
        MeasurementConfig::default(),
        capabilities(0.1),
    );
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let mut state = frame();
    state.eligible[2] = false;
    // The last element repeats a companion and stays masked as well.
    let walkers = [[0, 1, 2], [3, 0, 1], [1, 2, 3], [3, 1, 0], [3, 0, 0]];
    let elements = walkers.map(triplet);
    for &observable in TwistorObservable::ALL {
        let spec = twistor(observable, 0.8);
        let components = spec
            .signature(ElementKind::Triplet, &context)
            .unwrap()
            .components;
        let (mut out, mut valid) = (vec![0.; 5 * components], vec![true; 5]);
        spec.evaluate_all(&elements, &state, &context, &mut out, &mut valid);
        assert_eq!(valid, [false, true, false, true, false]);
        for row in [1, 3] {
            let single = value(observable, walkers[row], &state).unwrap();
            let batch = &out[row * components..(row + 1) * components];
            assert_eq!(
                batch.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                single.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
#[test]
fn catalog_describes_every_arm_in_a_run_without_velocities_or_time_step() {
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let mut jump = Capabilities {
        time_step: None,
        ..capabilities(0.1)
    };
    jump.missing
        .insert(Record::Velocities, "no velocities".into());
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &jump,
    };
    let specs: Vec<ChannelSpec> = TwistorObservable::ALL
        .iter()
        .map(|&observable| twistor(observable, 1.))
        .collect();
    let rows = catalog(&context, &specs, &Default::default()).unwrap();
    assert_eq!(rows.len(), 7);
    for (row, width) in rows.iter().zip([1, 1, 1, 3, 3, 5, 1]) {
        assert_eq!(row.kind, ElementKind::Triplet);
        assert_eq!(row.family, "twistor");
        assert!(!row.standard);
        let signature = row.signature.as_ref().unwrap();
        assert_eq!(signature.components, width);
        assert_eq!(signature.requires.dimension, Some(3));
        assert_eq!(row.availability.reason(), Some("no velocities"));
    }
}
#[test]
fn signatures_state_components_role_exchange_parity_fixed_normalisation_and_no_colour() {
    let (gas, measurement, capabilities) = (
        GasConfig::default(),
        MeasurementConfig::default(),
        Capabilities::nominal(),
    );
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    for (observable, components, exchange) in [
        (TwistorObservable::Scalar, 1, ExchangeParity::Odd),
        (TwistorObservable::Pseudoscalar, 1, ExchangeParity::Odd),
        (TwistorObservable::Abs2, 1, ExchangeParity::Even),
        (TwistorObservable::Vector, 3, ExchangeParity::Even),
        (TwistorObservable::Axial, 3, ExchangeParity::Odd),
        (TwistorObservable::Tensor, 5, ExchangeParity::Even),
        (TwistorObservable::TensorMean, 1, ExchangeParity::Even),
    ] {
        let spec = twistor(observable, 1.);
        assert_eq!(spec.kinds(PairSelection::Distance), [ElementKind::Triplet]);
        let signature = spec.signature(ElementKind::Triplet, &context).unwrap();
        assert_eq!(signature.components, components);
        assert_eq!(signature.exchange, exchange);
        assert_eq!(signature.normalization, Some(FrameNormalization::FixedN));
        assert!(signature.correlatable && !signature.degenerate);
        assert_eq!(
            signature.requires.records.iter().collect::<Vec<_>>(),
            [&Record::Velocities]
        );
        assert_eq!(signature.requires.dimension, Some(3));
        assert_eq!(
            signature.descriptor.book_label,
            "def-effective-twistor-operators"
        );
        assert_eq!(signature.descriptor.spatial_parity, None);
        assert!(!signature.descriptor.definition.is_empty());
        assert!(
            signature
                .descriptor
                .note
                .starts_with("velocity scale 1 in time units")
        );
    }
    assert_eq!(TwistorObservable::ALL.len(), 7);
}
#[test]
fn twistor_channels_need_triplets_a_time_step_three_dimensions_and_velocities_only() {
    let gas = GasConfig::default();
    let measurement = MeasurementConfig {
        channels: vec![twistor(TwistorObservable::Tensor, 0.5)],
        ..MeasurementConfig::default()
    };
    let listed = |capabilities: &Capabilities| {
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities,
        };
        let mut listed = channels(&context, &[]).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "twistor/tensor/a0.5/triplet");
        listed.remove(0)
    };
    // A run without colour or fitness records measures twistor channels.
    let mut colourless = capabilities(0.1);
    for record in [Record::Color, Record::Fitness, Record::Graph] {
        colourless.missing.insert(record, "not recorded".into());
    }
    let channel = listed(&colourless);
    assert!(channel.availability.is_available());
    let requires = channel.signature.unwrap().requires;
    assert_eq!(
        requires.records.into_iter().collect::<Vec<_>>(),
        [
            Record::Velocities,
            Record::DistanceCompanions,
            Record::CloningCompanions
        ]
    );
    let mut still = capabilities(0.1);
    still
        .missing
        .insert(Record::Velocities, "no velocities".into());
    assert_eq!(listed(&still).availability.reason(), Some("no velocities"));
    let planar = Capabilities {
        dimension: 2,
        ..capabilities(0.1)
    };
    assert!(!listed(&planar).availability.is_available());
    let stepless = Capabilities {
        time_step: None,
        ..capabilities(0.1)
    };
    let channel = listed(&stepless);
    assert!(channel.signature.is_none());
    let reason = channel.availability.reason().unwrap();
    assert!(reason.contains("time step"), "{reason}");
    let nominal = capabilities(0.1);
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &nominal,
    };
    let spec = twistor(TwistorObservable::Scalar, 1.);
    for kind in [
        ElementKind::Site,
        ElementKind::DistancePair,
        ElementKind::CloningPair,
    ] {
        assert!(matches!(
            spec.signature(kind, &context),
            Err(GasError::Capability(_))
        ));
    }
    for velocity_scale in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(matches!(
            twistor(TwistorObservable::Scalar, velocity_scale)
                .signature(ElementKind::Triplet, &context),
            Err(GasError::Configuration(_))
        ));
    }
}
#[test]
fn velocity_scale_enters_the_id_the_note_and_the_values() {
    let state = frame();
    let (gas, capabilities) = (GasConfig::default(), capabilities(0.1));
    let read = |velocity_scale, state: &FrameState| {
        let spec = twistor(TwistorObservable::Tensor, velocity_scale);
        values(&spec, [0, 1, 2], state, &gas, &capabilities).unwrap()
    };
    assert_eq!(
        twistor(TwistorObservable::Tensor, 1.).id(),
        "twistor/tensor"
    );
    assert_eq!(
        twistor(TwistorObservable::Tensor, 0.8).id(),
        "twistor/tensor/a0.8"
    );
    let measurement = MeasurementConfig::default();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let note = twistor(TwistorObservable::Tensor, 0.8)
        .signature(ElementKind::Triplet, &context)
        .unwrap()
        .descriptor
        .note;
    assert!(note.starts_with("velocity scale 0.8 in time units"));
    assert!((read(0.8, &state)[0] - read(1., &state)[0]).abs() > 1e-3);
    // Only the product of velocity scale and velocity enters.
    let mut slow = frame();
    for v in slow.v.as_mut().unwrap() {
        *v /= 4.;
    }
    assert_eq!(read(3.2, &slow), read(0.8, &state));
}
#[test]
fn periodic_box_displacement_is_the_minimum_image() {
    let periodic = GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-2.; 3],
                upper: vec![2.; 3],
            },
        },
        ..GasConfig::default()
    };
    let capabilities = capabilities(0.1);
    let spec = twistor(TwistorObservable::Tensor, 0.8);
    let state = frame();
    // Walker 1 leaves through one face and walker 2 through another: the
    // images of the two edges are the edges of the unwrapped frame.
    let mut wrapped = frame();
    wrapped.x[3] -= 4.;
    wrapped.x[7] += 4.;
    let read = |state, gas| values(&spec, [0, 1, 2], state, gas, &capabilities).unwrap();
    let unbounded = GasConfig::default();
    let (open, image, raw) = (
        read(&state, &unbounded),
        read(&wrapped, &periodic),
        read(&wrapped, &unbounded),
    );
    for m in 0..5 {
        close(image[m], open[m], 1e-12);
    }
    assert!((raw[0] - open[0]).abs() > 1e-3);
    let flat = GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-2.; 2],
                upper: vec![2.; 2],
            },
        },
        ..GasConfig::default()
    };
    assert_eq!(values(&spec, [0, 1, 2], &state, &flat, &capabilities), None);
}
#[test]
fn pauli_eigenstates_fix_the_axis_order_the_sign_of_the_second_matrix_and_the_spin_two_order() {
    let spinor = |lambda: [C; 2]| EdgeSpinor {
        lambda,
        mu: [C::ZERO; 2],
        column: 0,
    };
    let h = 0.5f64.sqrt();
    // An eigenstate of a Pauli matrix has its Bloch vector along that axis.
    for (lambda, bloch) in [
        ([C::from(h), C::from(h)], [1., 0., 0.]),
        ([C::from(h), C::new(0., h)], [0., 1., 0.]),
        ([C::from(h), C::new(0., -h)], [0., -1., 0.]),
        ([C::ONE, C::ZERO], [0., 0., 1.]),
        ([C::ZERO, C::ONE], [0., 0., -1.]),
    ] {
        let w = pauli_bilinear(&spinor(lambda), &spinor(lambda));
        for axis in 0..3 {
            close_complex(w[axis], [bloch[axis], 0.], 1e-15);
        }
    }
    // ⟨up|σ|down⟩ = (1, −i, 0) and ε(up, down) = +1, first spinor first.
    let (up, down) = (spinor([C::ONE, C::ZERO]), spinor([C::ZERO, C::ONE]));
    assert_eq!(
        pauli_bilinear(&up, &down),
        [C::ONE, C::new(0., -1.), C::ZERO]
    );
    assert_eq!(contraction(&up, &down), C::ONE);
    // The first spinor alone is conjugated: ⟨(1, 0)|σ|(i, 0)⟩ = (0, 0, i),
    // and the contraction conjugates nothing: ε((i, 0), (0, i)) = −1.
    let turned = spinor([C::new(0., 1.), C::ZERO]);
    assert_eq!(pauli_bilinear(&up, &turned)[2], C::new(0., 1.));
    assert_eq!(pauli_bilinear(&turned, &up)[2], C::new(0., -1.));
    let twisted = spinor([C::ZERO, C::new(0., 1.)]);
    assert_eq!(contraction(&turned, &twisted), C::from(-1.));
    // λ = (2, 1 + 2i)/3 has the Bloch vector (4, 8, −1)/9: five distinct
    // spin-2 components, each in its own slot.
    let generic = spinor([C::from(2. / 3.), C::new(1. / 3., 2. / 3.)]);
    let w = pauli_bilinear(&generic, &generic);
    for (axis, n) in [4. / 9., 8. / 9., -1. / 9.].into_iter().enumerate() {
        close_complex(w[axis], [n, 0.], 1e-15);
    }
    let expected = [
        32. / 81.,
        -4. / 81.,
        -8. / 81.,
        -48. / 81. / 2f64.sqrt(),
        -78. / 81. / 6f64.sqrt(),
    ];
    for (q, e) in spin_two(&w).into_iter().zip(expected) {
        close(q, e, 1e-15);
    }
}
#[test]
fn hand_computed_triplet_fixes_the_role_order_the_edge_direction_and_the_velocity_term() {
    // At dt = 1 and velocity scale 1/2 the edge (2, 0, 1) at rest has the
    // bispinor [[2, 2], [2, 0]] and the spinor (1, 1)/√2; the edge (0, 2, 1) at
    // rest and the edge (0, 0, 1) with velocity difference (4, 0, 0) both have
    // the dominant column (2, 2i) and the spinor (1, i)/√2. Hence
    // τ = (−1 + i)/2 and W = ((1 + i)/2, (1 + i)/2, (1 − i)/2). The anchor
    // moves and sits away from the origin: only differences may enter.
    let (anchor, drift) = ([5., -3., 7.], [1., 2., 3.]);
    let at = |offset: [f64; 3]| -> [f64; 3] { std::array::from_fn(|a| anchor[a] + offset[a]) };
    let state = FrameState {
        n: 4,
        d: 3,
        x: [anchor, at([2., 0., 1.]), at([0., 0., 1.]), at([0., 2., 1.])].concat(),
        v: Some([drift, drift, [5., 2., 3.], drift].concat()),
        eligible: vec![true; 4],
        ..FrameState::default()
    };
    let (gas, capabilities) = (GasConfig::default(), capabilities(1.));
    let read = |observable, walkers| {
        values(
            &twistor(observable, 0.5),
            walkers,
            &state,
            &gas,
            &capabilities,
        )
        .unwrap()
    };
    let expected: [(TwistorObservable, &[f64]); 7] = [
        (TwistorObservable::Scalar, &[-0.5]),
        (TwistorObservable::Pseudoscalar, &[0.5]),
        (TwistorObservable::Abs2, &[0.5]),
        (TwistorObservable::Vector, &[0.5, 0.5, 0.5]),
        (TwistorObservable::Axial, &[0.5, 0.5, -0.5]),
        (TwistorObservable::Tensor, &[0., 0.5, 0.5, 0., 0.]),
        (TwistorObservable::TensorMean, &[0.2]),
    ];
    for (observable, expected) in expected {
        for second in [2, 3] {
            let forward = read(observable, [0, 1, second]);
            assert_eq!(forward.len(), expected.len());
            for (f, e) in forward.iter().zip(expected) {
                close(*f, *e, 1e-15);
            }
        }
    }
    // The distance companion is the first spinor: with the roles swapped the
    // contraction is (1 − i)/2 and the imaginary part of W changes sign.
    close(read(TwistorObservable::Scalar, [0, 3, 1])[0], 0.5, 1e-15);
    close(
        read(TwistorObservable::Pseudoscalar, [0, 2, 1])[0],
        -0.5,
        1e-15,
    );
    for (a, e) in read(TwistorObservable::Axial, [0, 3, 1])
        .iter()
        .zip([-0.5, -0.5, 0.5])
    {
        close(*a, e, 1e-15);
    }
    // The same two companions seen from another anchor are other edges: the
    // edge to walker 0 is (−2, 0, −1) with bispinor [[0, −2], [−2, 2]] and
    // spinor (−1, 1)/√2 from column one, the edge to walker 3 is (−2, 2, 0)
    // with two columns of squared norm 9, so the tie keeps column zero
    // (1, −2 + 2i) and τ = (1 − 2i)/(3√2).
    let h = 0.5f64.sqrt();
    close(read(TwistorObservable::Scalar, [1, 0, 3])[0], h / 3., 1e-15);
    close(
        read(TwistorObservable::Pseudoscalar, [1, 0, 3])[0],
        -2. * h / 3.,
        1e-15,
    );
}
#[test]
fn minimum_image_acts_per_axis_on_positions_only_and_only_for_a_periodic_policy() {
    // Widths 3, 4, 5: a wrap with the width of another axis misses the image.
    let domain = BoxDomain {
        lower: vec![-1.5, -2., -2.5],
        upper: vec![1.5, 2., 2.5],
    };
    let periodic = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain: domain.clone(),
    };
    let boundary = |boundary| GasConfig {
        boundary,
        ..GasConfig::default()
    };
    // Velocities ten times larger at a tenth of the velocity scale: the same
    // observables, with velocity differences far beyond every half width.
    let mut wrapped = frame();
    for v in wrapped.v.as_mut().unwrap() {
        *v *= 10.;
    }
    let fast = wrapped.clone();
    for (index, shift) in [(3, -3.), (5, 5.), (7, 4.), (8, -5.)] {
        wrapped.x[index] += shift;
    }
    let capabilities = capabilities(0.1);
    let (_, tau, w, q, _) = TRIPLETS[0];
    let read = |observable, state: &FrameState, gas: &GasConfig| {
        values(
            &twistor(observable, 0.08),
            [0, 1, 2],
            state,
            gas,
            &capabilities,
        )
        .unwrap()
    };
    let composed = BoundaryPolicy::Composed {
        policies: vec![BoundaryPolicy::ExternalTermination, periodic.clone()],
    };
    for gas in [boundary(periodic), boundary(composed)] {
        for state in [&wrapped, &fast] {
            close(
                read(TwistorObservable::Scalar, state, &gas)[0],
                tau[0],
                1e-12,
            );
            close(
                read(TwistorObservable::Pseudoscalar, state, &gas)[0],
                tau[1],
                1e-12,
            );
            let (vector, axial) = (
                read(TwistorObservable::Vector, state, &gas),
                read(TwistorObservable::Axial, state, &gas),
            );
            for axis in 0..3 {
                close_complex(C::new(vector[axis], axial[axis]), w[axis], 1e-12);
            }
            for (a, b) in read(TwistorObservable::Tensor, state, &gas).iter().zip(q) {
                close(*a, b, 1e-12);
            }
        }
    }
    // An absorbing box of the same size wraps nothing.
    let absorbing = boundary(BoundaryPolicy::AbsorbingBox {
        field: "positions".into(),
        domain,
    });
    let open = GasConfig::default();
    for &observable in TwistorObservable::ALL {
        assert_eq!(
            read(observable, &wrapped, &absorbing),
            read(observable, &wrapped, &open)
        );
    }
    let scalar = read(TwistorObservable::Scalar, &wrapped, &absorbing)[0];
    assert!((scalar - tau[0]).abs() > 1e-3);
}
#[test]
fn walkers_beyond_the_eligibility_record_are_masked() {
    // Positions and velocities exist for every walker; the eligibility record
    // alone is short. An unknown walker is not an eligible walker.
    for length in [0, 1, 2] {
        let mut state = frame();
        state.eligible.truncate(length);
        for &observable in TwistorObservable::ALL {
            assert_eq!(value(observable, [0, 1, 2], &state), None);
            assert_eq!(value(observable, [2, 1, 0], &state), None);
        }
    }
    let mut state = frame();
    state.eligible.truncate(3);
    assert!(value(TwistorObservable::Scalar, [0, 1, 2], &state).is_some());
    assert_eq!(value(TwistorObservable::Scalar, [1, 2, 3], &state), None);
}
#[test]
fn descriptors_name_the_real_or_imaginary_part_of_their_arm_and_the_odd_arms_say_so() {
    let (gas, measurement, capabilities) = (
        GasConfig::default(),
        MeasurementConfig::default(),
        capabilities(0.1),
    );
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let (re, im) = (r"\operatorname{Re}", r"\operatorname{Im}");
    let (epsilon, sigma, mean) = (r"\epsilon^{AB}", r"\sigma^a", r"\tfrac15");
    let expected: [(TwistorObservable, &[&str], &[&str]); 7] = [
        (TwistorObservable::Scalar, &[re, epsilon], &[im, sigma]),
        (
            TwistorObservable::Pseudoscalar,
            &[im, epsilon],
            &[re, sigma],
        ),
        (TwistorObservable::Abs2, &[epsilon, "|^2"], &[re, im, sigma]),
        (TwistorObservable::Vector, &[re, sigma], &[im, epsilon]),
        (TwistorObservable::Axial, &[im, sigma], &[re, epsilon]),
        (
            TwistorObservable::Tensor,
            &[re, sigma, r"\sqrt6"],
            &[im, mean],
        ),
        (TwistorObservable::TensorMean, &[re, sigma, mean], &[im]),
    ];
    for (observable, present, absent) in expected {
        let signature = twistor(observable, 0.8)
            .signature(ElementKind::Triplet, &context)
            .unwrap();
        let definition = &signature.descriptor.definition;
        for text in present {
            assert!(definition.contains(text), "{definition}");
        }
        for text in absent {
            assert!(!definition.contains(text), "{definition}");
        }
        let note = &signature.descriptor.note;
        assert_eq!(
            note.contains("odd under exchange of the two companion roles"),
            signature.exchange == ExchangeParity::Odd,
            "{note}"
        );
        assert!(note.contains("no arm has a definite spatial parity"));
    }
}
