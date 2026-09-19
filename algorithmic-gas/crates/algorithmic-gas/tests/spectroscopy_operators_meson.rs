//! Meson and vector pair operators against closed forms, a reference swarm
//! and their exchange, colour-frame, inversion and rotation laws.
use algorithmic_gas::{
    GasConfig, GasError,
    boundary::{BoundaryPolicy, BoxDomain},
    physics::{
        qft::math::{C, adjoint, difference, dot, identity, mul},
        spectroscopy::{
            config::{
                ChannelSpec, Displacement, MeasurementConfig, MesonMode, MesonQuantum,
                PropagatorConfig, VectorProjection, VectorQuantum,
            },
            contract::{
                Capabilities, Element, ElementKind, ExchangeParity, FrameState, LocalOperator,
                OperatorContext, Record, Signature, SpatialParity, channel_id,
            },
            operators::channels,
        },
    },
    random::{RandomStream, Stream},
};

/// Reference swarm: six walkers in three dimensions, unnormalised colours as
/// (re, im) pairs, positions, and scores with the tie S_0 = S_4.
const RAW: [[[f64; 2]; 3]; 6] = [
    [[1., 0.], [0., 2.], [-1., 0.]],
    [[2., 1.], [1., 0.], [1., -1.]],
    [[0.5, 0.], [-1., 2.], [3., 0.]],
    [[-1., -1.], [2., 0.], [0., 0.5]],
    [[3., 0.], [1., 1.], [0., -2.]],
    [[1., 0.], [1., 0.], [1., 0.]],
];
const X: [[f64; 3]; 6] = [
    [0., 0., 0.],
    [1., 2., -1.],
    [-0.5, 1.5, 2.],
    [2., -1., 0.5],
    [-1.5, -0.5, 1.],
    [0.25, 0.75, -2.],
];
const S: [f64; 6] = [0.3, -0.2, 1.1, -0.7, 0.3, 0.05];
const INVOLUTION: [usize; 6] = [1, 0, 3, 2, 5, 4];
const CYCLE: [usize; 6] = [1, 2, 3, 4, 5, 0];

fn meson(quantum: MesonQuantum, mode: MesonMode) -> ChannelSpec {
    ChannelSpec::Meson { quantum, mode }
}
fn vector(
    quantum: VectorQuantum,
    projection: VectorProjection,
    displacement: Displacement,
) -> ChannelSpec {
    ChannelSpec::Vector {
        quantum,
        projection,
        displacement,
    }
}
fn full(quantum: VectorQuantum, displacement: Displacement) -> ChannelSpec {
    vector(quantum, VectorProjection::Full, displacement)
}
fn family() -> Vec<ChannelSpec> {
    ChannelSpec::all()
        .into_iter()
        .filter(|s| matches!(s.family(), "meson" | "vector"))
        .collect()
}
fn pair(i: usize, j: usize) -> Element {
    Element {
        walkers: [i as u32, j as u32, i as u32],
        kind: ElementKind::DistancePair,
        weight: 1.,
        generation: [0; 3],
    }
}
fn elements(map: &[usize]) -> Vec<Element> {
    map.iter().enumerate().map(|(i, &j)| pair(i, j)).collect()
}
fn ordered_pairs(n: usize) -> Vec<Element> {
    (0..n * n)
        .filter(|k| k / n != k % n)
        .map(|k| pair(k / n, k % n))
        .collect()
}
/// Mean of (S_j − S_i) r̂_ij over the listed companions of each walker with a
/// nonzero displacement: the record an operator reads, built independently.
fn gradient(state: &FrameState, maps: &[&[usize]]) -> Vec<f64> {
    let (d, score) = (state.d, state.score.as_ref().unwrap());
    let mut out = vec![0.; state.n * d];
    for i in 0..state.n {
        let mut count = 0.;
        for map in maps {
            let j = map[i];
            let r: Vec<f64> = (0..d)
                .map(|a| state.x[j * d + a] - state.x[i * d + a])
                .collect();
            let length = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            if length > 0. {
                count += 1.;
                for a in 0..d {
                    out[i * d + a] += (score[j] - score[i]) * r[a] / length;
                }
            }
        }
        for a in 0..d {
            out[i * d + a] /= f64::max(count, 1.);
        }
    }
    out
}
fn swarm(color: Vec<C>, x: Vec<f64>, score: Vec<f64>, maps: &[&[usize]]) -> FrameState {
    let n = score.len();
    let mut state = FrameState {
        n,
        d: x.len() / n,
        x,
        color,
        color_valid: vec![true; n],
        score: Some(score),
        score_valid: vec![true; n],
        eligible: vec![true; n],
        ..FrameState::default()
    };
    state.score_gradient = Some(gradient(&state, maps));
    state
}
fn reference(maps: &[&[usize]]) -> FrameState {
    let color = RAW
        .iter()
        .flat_map(|raw| {
            let length = raw
                .iter()
                .map(|z| z[0] * z[0] + z[1] * z[1])
                .sum::<f64>()
                .sqrt();
            raw.map(|z| C::new(z[0] / length, z[1] / length))
        })
        .collect();
    swarm(color, X.concat(), S.to_vec(), maps)
}
/// Substeps 0, 1 and 2 address the swarm, the matching and the colour frame
/// of one seed.
fn random(n: usize, seed: u64, maps: &[&[usize]]) -> FrameState {
    let (mut color, mut x, mut score) = (vec![], vec![], vec![]);
    for w in 0..n {
        let mut rng = RandomStream::new(seed, 0, Stream::Initialize, w as u64, 0);
        let raw: Vec<C> = (0..3)
            .map(|_| C::new(rng.gaussian(), rng.gaussian()))
            .collect();
        let length = dot(&raw, &raw).re.sqrt();
        color.extend(raw.into_iter().map(|z| z / length));
        x.extend((0..3).map(|_| rng.gaussian::<f64>()));
        score.push(rng.gaussian::<f64>());
    }
    swarm(color, x, score, maps)
}
fn matching(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    RandomStream::new(seed, 0, Stream::Initialize, 0, 1).shuffle(&mut order);
    let mut map: Vec<usize> = (0..n).collect();
    for ends in order.chunks_exact(2) {
        map[ends[0]] = ends[1];
        map[ends[1]] = ends[0];
    }
    map
}
/// Signature in the nominal context with positions of width `dimension`.
fn signature_in(
    spec: &ChannelSpec,
    kind: ElementKind,
    dimension: usize,
) -> algorithmic_gas::Result<Signature> {
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let capabilities = Capabilities {
        dimension,
        ..Capabilities::nominal()
    };
    spec.signature(
        kind,
        &OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        },
    )
}
fn evaluated_in(
    gas: &GasConfig,
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    out: &mut [f64],
) -> bool {
    let (measurement, capabilities) = (MeasurementConfig::default(), Capabilities::nominal());
    let context = OperatorContext {
        gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    spec.evaluate(element, state, &context, out)
}
/// Components of `spec` on `element`, `None` when masked.
fn value_in(
    gas: &GasConfig,
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
) -> Option<Vec<f64>> {
    let components = match spec {
        ChannelSpec::Vector { .. } => state.d,
        _ => 1,
    };
    let mut out = vec![0.; components];
    evaluated_in(gas, spec, element, state, &mut out).then_some(out)
}
fn value(spec: &ChannelSpec, element: &Element, state: &FrameState) -> Option<Vec<f64>> {
    value_in(&GasConfig::default(), spec, element, state)
}
fn scalar(spec: &ChannelSpec, i: usize, j: usize, state: &FrameState) -> Option<f64> {
    value(spec, &pair(i, j), state).map(|v| v[0])
}
/// `(Σ w m O, Σ w m)` of one frame through the batch entry point.
fn frame(spec: &ChannelSpec, elements: &[Element], state: &FrameState) -> (Vec<f64>, f64) {
    let (gas, measurement) = (GasConfig::default(), MeasurementConfig::default());
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let components = match spec {
        ChannelSpec::Vector { .. } => state.d,
        _ => 1,
    };
    let mut values = vec![0.; elements.len() * components];
    let mut valid = vec![false; elements.len()];
    spec.evaluate_all(elements, state, &context, &mut values, &mut valid);
    let (mut sum, mut weight) = (vec![0.; components], 0.);
    for ((element, row), ok) in elements
        .iter()
        .zip(values.chunks_exact(components))
        .zip(valid)
    {
        if ok {
            weight += element.weight;
            for (s, v) in sum.iter_mut().zip(row) {
                *s += element.weight * v;
            }
        }
    }
    (sum, weight)
}
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < tolerance, "{actual:?} != {expected:?}");
    }
}
/// Frame mean of the six elements of a companion map, all valid.
fn mean(spec: &ChannelSpec, map: &[usize], state: &FrameState) -> Vec<f64> {
    let (sum, weight) = frame(spec, &elements(map), state);
    assert_eq!(weight, 6.);
    sum.iter().map(|s| s / weight).collect()
}
fn unitary(seed: u64) -> Vec<C> {
    // Gram–Schmidt on a complex Gaussian matrix, twice for orthogonality to
    // roundoff; rows are the orthonormal vectors.
    let mut rng = RandomStream::new(seed, 0, Stream::Initialize, 0, 2);
    let mut u: Vec<C> = (0..9)
        .map(|_| C::new(rng.gaussian(), rng.gaussian()))
        .collect();
    for row in 0..3 {
        for _ in 0..2 {
            for earlier in 0..row {
                let overlap = dot(&u[earlier * 3..earlier * 3 + 3], &u[row * 3..row * 3 + 3]);
                for a in 0..3 {
                    u[row * 3 + a] = u[row * 3 + a] - u[earlier * 3 + a] * overlap;
                }
            }
        }
        let length = dot(&u[row * 3..row * 3 + 3], &u[row * 3..row * 3 + 3])
            .re
            .sqrt();
        for a in 0..3 {
            u[row * 3 + a] = u[row * 3 + a] / length;
        }
    }
    u
}
fn transformed(state: &FrameState, u: &[C]) -> FrameState {
    // (u c)_a = Σ_b u_ab c_b.
    let color = state
        .color
        .chunks_exact(3)
        .flat_map(|c| {
            let turned = |a: usize| (0..3).fold(C::ZERO, |s, b| s + u[a * 3 + b] * c[b]);
            [turned(0), turned(1), turned(2)]
        })
        .collect();
    FrameState {
        color,
        ..state.clone()
    }
}
const SCALAR: MesonQuantum = MesonQuantum::Scalar;
const PSEUDOSCALAR: MesonQuantum = MesonQuantum::Pseudoscalar;
const VECTOR: VectorQuantum = VectorQuantum::Vector;
const AXIAL: VectorQuantum = VectorQuantum::Axial;
/// Order of the rows of every table of full vector arms.
const FULL: [(VectorQuantum, Displacement); 8] = [
    (VECTOR, Displacement::Raw),
    (AXIAL, Displacement::Raw),
    (VECTOR, Displacement::Unit),
    (AXIAL, Displacement::Unit),
    (VECTOR, Displacement::ScoreGradient),
    (AXIAL, Displacement::ScoreGradient),
    (VECTOR, Displacement::ColorGamma),
    (AXIAL, Displacement::ColorGamma),
];
/// Order of the rows of every table of meson arms.
const MESONS: [(MesonQuantum, MesonMode); 9] = [
    (SCALAR, MesonMode::Standard),
    (PSEUDOSCALAR, MesonMode::Standard),
    (SCALAR, MesonMode::ScoreDirected),
    (PSEUDOSCALAR, MesonMode::ScoreDirected),
    (SCALAR, MesonMode::ScoreWeighted),
    (PSEUDOSCALAR, MesonMode::ScoreWeighted),
    (SCALAR, MesonMode::Gamma5Diagonal),
    (PSEUDOSCALAR, MesonMode::Gamma5Diagonal),
    (SCALAR, MesonMode::Abs2),
];

#[test]
fn pair_contractions_match_the_closed_forms_and_the_reference_table() {
    let state = reference(&[&CYCLE]);
    let pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 4),
        (5, 2),
    ];
    // Re q, Im q, |q|².
    let table = [
        [0.144337567297406, 0., 0.0208333333333333],
        [0.280975743474508, 0.421463615211762, 0.256578947368421],
        [-0.264906471413009, -0.317887765695611, 0.171228070175439],
        [-0.206559111797729, 0.516397779494322, 0.309333333333333],
        [0.596284793999944, 0.149071198499986, 0.377777777777778],
        [0., 0.471404520791032, 0.222222222222222],
        [0.52704627669473, 0., 0.277777777777778],
        [0.382359556450936, 0.305887645160749, 0.239766081871345],
    ];
    // Re g, Im g.
    let diagonal = [
        [0.144337567297406, 0.577350269189626],
        [0.468292905790847, 0.0468292905790847],
        [0.158943882847805, 0.529812942826018],
        [-0.619677335393187, 0.103279555898864],
        [0.298142396999972, 0.447213595499958],
        [0., -0.471404520791032],
        [0.105409255338946, 0.421637021355784],
        [0.688247201611685, -0.305887645160749],
    ];
    for (((i, j), q), g) in pairs.into_iter().zip(table).zip(diagonal) {
        let arm = |quantum, mode| scalar(&meson(quantum, mode), i, j, &state).unwrap();
        let measured = [
            arm(SCALAR, MesonMode::Standard),
            arm(PSEUDOSCALAR, MesonMode::Standard),
            arm(SCALAR, MesonMode::Abs2),
        ];
        close(&measured, &q, 1e-12);
        assert!((measured[2] - (q[0] * q[0] + q[1] * q[1])).abs() < 1e-12);
        let measured = [
            arm(SCALAR, MesonMode::Gamma5Diagonal),
            arm(PSEUDOSCALAR, MesonMode::Gamma5Diagonal),
        ];
        close(&measured, &g, 1e-12);
    }
    let re = |i, j| scalar(&meson(SCALAR, MesonMode::Standard), i, j, &state).unwrap();
    let im = |i, j| scalar(&meson(PSEUDOSCALAR, MesonMode::Standard), i, j, &state).unwrap();
    assert!((re(0, 1) - 1. / 48_f64.sqrt()).abs() < 1e-15 && im(0, 1).abs() < 1e-15);
    assert!((im(5, 0) - 2_f64.sqrt() / 3.).abs() < 1e-15 && re(5, 0).abs() < 1e-15);
    assert!((re(0, 4) - 5. / 90_f64.sqrt()).abs() < 1e-15);
    // In three dimensions g = q − 2 conj(c_i^1) c_j^1.
    for e in ordered_pairs(6) {
        let (i, j) = (e.walkers[0] as usize, e.walkers[1] as usize);
        let odd = state.color[i * 3 + 1].conj() * state.color[j * 3 + 1];
        let arm = |quantum| scalar(&meson(quantum, MesonMode::Gamma5Diagonal), i, j, &state);
        assert!((arm(SCALAR).unwrap() - (re(i, j) - 2. * odd.re)).abs() < 1e-15);
        assert!((arm(PSEUDOSCALAR).unwrap() - (im(i, j) - 2. * odd.im)).abs() < 1e-15);
    }
    // The element kind selects the pairs, not the value on a pair.
    for spec in family() {
        let cloning = Element {
            kind: ElementKind::CloningPair,
            ..pair(2, 3)
        };
        assert_eq!(
            value(&spec, &cloning, &state),
            value(&spec, &pair(2, 3), &state)
        );
    }
}

#[test]
fn score_modes_orient_and_weight_the_pair_from_the_scores_of_the_evaluated_state() {
    let state = reference(&[&CYCLE]);
    let arm = |quantum, mode, i, j, state: &FrameState| scalar(&meson(quantum, mode), i, j, state);
    let directed =
        |i, j, state: &FrameState| arm(PSEUDOSCALAR, MesonMode::ScoreDirected, i, j, state);
    // S_3 − S_2 = −1.8: the pair (2, 3) is read from 3 to 2.
    assert!((directed(2, 3, &state).unwrap() - 0.317887765695611).abs() < 1e-12);
    assert_eq!(directed(2, 3, &state), directed(3, 2, &state));
    assert_eq!(
        arm(SCALAR, MesonMode::ScoreDirected, 2, 3, &state),
        arm(SCALAR, MesonMode::Standard, 2, 3, &state)
    );
    let weighted = arm(SCALAR, MesonMode::ScoreWeighted, 2, 3, &state).unwrap();
    assert!((weighted + 0.476831648543416).abs() < 1e-12);
    let weighted = arm(PSEUDOSCALAR, MesonMode::ScoreWeighted, 2, 3, &state).unwrap();
    assert!((weighted + 0.572197978252100).abs() < 1e-12);
    // The tie S_0 = S_4 masks the directed arms at both ends; a weighted arm
    // keeps the element with the value 0.
    for (i, j) in [(0, 4), (4, 0)] {
        assert_eq!(directed(i, j, &state), None);
        assert_eq!(arm(SCALAR, MesonMode::ScoreDirected, i, j, &state), None);
        assert_eq!(
            arm(SCALAR, MesonMode::ScoreWeighted, i, j, &state),
            Some(0.)
        );
    }
    // A sink state with the scores reversed reverses the orientation of the
    // same frozen element.
    let mut sink = state.clone();
    sink.score = Some(S.iter().map(|s| -s).collect());
    assert_eq!(
        directed(2, 3, &sink).unwrap().to_bits(),
        (-directed(2, 3, &state).unwrap()).to_bits()
    );
    // A walker without a valid score masks every score mode and no other.
    let mut unscored = state.clone();
    unscored.score_valid[3] = false;
    for mode in [MesonMode::ScoreDirected, MesonMode::ScoreWeighted] {
        assert_eq!(arm(PSEUDOSCALAR, mode, 2, 3, &unscored), None);
        assert_eq!(arm(PSEUDOSCALAR, mode, 3, 2, &unscored), None);
        assert!(arm(PSEUDOSCALAR, mode, 1, 2, &unscored).is_some());
    }
    assert!(arm(PSEUDOSCALAR, MesonMode::Standard, 2, 3, &unscored).is_some());
    unscored.score = None;
    assert_eq!(arm(SCALAR, MesonMode::ScoreWeighted, 1, 2, &unscored), None);
}

#[test]
fn vector_components_match_the_reference_table() {
    let state = reference(&[&CYCLE]);
    let arm = |quantum, displacement, i, j, state: &FrameState| {
        value(&full(quantum, displacement), &pair(i, j), state).unwrap()
    };
    // Rows in the order of `FULL`, on the pairs (1, 2) and (2, 3).
    let rows = [
        [-0.421463615211762, -0.140487871737254, 0.842927230423525],
        [-0.632195422817643, -0.210731807605881, 1.26439084563529],
        [-0.124282839749874, -0.0414276132499581, 0.248565679499748],
        [-0.186424259624811, -0.0621414198749371, 0.372848519249623],
        [-0.161567691674836, -0.0538558972249455, 0.323135383349673],
        [-0.242351537512255, -0.0807838458374182, 0.484703075024509],
    ];
    for ((quantum, displacement), expected) in FULL.into_iter().zip(rows) {
        close(&arm(quantum, displacement, 1, 2, &state), &expected, 1e-12);
    }
    let rows = [
        [-0.662266178532522, 0.662266178532522, 0.397359707119513],
        [-0.794719414239026, 0.794719414239026, 0.476831648543416],
        [-0.172439425125162, 0.172439425125162, 0.103463655075097],
        [-0.206927310150194, 0.206927310150194, 0.124156386090117],
        [0.310390965225291, -0.310390965225291, -0.186234579135175],
        [0.37246915827035, -0.37246915827035, -0.22348149496221],
    ];
    for ((quantum, displacement), expected) in FULL.into_iter().zip(rows) {
        close(&arm(quantum, displacement, 2, 3, &state), &expected, 1e-12);
    }
    let raw = Displacement::Raw;
    let expected = [0.722956891292051, -0.103279555898864, -0.103279555898864];
    close(&arm(VECTOR, raw, 3, 4, &state), &expected, 1e-12);
    let expected = [-1.80739222823013, 0.258198889747161, 0.258198889747161];
    close(&arm(AXIAL, raw, 3, 4, &state), &expected, 1e-12);
    // q_50 is imaginary.
    close(&arm(VECTOR, raw, 5, 0, &state), &[0.; 3], 1e-15);
    let expected = [-0.117851130197758, -0.353553390593274, 0.942809041582064];
    close(&arm(AXIAL, raw, 5, 0, &state), &expected, 1e-12);
    // Norm identities on (1, 2): |q|² = 0.2565…, |r|² = 11.5, ΔS² = 1.69.
    let square = |v: Vec<f64>| v.iter().map(|x| x * x).sum::<f64>();
    let both = |displacement| {
        square(arm(VECTOR, displacement, 1, 2, &state))
            + square(arm(AXIAL, displacement, 1, 2, &state))
    };
    let abs2 = 0.256578947368421;
    assert!((both(Displacement::Raw) - abs2 * 11.5).abs() < 1e-12);
    assert!((both(Displacement::Unit) - abs2).abs() < 1e-12);
    assert!((both(Displacement::ScoreGradient) - abs2 * 1.69).abs() < 1e-12);
    // On the mirrored element the gradient of walker 3 alone is the same
    // vector: the raw vector is odd, the raw axial even.
    let mirrored = reference(&[&INVOLUTION]);
    let sign = |quantum, displacement, sign: f64| {
        let expected: Vec<f64> = arm(quantum, displacement, 2, 3, &mirrored)
            .iter()
            .map(|v| sign * v)
            .collect();
        close(
            &arm(quantum, displacement, 3, 2, &mirrored),
            &expected,
            1e-15,
        );
    };
    sign(VECTOR, Displacement::Raw, -1.);
    sign(AXIAL, Displacement::Raw, 1.);
    sign(VECTOR, Displacement::ScoreGradient, 1.);
    sign(AXIAL, Displacement::ScoreGradient, -1.);
}

#[test]
fn exchanging_the_walkers_of_a_pair_conjugates_every_contraction_bit_for_bit() {
    for state in [reference(&[&CYCLE]), random(16, 7, &[&matching(16, 3)])] {
        for spec in family() {
            let Ok(signature) = signature_in(&spec, ElementKind::DistancePair, 3) else {
                continue;
            };
            let sign = match signature.exchange {
                ExchangeParity::Even => 1.,
                ExchangeParity::Odd => -1.,
                ExchangeParity::Mixed => continue,
            };
            let mut measured = 0;
            for e in ordered_pairs(state.n) {
                let (i, j) = (e.walkers[0] as usize, e.walkers[1] as usize);
                let (here, there) = (value(&spec, &e, &state), value(&spec, &pair(j, i), &state));
                assert_eq!(here.is_some(), there.is_some(), "{}", spec.id());
                for (a, b) in here.iter().flatten().zip(there.iter().flatten()) {
                    // Adding 0 folds the two zeros, which are one value.
                    assert_eq!(
                        (a + 0.).to_bits(),
                        (sign * b + 0.).to_bits(),
                        "{}",
                        spec.id()
                    );
                    measured += usize::from(*a != 0.);
                }
            }
            assert!(measured > 0, "{}", spec.id());
        }
    }
}

#[test]
fn exchange_odd_frame_sums_cancel_on_every_involution_and_survive_on_a_cycle() {
    let odd: Vec<ChannelSpec> = family()
        .into_iter()
        .filter(|s| {
            signature_in(s, ElementKind::CloningPair, 3)
                .is_ok_and(|g| g.exchange == ExchangeParity::Odd)
        })
        .collect();
    let mut ids: Vec<String> = odd.iter().map(ChannelSpec::id).collect();
    ids.sort();
    assert_eq!(
        ids,
        [
            "meson/pseudoscalar/gamma5_diagonal",
            "meson/pseudoscalar/score_weighted",
            "meson/pseudoscalar/standard",
            "vector/axial/full/color_gamma",
            "vector/vector/full/raw",
            "vector/vector/full/unit",
        ]
    );
    // In pair order every mirrored element cancels its partner exactly.
    let state = reference(&[&INVOLUTION]);
    for spec in &odd {
        let (sum, weight) = frame(spec, &elements(&INVOLUTION), &state);
        assert_eq!(weight, 6.);
        assert!(sum.iter().all(|s| *s == 0.), "{}", spec.id());
    }
    // Random perfect matchings; with an odd population one walker is its own
    // companion and is masked without disturbing the rest.
    for (n, seed) in [(64, 1), (64, 2), (63, 3), (6, 4)] {
        let map = matching(n, seed);
        let state = random(n, seed + 10, &[&map]);
        let shift: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
        for spec in &odd {
            let (sum, weight) = frame(spec, &elements(&map), &state);
            assert_eq!(weight, (n - n % 2) as f64);
            assert!(
                sum.iter().all(|s| s.abs() < 1e-14 * n as f64),
                "{}",
                spec.id()
            );
            // A cyclic companion map has no mirrored element.
            let (sum, weight) = frame(spec, &elements(&shift), &state);
            assert_eq!(weight, n as f64);
            assert!(sum.iter().any(|s| s.abs() > 1e-3), "{}", spec.id());
        }
    }
}

#[test]
fn unequal_weights_at_the_two_ends_of_a_pair_leave_the_stated_residual() {
    // Walker 2 has the two companions 1 and 3 at weight 1/2; they have it alone.
    let state = reference(&[&CYCLE]);
    let weighted = |i, j, weight| Element {
        weight,
        ..pair(i, j)
    };
    let set = [
        weighted(2, 1, 0.5),
        weighted(2, 3, 0.5),
        weighted(1, 2, 1.),
        weighted(3, 2, 1.),
    ];
    let (sum, weight) = frame(&meson(PSEUDOSCALAR, MesonMode::Standard), &set, &state);
    assert_eq!(weight, 3.);
    close(&sum, &[0.369675690453686], 1e-12);
    close(&[sum[0] / weight], &[0.123225230151229], 1e-12);
    let (sum, _) = frame(&full(VECTOR, Displacement::Raw), &set, &state);
    close(
        &sum,
        &[0.12040128166038, -0.401377025134888, 0.222783761652006],
        1e-12,
    );
    let (sum, _) = frame(&full(AXIAL, Displacement::Raw), &set, &state);
    let mean: Vec<f64> = sum.iter().map(|s| s / 3.).collect();
    close(
        &mean,
        &[-0.713457418528335, 0.291993803316573, 0.870611247089351],
        1e-12,
    );
}

#[test]
fn frame_means_match_the_reference_frames() {
    // All six elements are valid. The gradient of each walker is built from
    // the one companion of the frame, which makes it the same vector at both
    // ends of a mirrored pair.
    let (paired, cyclic) = (reference(&[&INVOLUTION]), reference(&[&CYCLE]));
    // Rows in the order of `MESONS`: mean on the involution, mean on the cycle.
    let means = [
        [0.158571963294781, 0.0916887535935201],
        [0., 0.206741558050249],
        [0.158571963294781, 0.0916887535935201],
        [0.0562721890652082, 0.263013747115457],
        [-0.0851972221315756, -0.0161470519459325],
        [0., 0.107870238473378],
        [0.200474615715061, 0.075006569590474],
        [0., 0.20551352220042],
        [0.189946393762183, 0.226328947368421],
    ];
    for ((quantum, mode), [involution, cycle]) in MESONS.into_iter().zip(means) {
        let spec = meson(quantum, mode);
        close(&mean(&spec, &INVOLUTION, &paired), &[involution], 1e-12);
        close(&mean(&spec, &CYCLE, &cyclic), &[cycle], 1e-12);
    }
    let directed = meson(PSEUDOSCALAR, MesonMode::ScoreDirected);
    let by_hand = 2. * (0.317887765695611 - 0.149071198499986) / 6.;
    close(&mean(&directed, &INVOLUTION, &paired), &[by_hand], 1e-12);
    // Rows in the order of `FULL`.
    let involution = [
        [0., 0., 0.],
        [-0.177948272288017, 0.327019470788003, 0.00987268434781929],
        [0., 0., 0.],
        [
            -0.0454175604405776,
            0.0858030626282698,
            0.000999959842346664,
        ],
        [0.0700845179491302, -0.139932802686262, -0.0118717633408863],
        [0., 0., 0.],
        [-0.136177900664667, 0.262956674792458, 0.0154120052789798],
        [0., 0., 0.],
    ];
    let cycle = [
        [0.137843842390846, 0.242088312998524, -0.132697427942177],
        [-0.515213933018263, 0.112495350652003, 0.415836138334662],
        [0.0412283738058501, 0.0703110653281559, -0.0367410529698045],
        [-0.147274611937532, 0.0171964888851223, 0.147759308477306],
        [0.0418590482449944, -0.0837630610543698, 0.0430993387140878],
        [-0.0679036700502219, -0.0724438446269237, 0.078903364029921],
        [-0.22470518809598, 0.2859019286621, 0.0133182379048954],
        [-0.123839665222091, -0.041744977169895, -0.241398475077403],
    ];
    for (((quantum, displacement), involution), cycle) in
        FULL.into_iter().zip(involution).zip(cycle)
    {
        let spec = full(quantum, displacement);
        close(&mean(&spec, &INVOLUTION, &paired), &involution, 1e-12);
        close(&mean(&spec, &CYCLE, &cyclic), &cycle, 1e-12);
    }
}

#[test]
fn a_masked_frame_separates_the_valid_count_from_the_fixed_normalisation() {
    // Walker 1 is its own companion and walker 3 has no colour: the valid
    // elements are (0, 4), (4, 0) and (5, 2), and (0, 4) is a score tie.
    let map = [4, 1, 3, 2, 0, 2];
    let mut state = reference(&[&map]);
    state.color_valid[3] = false;
    state.color[9..12].fill(C::ZERO);
    let check = |spec: &ChannelSpec, count: f64, expected: &[f64]| {
        let (sum, weight) = frame(spec, &elements(&map), &state);
        assert_eq!(weight, count, "{}", spec.id());
        let valid_count: Vec<f64> = sum.iter().map(|s| s / weight).collect();
        close(&valid_count, expected, 1e-12);
    };
    // Rows in the order of `MESONS`: valid weight and valid-count mean. The
    // tie is masked for the directed arms, which keep (5, 2) alone, and
    // counted with the value 0 by the weighted arms.
    let means = [
        [3., 0.478817369946799],
        [3., 0.101962548386916],
        [1., 0.382359556450936],
        [1., 0.305887645160749],
        [3., 0.133825844757828],
        [3., 0.107060675806262],
        [3., 0.299688570763192],
        [3., -0.101962548386916],
        [3., 0.265107212475634],
    ];
    for ((quantum, mode), [count, expected]) in MESONS.into_iter().zip(means) {
        check(&meson(quantum, mode), count, &[expected]);
    }
    // The fixed normalisation divides the same sums by all six slots.
    let fixed = |spec: &ChannelSpec| frame(spec, &elements(&map), &state).0[0] / 6.;
    let scalar = fixed(&meson(SCALAR, MesonMode::Standard));
    assert!((scalar - 0.239408684973399).abs() < 1e-12);
    let directed = fixed(&meson(PSEUDOSCALAR, MesonMode::ScoreDirected));
    assert!((directed - 0.0509812741934582).abs() < 1e-12);
    // Rows in the order of `FULL`, displacement arms.
    let whole = [
        [-0.0955898891127341, 0.0955898891127341, 0.509812741934582],
        [-0.0764719112901873, 0.0764719112901873, 0.407850193547665],
        [-0.0230991855647938, 0.0230991855647938, 0.123195656345567],
        [-0.018479348451835, 0.018479348451835, 0.0985565250764534],
    ];
    // Walkers 0 and 4 have a zero score gradient, which has no direction:
    // every arm that reads it keeps (5, 2) alone.
    let along = [
        [-0.286769667338202, 0.286769667338202, 1.52943822580375],
        [-0.229415733870562, 0.229415733870562, 1.223550580643],
        [-0.0692975566943813, 0.0692975566943813, 0.3695869690367],
        [-0.055438045355505, 0.055438045355505, 0.29566957522936],
        [-0.0727624345291004, 0.0727624345291004, 0.388066317488535],
        [-0.0582099476232803, 0.0582099476232803, 0.310453053990828],
    ];
    for ((quantum, displacement), expected) in FULL.into_iter().zip(whole) {
        check(&full(quantum, displacement), 3., &expected);
    }
    for ((quantum, displacement), expected) in FULL.into_iter().zip(along) {
        let projection = VectorProjection::Longitudinal;
        check(&vector(quantum, projection, displacement), 1., &expected);
        if displacement == Displacement::ScoreGradient {
            check(&full(quantum, displacement), 1., &expected);
        }
    }
    // No valid element: the frame has no weight and therefore no value.
    state.color_valid = vec![false; 6];
    let (_, weight) = frame(&meson(SCALAR, MesonMode::Standard), &elements(&map), &state);
    assert_eq!(weight, 0.);
}

#[test]
fn a_common_unitary_frame_preserves_the_overlap_and_the_bounds_hold() {
    for seed in [7, 8, 9] {
        let u = unitary(seed);
        assert!(difference(&mul(&adjoint(&u, 3), &u, 3), &identity(3)) < 1e-14);
        let state = random(12, seed, &[&matching(12, seed)]);
        let turned = transformed(&state, &u);
        let mut moved: f64 = 0.;
        for e in ordered_pairs(12) {
            for spec in [
                meson(SCALAR, MesonMode::Standard),
                meson(PSEUDOSCALAR, MesonMode::Standard),
                meson(SCALAR, MesonMode::Abs2),
                meson(PSEUDOSCALAR, MesonMode::ScoreDirected),
                meson(SCALAR, MesonMode::ScoreWeighted),
                full(VECTOR, Displacement::Raw),
                full(AXIAL, Displacement::Unit),
                vector(AXIAL, VectorProjection::Transverse, Displacement::Raw),
            ] {
                let before = value(&spec, &e, &state).unwrap();
                let after = value(&spec, &e, &turned).unwrap();
                let scale = before.iter().fold(1., |m: f64, v| m.max(v.abs()));
                close(&after, &before, 1e-14 * scale);
            }
            let arm = |quantum, mode, state: &FrameState| {
                value(&meson(quantum, mode), &e, state).unwrap()[0]
            };
            let (re, im) = (
                arm(SCALAR, MesonMode::Standard, &state),
                arm(PSEUDOSCALAR, MesonMode::Standard, &state),
            );
            assert!(re * re + im * im <= 1. + 1e-15);
            assert!((0. ..=1. + 1e-15).contains(&arm(SCALAR, MesonMode::Abs2, &state)));
            // The γ5-diagonal contraction is not a colour singlet.
            let shift = arm(SCALAR, MesonMode::Gamma5Diagonal, &turned)
                - arm(SCALAR, MesonMode::Gamma5Diagonal, &state);
            moved = moved.max(shift.abs());
        }
        assert!(moved > 0.1);
    }
}

#[test]
fn a_component_shift_preserves_the_overlap_and_moves_the_gamma5_contraction() {
    // c'^{(a+1) mod 3} = c^a is a permutation, hence unitary.
    let state = reference(&[&CYCLE]);
    let color = state
        .color
        .chunks_exact(3)
        .flat_map(|c| [c[2], c[0], c[1]])
        .collect();
    let shifted = FrameState {
        color,
        ..state.clone()
    };
    let arm =
        |quantum, mode, state: &FrameState| scalar(&meson(quantum, mode), 2, 3, state).unwrap();
    for quantum in [SCALAR, PSEUDOSCALAR] {
        let (before, after) = (
            arm(quantum, MesonMode::Standard, &state),
            arm(quantum, MesonMode::Standard, &shifted),
        );
        assert!((before - after).abs() < 1e-15);
    }
    let g = [
        arm(SCALAR, MesonMode::Gamma5Diagonal, &shifted),
        arm(PSEUDOSCALAR, MesonMode::Gamma5Diagonal, &shifted),
    ];
    close(&g, &[-0.158943882847805, -0.211925177130407], 1e-12);
}

#[test]
fn a_rephasing_of_each_walker_rotates_the_overlap_and_preserves_its_modulus() {
    let state = random(8, 5, &[&matching(8, 5)]);
    let alpha: Vec<f64> = (0..8).map(|i| 0.37 * i as f64 - 1.1).collect();
    let color = state
        .color
        .chunks_exact(3)
        .zip(&alpha)
        .flat_map(|(c, a)| c.iter().map(|z| *z * C::phase(*a)).collect::<Vec<_>>())
        .collect();
    let rephased = FrameState {
        color,
        ..state.clone()
    };
    for e in ordered_pairs(8) {
        let (i, j) = (e.walkers[0] as usize, e.walkers[1] as usize);
        let q = |state: &FrameState| {
            C::new(
                scalar(&meson(SCALAR, MesonMode::Standard), i, j, state).unwrap(),
                scalar(&meson(PSEUDOSCALAR, MesonMode::Standard), i, j, state).unwrap(),
            )
        };
        let expected = q(&state) * C::phase(alpha[j] - alpha[i]);
        assert!((q(&rephased) - expected).abs() < 1e-14);
        let abs2 =
            |state: &FrameState| scalar(&meson(SCALAR, MesonMode::Abs2), i, j, state).unwrap();
        assert!((abs2(&rephased) - abs2(&state)).abs() < 1e-15);
    }
}

#[test]
fn spatial_inversion_gives_every_arm_the_sign_its_descriptor_states() {
    // Inversion: c → −conj(c), x → −x; scores are scalars, so the score
    // gradient changes sign with the displacements.
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let inverted = FrameState {
        color: state.color.iter().map(|c| -c.conj()).collect(),
        x: state.x.iter().map(|x| -x).collect(),
        score_gradient: state
            .score_gradient
            .as_ref()
            .map(|g| g.iter().map(|g| -g).collect()),
        ..state.clone()
    };
    let mut arms = 0;
    for spec in family() {
        let Ok(signature) = signature_in(&spec, ElementKind::DistancePair, 3) else {
            continue;
        };
        let parity = signature.descriptor.spatial_parity.unwrap();
        let expected = match &spec {
            ChannelSpec::Meson {
                quantum: MesonQuantum::Scalar,
                ..
            }
            | ChannelSpec::Vector {
                quantum: VectorQuantum::Axial,
                ..
            } => SpatialParity::Even,
            _ => SpatialParity::Odd,
        };
        assert_eq!(parity, expected, "{}", spec.id());
        let sign = if parity == SpatialParity::Even {
            1.
        } else {
            -1.
        };
        let mut measured = 0;
        for e in ordered_pairs(6) {
            let (before, after) = (value(&spec, &e, &state), value(&spec, &e, &inverted));
            assert_eq!(before.is_some(), after.is_some());
            for (b, a) in before.iter().flatten().zip(after.iter().flatten()) {
                assert!((a - sign * b).abs() < 1e-15, "{}", spec.id());
                measured += usize::from(b.abs() > 1e-3);
            }
        }
        assert!(measured > 0, "{}", spec.id());
        arms += 1;
    }
    // 9 meson arms and 6 + 6 + 4 + 2 vector arms exist in three dimensions.
    assert_eq!(arms, 27);
}

#[test]
fn projections_split_every_displacement_along_and_across_the_anchor_gradient() {
    // Two companions per walker: the gradient is not parallel to the pair.
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let gradient = state.score_gradient.clone().unwrap();
    let mut moved: f64 = 0.;
    for e in ordered_pairs(6) {
        let g = &gradient[e.walkers[0] as usize * 3..e.walkers[0] as usize * 3 + 3];
        let length = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        for quantum in [VECTOR, AXIAL] {
            for displacement in [
                Displacement::Raw,
                Displacement::Unit,
                Displacement::ScoreGradient,
            ] {
                let arm =
                    |projection| value(&vector(quantum, projection, displacement), &e, &state);
                let whole = arm(VectorProjection::Full).unwrap();
                let along = arm(VectorProjection::Longitudinal).unwrap();
                let shadow: f64 = whole.iter().zip(g).map(|(w, g)| w * g / length).sum();
                let expected: Vec<f64> = g.iter().map(|g| shadow * g / length).collect();
                close(&along, &expected, 1e-14);
                if displacement == Displacement::ScoreGradient {
                    close(&along, &whole, 1e-14);
                    assert_eq!(arm(VectorProjection::Transverse), None);
                    continue;
                }
                let across = arm(VectorProjection::Transverse).unwrap();
                let sum: Vec<f64> = along.iter().zip(&across).map(|(a, b)| a + b).collect();
                close(&sum, &whole, 1e-14);
                let leak: f64 = across.iter().zip(g).map(|(a, g)| a * g / length).sum();
                assert!(leak.abs() < 1e-14);
                // The projection acts in the standard displacement modes too.
                moved = across.iter().fold(moved, |m, a| m.max(a.abs()));
            }
        }
    }
    assert!(moved > 0.1);
    // A gradient built from the pair alone is parallel to it: the part along
    // it is the whole displacement and nothing is left across.
    let local = reference(&[&CYCLE]);
    for e in elements(&CYCLE) {
        for displacement in [Displacement::Raw, Displacement::Unit] {
            let arm =
                |projection| value(&vector(AXIAL, projection, displacement), &e, &local).unwrap();
            let whole = arm(VectorProjection::Full);
            let scale = whole.iter().fold(1., |m: f64, w| m.max(w.abs()));
            close(&arm(VectorProjection::Longitudinal), &whole, 1e-14 * scale);
            close(&arm(VectorProjection::Transverse), &[0.; 3], 1e-14 * scale);
        }
    }
    // The direction of a gradient does not depend on its length.
    let mut faint = state.clone();
    let scaled = faint.score_gradient.as_mut().unwrap();
    scaled.iter_mut().for_each(|g| *g *= 1e-200);
    let along = vector(AXIAL, VectorProjection::Longitudinal, Displacement::Raw);
    let expected = value(&along, &pair(1, 2), &state).unwrap();
    close(
        &value(&along, &pair(1, 2), &faint).unwrap(),
        &expected,
        1e-14,
    );
    // Without a gradient, or without a valid score, the anchor has no
    // direction; the unprojected displacement arms do not read either.
    let strips: [fn(&mut FrameState); 3] = [
        |s| s.score_gradient.as_mut().unwrap()[3..6].fill(0.),
        |s| s.score_valid[1] = false,
        |s| s.score_gradient = None,
    ];
    for strip in strips {
        let mut bare = state.clone();
        strip(&mut bare);
        for spec in family() {
            let ChannelSpec::Vector {
                projection,
                displacement,
                ..
            } = &spec
            else {
                continue;
            };
            if signature_in(&spec, ElementKind::DistancePair, 3).is_err() {
                continue;
            }
            let reads = *projection != VectorProjection::Full
                || *displacement == Displacement::ScoreGradient;
            assert_eq!(
                value(&spec, &pair(1, 2), &bare).is_none(),
                reads,
                "{}",
                spec.id()
            );
        }
    }
}

#[test]
fn rotating_the_positions_rotates_the_components_and_preserves_their_contraction() {
    let rotation = [0.6, -0.48, 0.64, 0.8, 0.36, -0.48, 0., 0.8, 0.6];
    let rotate = |v: &[f64]| -> Vec<f64> {
        v.chunks_exact(3)
            .flat_map(|r| {
                (0..3).map(move |a| (0..3).map(|b| rotation[a * 3 + b] * r[b]).sum::<f64>())
            })
            .collect()
    };
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let mut later = random(6, 11, &[&CYCLE, &INVOLUTION]);
    later.color_valid = vec![true; 6];
    let turn = |state: &FrameState| FrameState {
        x: rotate(&state.x),
        score_gradient: state.score_gradient.as_ref().map(|g| rotate(g)),
        ..state.clone()
    };
    for spec in family() {
        let is_spatial = matches!(
            &spec,
            ChannelSpec::Vector { displacement, .. } if *displacement != Displacement::ColorGamma
        );
        if !is_spatial || signature_in(&spec, ElementKind::DistancePair, 3).is_err() {
            continue;
        }
        let e = pair(1, 2);
        let (source, sink) = (
            value(&spec, &e, &state).unwrap(),
            value(&spec, &e, &later).unwrap(),
        );
        let (turned_source, turned_sink) = (
            value(&spec, &e, &turn(&state)).unwrap(),
            value(&spec, &e, &turn(&later)).unwrap(),
        );
        close(&turned_source, &rotate(&source), 1e-14);
        let contract = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        let invariant = contract(&turned_source, &turned_sink) - contract(&source, &sink);
        assert!(invariant.abs() < 1e-14, "{}", spec.id());
        // The component mean is not a rotation invariant.
        let mean = |v: &[f64]| v.iter().sum::<f64>() / 3.;
        let varying = mean(&turned_source) * mean(&turned_sink) - mean(&source) * mean(&sink);
        assert!(varying.abs() > 1e-6, "{}", spec.id());
    }
}

#[test]
fn colour_gamma_components_are_the_cross_product_of_the_colours() {
    let state = reference(&[&CYCLE]);
    let arm = |quantum, i, j| {
        value(
            &full(quantum, Displacement::ColorGamma),
            &pair(i, j),
            &state,
        )
        .unwrap()
    };
    close(
        &arm(VECTOR, 2, 3),
        &[0.317887765695611, 0.0529812942826018, 0.344378412836911],
        1e-12,
    );
    close(
        &arm(AXIAL, 2, 3),
        &[0.211925177130407, -0.529812942826018, -0.317887765695611],
        1e-12,
    );
    close(&arm(VECTOR, 3, 2), &arm(VECTOR, 2, 3), 1e-15);
    // h^μ = i (conj(c_i) × c_j)_{μ + 2 mod 3}.
    for e in ordered_pairs(6) {
        let (i, j) = (e.walkers[0] as usize, e.walkers[1] as usize);
        let (a, b) = (
            &state.color[i * 3..i * 3 + 3],
            &state.color[j * 3..j * 3 + 3],
        );
        let cross = |k: usize| {
            let (m, n) = ((k + 1) % 3, (k + 2) % 3);
            a[m].conj() * b[n] - a[n].conj() * b[m]
        };
        let h: Vec<C> = (0..3)
            .map(|mu| C::new(0., 1.) * cross((mu + 2) % 3))
            .collect();
        close(
            &arm(VECTOR, i, j),
            &h.iter().map(|h| h.re).collect::<Vec<_>>(),
            1e-15,
        );
        close(
            &arm(AXIAL, i, j),
            &h.iter().map(|h| h.im).collect::<Vec<_>>(),
            1e-15,
        );
    }
    // Two colour components leave Γ_1 = −Γ_0: the operator does not exist.
    for quantum in [VECTOR, AXIAL] {
        let spec = full(quantum, Displacement::ColorGamma);
        assert!(matches!(
            signature_in(&spec, ElementKind::CloningPair, 2),
            Err(GasError::Capability(_))
        ));
        let plane = FrameState {
            d: 2,
            x: vec![0.; 4],
            color: vec![C::ONE, C::ZERO, C::ZERO, C::ONE],
            color_valid: vec![true; 2],
            n: 2,
            ..FrameState::default()
        };
        assert_eq!(value(&spec, &pair(0, 1), &plane), None);
    }
    // Any other projection of it is an invalid specification, not a missing capability.
    let projected = vector(
        VECTOR,
        VectorProjection::Longitudinal,
        Displacement::ColorGamma,
    );
    assert!(matches!(
        signature_in(&projected, ElementKind::CloningPair, 3),
        Err(GasError::Configuration(_))
    ));
    assert_eq!(value(&projected, &pair(2, 3), &state), None);
}

#[test]
fn displacements_take_the_minimum_image_of_a_periodic_box() {
    let boxed = |width: usize| GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![0.; width],
                upper: vec![4.; width],
            },
        },
        ..GasConfig::default()
    };
    let mut state = reference(&[&CYCLE]);
    state.x[..6].copy_from_slice(&[0.5, 3., 0.25, 3.5, 0.5, 1.25]);
    let re = scalar(&meson(SCALAR, MesonMode::Standard), 0, 1, &state).unwrap();
    // Differences (3, −2.5, 1) wrap to (−1, 1.5, 1) in a box of length 4.
    let image = [-1., 1.5, 1.];
    let raw = full(VECTOR, Displacement::Raw);
    let wrapped = value_in(&boxed(3), &raw, &pair(0, 1), &state).unwrap();
    close(&wrapped, &image.map(|r| re * r), 1e-15);
    let mirrored = value_in(&boxed(3), &raw, &pair(1, 0), &state).unwrap();
    close(&mirrored, &image.map(|r| -re * r), 1e-15);
    let unit = full(VECTOR, Displacement::Unit);
    let unit = value_in(&boxed(3), &unit, &pair(0, 1), &state).unwrap();
    close(&unit, &image.map(|r| re * r / 4.25_f64.sqrt()), 1e-15);
    let open = value(&raw, &pair(0, 1), &state).unwrap();
    close(&open, &[3., -2.5, 1.].map(|r| re * r), 1e-15);
    // A box of another dimension cannot wrap these positions.
    assert_eq!(value_in(&boxed(2), &raw, &pair(0, 1), &state), None);
}

#[test]
fn elements_without_two_valid_distinct_walkers_are_masked() {
    let mut state = reference(&[&CYCLE]);
    state.color_valid[3] = false;
    // Walker 5 sits on walker 4, as a clone without jitter does.
    let (left, right) = state.x.split_at_mut(15);
    right.copy_from_slice(&left[12..]);
    for spec in family() {
        let Ok(signature) = signature_in(&spec, ElementKind::DistancePair, 3) else {
            continue;
        };
        let masked = |e: &Element| value(&spec, e, &state).is_none();
        assert!(
            masked(&pair(1, 1)) && masked(&pair(2, 3)) && masked(&pair(3, 2)),
            "{}",
            spec.id()
        );
        assert!(masked(&pair(1, 6)) && masked(&pair(6, 1)), "{}", spec.id());
        for kind in [ElementKind::Site, ElementKind::Triplet] {
            assert!(masked(&Element { kind, ..pair(1, 2) }));
            assert!(matches!(
                signature_in(&spec, kind, 3),
                Err(GasError::Capability(_))
            ));
        }
        let mut short = vec![0.; signature.components + 1];
        let gas = GasConfig::default();
        assert!(!evaluated_in(&gas, &spec, &pair(1, 2), &state, &mut short));
    }
    // A zero displacement is a value of the raw arms and has no unit vector.
    let raw = value(&full(AXIAL, Displacement::Raw), &pair(4, 5), &state);
    assert_eq!(raw, Some(vec![0.; 3]));
    assert_eq!(
        value(&full(AXIAL, Displacement::Unit), &pair(4, 5), &state,),
        None
    );
    let along = vector(AXIAL, VectorProjection::Longitudinal, Displacement::Unit);
    assert_eq!(value(&along, &pair(4, 5), &state), None);
}

#[test]
fn the_product_of_an_odd_operator_at_two_times_on_a_frozen_pair_is_exchange_even() {
    let (source, sink) = (random(6, 20, &[&INVOLUTION]), random(6, 21, &[&INVOLUTION]));
    for spec in [
        meson(PSEUDOSCALAR, MesonMode::Standard),
        full(VECTOR, Displacement::Raw),
    ] {
        let product = |i, j| -> f64 {
            let at = |state| value(&spec, &pair(i, j), state).unwrap();
            at(&source).iter().zip(at(&sink)).map(|(a, b)| a * b).sum()
        };
        for e in elements(&INVOLUTION) {
            let (i, j) = (e.walkers[0] as usize, e.walkers[1] as usize);
            assert_eq!(product(i, j).to_bits(), product(j, i).to_bits());
            assert!(product(i, j).abs() > 1e-6);
        }
    }
}

#[test]
fn four_dimensions_index_walkers_and_components_by_the_state_dimension() {
    // Walker 0 only shifts the rows; c_1 ∝ (1, i, −1, 2), c_2 ∝ (2 − i, 1, i, 1 + i),
    // so that 3√7 q_12 = 4 − i, 3√7 g_12 = −3i and r_12 = (1, −2, 2, 4), |r_12| = 5.
    let raw = [
        [[1., 0.], [0., 0.], [0., 0.], [0., 0.]],
        [[1., 0.], [0., 1.], [-1., 0.], [2., 0.]],
        [[2., -1.], [1., 0.], [0., 1.], [1., 1.]],
    ];
    let color = raw
        .iter()
        .flat_map(|row| {
            let length = row
                .iter()
                .map(|z| z[0] * z[0] + z[1] * z[1])
                .sum::<f64>()
                .sqrt();
            row.map(|z| C::new(z[0] / length, z[1] / length))
        })
        .collect();
    let x = vec![9., 9., 9., 9., 0.5, 1., -1., 0., 1.5, -1., 1., 4.];
    let state = swarm(color, x, vec![0., 0.2, 0.9], &[&[1, 2, 1]]);
    let scale = 3. * 7_f64.sqrt();
    let arm = |quantum, mode| scalar(&meson(quantum, mode), 1, 2, &state).unwrap();
    assert!((arm(SCALAR, MesonMode::Standard) - 4. / scale).abs() < 1e-15);
    assert!((arm(PSEUDOSCALAR, MesonMode::Standard) + 1. / scale).abs() < 1e-15);
    assert!((arm(SCALAR, MesonMode::Abs2) - 17. / 63.).abs() < 1e-15);
    // Signs (+, −, +, −) on the four components.
    assert!(arm(SCALAR, MesonMode::Gamma5Diagonal).abs() < 1e-15);
    assert!((arm(PSEUDOSCALAR, MesonMode::Gamma5Diagonal) + 3. / scale).abs() < 1e-15);
    assert!((arm(PSEUDOSCALAR, MesonMode::ScoreWeighted) + 0.7 / scale).abs() < 1e-15);
    let r = [1., -2., 2., 4.];
    let arm = |quantum, displacement| value(&full(quantum, displacement), &pair(1, 2), &state);
    let expected = r.map(|r| 4. * r / scale);
    close(&arm(VECTOR, Displacement::Raw).unwrap(), &expected, 1e-15);
    let expected = r.map(|r| -r / (5. * scale));
    close(&arm(AXIAL, Displacement::Unit).unwrap(), &expected, 1e-15);
    let expected = r.map(|r| 4. * 0.7 * r / (5. * scale));
    close(
        &arm(VECTOR, Displacement::ScoreGradient).unwrap(),
        &expected,
        1e-15,
    );
    // The last colour-space matrix couples the components 3 and 0.
    let expected = [-2., 0., 3., 3.].map(|h| h / scale);
    close(
        &arm(VECTOR, Displacement::ColorGamma).unwrap(),
        &expected,
        1e-15,
    );
    let expected = [2., 2., -1., 3.].map(|h| h / scale);
    close(
        &arm(AXIAL, Displacement::ColorGamma).unwrap(),
        &expected,
        1e-15,
    );
    let along = vector(VECTOR, VectorProjection::Longitudinal, Displacement::Raw);
    let expected = r.map(|r| 4. * r / scale);
    close(
        &value(&along, &pair(1, 2), &state).unwrap(),
        &expected,
        1e-14,
    );
    for displacement in [Displacement::Raw, Displacement::ColorGamma] {
        let spec = full(AXIAL, displacement);
        let signature = signature_in(&spec, ElementKind::CloningPair, 4).unwrap();
        assert_eq!(signature.components, 4);
    }
}

#[test]
fn one_dimension_offers_neither_the_gamma5_contraction_nor_a_part_across_the_gradient() {
    // c_0 = e^{0.3i}, c_1 = e^{1.1i}: q_01 = e^{0.8i}; r_01 = 1.5, G_0 = S_1 − S_0 = 0.5.
    let color = vec![C::phase(0.3), C::phase(1.1)];
    let state = swarm(color, vec![0.5, 2.], vec![0.2, 0.7], &[&[1, 0]]);
    let mut available = 0;
    for spec in family() {
        let absent = matches!(
            &spec,
            ChannelSpec::Meson {
                mode: MesonMode::Gamma5Diagonal,
                ..
            } | ChannelSpec::Vector {
                projection: VectorProjection::Transverse,
                ..
            }
        );
        let signature = signature_in(&spec, ElementKind::DistancePair, 1);
        if signature.is_ok() {
            available += 1;
            assert!(value(&spec, &pair(0, 1), &state).is_some(), "{}", spec.id());
        } else {
            assert!(
                matches!(signature, Err(GasError::Capability(_))),
                "{}",
                spec.id()
            );
            assert_eq!(value(&spec, &pair(0, 1), &state), None, "{}", spec.id());
        }
        if absent {
            assert!(signature.is_err(), "{}", spec.id());
        }
    }
    // 7 meson arms; the full and the longitudinal part of three displacements.
    assert_eq!(available, 7 + 2 * 2 * 3);
    let re = scalar(&meson(SCALAR, MesonMode::Standard), 0, 1, &state).unwrap();
    let im = scalar(&meson(PSEUDOSCALAR, MesonMode::Standard), 0, 1, &state).unwrap();
    assert!((re - 0.8_f64.cos()).abs() < 1e-15 && (im - 0.8_f64.sin()).abs() < 1e-15);
    for projection in [VectorProjection::Full, VectorProjection::Longitudinal] {
        let arm = |displacement| {
            value(
                &vector(VECTOR, projection, displacement),
                &pair(0, 1),
                &state,
            )
            .unwrap()[0]
        };
        assert!((arm(Displacement::Raw) - 1.5 * re).abs() < 1e-15);
        assert!((arm(Displacement::Unit) - re).abs() < 1e-15);
        assert!((arm(Displacement::ScoreGradient) - 0.5 * re).abs() < 1e-15);
    }
}

#[test]
fn nonfinite_inputs_mask_the_element_and_never_reach_a_series() {
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let available: Vec<ChannelSpec> = family()
        .into_iter()
        .filter(|s| signature_in(s, ElementKind::DistancePair, 3).is_ok())
        .collect();
    let spoils: [fn(&mut FrameState); 5] = [
        |s| s.color[4] = C::new(f64::NAN, 0.),
        |s| s.color[7] = C::new(0., f64::INFINITY),
        |s| s.x[5] = f64::NAN,
        |s| s.score.as_mut().unwrap()[2] = f64::NAN,
        |s| s.score_gradient.as_mut().unwrap()[3] = f64::INFINITY,
    ];
    for (case, spoil) in spoils.into_iter().enumerate() {
        let mut spoiled = state.clone();
        spoil(&mut spoiled);
        for spec in &available {
            let (spatial, directed, scored) = match spec {
                ChannelSpec::Meson { mode, .. } => (
                    false,
                    false,
                    matches!(mode, MesonMode::ScoreDirected | MesonMode::ScoreWeighted),
                ),
                ChannelSpec::Vector {
                    projection,
                    displacement,
                    ..
                } => (
                    matches!(displacement, Displacement::Raw | Displacement::Unit),
                    *projection != VectorProjection::Full
                        || *displacement == Displacement::ScoreGradient,
                    false,
                ),
                _ => continue,
            };
            let masked = [true, true, spatial, scored, directed][case];
            let measured = value(spec, &pair(1, 2), &spoiled);
            assert_eq!(measured.is_none(), masked, "{} {case}", spec.id());
            assert!(measured.iter().flatten().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn equal_scores_leave_the_directed_arms_without_a_frame_value() {
    let mut state = reference(&[&INVOLUTION]);
    state.score = Some(vec![0.4; 6]);
    for quantum in [SCALAR, PSEUDOSCALAR] {
        let directed = meson(quantum, MesonMode::ScoreDirected);
        let (_, weight) = frame(&directed, &elements(&INVOLUTION), &state);
        assert_eq!(weight, 0.);
        let weighted = meson(quantum, MesonMode::ScoreWeighted);
        let (sum, weight) = frame(&weighted, &elements(&INVOLUTION), &state);
        assert_eq!((sum[0], weight), (0., 6.));
    }
}

#[test]
fn signatures_state_components_requirements_and_exchange_parities() {
    let scored = [Record::Color, Record::Fitness, Record::CloningCompanions];
    let mut available = 0;
    for spec in family() {
        let expected = match &spec {
            ChannelSpec::Meson {
                quantum: MesonQuantum::Pseudoscalar,
                mode: MesonMode::Abs2,
            }
            | ChannelSpec::Vector {
                projection: VectorProjection::Transverse,
                displacement: Displacement::ScoreGradient,
                ..
            } => None,
            ChannelSpec::Meson { quantum, mode } => Some((
                1,
                matches!(mode, MesonMode::ScoreDirected | MesonMode::ScoreWeighted),
                match (quantum, mode) {
                    (MesonQuantum::Scalar, _) | (_, MesonMode::ScoreDirected) => {
                        ExchangeParity::Even
                    }
                    _ => ExchangeParity::Odd,
                },
            )),
            ChannelSpec::Vector {
                quantum,
                projection,
                displacement,
            } => {
                let spatial = matches!(displacement, Displacement::Raw | Displacement::Unit);
                let plain = *projection == VectorProjection::Full;
                let odd = (*quantum == VectorQuantum::Vector) == spatial;
                Some(
                    match (plain && *displacement != Displacement::ScoreGradient, odd) {
                        (true, true) => (3, false, ExchangeParity::Odd),
                        (true, false) => (3, false, ExchangeParity::Even),
                        (false, _) => (3, true, ExchangeParity::Mixed),
                    },
                )
            }
            _ => None,
        };
        for kind in [ElementKind::DistancePair, ElementKind::CloningPair] {
            let signature = signature_in(&spec, kind, 3);
            let Some((components, reads_scores, exchange)) = expected else {
                assert!(
                    matches!(signature, Err(GasError::Capability(_))),
                    "{}",
                    spec.id()
                );
                continue;
            };
            let signature = signature.unwrap();
            assert_eq!(signature.components, components, "{}", spec.id());
            assert_eq!(signature.exchange, exchange, "{}", spec.id());
            let records: Vec<Record> = signature.requires.records.iter().copied().collect();
            assert_eq!(
                records,
                if reads_scores {
                    scored.to_vec()
                } else {
                    vec![Record::Color]
                }
            );
            assert!(signature.correlatable && !signature.degenerate);
            assert_eq!(signature.normalization, None);
            assert!(!signature.descriptor.definition.is_empty());
            available += 1;
        }
    }
    assert_eq!(available, 2 * 27);
    // The number of kept components follows the position dimension.
    let raw = full(VECTOR, Displacement::Raw);
    let planar = signature_in(&raw, ElementKind::CloningPair, 2).unwrap();
    assert_eq!(planar.components, 2);
    // Arms without an operator stay in the channel list as unavailable
    // channels; the others carry the companion record of their element kind.
    let (gas, capabilities) = (GasConfig::default(), Capabilities::nominal());
    let measurement = MeasurementConfig {
        channels: family(),
        ..MeasurementConfig::default()
    };
    let listed = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let all = channels(&listed, &[]).unwrap();
    assert_eq!(all.len(), 2 * family().len());
    for channel in &all {
        assert_eq!(channel.id, channel_id(&channel.spec.id(), channel.kind));
        assert_eq!(
            channel.availability.is_available(),
            channel.signature.is_some()
        );
    }
    assert_eq!(all.iter().filter(|c| c.signature.is_none()).count(), 2 * 3);
}

#[test]
fn exchange_odd_members_of_the_standard_set_are_propagated_by_default() {
    let propagators = PropagatorConfig::default();
    let mut odd = vec![];
    for spec in ChannelSpec::standard_set() {
        if !matches!(spec.family(), "meson" | "vector") {
            continue;
        }
        for kind in [ElementKind::DistancePair, ElementKind::CloningPair] {
            let signature = signature_in(&spec, kind, 3).unwrap();
            let enabled = propagators.enabled(&spec.id(), &channel_id(&spec.id(), kind));
            assert_eq!(
                enabled,
                signature.exchange == ExchangeParity::Odd,
                "{}",
                spec.id()
            );
        }
        if propagators.enabled_for.contains(&spec.id()) {
            odd.push(spec.id());
        }
    }
    assert_eq!(
        odd,
        ["meson/pseudoscalar/standard", "vector/vector/full/raw"]
    );
    // Every default entry of these families names a member of the standard set.
    let named = propagators
        .enabled_for
        .iter()
        .filter(|id| id.starts_with("meson/") || id.starts_with("vector/"));
    assert_eq!(named.count(), odd.len());
}
