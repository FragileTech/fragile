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
                Auxiliary, Capabilities, Companions, Element, ElementKind, ExchangeParity, Frame,
                FrameState, LocalOperator, OperatorContext, Record, Signature, SpatialParity,
                channel_id,
            },
            fields::score,
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
                    out[i * d + a] += (score[j] - score[i]) * r[a] / (length * length);
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
fn raw_in(
    gas: &GasConfig,
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
) -> Option<Vec<f64>> {
    let mut out = vec![0.; width_of(spec, element.kind, state.d)];
    evaluated_in(gas, spec, element, state, &mut out).then_some(out)
}
fn raw(spec: &ChannelSpec, element: &Element, state: &FrameState) -> Option<Vec<f64>> {
    raw_in(&GasConfig::default(), spec, element, state)
}
/// Width of one row of `evaluate`: the components of the arm in `dimension`
/// position dimensions, and its auxiliary column when it has one.
fn width_of(spec: &ChannelSpec, kind: ElementKind, dimension: usize) -> usize {
    signature_in(spec, kind, dimension).map_or(1, |signature| signature.width())
}
/// Components of `spec` on `element` with its auxiliary column folded in, as
/// the measurement folds it at a source time: the score factor multiplies the
/// value, and an element the frame cannot orient is masked.
fn value_in(
    gas: &GasConfig,
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
) -> Option<Vec<f64>> {
    let row = raw_in(gas, spec, element, state)?;
    let signature = signature_in(spec, element.kind, state.d).ok()?;
    let Some(auxiliary) = signature.auxiliary else {
        return Some(row);
    };
    let factor = row[signature.components];
    if !factor.is_finite() || (auxiliary == Auxiliary::ScoreOrientation && factor == 0.) {
        return None;
    }
    Some(
        row[..signature.components]
            .iter()
            .map(|value| value * factor)
            .collect(),
    )
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
    let signature = signature_in(spec, elements[0].kind, state.d).unwrap();
    let (components, width) = (signature.components, signature.width());
    let mut values = vec![0.; elements.len() * width];
    let mut valid = vec![false; elements.len()];
    spec.evaluate_all(elements, state, &context, &mut values, &mut valid);
    let (mut sum, mut weight) = (vec![0.; components], 0.);
    for ((element, row), ok) in elements.iter().zip(values.chunks_exact(width)).zip(valid) {
        // The auxiliary column folds into the value exactly as it does in the
        // measurement, and an element without a score factor is masked.
        let factor = match signature.auxiliary {
            None => 1.,
            Some(auxiliary) => {
                let factor = row[components];
                if !factor.is_finite() || (auxiliary == Auxiliary::ScoreOrientation && factor == 0.)
                {
                    continue;
                }
                factor
            }
        };
        if ok {
            weight += element.weight;
            for (s, v) in sum.iter_mut().zip(&row[..components]) {
                *s += element.weight * v * factor;
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
fn score_modes_report_the_score_factor_apart_from_the_contraction() {
    // The contraction is read in the sampled order and the factor the score
    // contributes is the second column, so that the measurement can freeze it
    // with the element instead of re-deriving it at a sink.
    let state = reference(&[&CYCLE]);
    let arm =
        |quantum, mode, i, j, state: &FrameState| raw(&meson(quantum, mode), &pair(i, j), state);
    let directed =
        |i, j, state: &FrameState| arm(PSEUDOSCALAR, MesonMode::ScoreDirected, i, j, state);
    // S_3 − S_2 = −1.8: the pair (2, 3) is read from 3 to 2, which is the sign
    // −1 on the imaginary part and no factor at all on the real part.
    let row = directed(2, 3, &state).unwrap();
    close(&row, &[-0.317887765695611, -1.], 1e-12);
    close(
        &directed(3, 2, &state).unwrap(),
        &[0.317887765695611, 1.],
        1e-12,
    );
    close(
        &arm(SCALAR, MesonMode::ScoreDirected, 2, 3, &state).unwrap(),
        &[-0.264906471413009, 1.],
        1e-12,
    );
    close(
        &arm(SCALAR, MesonMode::ScoreWeighted, 2, 3, &state).unwrap(),
        &[-0.264906471413009, 1.8],
        1e-12,
    );
    close(
        &arm(PSEUDOSCALAR, MesonMode::ScoreWeighted, 2, 3, &state).unwrap(),
        &[-0.317887765695611, 1.8],
        1e-12,
    );
    // The tie S_0 = S_4 leaves the pair without an orientation and without a
    // score dispersion: the factor is exactly 0 at both ends.
    for (i, j) in [(0, 4), (4, 0)] {
        for mode in [MesonMode::ScoreDirected, MesonMode::ScoreWeighted] {
            for quantum in [SCALAR, PSEUDOSCALAR] {
                assert_eq!(arm(quantum, mode, i, j, &state).unwrap()[1], 0.);
            }
        }
    }
    // A sink state with the scores reversed reverses the factor of the same
    // element; only the frozen factor of the source reaches a propagator.
    let mut sink = state.clone();
    sink.score = Some(S.iter().map(|s| -s).collect());
    assert_eq!(directed(2, 3, &sink).unwrap()[1], 1.);
    assert_eq!(directed(2, 3, &sink).unwrap()[0], row[0]);
    // A walker without a valid score is missing data, not a tie: the factor is
    // not a number, and no other mode reads it.
    let mut unscored = state.clone();
    unscored.score_valid[3] = false;
    for mode in [MesonMode::ScoreDirected, MesonMode::ScoreWeighted] {
        assert!(arm(PSEUDOSCALAR, mode, 2, 3, &unscored).unwrap()[1].is_nan());
        assert!(arm(PSEUDOSCALAR, mode, 3, 2, &unscored).unwrap()[1].is_nan());
        assert!(arm(PSEUDOSCALAR, mode, 1, 2, &unscored).unwrap()[1].is_finite());
    }
    assert!(arm(PSEUDOSCALAR, MesonMode::Standard, 2, 3, &unscored).is_some());
    unscored.score = None;
    assert!(arm(SCALAR, MesonMode::ScoreWeighted, 1, 2, &unscored).unwrap()[1].is_nan());
    // The orientation is frozen with the element, so it enters the lag product
    // squared and the source-frozen propagator of a directed arm is the one of
    // the standard arm over the same elements: the arm is a frame-mean channel
    // only, exactly as `baryon/score_ordered` is. The dispersion of a weighted
    // arm is no sign and its propagator stays its own.
    for quantum in [SCALAR, PSEUDOSCALAR] {
        for mode in [
            MesonMode::Standard,
            MesonMode::ScoreDirected,
            MesonMode::ScoreWeighted,
        ] {
            let spec = meson(quantum, mode);
            let signature = signature_in(&spec, ElementKind::CloningPair, 3).unwrap();
            assert!(signature.correlatable);
            assert_eq!(
                signature.propagatable,
                mode != MesonMode::ScoreDirected,
                "{}",
                spec.id()
            );
        }
    }
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
        [-0.0476437130239384, -0.0158812376746461, 0.0952874260478767],
        [-0.0714655695359075, -0.0238218565119692, 0.142931139071815],
    ];
    for ((quantum, displacement), expected) in FULL.into_iter().zip(rows) {
        close(&arm(quantum, displacement, 1, 2, &state), &expected, 1e-12);
    }
    let rows = [
        [-0.662266178532522, 0.662266178532522, 0.397359707119513],
        [-0.794719414239026, 0.794719414239026, 0.476831648543416],
        [-0.172439425125162, 0.172439425125162, 0.103463655075097],
        [-0.206927310150194, 0.206927310150194, 0.124156386090117],
        [0.0808189234819349, -0.0808189234819349, -0.0484913540891609],
        [0.0969827081783219, -0.0969827081783219, -0.0581896249069931],
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
    // Norm identities on (1, 2): |q|² = 0.2565…, |r|² = 11.5, ΔS² = 1.69. The
    // gradient of walker 1 is (ΔS / |r|) r̂, so it carries 1/|r|² of the square.
    let square = |v: Vec<f64>| v.iter().map(|x| x * x).sum::<f64>();
    let both = |displacement| {
        square(arm(VECTOR, displacement, 1, 2, &state))
            + square(arm(AXIAL, displacement, 1, 2, &state))
    };
    let abs2 = 0.256578947368421;
    assert!((both(Displacement::Raw) - abs2 * 11.5).abs() < 1e-12);
    assert!((both(Displacement::Unit) - abs2).abs() < 1e-12);
    assert!((both(Displacement::ScoreGradient) - abs2 * 1.69 / 11.5).abs() < 1e-12);
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
    // When both companion maps are the same involution the score gradient of a
    // pair is the same vector at both of its ends, so an arm that multiplies it
    // by an exchange-odd colour part cancels element by element although its
    // declared parity is `Mixed` and the mirror rule does not apply to it. The
    // descriptor of the arm states it; the frame sum is exactly zero.
    let state = reference(&[&INVOLUTION, &INVOLUTION]);
    let gradient = state.score_gradient.as_ref().unwrap();
    for (i, &j) in INVOLUTION.iter().enumerate() {
        assert_eq!(gradient[i * 3..i * 3 + 3], gradient[j * 3..j * 3 + 3]);
    }
    let axial = full(AXIAL, Displacement::ScoreGradient);
    assert_eq!(
        signature_in(&axial, ElementKind::DistancePair, 3)
            .unwrap()
            .exchange,
        ExchangeParity::Mixed
    );
    let (sum, weight) = frame(&axial, &elements(&INVOLUTION), &state);
    assert_eq!(weight, 6.);
    assert_eq!(sum, vec![0.; 3]);
    let vector = full(VECTOR, Displacement::ScoreGradient);
    let (sum, _) = frame(&vector, &elements(&INVOLUTION), &state);
    assert!(sum.iter().any(|s| s.abs() > 1e-3));
    assert!(
        signature_in(&axial, ElementKind::DistancePair, 3)
            .unwrap()
            .descriptor
            .note
            .contains("cancels element by element over a mutual pairing")
    );
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
        [
            0.0165480111444842,
            -0.0395171471470958,
            -0.00121340243252019,
        ],
        [0., 0., 0.],
        [-0.136177900664667, 0.262956674792458, 0.0154120052789798],
        [0., 0., 0.],
    ];
    let cycle = [
        [0.137843842390846, 0.242088312998524, -0.132697427942177],
        [-0.515213933018263, 0.112495350652003, 0.415836138334662],
        [0.0412283738058501, 0.0703110653281559, -0.0367410529698045],
        [-0.147274611937532, 0.0171964888851223, 0.147759308477306],
        [0.00978380361455368, -0.0237555060735558, 0.0139244769041525],
        [-0.0212326892957063, -0.0205139549414484, 0.0273601354034206],
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
        [-0.017582957705408, 0.017582957705408, 0.0937757744288428],
        [-0.0140663661643264, 0.0140663661643264, 0.0750206195430742],
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
    // A box of another dimension cannot wrap these positions, whether it has
    // fewer axes or more.
    for width in [2, 4] {
        for displacement in [Displacement::Raw, Displacement::Unit] {
            let spec = full(AXIAL, displacement);
            assert_eq!(value_in(&boxed(width), &spec, &pair(1, 2), &state), None);
        }
        assert_eq!(value_in(&boxed(width), &raw, &pair(0, 1), &state), None);
    }
    // Each axis wraps by its own length, wherever the box sits, and a periodic
    // box inside a composed policy wraps as it does alone: the differences
    // (1.5, 4, −2) against the lengths (2, 10, 3) become (−0.5, 4, 1).
    let periodic = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![-1., 0., 2.],
            upper: vec![1., 10., 5.],
        },
    };
    let composed = BoundaryPolicy::Composed {
        policies: vec![BoundaryPolicy::ExternalTermination, periodic.clone()],
    };
    state.x[..6].copy_from_slice(&[-0.75, 1., 4.5, 0.75, 5., 2.5]);
    let image = [-0.5, 4., 1.];
    for boundary in [periodic, composed] {
        let gas = GasConfig {
            boundary,
            ..GasConfig::default()
        };
        let wrapped = value_in(&gas, &raw, &pair(0, 1), &state).unwrap();
        close(&wrapped, &image.map(|r| re * r), 1e-15);
        let mirrored = value_in(&gas, &raw, &pair(1, 0), &state).unwrap();
        close(&mirrored, &image.map(|r| -re * r), 1e-15);
        let unit = full(VECTOR, Displacement::Unit);
        let unit = value_in(&gas, &unit, &pair(0, 1), &state).unwrap();
        close(&unit, &image.map(|r| re * r / 17.25_f64.sqrt()), 1e-15);
    }
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
        let mut short = vec![0.; signature.width() + 1];
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
    let expected = r.map(|r| 4. * 0.7 * r / (25. * scale));
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
        assert!((arm(Displacement::ScoreGradient) - (0.5 / 1.5) * re).abs() < 1e-15);
    }
}

#[test]
fn two_dimensions_keep_the_gamma5_contraction_and_the_part_across_the_gradient() {
    // c_0 = (3, 4i)/5, c_1 = (1 + 2i, 2 − i)/√10: 5√10 q_01 = −1 − 2i and
    // 5√10 g_01 = 7 + 14i. r_01 = (3, 4); the recorded gradients (0, 2) of
    // walker 0 and (0, −3) of walker 1 lie on the second axis.
    let root = 10_f64.sqrt();
    let color = vec![
        C::new(0.6, 0.),
        C::new(0., 0.8),
        C::new(1. / root, 2. / root),
        C::new(2. / root, -1. / root),
    ];
    let mut state = swarm(color, vec![0., 0., 3., 4.], vec![0.1, 0.6], &[&[1, 0]]);
    state.score_gradient = Some(vec![0., 2., 0., -3.]);
    let scale = 5. * root;
    let arm = |quantum, mode| scalar(&meson(quantum, mode), 0, 1, &state).unwrap();
    assert!((arm(SCALAR, MesonMode::Standard) + 1. / scale).abs() < 1e-15);
    assert!((arm(PSEUDOSCALAR, MesonMode::Standard) + 2. / scale).abs() < 1e-15);
    assert!((arm(SCALAR, MesonMode::Gamma5Diagonal) - 7. / scale).abs() < 1e-15);
    assert!((arm(PSEUDOSCALAR, MesonMode::Gamma5Diagonal) - 14. / scale).abs() < 1e-15);
    assert!((arm(SCALAR, MesonMode::Abs2) - 0.02).abs() < 1e-15);
    assert!((arm(SCALAR, MesonMode::ScoreWeighted) + 0.5 / scale).abs() < 1e-15);
    // Rows: quantum, projection, displacement, anchor, scale × components.
    let (along, across) = (VectorProjection::Longitudinal, VectorProjection::Transverse);
    let rows = [
        (
            VECTOR,
            VectorProjection::Full,
            Displacement::Raw,
            0,
            [-3., -4.],
        ),
        (VECTOR, along, Displacement::Raw, 0, [0., -4.]),
        (VECTOR, across, Displacement::Raw, 0, [-3., 0.]),
        (AXIAL, along, Displacement::Unit, 0, [0., -1.6]),
        (AXIAL, across, Displacement::Unit, 0, [-1.2, 0.]),
        (
            VECTOR,
            VectorProjection::Full,
            Displacement::ScoreGradient,
            0,
            [0., -2.],
        ),
        (AXIAL, along, Displacement::ScoreGradient, 0, [0., -4.]),
        // From walker 1: q_10 = conj(q_01), r_10 = −r_01.
        (VECTOR, along, Displacement::Raw, 1, [0., 4.]),
        (AXIAL, across, Displacement::Raw, 1, [-6., 0.]),
        (
            VECTOR,
            VectorProjection::Full,
            Displacement::ScoreGradient,
            1,
            [0., 3.],
        ),
    ];
    for (quantum, projection, displacement, i, expected) in rows {
        let spec = vector(quantum, projection, displacement);
        let measured = value(&spec, &pair(i, 1 - i), &state).unwrap();
        close(&measured, &expected.map(|v| v / scale), 1e-15);
    }
    // Only the colour-space matrices and the two arms without an operator in
    // any dimension are refused.
    let mut available = 0;
    for spec in family() {
        let refused = matches!(
            &spec,
            ChannelSpec::Meson {
                quantum: MesonQuantum::Pseudoscalar,
                mode: MesonMode::Abs2,
            } | ChannelSpec::Vector {
                displacement: Displacement::ColorGamma,
                ..
            } | ChannelSpec::Vector {
                projection: VectorProjection::Transverse,
                displacement: Displacement::ScoreGradient,
                ..
            }
        );
        let signature = signature_in(&spec, ElementKind::CloningPair, 2);
        assert_eq!(signature.is_err(), refused, "{}", spec.id());
        assert_eq!(
            value(&spec, &pair(0, 1), &state).is_none(),
            refused,
            "{}",
            spec.id()
        );
        available += usize::from(signature.is_ok());
    }
    assert_eq!(available, 9 + 2 * (3 * 2 + 2));
    let spec = vector(AXIAL, across, Displacement::Unit);
    let signature = signature_in(&spec, ElementKind::DistancePair, 2).unwrap();
    assert_eq!(
        (signature.components, signature.exchange),
        (2, ExchangeParity::Mixed)
    );
}

#[test]
fn only_the_anchor_lends_its_score_and_gradient_to_a_projected_arm() {
    // Walker 2 has neither a valid score nor a gradient: it anchors no arm
    // that reads them and remains the companion of walker 1 in all of them.
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let mut bare = state.clone();
    bare.score_valid[2] = false;
    bare.score.as_mut().unwrap()[2] = 0.;
    bare.score_gradient.as_mut().unwrap()[6..9].fill(0.);
    let mut reading = 0;
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
        let reads =
            *projection != VectorProjection::Full || *displacement == Displacement::ScoreGradient;
        let kept = value(&spec, &pair(1, 2), &bare);
        assert!(kept.is_some(), "{}", spec.id());
        assert_eq!(kept, value(&spec, &pair(1, 2), &state), "{}", spec.id());
        let anchored = value(&spec, &pair(2, 1), &bare);
        assert_eq!(anchored.is_none(), reads, "{}", spec.id());
        reading += usize::from(reads);
    }
    assert_eq!(reading, 2 * (2 * 2 + 2));
    // The score modes of the overlap need the scores of both walkers.
    for mode in [MesonMode::ScoreDirected, MesonMode::ScoreWeighted] {
        assert_eq!(scalar(&meson(SCALAR, mode), 1, 2, &bare), None);
    }
    // Walker 5 sits on walker 4, which keeps its recorded gradient: a zero
    // displacement is the value 0 of every raw arm, along or across the
    // gradient, and has no unit vector to project.
    let mut stacked = state.clone();
    let (left, right) = stacked.x.split_at_mut(15);
    right.copy_from_slice(&left[12..]);
    for projection in [
        VectorProjection::Full,
        VectorProjection::Longitudinal,
        VectorProjection::Transverse,
    ] {
        let arm = |displacement| {
            value(
                &vector(AXIAL, projection, displacement),
                &pair(4, 5),
                &stacked,
            )
        };
        assert_eq!(arm(Displacement::Raw), Some(vec![0.; 3]));
        assert_eq!(arm(Displacement::Unit), None);
    }
    let gradient = value(
        &full(AXIAL, Displacement::ScoreGradient),
        &pair(4, 5),
        &stacked,
    );
    assert!(gradient.unwrap().iter().any(|v| v.abs() > 1e-3));
}

#[test]
fn a_displacement_of_any_nonzero_length_has_a_unit_vector() {
    // q_01 = 1/√48 is real; walker 0 sits at the origin.
    let mut state = reference(&[&CYCLE]);
    let re = 1. / 48_f64.sqrt();
    let arm =
        |displacement, state: &FrameState| value(&full(VECTOR, displacement), &pair(0, 1), state);
    for length in [1e-200, 1e-12, 1e200] {
        state.x[3..6].copy_from_slice(&[3. * length, -4. * length, 0.]);
        let unit = arm(Displacement::Unit, &state).unwrap();
        close(&unit, &[0.6 * re, -0.8 * re, 0.], 1e-15);
        let raw = arm(Displacement::Raw, &state).unwrap();
        let scaled: Vec<f64> = raw.iter().map(|r| r / length).collect();
        close(&scaled, &[3. * re, -4. * re, 0.], 1e-15);
    }
    // A difference beyond the largest number is not a displacement.
    state.x[..6].copy_from_slice(&[-1.5e308, 0., 0., 1.5e308, 0., 0.]);
    assert_eq!(arm(Displacement::Raw, &state), None);
    assert_eq!(arm(Displacement::Unit, &state), None);
}

#[test]
fn descriptors_cite_the_book_definitions_and_refusals_are_plain_ascii() {
    let twins = "never shares a fit basis";
    for spec in family() {
        let (label, twin) = match &spec {
            ChannelSpec::Meson { quantum, mode } => (
                match mode {
                    MesonMode::Standard => "def-sm-direct-color-contractions",
                    MesonMode::Gamma5Diagonal => "def-qft-color-gamma-operators",
                    MesonMode::Abs2 => "cor-sm-direct-exchange-parity",
                    _ => "",
                },
                (*quantum, *mode) == (SCALAR, MesonMode::ScoreDirected),
            ),
            ChannelSpec::Vector {
                projection,
                displacement,
                ..
            } => (
                match (projection, displacement) {
                    (_, Displacement::ColorGamma) => "def-qft-color-gamma-operators",
                    (VectorProjection::Full, Displacement::Raw | Displacement::Unit) => {
                        "def-sm-direct-color-contractions"
                    }
                    _ => "",
                },
                (*projection, *displacement)
                    == (VectorProjection::Longitudinal, Displacement::ScoreGradient),
            ),
            _ => continue,
        };
        for dimension in 1..=4 {
            for kind in [
                ElementKind::DistancePair,
                ElementKind::CloningPair,
                ElementKind::Site,
                ElementKind::Triplet,
            ] {
                match signature_in(&spec, kind, dimension) {
                    Ok(signature) => {
                        let descriptor = signature.descriptor;
                        assert_eq!(descriptor.book_label, label, "{}", spec.id());
                        // A series equal to another element by element says so.
                        assert_eq!(descriptor.note.contains(twins), twin, "{}", spec.id());
                    }
                    Err(error) => assert!(error.to_string().is_ascii(), "{error}"),
                }
            }
        }
    }
}

#[test]
fn nonfinite_inputs_mask_the_element_and_never_reach_a_series() {
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let available: Vec<ChannelSpec> = family()
        .into_iter()
        .filter(|s| signature_in(s, ElementKind::DistancePair, 3).is_ok())
        .collect();
    let spoils: [fn(&mut FrameState); 7] = [
        |s| s.color[4] = C::new(f64::NAN, 0.),
        |s| s.color[7] = C::new(0., f64::INFINITY),
        |s| s.x[5] = f64::NAN,
        |s| s.score.as_mut().unwrap()[2] = f64::NAN,
        |s| s.score_gradient.as_mut().unwrap()[3] = f64::INFINITY,
        |s| s.x[6] = f64::NEG_INFINITY,
        |s| s.score_gradient.as_mut().unwrap()[4] = f64::NAN,
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
            let masked = [true, true, spatial, scored, directed, spatial, directed][case];
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
fn a_score_difference_of_any_nonzero_size_orients_the_pair() {
    // Im q_23 = −0.3178…. One unit in the last place of S_2 = 1.1 decides the
    // orientation, as does the smallest positive number against zero: only
    // S_2 = S_3 is a tie. The fitness runs the other way and is not read.
    let state = reference(&[&CYCLE]);
    let im = -0.317887765695611;
    let directed = meson(PSEUDOSCALAR, MesonMode::ScoreDirected);
    let above = f64::from_bits(1.1_f64.to_bits() + 1);
    let cases = [
        (1.1, above, 1.),
        (above, 1.1, -1.),
        (0., 5e-324, 1.),
        (5e-324, 0., -1.),
        (-1e300, 1e300, 1.),
    ];
    for (lower, upper, sign) in cases {
        let mut near = state.clone();
        let score = near.score.as_mut().unwrap();
        (score[2], score[3]) = (lower, upper);
        near.fitness = Some(score.iter().map(|s| -s).collect());
        let value = scalar(&directed, 2, 3, &near).unwrap();
        assert!((value - sign * im).abs() < 1e-12, "{lower} {upper}");
        assert_eq!(scalar(&directed, 3, 2, &near), Some(value));
        let even = meson(SCALAR, MesonMode::ScoreDirected);
        assert!((scalar(&even, 2, 3, &near).unwrap() + 0.264906471413009).abs() < 1e-12);
    }
    // The weighted arm carries the difference itself, here 2⁻⁵² exactly.
    let mut near = state.clone();
    near.score.as_mut().unwrap()[3] = above;
    let weighted = meson(PSEUDOSCALAR, MesonMode::ScoreWeighted);
    for (i, j, sign) in [(2, 3, 1.), (3, 2, -1.)] {
        let value = scalar(&weighted, i, j, &near).unwrap();
        assert!((value / f64::EPSILON - sign * im).abs() < 1e-12);
    }
}

#[test]
fn records_shorter_than_the_population_mask_the_walkers_they_do_not_cover() {
    // Each record in turn covers the walkers 0 and 1 alone. Walker 2 is then
    // outside that record: no arm reading it there has a value, no other arm
    // changes, and nothing is indexed out of range.
    let state = reference(&[&CYCLE, &INVOLUTION]);
    let cuts: [fn(&mut FrameState); 6] = [
        |s| s.color.truncate(6),
        |s| s.color_valid.truncate(2),
        |s| s.x.truncate(6),
        |s| s.score.as_mut().unwrap().truncate(2),
        |s| s.score_valid.truncate(2),
        |s| s.score_gradient.as_mut().unwrap().truncate(6),
    ];
    for (case, cut) in cuts.into_iter().enumerate() {
        let mut short = state.clone();
        cut(&mut short);
        for spec in family() {
            if signature_in(&spec, ElementKind::DistancePair, 3).is_err() {
                continue;
            }
            let (spatial, directed, scored) = match &spec {
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
            // Masked with walker 2 as the companion, then as the anchor: the
            // gradient and its validity are those of the anchor alone.
            let masked = [
                [true; 2],
                [true; 2],
                [spatial; 2],
                [scored; 2],
                [scored, scored || directed],
                [false, directed],
            ][case];
            for ((i, j), masked) in [(1, 2), (2, 1)].into_iter().zip(masked) {
                let measured = value(&spec, &pair(i, j), &short);
                assert_eq!(measured.is_none(), masked, "{} {case}", spec.id());
                if !masked {
                    assert_eq!(measured, value(&spec, &pair(i, j), &state));
                }
            }
        }
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

/// A frame whose cloning scores are the prescribed values: with `fitness = 1`
/// and `epsilon = 0` the score `(V_c − V_i) / (V_i + ε)` is `V_c − 1`, so a
/// companion fitness of `1 + S_i` gives walker `i` the score `S_i` exactly.
/// Positions are the rows of `x`, scaled by `stretch`.
fn scored_frame(x: &[[f64; 3]], s: &[f64], stretch: f64) -> (GasConfig, Frame, FrameState) {
    let (n, d) = (s.len(), 3);
    let mut gas = GasConfig::default();
    gas.clone_decision.epsilon = 0.;
    let positions: Vec<f64> = x.iter().flatten().map(|a| a * stretch).collect();
    let cloning = Companions {
        count: 1,
        slot: (0..n as u32).map(|i| (i + 1) % n as u32).collect(),
        generation: vec![0; n],
        valid: vec![true; n],
        historical: vec![false; n],
        mutual: false,
    };
    let frame = Frame {
        step: 1,
        n,
        d,
        x: positions.clone(),
        eligible: vec![true; n],
        generation: vec![0; n],
        fitness: Some(vec![1.; n]),
        cloning: Some(cloning),
        companion_fitness: Some(s.iter().map(|s| 1. + s).collect()),
        cloned: vec![false; n],
        revived: vec![false; n],
        ..Frame::default()
    };
    frame.validate().unwrap();
    let state = FrameState {
        n,
        d,
        x: positions,
        eligible: vec![true; n],
        // Walker 0 sees walker 1 through the distance map and walker 2 through
        // the cloning map: the two companions of the linear-field oracle.
        distance_companion: Some((0..n as u32).map(|i| (i + 1) % n as u32).collect()),
        cloning_companion: Some((0..n as u32).map(|i| (i + 2) % n as u32).collect()),
        ..FrameState::default()
    };
    (gas, frame, state)
}
/// Anchor at the origin with the companions `(4, 0, 0)` and `(0, 1, 0)` of the
/// linear score field `S(x) = (1, 1, 0) · x`.
fn linear_score_field(stretch: f64) -> (GasConfig, Frame, FrameState) {
    scored_frame(
        &[[0., 0., 0.], [4., 0., 0.], [0., 1., 0.]],
        &[0., 4., 1.],
        stretch,
    )
}
fn filled_gradient(stretch: f64) -> Vec<f64> {
    let (gas, frame, mut state) = linear_score_field(stretch);
    assert!(score::fill(&frame, &gas, &mut state).is_available());
    assert_eq!(state.score_valid, vec![true; 3]);
    state.score_gradient.unwrap()
}

#[test]
fn the_score_gradient_is_the_finite_difference_quotient_of_the_score_field() {
    // S(x) = x + y, anchor at the origin, companions at (4, 0, 0) and (0, 1, 0):
    // mean_p (ΔS_p / |r_p|) r̂_p = ((1, 0, 0) + (0, 1, 0)) / 2.
    let gradient = filled_gradient(1.);
    close(&gradient[..3], &[0.5, 0.5, 0.], 1e-15);
    // The angle against the true direction (1, 1, 0)/√2, through the cross
    // product, which keeps its digits near zero where an arc cosine loses them.
    let length = gradient[..3].iter().map(|g| g * g).sum::<f64>().sqrt();
    let unit = [1. / 2f64.sqrt(), 1. / 2f64.sqrt(), 0.];
    let cross = [
        gradient[1] * unit[2] - gradient[2] * unit[1],
        gradient[2] * unit[0] - gradient[0] * unit[2],
        gradient[0] * unit[1] - gradient[1] * unit[0],
    ];
    let sine = cross.iter().map(|c| c * c).sum::<f64>().sqrt() / length;
    let degrees = sine.asin().to_degrees();
    assert!(degrees < 1e-5, "{degrees} degrees off the true direction");
}

#[test]
fn the_score_gradient_scales_as_the_inverse_of_a_length() {
    let (plain, stretched) = (filled_gradient(1.), filled_gradient(7.));
    for (a, b) in plain.iter().zip(&stretched) {
        assert!((a / 7. - b).abs() < 1e-14, "{a} vs {b}");
    }
    assert!(plain.iter().any(|g| g.abs() > 1e-3));
}

#[test]
fn vector_series_are_invariant_under_a_translation_of_the_swarm() {
    let state = reference(&[&CYCLE]);
    for offset in [1e3, 1e5] {
        let moved = FrameState {
            x: state.x.iter().map(|a| a + offset).collect(),
            ..state.clone()
        };
        for quantum in [VECTOR, AXIAL] {
            let spec = full(quantum, Displacement::Raw);
            let here = mean(&spec, &CYCLE, &state);
            let there = mean(&spec, &CYCLE, &moved);
            for (a, b) in here.iter().zip(&there) {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(1e-12),
                    "{a} vs {b} at offset {offset}"
                );
            }
            assert!(here.iter().any(|v| v.abs() > 1e-3));
        }
    }
}
