//! Baryon and glueball arms against closed-form colour triples, a recorded
//! eight-walker frame evaluated from the definitions, the transformation laws
//! of the determinant and the plaquette, and the moments of Haar colours.
use algorithmic_gas::{
    GasConfig, GasError,
    boundary::{BoundaryPolicy, BoxDomain},
    physics::{
        qft::math::{C, determinant, dot, mul},
        spectroscopy::{
            config::{
                BaryonMode, ChannelSpec, GlueballObservable, MeasurementConfig, Momentum,
                MomentumPhase,
            },
            contract::{
                Auxiliary, Capabilities, Element, ElementKind, ExchangeParity, FrameState,
                LocalOperator, OperatorContext, Record, SpatialParity,
            },
            fields::color::det3,
            operators::channels,
        },
    },
    random::{RandomStream, Stream},
};
use std::collections::{BTreeMap, BTreeSet};

const KAPPA: f64 = 1.3;
const FORCE: [f64; 24] = [
    1., 2., 3., -1., 0.5, 2., 0.3, -0.7, 0.2, 2., 1., -1., 0., 0., 0., -0.5, 1.5, 1., 0.8, -0.4,
    1.6, 1.2, 0.9, -0.3,
];
const VELOCITY: [f64; 24] = [
    0.1, -0.2, 0.3, 1., 0.5, -1., -0.4, 0.8, 0.6, 0.2, 0.2, -0.9, 1., 1., 1., -1.2, 0.3, 0.7, 0.5,
    -0.6, 0.15, -0.3, 1.4, 0.45,
];
const DISTANCE: [u32; 8] = [1, 0, 0, 0, 1, 5, 3, 4];
const CLONING: [u32; 8] = [2, 2, 1, 0, 2, 2, 5, 3];
const SCORE: [f64; 8] = [0.5, -0.2, 1.1, 0., 0.3, -0.7, 0.9, 0.4];
const AXIS: [f64; 8] = [-1.5, 0.25, 2.75, 1., -0.5, 2., -1.9, 0.6];
const VALID: [usize; 4] = [0, 1, 2, 6];
const PERMUTATIONS: [([usize; 3], f64); 6] = [
    ([0, 1, 2], 1.),
    ([1, 2, 0], 1.),
    ([2, 0, 1], 1.),
    ([1, 0, 2], -1.),
    ([0, 2, 1], -1.),
    ([2, 1, 0], -1.),
];
fn baryon(mode: BaryonMode) -> ChannelSpec {
    ChannelSpec::Baryon {
        mode,
        flux_alpha: 1.,
    }
}
fn flux(flux_alpha: f64) -> ChannelSpec {
    ChannelSpec::Baryon {
        mode: BaryonMode::FluxWeighted,
        flux_alpha,
    }
}
fn glueball(observable: GlueballObservable) -> ChannelSpec {
    ChannelSpec::Glueball {
        observable,
        momentum: None,
    }
}
fn projected(observable: GlueballObservable, mode: u32, phase: MomentumPhase) -> ChannelSpec {
    ChannelSpec::Glueball {
        observable,
        momentum: Some(Momentum {
            axis: 0,
            mode,
            phase,
        }),
    }
}
fn every_triplet_arm() -> Vec<ChannelSpec> {
    let mut out: Vec<_> = BaryonMode::ALL.iter().map(|&mode| baryon(mode)).collect();
    out.push(flux(0.5));
    out.extend(
        GlueballObservable::ALL
            .iter()
            .filter(|&&o| o != GlueballObservable::ForceNorm)
            .map(|&o| glueball(o)),
    );
    out
}
fn periodic() -> GasConfig {
    GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "x".into(),
            domain: BoxDomain {
                lower: vec![-2.; 3],
                upper: vec![3.; 3],
            },
        },
        ..GasConfig::default()
    }
}
fn triplet(i: usize, j: usize, k: usize) -> Element {
    Element {
        walkers: [i as u32, j as u32, k as u32],
        kind: ElementKind::Triplet,
        weight: 1.,
        generation: [0; 3],
    }
}
fn site(i: usize) -> Element {
    Element {
        walkers: [i as u32; 3],
        kind: ElementKind::Site,
        weight: 1.,
        generation: [0; 3],
    }
}
fn sampled() -> Vec<Element> {
    (0..8)
        .map(|i| triplet(i, DISTANCE[i] as usize, CLONING[i] as usize))
        .collect()
}
/// One row of `evaluate` on `element`, auxiliary column included, `None` when
/// masked. An element of a kind the specification is not measured on has no
/// signature.
fn raw_in(
    gas: &GasConfig,
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
) -> Option<Vec<f64>> {
    let (measurement, capabilities) = (MeasurementConfig::default(), Capabilities::nominal());
    let context = OperatorContext {
        gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let width = spec
        .signature(element.kind, &context)
        .map_or(1, |signature| signature.width());
    let mut out = vec![0.; width];
    spec.evaluate(element, state, &context, &mut out)
        .then_some(out)
}
fn raw(spec: &ChannelSpec, element: &Element, state: &FrameState) -> Option<Vec<f64>> {
    raw_in(&GasConfig::default(), spec, element, state)
}
/// Components of `spec` on `element` with its auxiliary column folded in, as
/// the measurement folds it at a source time: the momentum weight or the score
/// factor multiplies the value, an element without a score factor is masked,
/// and a momentum arm is shown here without the frame-mean subtraction that
/// the measurement applies before projecting.
fn value_in(
    gas: &GasConfig,
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
) -> Option<Vec<f64>> {
    let (measurement, capabilities) = (MeasurementConfig::default(), Capabilities::nominal());
    let context = OperatorContext {
        gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let row = raw_in(gas, spec, element, state)?;
    let signature = spec.signature(element.kind, &context).ok()?;
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
/// `evaluate_all` with the auxiliary column folded in element by element, as
/// `value_in` folds one row: `[elements, components]` and `[elements]`.
fn values(spec: &ChannelSpec, elements: &[Element], state: &FrameState) -> (Vec<f64>, Vec<bool>) {
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
    let signature = spec.signature(elements[0].kind, &context).unwrap();
    let (components, width) = (signature.components, signature.width());
    let mut raw = vec![0.; elements.len() * width];
    let mut valid = vec![false; elements.len()];
    spec.evaluate_all(elements, state, &context, &mut raw, &mut valid);
    let mut values = vec![0.; elements.len() * components];
    for (index, row) in raw.chunks_exact(width).enumerate() {
        let factor = match signature.auxiliary {
            None => 1.,
            Some(auxiliary) => {
                let factor = row[components];
                if !factor.is_finite() || (auxiliary == Auxiliary::ScoreOrientation && factor == 0.)
                {
                    valid[index] = false;
                }
                factor
            }
        };
        for k in 0..components {
            values[index * components + k] = row[k] * factor;
        }
    }
    (values, valid)
}
fn scalar(spec: &ChannelSpec, element: &Element, state: &FrameState) -> f64 {
    value(spec, element, state).unwrap()[0]
}
fn determinant_of(element: &Element, state: &FrameState) -> C {
    let b = value(&baryon(BaryonMode::Complex), element, state).unwrap();
    C::new(b[0], b[1])
}
/// The eight recorded walkers with colours `F_a e^{i κ v_a} / |F|`, invalid at zero force.
fn frame_state(force: &[f64], velocity: &[f64]) -> FrameState {
    let n = force.len() / 3;
    let mut state = FrameState {
        n,
        d: 3,
        x: vec![0.; n * 3],
        color: vec![C::ZERO; n * 3],
        color_valid: vec![false; n],
        force: Some(force.to_vec()),
        force_valid: vec![true; n],
        score: Some(SCORE.to_vec()),
        score_valid: vec![true; n],
        eligible: vec![true; n],
        ..FrameState::default()
    };
    for i in 0..n {
        let f = &force[i * 3..i * 3 + 3];
        let norm = f.iter().map(|f| f * f).sum::<f64>().sqrt();
        state.x[i * 3] = AXIS[i];
        state.color_valid[i] = norm > 1e-12;
        if state.color_valid[i] {
            for a in 0..3 {
                state.color[i * 3 + a] = C::phase(KAPPA * velocity[i * 3 + a]) * (f[a] / norm);
            }
        }
    }
    state
}
fn recorded() -> FrameState {
    frame_state(&FORCE, &VELOCITY)
}
fn color_state(d: usize, colors: &[C]) -> FrameState {
    let n = colors.len() / d;
    FrameState {
        n,
        d,
        x: vec![0.; n * d],
        color: colors.to_vec(),
        color_valid: vec![true; n],
        score: Some((0..n).map(|i| i as f64).collect()),
        score_valid: vec![true; n],
        eligible: vec![true; n],
        ..FrameState::default()
    }
}
fn scaled(vector: [C; 3], norm: f64) -> [C; 3] {
    vector.map(|z| z / norm)
}
fn book_triple() -> Vec<C> {
    let (one, i) = (C::ONE, C::new(0., 1.));
    [
        [one, C::ZERO, C::ZERO],
        scaled([one, one, C::ZERO], 2f64.sqrt()),
        scaled([one, i, C::ZERO], 2f64.sqrt()),
    ]
    .concat()
}
fn lifted_triple() -> Vec<C> {
    let (one, i) = (C::ONE, C::new(0., 1.));
    let third = C::phase(std::f64::consts::FRAC_PI_3);
    [
        [one, C::ZERO, C::ZERO],
        scaled([one, one, C::ZERO], 2f64.sqrt()),
        scaled([one, i, third], 3f64.sqrt()),
    ]
    .concat()
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}
fn rotation(p: usize, q: usize, angle: f64) -> Vec<C> {
    let mut r: Vec<C> = (0..9)
        .map(|k| if k / 3 == k % 3 { C::ONE } else { C::ZERO })
        .collect();
    r[p * 3 + p] = angle.cos().into();
    r[q * 3 + q] = angle.cos().into();
    r[p * 3 + q] = (-angle.sin()).into();
    r[q * 3 + p] = angle.sin().into();
    r
}
fn diagonal(phases: [f64; 3]) -> Vec<C> {
    (0..9)
        .map(|k| {
            if k / 3 == k % 3 {
                C::phase(phases[k / 3])
            } else {
                C::ZERO
            }
        })
        .collect()
}
fn special_unitary() -> Vec<C> {
    [
        diagonal([0.4, -1.1, 0.7]),
        rotation(0, 1, 0.7),
        diagonal([-0.3, 0.9, -0.6]),
        rotation(1, 2, -1.2),
        rotation(0, 2, 0.5),
    ]
    .into_iter()
    .reduce(|a, b| mul(&a, &b, 3))
    .unwrap()
}
fn rotated(state: &FrameState, matrix: &[C]) -> FrameState {
    let mut out = state.clone();
    for (row, source) in out
        .color
        .chunks_exact_mut(3)
        .zip(state.color.chunks_exact(3))
    {
        for a in 0..3 {
            row[a] = (0..3).fold(C::ZERO, |s, b| s + matrix[a * 3 + b] * source[b]);
        }
    }
    out
}
/// Every perfect matching of `items`, as the involution it defines on eight slots.
fn matchings(items: &[usize]) -> Vec<[usize; 8]> {
    let Some((&a, rest)) = items.split_first() else {
        return vec![[0; 8]];
    };
    let mut out = vec![];
    for (at, &b) in rest.iter().enumerate() {
        let others: Vec<usize> = rest
            .iter()
            .enumerate()
            .filter(|&(e, _)| e != at)
            .map(|(_, &w)| w)
            .collect();
        for mut map in matchings(&others) {
            (map[a], map[b]) = (b, a);
            out.push(map);
        }
    }
    out
}
#[test]
fn determinant_is_alternating_vanishes_on_a_repeated_column_and_matches_elimination() {
    let state = recorded();
    let c = |w: usize| &state.color[w * 3..w * 3 + 3];
    let reference = det3(c(0), c(1), c(2));
    assert!(close(reference.re, -0.238425106641115) && close(reference.im, -0.0185258398336716));
    for (p, sign) in PERMUTATIONS {
        let permuted = det3(c(p[0]), c(p[1]), c(p[2]));
        assert!((permuted - reference * sign).abs() < 1e-15, "{p:?}");
    }
    for repeated in [[0, 0, 1], [0, 1, 0], [1, 0, 0], [2, 2, 2]] {
        let [i, j, k] = repeated;
        assert!(det3(c(i), c(j), c(k)).abs() < 1e-15, "{repeated:?}");
    }
    for [i, j, k] in [[0, 1, 2], [6, 3, 5], [7, 2, 5]] {
        let columns: Vec<C> = (0..9).map(|e| c([i, j, k][e % 3])[e / 3]).collect();
        assert!((det3(c(i), c(j), c(k)) - determinant(&columns, 3)).abs() < 1e-14);
    }
}
#[test]
fn analytic_triples_give_the_closed_form_determinants_and_plaquettes() {
    let element = triplet(0, 1, 2);
    let book = color_state(3, &book_triple());
    assert!(determinant_of(&element, &book).abs() < 1e-15);
    assert!(close(
        scalar(&glueball(GlueballObservable::RePlaquette), &element, &book),
        0.25
    ));
    assert!(close(
        scalar(&glueball(GlueballObservable::OneMinusCos), &element, &book),
        0.292893218813452
    ));
    assert!(close(
        scalar(&glueball(GlueballObservable::Sin2), &element, &book),
        0.5
    ));
    let lifted = color_state(3, &lifted_triple());
    let b = determinant_of(&element, &lifted);
    let exact = C::phase(std::f64::consts::FRAC_PI_3) / 6f64.sqrt();
    assert!((b - exact).abs() < 1e-14);
    assert!(close(b.re, 0.204124145231932) && close(b.im, 0.353553390593274));
    for (spec, expected) in [
        (baryon(BaryonMode::Abs), 0.408248290463863),
        (baryon(BaryonMode::Abs2), 1. / 6.),
        (flux(1.), 0.547175046538009),
        (flux(0.5), 0.472634401343815),
        (glueball(GlueballObservable::RePlaquette), 1. / 6.),
        (glueball(GlueballObservable::OneMinusRe), 0.833333333333333),
        (glueball(GlueballObservable::OneMinusCos), 0.292893218813452),
        (glueball(GlueballObservable::Sin2), 0.5),
    ] {
        assert!(
            close(scalar(&spec, &element, &lifted), expected),
            "{}",
            spec.id()
        );
    }
}
#[test]
fn orthonormal_colours_saturate_the_determinant_and_have_no_plaquette_phase() {
    let identity: Vec<C> = (0..9)
        .map(|k| if k / 3 == k % 3 { C::ONE } else { C::ZERO })
        .collect();
    let state = color_state(3, &identity);
    for (p, sign) in PERMUTATIONS {
        let element = triplet(p[0], p[1], p[2]);
        assert_eq!(scalar(&baryon(BaryonMode::Real), &element, &state), sign);
        assert_eq!(scalar(&baryon(BaryonMode::Imag), &element, &state), 0.);
        assert_eq!(scalar(&baryon(BaryonMode::Abs2), &element, &state), 1.);
        assert_eq!(scalar(&baryon(BaryonMode::Abs), &element, &state), 1.);
        assert_eq!(
            scalar(&glueball(GlueballObservable::RePlaquette), &element, &state),
            0.
        );
        assert_eq!(
            scalar(&glueball(GlueballObservable::OneMinusRe), &element, &state),
            1.
        );
        for masked in [
            glueball(GlueballObservable::OneMinusCos),
            glueball(GlueballObservable::Sin2),
            flux(1.),
            flux(0.),
        ] {
            assert_eq!(value(&masked, &element, &state), None, "{}", masked.id());
        }
    }
}
#[test]
fn equal_colours_have_a_unit_plaquette_and_a_vanishing_determinant() {
    let lifted = lifted_triple();
    let colors = [&lifted[6..9], &lifted[6..9], &lifted[6..9]].concat();
    let (state, element) = (color_state(3, &colors), triplet(0, 1, 2));
    assert!(determinant_of(&element, &state).abs() < 1e-15);
    for (observable, expected) in [
        (GlueballObservable::RePlaquette, 1.),
        (GlueballObservable::OneMinusRe, 0.),
        (GlueballObservable::OneMinusCos, 0.),
        (GlueballObservable::Sin2, 0.),
    ] {
        assert!(close(
            scalar(&glueball(observable), &element, &state),
            expected
        ));
    }
    assert!(close(scalar(&flux(1.), &element, &state), 0.));
}
#[test]
fn coplanar_colours_at_equal_angles_attain_the_lower_bound_of_the_plaquette() {
    let h = 0.75f64.sqrt();
    let colors: Vec<C> = [1., 0., 0., 0.5, h, 0., 0.5, -h, 0.]
        .into_iter()
        .map(C::from)
        .collect();
    let (state, element) = (color_state(3, &colors), triplet(0, 1, 2));
    assert!(determinant_of(&element, &state).abs() < 1e-15);
    for (observable, expected) in [
        (GlueballObservable::RePlaquette, -0.125),
        (GlueballObservable::OneMinusRe, 1.125),
        (GlueballObservable::OneMinusCos, 2.),
        (GlueballObservable::Sin2, 0.),
    ] {
        assert!(close(
            scalar(&glueball(observable), &element, &state),
            expected
        ));
    }
}
#[test]
fn recorded_frame_reproduces_every_arm_and_both_frame_normalisations() {
    let state = recorded();
    for (w, expected) in [
        (
            0,
            [
                (0.26500606314725, 0.034646181950792),
                (0.516557171455901, -0.137415335140458),
                (0.741577031980555, 0.304828883959597),
            ],
        ),
        (
            7,
            [
                (0.72555836069403, -0.298244330393064),
                (-0.145105726293871, 0.570173810649502),
                (-0.16350439210216, -0.108295208688297),
            ],
        ),
    ] {
        for (a, (re, im)) in expected.into_iter().enumerate() {
            let c = state.color[w * 3 + a];
            assert!(close(c.re, re) && close(c.im, im));
        }
    }
    let elements = sampled();
    let table = [
        (
            baryon(BaryonMode::Real),
            [
                -0.238425106641115,
                0.238425106641115,
                -0.238425106641115,
                0.650511165846924,
            ],
            0.103021514801452,
        ),
        (
            baryon(BaryonMode::Imag),
            [
                -0.0185258398336716,
                0.0185258398336715,
                -0.0185258398336716,
                0.443731074962497,
            ],
            0.106301308782206,
        ),
        (
            baryon(BaryonMode::Abs2),
            [
                0.0571897382183701,
                0.0571897382183701,
                0.0571897382183701,
                0.620062043778897,
            ],
            0.197907814608502,
        ),
        (
            baryon(BaryonMode::Abs),
            [
                0.239143760567509,
                0.239143760567509,
                0.239143760567509,
                0.787440184254587,
            ],
            0.376217866489278,
        ),
        (
            baryon(BaryonMode::ScoreOrdered),
            [
                0.238425106641115,
                0.238425106641115,
                0.238425106641115,
                -0.650511165846924,
            ],
            0.0161910385191056,
        ),
        (
            flux(1.),
            [
                0.897480307221536,
                0.897480307221536,
                0.897480307221536,
                0.792182139150765,
            ],
            0.871155765203843,
        ),
        (
            flux(0.5),
            [
                0.463278335025761,
                0.463278335025761,
                0.463278335025761,
                0.789807602911032,
            ],
            0.544910651997079,
        ),
        (
            glueball(GlueballObservable::RePlaquette),
            [
                -0.039889462413726,
                -0.039889462413726,
                -0.039889462413726,
                0.052202244582047,
            ],
            -0.0168665356647827,
        ),
        (
            glueball(GlueballObservable::OneMinusRe),
            [
                1.03988946241373,
                1.03988946241373,
                1.03988946241373,
                0.947797755417953,
            ],
            1.01686653566478,
        ),
        (
            glueball(GlueballObservable::OneMinusCos),
            [
                1.3225262985589,
                1.3225262985589,
                1.3225262985589,
                0.00600392773690783,
            ],
            0.993395705853405,
        ),
        (
            glueball(GlueballObservable::Sin2),
            [
                0.895976786737892,
                0.895976786737892,
                0.895976786737892,
                0.0119718083255458,
            ],
            0.674975542134806,
        ),
    ];
    for (spec, expected, mean) in table {
        let (values, valid) = values(&spec, &elements, &state);
        let kept: Vec<usize> = (0..8).filter(|&i| valid[i]).collect();
        assert_eq!(kept, VALID, "{}", spec.id());
        for (&i, e) in VALID.iter().zip(expected) {
            assert!(close(values[i], e), "{} anchor {i}", spec.id());
        }
        let sum: f64 = VALID.iter().map(|&i| values[i]).sum();
        assert!(close(sum / 4., mean), "{}", spec.id());
        assert!(close(sum / 8., mean / 2.), "{}", spec.id());
    }
    let (values, valid) = values(&baryon(BaryonMode::Complex), &elements, &state);
    assert_eq!(values.len(), 16);
    for &i in &VALID {
        assert!(valid[i]);
        assert_eq!(
            values[i * 2],
            scalar(&baryon(BaryonMode::Real), &elements[i], &state)
        );
        assert_eq!(
            values[i * 2 + 1],
            scalar(&baryon(BaryonMode::Imag), &elements[i], &state)
        );
    }
}
#[test]
fn invalid_colours_ineligible_walkers_and_coinciding_slots_mask_every_triplet_arm() {
    let state = recorded();
    let mut dead = recorded();
    dead.eligible[2] = false;
    let planar: Vec<C> = book_triple()
        .chunks_exact(3)
        .flat_map(|c| [c[0], c[1]])
        .collect();
    let planar = color_state(2, &planar);
    for spec in every_triplet_arm() {
        for element in [
            triplet(4, 1, 2),
            triplet(7, 4, 3),
            triplet(3, 0, 0),
            triplet(5, 5, 2),
        ] {
            assert_eq!(value(&spec, &element, &state), None, "{}", spec.id());
        }
        assert_eq!(
            value(&spec, &triplet(0, 1, 2), &dead),
            None,
            "{}",
            spec.id()
        );
        let site_kind = Element {
            kind: ElementKind::Site,
            ..triplet(0, 1, 2)
        };
        assert_eq!(value(&spec, &site_kind, &state), None);
        let outside = value(&spec, &triplet(0, 1, 2), &planar);
        match spec {
            ChannelSpec::Baryon { .. } => assert_eq!(outside, None),
            _ => assert!(outside.is_some()),
        }
    }
    let planar_plaquette = scalar(
        &glueball(GlueballObservable::RePlaquette),
        &triplet(0, 1, 2),
        &planar,
    );
    assert!(close(planar_plaquette, 0.25));
}
#[test]
fn relabelling_a_triplet_flips_the_determinant_parts_and_fixes_every_other_arm() {
    let state = recorded();
    for base in [[0, 1, 2], [6, 3, 5]] {
        let identity = triplet(base[0], base[1], base[2]);
        for spec in every_triplet_arm() {
            let odd = matches!(
                spec,
                ChannelSpec::Baryon {
                    mode: BaryonMode::Real | BaryonMode::Imag | BaryonMode::Complex,
                    ..
                }
            );
            let reference = value(&spec, &identity, &state).unwrap();
            for (p, sign) in PERMUTATIONS {
                let element = triplet(base[p[0]], base[p[1]], base[p[2]]);
                let permuted = value(&spec, &element, &state).unwrap();
                for (a, b) in permuted.iter().zip(&reference) {
                    let expected = if odd { sign * b } else { *b };
                    assert!(close(*a, expected), "{} {p:?}", spec.id());
                }
            }
        }
    }
}
#[test]
fn signatures_state_the_relabelling_parity_the_measured_values_obey() {
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
    for &mode in BaryonMode::ALL {
        let spec = baryon(mode);
        let signature = spec.signature(ElementKind::Triplet, &context).unwrap();
        let odd = matches!(
            mode,
            BaryonMode::Real | BaryonMode::Imag | BaryonMode::Complex
        );
        assert_eq!(signature.exchange == ExchangeParity::Odd, odd);
        assert_eq!(signature.exchange == ExchangeParity::Even, !odd);
        assert_eq!(
            signature.components,
            if mode == BaryonMode::Complex { 2 } else { 1 }
        );
        assert_eq!(signature.requires.dimension, Some(3));
        assert!(signature.requires.records.contains(&Record::Color));
        assert_eq!(
            signature.requires.records.contains(&Record::Fitness)
                && signature
                    .requires
                    .records
                    .contains(&Record::CloningCompanions),
            mode == BaryonMode::ScoreOrdered
        );
        assert!(signature.correlatable && !signature.degenerate);
        assert_eq!(signature.normalization, None);
        // The column order of `score_ordered` is frozen with the element, so
        // it enters the lag product squared and the source-frozen propagator
        // of the arm is the one of `baryon/real`: the arm is a frame-mean
        // channel only, and no configuration gives it a propagator of its own.
        assert_eq!(
            signature.propagatable,
            mode != BaryonMode::ScoreOrdered,
            "{}",
            spec.id()
        );
        let book = !matches!(
            mode,
            BaryonMode::Abs | BaryonMode::ScoreOrdered | BaryonMode::FluxWeighted
        );
        assert_eq!(!signature.descriptor.book_label.is_empty(), book);
        assert!(!signature.descriptor.definition.is_empty());
        assert!(matches!(
            spec.signature(ElementKind::Site, &context),
            Err(GasError::Capability(_))
        ));
    }
    let parity = |mode| {
        baryon(mode)
            .signature(ElementKind::Triplet, &context)
            .unwrap()
            .descriptor
            .spatial_parity
    };
    assert_eq!(parity(BaryonMode::Real), Some(SpatialParity::Odd));
    assert_eq!(parity(BaryonMode::Imag), Some(SpatialParity::Even));
    assert_eq!(parity(BaryonMode::Complex), None);
    for &observable in GlueballObservable::ALL {
        let spec = glueball(observable);
        let (kind, other) = if observable == GlueballObservable::ForceNorm {
            (ElementKind::Site, ElementKind::Triplet)
        } else {
            (ElementKind::Triplet, ElementKind::Site)
        };
        let signature = spec.signature(kind, &context).unwrap();
        assert_eq!(signature.exchange, ExchangeParity::Even);
        assert_eq!(signature.components, 1);
        assert_eq!(signature.requires.dimension, None);
        assert_eq!(
            signature.requires.records.iter().collect::<Vec<_>>(),
            [&Record::Color]
        );
        assert!(!signature.descriptor.book_label.is_empty());
        assert_eq!(
            signature.descriptor.spatial_parity,
            Some(SpatialParity::Even)
        );
        assert!(matches!(
            spec.signature(other, &context),
            Err(GasError::Capability(_))
        ));
    }
}
#[test]
fn default_propagators_hold_exactly_the_exchange_odd_standard_channels_of_these_families() {
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
    let mut seen = 0;
    for spec in ChannelSpec::standard_set() {
        if !matches!(
            spec,
            ChannelSpec::Baryon { .. } | ChannelSpec::Glueball { .. }
        ) {
            continue;
        }
        for kind in spec.kinds(measurement.pairs) {
            let signature = spec.signature(kind, &context).unwrap();
            assert_eq!(
                measurement.propagators.enabled_for.contains(&spec.id()),
                signature.exchange == ExchangeParity::Odd,
                "{}",
                spec.id()
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 4);
}
#[test]
fn common_special_unitary_rotation_fixes_every_arm() {
    let matrix = special_unitary();
    for (entry, (re, im)) in [
        (0, (0.590855515708062, -0.210364679227999)),
        (4, (0.271622019378678, -0.0550605091931892)),
        (8, (0.316410176815617, 0.0317469113352943)),
    ] {
        assert!(close(matrix[entry].re, re) && close(matrix[entry].im, im));
    }
    let state = recorded();
    let turned = rotated(&state, &matrix);
    for spec in every_triplet_arm() {
        for &i in &VALID {
            let element = &sampled()[i];
            let (a, b) = (
                value(&spec, element, &state).unwrap(),
                value(&spec, element, &turned).unwrap(),
            );
            assert!(
                a.iter().zip(&b).all(|(a, b)| close(*a, *b)),
                "{}",
                spec.id()
            );
        }
    }
}
#[test]
fn unitary_rotation_multiplies_the_determinant_by_its_determinant_phase() {
    let state = recorded();
    let matrix = mul(&diagonal([0.9, 0., 0.]), &special_unitary(), 3);
    let turned = rotated(&state, &matrix);
    let element = triplet(0, 1, 2);
    let b = determinant_of(&element, &turned);
    assert!(close(b.re, -0.13369563410895) && close(b.im, -0.198280648673984));
    for &i in &VALID {
        let element = &sampled()[i];
        let expected = determinant_of(element, &state) * C::phase(0.9);
        assert!((determinant_of(element, &turned) - expected).abs() < 1e-12);
    }
    for spec in every_triplet_arm() {
        let invariant = !matches!(
            spec,
            ChannelSpec::Baryon {
                mode: BaryonMode::Real
                    | BaryonMode::Imag
                    | BaryonMode::Complex
                    | BaryonMode::ScoreOrdered,
                ..
            }
        );
        let (a, b) = (
            scalar(&spec, &element, &state),
            scalar(&spec, &element, &turned),
        );
        assert_eq!(close(a, b), invariant, "{}", spec.id());
    }
}
#[test]
fn per_site_rephasing_fixes_the_plaquette_and_shifts_the_determinant_by_the_summed_phase() {
    let state = recorded();
    let mut rephased = state.clone();
    for (w, alpha) in [(0, 0.3), (1, -1.7), (2, 2.2)] {
        for c in &mut rephased.color[w * 3..w * 3 + 3] {
            *c = *c * C::phase(alpha);
        }
    }
    let element = triplet(0, 1, 2);
    let b = determinant_of(&element, &rephased);
    assert!(close(b.re, -0.152822747429965) && close(b.im, -0.183942779380782));
    let link = |s: &FrameState| dot(&s.color[0..3], &s.color[3..6]);
    assert!((link(&rephased) - link(&state) * C::phase(-2.)).abs() < 1e-12);
    for spec in every_triplet_arm() {
        let invariant = !matches!(
            spec,
            ChannelSpec::Baryon {
                mode: BaryonMode::Real
                    | BaryonMode::Imag
                    | BaryonMode::Complex
                    | BaryonMode::ScoreOrdered,
                ..
            }
        );
        let (a, b) = (
            scalar(&spec, &element, &state),
            scalar(&spec, &element, &rephased),
        );
        assert_eq!(close(a, b), invariant, "{}", spec.id());
    }
}
#[test]
fn common_velocity_boost_rotates_the_determinant_and_fixes_the_plaquette() {
    let state = recorded();
    let boosted: Vec<f64> = VELOCITY
        .iter()
        .enumerate()
        .map(|(e, v)| v + [0.3, -0.5, 0.45][e % 3])
        .collect();
    let boosted = frame_state(&FORCE, &boosted);
    let b = determinant_of(&triplet(0, 1, 2), &boosted);
    assert!(close(b.re, -0.220028262079712) && close(b.im, -0.0936872568952239));
    for &i in &VALID {
        let element = &sampled()[i];
        let expected = determinant_of(element, &state) * C::phase(KAPPA * 0.25);
        assert!((determinant_of(element, &boosted) - expected).abs() < 1e-12);
        for &observable in &GlueballObservable::ALL[..4] {
            let spec = glueball(observable);
            assert!(close(
                scalar(&spec, element, &state),
                scalar(&spec, element, &boosted)
            ));
        }
    }
}
#[test]
fn spatial_parity_flips_the_real_part_of_the_determinant_exactly_and_nothing_else() {
    let state = recorded();
    let negate = |x: &[f64]| x.iter().map(|x| -x).collect::<Vec<_>>();
    let mirrored = frame_state(&negate(&FORCE), &negate(&VELOCITY));
    for &i in &VALID {
        let element = &sampled()[i];
        for spec in every_triplet_arm() {
            let (a, b) = (
                value(&spec, element, &state).unwrap(),
                value(&spec, element, &mirrored).unwrap(),
            );
            match spec {
                ChannelSpec::Baryon {
                    mode: BaryonMode::Real | BaryonMode::ScoreOrdered,
                    ..
                } => assert_eq!(b[0], -a[0]),
                ChannelSpec::Baryon {
                    mode: BaryonMode::Complex,
                    ..
                } => assert_eq!((b[0], b[1]), (-a[0], a[1])),
                _ => assert_eq!(a, b, "{}", spec.id()),
            }
        }
    }
}
#[test]
fn determinant_sums_to_zero_over_the_ordered_companion_pairs_of_an_anchor() {
    let state = recorded();
    let companions = [1, 2, 3, 5, 6, 7];
    let (mut sum, mut plaquette, mut modulus) = (C::ZERO, 0., 0.);
    for &j in &companions {
        for &k in companions.iter().filter(|&&k| k != j) {
            let element = triplet(0, j, k);
            sum = sum + determinant_of(&element, &state);
            plaquette += scalar(&glueball(GlueballObservable::RePlaquette), &element, &state);
            modulus += scalar(&baryon(BaryonMode::Abs), &element, &state);
        }
    }
    assert!(sum.abs() < 1e-12);
    assert!(close(plaquette / 30., 0.12418809639643));
    assert!(close(modulus / 30., 0.404515839338919));
}
#[test]
fn source_frozen_products_are_relabelling_even_and_equal_the_gram_determinant() {
    let source = recorded();
    let kicked: Vec<f64> = VELOCITY
        .iter()
        .zip(&FORCE)
        .map(|(v, f)| v + 0.1 * f)
        .collect();
    let sink = frame_state(&FORCE, &kicked);
    let elements = sampled();
    let b = determinant_of(&elements[0], &sink);
    assert!(close(b.re, -0.150370197825585) && close(b.im, -0.055579978725511));
    for (i, real, imaginary, contracted) in [
        (
            0,
            0.0358520304522107,
            0.00102966578382769,
            0.0368816962360384,
        ),
        (
            1,
            0.0358520304522107,
            0.00102966578382769,
            0.0368816962360384,
        ),
        (
            2,
            0.0358520304522107,
            0.00102966578382769,
            0.0368816962360384,
        ),
        (6, 0.181148102235978, 0.280870908278755, 0.462019010514733),
    ] {
        let (s, t) = (
            determinant_of(&elements[i], &source),
            determinant_of(&elements[i], &sink),
        );
        assert!(close(s.re * t.re, real) && close(s.im * t.im, imaginary));
        assert!(close(s.re * t.re + s.im * t.im, contracted));
        let walkers = elements[i].walkers.map(|w| w as usize);
        let gram: Vec<C> = (0..9)
            .map(|e| {
                let (a, b) = (walkers[e / 3] * 3, walkers[e % 3] * 3);
                dot(&source.color[a..a + 3], &sink.color[b..b + 3])
            })
            .collect();
        assert!((determinant(&gram, 3) - s.conj() * t).abs() < 1e-12);
    }
}
#[test]
fn two_mutual_pairings_share_no_triplet_so_the_determinant_frame_sum_does_not_cancel() {
    let state = recorded();
    let walkers = [0, 1, 2, 3, 5, 6];
    let pairings = matchings(&walkers);
    assert_eq!(pairings.len(), 15);
    let (mut elements, mut frames) = (0, 0);
    for distance in &pairings {
        for cloning in &pairings {
            let sampled: Vec<Element> = walkers
                .iter()
                .filter(|&&i| distance[i] != cloning[i])
                .map(|&i| triplet(i, distance[i], cloning[i]))
                .collect();
            let unordered: BTreeSet<[u32; 3]> = sampled
                .iter()
                .map(|e| {
                    let mut w = e.walkers;
                    w.sort_unstable();
                    w
                })
                .collect();
            assert_eq!(unordered.len(), sampled.len());
            if sampled.is_empty() {
                assert_eq!(distance, cloning);
                continue;
            }
            let sum = sampled
                .iter()
                .fold(C::ZERO, |s, e| s + determinant_of(e, &state));
            assert!(sum.abs() > 0.05, "{distance:?} {cloning:?}");
            elements += sampled.len();
            frames += 1;
        }
    }
    assert_eq!((elements, frames), (1080, 210));
    let (distance, cloning) = ([1, 0, 3, 2, 0, 6, 5, 0], [2, 5, 0, 6, 0, 1, 3, 0]);
    let sum = walkers.iter().fold(C::ZERO, |s, &i| {
        s + determinant_of(&triplet(i, distance[i], cloning[i]), &state)
    });
    assert!(close(sum.re, -1.37315054429134) && close(sum.im, -0.802600196918104));
    let mirrored: f64 = walkers
        .iter()
        .map(|&i| {
            dot(
                &state.color[i * 3..i * 3 + 3],
                &state.color[distance[i] * 3..][..3],
            )
            .im
        })
        .sum();
    assert!(mirrored.abs() < 1e-14);
}
#[test]
fn descriptors_state_the_algebra_and_never_a_particle_name() {
    let (gas, measurement, capabilities) = (
        periodic(),
        MeasurementConfig::default(),
        Capabilities::nominal(),
    );
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let mut specs = every_triplet_arm();
    specs.push(glueball(GlueballObservable::ForceNorm));
    specs.push(projected(GlueballObservable::Sin2, 2, MomentumPhase::Sin));
    specs.push(projected(
        GlueballObservable::ForceNorm,
        1,
        MomentumPhase::Cos,
    ));
    for spec in specs {
        let kind = spec.kinds(measurement.pairs)[0];
        let descriptor = spec.signature(kind, &context).unwrap().descriptor;
        let text = format!("{} {}", descriptor.definition, descriptor.note).to_lowercase();
        for name in ["nucleon", "proton", "neutron", "glueball", "mass", "0++"] {
            assert!(!text.contains(name), "{} names {name}", spec.id());
        }
        assert!(!descriptor.definition.is_empty() && !descriptor.note.is_empty());
        let book = [
            "",
            "def-sm-direct-color-contractions",
            "thm-sm-direct-color-invariants",
            "prop-sm-baryon-exterior-correlator",
            "sec-qft-calibration-channel-derivations",
        ];
        assert!(book.contains(&descriptor.book_label.as_str()));
    }
    let weighted = flux(2.5).signature(ElementKind::Triplet, &context).unwrap();
    assert!(weighted.descriptor.definition.contains("2.5"));
    let sine = projected(GlueballObservable::Sin2, 2, MomentumPhase::Sin)
        .signature(ElementKind::Triplet, &context)
        .unwrap();
    assert!(
        sine.descriptor
            .definition
            .contains(r"\sin(2\pi\, 2\, x_i^{(0)} / L)")
    );
}
#[test]
fn walkers_outside_the_state_and_nonfinite_inputs_mask_the_element_explicitly() {
    let (gas, state) = (periodic(), recorded());
    let cosine = projected(GlueballObservable::RePlaquette, 1, MomentumPhase::Cos);
    let force = glueball(GlueballObservable::ForceNorm);
    let projected_force = projected(GlueballObservable::ForceNorm, 1, MomentumPhase::Sin);
    for spec in every_triplet_arm().iter().chain([&cosine]) {
        for element in [triplet(0, 1, 99), triplet(99, 1, 2), triplet(0, 99, 2)] {
            assert_eq!(
                value_in(&gas, spec, &element, &state),
                None,
                "{}",
                spec.id()
            );
        }
    }
    for spec in [&force, &projected_force] {
        assert_eq!(value_in(&gas, spec, &site(99), &state), None);
    }
    let mut lost = state.clone();
    lost.x[0] = f64::NAN;
    lost.x[6 * 3] = f64::INFINITY;
    for i in [0, 6] {
        let element = &sampled()[i];
        assert_eq!(value_in(&gas, &cosine, element, &lost), None);
        assert_eq!(value_in(&gas, &projected_force, &site(i), &lost), None);
        assert!(
            value_in(
                &gas,
                &glueball(GlueballObservable::RePlaquette),
                element,
                &lost
            )
            .is_some()
        );
        assert!(value_in(&gas, &force, &site(i), &lost).is_some());
    }
    let mut huge = state.clone();
    huge.force.as_mut().unwrap()[3] = 1e200;
    assert_eq!(value(&force, &site(1), &huge), None);
    assert!(value(&force, &site(0), &huge).is_some());
    let flat = GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "x".into(),
            domain: BoxDomain {
                lower: vec![1.; 3],
                upper: vec![1.; 3],
            },
        },
        ..GasConfig::default()
    };
    assert_eq!(value_in(&flat, &cosine, &sampled()[0], &state), None);
    let short = GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "x".into(),
            domain: BoxDomain {
                lower: vec![],
                upper: vec![],
            },
        },
        ..GasConfig::default()
    };
    assert_eq!(value_in(&short, &cosine, &sampled()[0], &state), None);
}
#[test]
fn score_order_reports_the_permutation_sign_apart_from_the_determinant() {
    // The columns stay in the sampled order and the sign of the permutation
    // that sorts them by score is the second column, so that the measurement
    // can freeze it with the element instead of re-deriving it at a sink.
    let spec = baryon(BaryonMode::ScoreOrdered);
    let (state, element) = (recorded(), triplet(0, 1, 2));
    let real = scalar(&baryon(BaryonMode::Real), &element, &state);
    assert_eq!(raw(&spec, &element, &state), Some(vec![real, -1.]));
    let mut crossed = state.clone();
    crossed.score.as_mut().unwrap()[1] = 0.7;
    assert_eq!(raw(&spec, &element, &crossed), Some(vec![real, 1.]));
    // A tie leaves the element without an orientation: the factor is 0 and the
    // measurement masks it at a source time.
    let mut tied = state.clone();
    tied.score.as_mut().unwrap()[1] = 0.5;
    assert_eq!(raw(&spec, &element, &tied), Some(vec![real, 0.]));
    assert_eq!(value(&spec, &element, &tied), None);
    tied.score.as_mut().unwrap()[1] = 1.1;
    assert_eq!(raw(&spec, &element, &tied), Some(vec![real, 0.]));
    // A walker without a score is missing data, not a tie: the factor is not
    // a number and the measurement masks the element wherever it reads one.
    let mut unscored = state.clone();
    unscored.score_valid[2] = false;
    let row = raw(&spec, &element, &unscored).unwrap();
    assert_eq!(row[0], real);
    assert!(row[1].is_nan());
    assert_eq!(value(&spec, &element, &unscored), None);
    unscored.score_valid[2] = true;
    unscored.score = None;
    assert!(raw(&spec, &element, &unscored).unwrap()[1].is_nan());
    assert!(value(&baryon(BaryonMode::Real), &element, &unscored).is_some());
}
#[test]
fn flux_weight_with_a_zero_exponent_is_the_modulus_and_an_overflowing_weight_is_masked() {
    let state = recorded();
    for &i in &VALID {
        let element = &sampled()[i];
        assert_eq!(
            scalar(&flux(0.), element, &state),
            scalar(&baryon(BaryonMode::Abs), element, &state)
        );
    }
    assert_eq!(value(&flux(1e6), &sampled()[0], &state), None);
    assert_eq!(flux(1.).id(), "baryon/flux_weighted");
    assert_eq!(flux(2.).id(), "baryon/flux_weighted/a2");
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
    for exponent in [-1., f64::NAN, f64::INFINITY] {
        assert!(matches!(
            flux(exponent).validate(),
            Err(GasError::Configuration(_))
        ));
        assert!(matches!(
            flux(exponent).signature(ElementKind::Triplet, &context),
            Err(GasError::Configuration(_))
        ));
    }
}
#[test]
fn haar_colours_obey_the_gram_identity_the_bounds_and_the_incoherent_moments() {
    let triples = 40_000;
    let mut colors = Vec::with_capacity(triples * 9);
    for w in 0..triples * 3 {
        let mut rng = RandomStream::new(7, 0, Stream::Initialize, w as u64, 0);
        let raw = [(); 3].map(|_| C::new(rng.gaussian(), rng.gaussian()));
        let norm = raw.iter().map(|z| z.abs2()).sum::<f64>().sqrt();
        colors.extend(raw.map(|z| z / norm));
    }
    let state = color_state(3, &colors);
    let elements: Vec<Element> = (0..triples)
        .map(|t| triplet(3 * t, 3 * t + 1, 3 * t + 2))
        .collect();
    let of = |spec: ChannelSpec| {
        let (values, valid) = values(&spec, &elements, &state);
        assert!(valid.iter().all(|&v| v), "{}", spec.id());
        values
    };
    let determinants = of(baryon(BaryonMode::Complex));
    let (modulus, squared, weighted) = (
        of(baryon(BaryonMode::Abs)),
        of(baryon(BaryonMode::Abs2)),
        of(flux(0.5)),
    );
    let (re, one_minus_re, one_minus_cos, sin2) = (
        of(glueball(GlueballObservable::RePlaquette)),
        of(glueball(GlueballObservable::OneMinusRe)),
        of(glueball(GlueballObservable::OneMinusCos)),
        of(glueball(GlueballObservable::Sin2)),
    );
    let (mut real, mut abs, mut abs2, mut plaquette) = (0., 0., 0., 0.);
    for t in 0..triples {
        let c = |w: usize| &colors[(3 * t + w) * 3..(3 * t + w) * 3 + 3];
        let overlaps: f64 = [(0, 1), (1, 2), (2, 0)]
            .iter()
            .map(|&(a, b)| dot(c(a), c(b)).abs2())
            .sum();
        let b = C::new(determinants[2 * t], determinants[2 * t + 1]);
        assert!((squared[t] - (1. - overlaps + 2. * re[t])).abs() < 1e-12);
        assert!(close(b.abs2(), squared[t]) && close(b.abs(), modulus[t]));
        assert!(squared[t] <= 1. + 1e-12 && (-0.125 - 1e-12..=1. + 1e-12).contains(&re[t]));
        assert!(close(one_minus_re[t] + re[t], 1.) && one_minus_re[t] <= 1.125 + 1e-12);
        assert!((0. ..=2.).contains(&one_minus_cos[t]) && (0. ..=1.).contains(&sin2[t]));
        assert!(close((1. - one_minus_cos[t]).powi(2) + sin2[t], 1.));
        assert!(weighted[t] >= 0. && weighted[t] < 1f64.exp());
        real += b.re;
        abs += modulus[t];
        abs2 += squared[t];
        plaquette += re[t];
    }
    // Six standard errors over 40000 independent triples. Exact moments of
    // |b|² = X₂ X₃ with X₂ ~ Beta(2,1), X₃ ~ Beta(1,2): E|b| = 32/75,
    // E|b|² = 2/9, E|b|⁴ = 1/12, so σ(|b|) = 0.2005, σ(|b|²) = 0.1842 and
    // σ(Re b) = 1/3 by phase symmetry; σ(Re Π) ≤ (E|q|²)^½ = 0.5774.
    let n = triples as f64;
    assert!((real / n).abs() < 6. * (1. / 3.) / n.sqrt());
    assert!((abs / n - 32. / 75.).abs() < 6. * 0.2005 / n.sqrt());
    assert!((abs2 / n - 2. / 9.).abs() < 6. * 0.1842 / n.sqrt());
    assert!((plaquette / n - 1. / 9.).abs() < 6. * 0.5774 / n.sqrt());
}
#[test]
fn force_norm_is_the_squared_recorded_force_and_keeps_a_zero_force() {
    let spec = glueball(GlueballObservable::ForceNorm);
    let state = recorded();
    let expected = [14., 5.25, 0.62, 6., 0., 3.5, 3.36, 2.34];
    let mut sum = 0.;
    for (i, e) in expected.into_iter().enumerate() {
        let norm = scalar(&spec, &site(i), &state);
        assert!(close(norm, e));
        sum += norm;
    }
    assert!(!state.color_valid[4] && close(sum / 8., 4.38375));
    let mut uncovered = state.clone();
    uncovered.force_valid[3] = false;
    uncovered.eligible[5] = false;
    assert_eq!(value(&spec, &site(3), &uncovered), None);
    assert_eq!(value(&spec, &site(5), &uncovered), None);
    assert!(value(&spec, &site(4), &uncovered).is_some());
    uncovered.force = None;
    assert_eq!(value(&spec, &site(0), &uncovered), None);
    let as_triplet = Element {
        kind: ElementKind::Triplet,
        ..site(0)
    };
    assert_eq!(value(&spec, &as_triplet, &state), None);
}
#[test]
fn momentum_projection_weights_the_anchor_coordinate_on_the_periodic_box() {
    let (gas, state, elements) = (periodic(), recorded(), sampled());
    let mean = |spec: &ChannelSpec, elements: &[Element], count: f64| {
        elements
            .iter()
            .filter_map(|e| value_in(&gas, spec, e, &state))
            .map(|v| v[0])
            .sum::<f64>()
            / count
    };
    let plaquette = GlueballObservable::RePlaquette;
    // Mode 0 is no arm of its own: the connected projection vanishes on it in
    // both phases, and `the_connected_projection_refuses_the_vanishing_cosine_
    // of_mode_zero` pins the refusal.
    for (mode, phase, expected) in [
        (1, MomentumPhase::Cos, -0.00643181919974239),
        (1, MomentumPhase::Sin, 0.00055055942711571),
        (2, MomentumPhase::Cos, -0.00724836173282051),
        (2, MomentumPhase::Sin, -0.00456001945228907),
    ] {
        let spec = projected(plaquette, mode, phase);
        assert!(close(mean(&spec, &elements, 4.), expected), "{}", spec.id());
    }
    let sites: Vec<Element> = (0..8).map(site).collect();
    for (mode, phase, expected) in [
        (1, MomentumPhase::Cos, -0.205481470604397),
        (1, MomentumPhase::Sin, -0.602336638928604),
        (2, MomentumPhase::Cos, -1.24899308611108),
        (2, MomentumPhase::Sin, 2.1957573777268),
    ] {
        let spec = projected(GlueballObservable::ForceNorm, mode, phase);
        assert!(close(mean(&spec, &sites, 8.), expected), "{}", spec.id());
    }
    let mut wrapped = state.clone();
    for i in 0..8 {
        wrapped.x[i * 3] += 5.;
    }
    for &i in &VALID {
        // The observable column of a projected row is the unprojected arm
        // itself, whatever the mode: the weight travels beside it.
        assert_eq!(
            raw_in(
                &gas,
                &projected(plaquette, 2, MomentumPhase::Cos),
                &elements[i],
                &state
            )
            .map(|row| row[0]),
            value(&glueball(plaquette), &elements[i], &state).map(|row| row[0])
        );
        for phase in [MomentumPhase::Cos, MomentumPhase::Sin] {
            let spec = projected(plaquette, 2, phase);
            let (a, b) = (
                value_in(&gas, &spec, &elements[i], &state).unwrap()[0],
                value_in(&gas, &spec, &elements[i], &wrapped).unwrap()[0],
            );
            assert!(close(a, b));
        }
    }
    let spec = projected(plaquette, 1, MomentumPhase::Cos);
    assert_eq!(value(&spec, &elements[0], &state), None);
}
#[test]
fn momentum_projection_requires_a_periodic_box_and_rejects_the_vanishing_sine_mode() {
    let plaquette = GlueballObservable::RePlaquette;
    let cosine = projected(plaquette, 1, MomentumPhase::Cos);
    let sine = projected(GlueballObservable::ForceNorm, 1, MomentumPhase::Sin);
    let outside = ChannelSpec::Glueball {
        observable: plaquette,
        momentum: Some(Momentum {
            axis: 3,
            mode: 1,
            phase: MomentumPhase::Cos,
        }),
    };
    let vanishing = projected(plaquette, 0, MomentumPhase::Sin);
    assert!(matches!(
        vanishing.validate(),
        Err(GasError::Configuration(_))
    ));
    let gas = GasConfig::default();
    let measurement = MeasurementConfig {
        channels: vec![
            cosine.clone(),
            sine.clone(),
            outside,
            glueball(plaquette),
            baryon(BaryonMode::Complex),
        ],
        ..MeasurementConfig::default()
    };
    let reason = "momentum projection needs a periodic box";
    let capabilities = Capabilities {
        missing: BTreeMap::from([(Record::PeriodicBox, reason.to_string())]),
        ..Capabilities::nominal()
    };
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    assert!(matches!(
        vanishing.signature(ElementKind::Triplet, &context),
        Err(GasError::Configuration(_))
    ));
    let listed = channels(&context, &[]).unwrap();
    let ids: Vec<&str> = listed.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "glueball/re_plaquette/p0_cos1/triplet",
            "glueball/force_norm/p0_sin1/site",
            "glueball/re_plaquette/p3_cos1/triplet",
            "glueball/re_plaquette/triplet",
            "baryon/complex/triplet",
        ]
    );
    assert_eq!(listed[0].availability.reason(), Some(reason));
    assert_eq!(listed[1].availability.reason(), Some(reason));
    assert_eq!(
        listed[2].availability.reason(),
        Some("momentum axis 3 is outside the 3 position coordinates")
    );
    assert!(listed[3].availability.is_available() && listed[4].availability.is_available());
    for (channel, parity) in [
        (&listed[0], SpatialParity::Even),
        (&listed[1], SpatialParity::Odd),
    ] {
        let signature = channel.signature.as_ref().unwrap();
        assert!(signature.requires.records.contains(&Record::PeriodicBox));
        assert!(signature.descriptor.book_label.is_empty());
        assert_eq!(signature.descriptor.spatial_parity, Some(parity));
    }
    // The determinant is defined in three colour components and the plaquette
    // in every dimension, on both sides of three: a run of four coordinates
    // refuses the baryon and keeps the plaquette, exactly as a planar one does.
    for dimension in [2, 4] {
        let other = Capabilities {
            dimension,
            euclidean_axis: Some(dimension - 1),
            ..Capabilities::nominal()
        };
        let context = OperatorContext {
            capabilities: &other,
            ..context
        };
        let listed = channels(&context, &[]).unwrap();
        assert!(listed[3].availability.is_available(), "{dimension}");
        assert_eq!(
            listed[4].availability.reason(),
            Some(format!("defined in 3 position dimensions; the run has {dimension}").as_str()),
            "{dimension}"
        );
    }
}
#[test]
fn the_connected_projection_refuses_the_vanishing_cosine_of_mode_zero() {
    // The projection is taken of the element minus the frame mean, so the
    // cosine of mode 0 — which weights every element by 1 — sums to zero by
    // construction, exactly as the sine of mode 0 does. A channel of exact
    // zeros carried at full weight is the failure the score arms are refused
    // for, so the arm is a `Capability` refusal and the reader is sent to the
    // unprojected channel, which is the zero-momentum observable.
    let plaquette = GlueballObservable::RePlaquette;
    let gas = GasConfig::default();
    let measurement = MeasurementConfig {
        channels: vec![
            projected(plaquette, 0, MomentumPhase::Cos),
            projected(GlueballObservable::ForceNorm, 0, MomentumPhase::Cos),
            projected(plaquette, 1, MomentumPhase::Cos),
        ],
        ..MeasurementConfig::default()
    };
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let refusal = "the connected projection on momentum mode 0 vanishes identically; measure \
                   the unprojected arm for the zero mode";
    for (spec, kind) in [
        (&measurement.channels[0], ElementKind::Triplet),
        (&measurement.channels[1], ElementKind::Site),
    ] {
        // The configuration is well formed: only the operator knows that the
        // combination has no value.
        spec.validate().unwrap();
        match spec.signature(kind, &context) {
            Err(GasError::Capability(reason)) => assert_eq!(reason, refusal),
            other => panic!("{}: {other:?}", spec.id()),
        }
    }
    // Every other mode of the same axis is untouched.
    measurement.channels[2]
        .signature(ElementKind::Triplet, &context)
        .unwrap();
    // The refusal reaches the channel list, which keeps the arm with its
    // reason instead of publishing a series of zeros.
    let listed = channels(&context, &[]).unwrap();
    assert_eq!(listed[0].availability.reason(), Some(refusal));
    assert_eq!(listed[1].availability.reason(), Some(refusal));
    assert!(listed[2].availability.is_available());
}

#[test]
fn every_slot_of_a_triplet_is_checked_for_coincidence_eligibility_and_colour_validity() {
    let state = recorded();
    for spec in every_triplet_arm() {
        assert!(value(&spec, &triplet(0, 1, 2), &state).is_some());
        for element in [triplet(0, 1, 0), triplet(0, 0, 2), triplet(0, 1, 1)] {
            assert_eq!(value(&spec, &element, &state), None, "{}", spec.id());
        }
        for slot in 0..3 {
            let (mut dead, mut colourless) = (state.clone(), state.clone());
            dead.eligible[slot] = false;
            colourless.color_valid[slot] = false;
            for masked in [&dead, &colourless] {
                assert_eq!(
                    value(&spec, &triplet(0, 1, 2), masked),
                    None,
                    "{} slot {slot}",
                    spec.id()
                );
                let untouched = triplet(6, 3, 5);
                assert_eq!(
                    value(&spec, &untouched, masked),
                    value(&spec, &untouched, &state)
                );
            }
        }
    }
}
#[test]
fn score_order_is_the_real_part_of_the_determinant_with_columns_sorted_by_score() {
    let spec = baryon(BaryonMode::ScoreOrdered);
    let state = recorded();
    for base in [[0, 1, 2], [6, 3, 5]] {
        let element = triplet(base[0], base[1], base[2]);
        let real = scalar(&baryon(BaryonMode::Real), &element, &state);
        for (ranks, sign) in PERMUTATIONS {
            let mut ranked = state.clone();
            let mut sorted = [0; 3];
            for slot in 0..3 {
                ranked.score.as_mut().unwrap()[base[slot]] = ranks[slot] as f64 - 1.;
                sorted[ranks[slot]] = base[slot];
            }
            let c = |w: usize| &state.color[w * 3..w * 3 + 3];
            let ordered = det3(c(sorted[0]), c(sorted[1]), c(sorted[2])).re;
            let measured = scalar(&spec, &element, &ranked);
            assert!(close(measured, ordered), "{base:?} {ranks:?}");
            assert_eq!(measured, sign * real, "{base:?} {ranks:?}");
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let mut tied = state.clone();
            let scores = tied.score.as_mut().unwrap();
            scores[base[a]] = scores[base[b]];
            assert_eq!(value(&spec, &element, &tied), None, "{base:?} {a} {b}");
        }
        for slot in 0..3 {
            let mut unscored = state.clone();
            unscored.score_valid[base[slot]] = false;
            assert_eq!(value(&spec, &element, &unscored), None, "{base:?} {slot}");
            assert!(value(&baryon(BaryonMode::Real), &element, &unscored).is_some());
        }
    }
}
#[test]
fn plaquette_phase_floor_is_tested_on_every_link_and_never_on_the_product() {
    let element = triplet(0, 1, 2);
    let (one, i) = (C::ONE, C::new(0., 1.));
    let phase_arms = [
        glueball(GlueballObservable::OneMinusCos),
        glueball(GlueballObservable::Sin2),
        flux(1.),
    ];
    // q_ab = ε, q_bc = (ε + i s) / √2, q_ca = 1 / √2 with s² = 1 - ε²: the
    // plaquette is ε (ε + i s) / 2, of phase ε + i s.
    let one_small_link = |epsilon: f64| {
        let s = (1. - epsilon * epsilon).sqrt();
        let colors = [
            [one, C::ZERO, C::ZERO],
            [epsilon.into(), s.into(), C::ZERO],
            scaled([one, i, C::ZERO], 2f64.sqrt()),
        ]
        .concat();
        color_state(3, &colors)
    };
    for epsilon in [0., 1e-13, 1e-12] {
        let state = one_small_link(epsilon);
        for spec in &phase_arms {
            assert_eq!(
                value(spec, &element, &state),
                None,
                "{} {epsilon}",
                spec.id()
            );
        }
        let re = scalar(&glueball(GlueballObservable::RePlaquette), &element, &state);
        assert!((re - epsilon * epsilon / 2.).abs() < 1e-30);
        assert!(value(&glueball(GlueballObservable::OneMinusRe), &element, &state).is_some());
        assert_eq!(scalar(&baryon(BaryonMode::Abs2), &element, &state), 0.);
    }
    for epsilon in [2e-12, 1e-9, 0.3] {
        let state = one_small_link(epsilon);
        let cosine = scalar(&phase_arms[0], &element, &state);
        let sine = scalar(&phase_arms[1], &element, &state);
        assert!((cosine - (1. - epsilon)).abs() < 1e-14, "{epsilon}");
        assert!((sine - (1. - epsilon * epsilon)).abs() < 1e-14, "{epsilon}");
        assert_eq!(scalar(&phase_arms[2], &element, &state), 0.);
    }
    // Two links of modulus 1e-7 and one of modulus one: |Π| = 1e-14 is below
    // the floor, every link is above it, and the phase is i to 1e-14.
    let epsilon = 1e-7f64;
    let s = (1. - epsilon * epsilon).sqrt();
    let colors = [
        [one, C::ZERO, C::ZERO],
        [epsilon.into(), s.into(), C::ZERO],
        [epsilon.into(), i * s, C::ZERO],
    ]
    .concat();
    let state = color_state(3, &colors);
    let re = scalar(&glueball(GlueballObservable::RePlaquette), &element, &state);
    assert!(re > 0. && re < 1e-27);
    assert!(close(scalar(&phase_arms[0], &element, &state), 1.));
    assert!(close(scalar(&phase_arms[1], &element, &state), 1.));
    assert_eq!(scalar(&phase_arms[2], &element, &state), 0.);
}
#[test]
fn momentum_weight_reads_its_own_axis_of_the_anchor_and_the_box_length_of_that_axis() {
    let gas = GasConfig {
        boundary: BoundaryPolicy::PeriodicBox {
            field: "x".into(),
            domain: BoxDomain {
                lower: vec![-2., -1., 0., -2.],
                upper: vec![3., 3., 6., 3.],
            },
        },
        ..GasConfig::default()
    };
    let mut state = recorded();
    state.x[1] = 2. / 3.;
    state.x[2] = 1.5;
    let along = |observable, axis, mode, phase| ChannelSpec::Glueball {
        observable,
        momentum: Some(Momentum { axis, mode, phase }),
    };
    // ‖F_0‖² = 14; the angles are 2π(-1.5)/5 = -108°, 2π(2/3)/4 = 60°,
    // 2π(1.5)/6 = 90° and its double.
    let force = GlueballObservable::ForceNorm;
    for (axis, mode, phase, expected) in [
        (0, 1, MomentumPhase::Cos, -14. * (5f64.sqrt() - 1.) / 4.),
        (1, 1, MomentumPhase::Cos, 7.),
        (1, 1, MomentumPhase::Sin, 7. * 3f64.sqrt()),
        (2, 1, MomentumPhase::Cos, 0.),
        (2, 1, MomentumPhase::Sin, 14.),
        (2, 2, MomentumPhase::Cos, -14.),
    ] {
        let spec = along(force, axis, mode, phase);
        let measured = value_in(&gas, &spec, &site(0), &state).unwrap()[0];
        assert!(close(measured, expected), "{}", spec.id());
    }
    let spec = along(GlueballObservable::RePlaquette, 1, 1, MomentumPhase::Cos);
    let measured = value_in(&gas, &spec, &sampled()[0], &state).unwrap()[0];
    assert!(close(measured, -0.039889462413726 / 2.));
    // A fourth box axis is no coordinate of a three-dimensional state.
    for phase in [MomentumPhase::Cos, MomentumPhase::Sin] {
        let spec = along(force, 3, 1, phase);
        assert_eq!(value_in(&gas, &spec, &site(0), &state), None);
    }
}
#[test]
fn declared_spatial_parity_is_the_one_the_mirrored_frame_obeys() {
    let (gas, measurement, capabilities) = (
        periodic(),
        MeasurementConfig::default(),
        Capabilities::nominal(),
    );
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let state = recorded();
    let negate = |x: &[f64]| x.iter().map(|x| -x).collect::<Vec<_>>();
    let mut mirrored = frame_state(&negate(&FORCE), &negate(&VELOCITY));
    mirrored.x = negate(&state.x);
    let mut specs = every_triplet_arm();
    for observable in [GlueballObservable::Sin2, GlueballObservable::ForceNorm] {
        specs.push(glueball(observable));
        specs.push(projected(observable, 1, MomentumPhase::Cos));
        specs.push(projected(observable, 2, MomentumPhase::Sin));
    }
    let mut odd = vec![];
    for spec in specs {
        let kind = spec.kinds(measurement.pairs)[0];
        let declared = spec
            .signature(kind, &context)
            .unwrap()
            .descriptor
            .spatial_parity;
        for &i in &VALID {
            let element = if kind == ElementKind::Site {
                site(i)
            } else {
                sampled()[i]
            };
            let (a, b) = (
                value_in(&gas, &spec, &element, &state).unwrap(),
                value_in(&gas, &spec, &element, &mirrored).unwrap(),
            );
            match declared {
                Some(SpatialParity::Even) => assert!(close(a[0], b[0]), "{}", spec.id()),
                Some(SpatialParity::Odd) => {
                    assert!(close(a[0], -b[0]) && a[0].abs() > 1e-3, "{}", spec.id());
                }
                None => assert!(a.len() == 2 && close(a[0], -b[0]) && close(a[1], b[1])),
            }
        }
        if declared == Some(SpatialParity::Odd) {
            odd.push(spec.id());
        }
    }
    assert_eq!(
        odd,
        [
            "baryon/real",
            "baryon/score_ordered",
            "glueball/sin2/p0_sin2",
            "glueball/force_norm/p0_sin2"
        ]
    );
}

#[test]
fn a_projected_arm_reports_the_observable_and_its_mode_weight_apart() {
    // The momentum weight is the second column, so that the measurement can
    // subtract the frame mean of the observable before projecting it.
    let (gas, state, elements) = (periodic(), recorded(), sampled());
    let length = 5.;
    for observable in [
        GlueballObservable::RePlaquette,
        GlueballObservable::OneMinusRe,
    ] {
        for (mode, phase) in [
            (1, MomentumPhase::Cos),
            (1, MomentumPhase::Sin),
            (2, MomentumPhase::Cos),
        ] {
            let spec = projected(observable, mode, phase);
            for &i in &VALID {
                let row = raw_in(&gas, &spec, &elements[i], &state).unwrap();
                assert_eq!(row.len(), 2, "{}", spec.id());
                let bare = value(&glueball(observable), &elements[i], &state).unwrap()[0];
                let angle = std::f64::consts::TAU * f64::from(mode) * state.x[i * 3] / length;
                let weight = match phase {
                    MomentumPhase::Cos => angle.cos(),
                    MomentumPhase::Sin => angle.sin(),
                };
                assert!(close(row[0], bare), "{} {i}", spec.id());
                assert!(close(row[1], weight), "{} {i}", spec.id());
            }
        }
    }
    // An additive shift of the observable moves only the first column, so the
    // element sum of the connected mode is the same for the two arms.
    let connected = |observable| {
        let spec = projected(observable, 1, MomentumPhase::Cos);
        let rows: Vec<Vec<f64>> = VALID
            .iter()
            .map(|&i| raw_in(&gas, &spec, &elements[i], &state).unwrap())
            .collect();
        let bare = rows.iter().map(|r| r[0]).sum::<f64>() / rows.len() as f64;
        rows.iter().map(|r| (r[0] - bare) * r[1]).sum::<f64>() / rows.len() as f64
    };
    let re = connected(GlueballObservable::RePlaquette);
    let one_minus = connected(GlueballObservable::OneMinusRe);
    assert!(close(re, -one_minus) && re.abs() > 1e-9, "{re} {one_minus}");
}
