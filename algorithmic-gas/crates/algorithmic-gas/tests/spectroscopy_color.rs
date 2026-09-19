//! Colour map oracles, alignment masks and bit parity with the recorded-colour experiment.
use algorithmic_gas::{
    ExecutionContext, GasBuilder, GasConfig, GasError, InputBatch, ObservationBatch, Population,
    Precision, Provenance, RewardBatch, RunArchive, TensorBatch,
    domain::{GradientProvider, OperatorFuture, RewardSource},
    kinetic::{KineticKind, KineticOperator, QftExecutionConfig, ViscousForceConfig},
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry},
    physics::{
        partvi::{ExperimentRequest, analyze_archive},
        qft::math::{C, dot},
        spectroscopy::{
            config::{
                ColorAlignment, ColorSource, KickStage, LengthScale, PhaseScale, StepAlignment,
            },
            contract::{Availability, FieldSource, Frame, FrameState, Kick, StageKick},
            fields::color::{color_vector, det3, kappa, phase_length, phase_wrapping},
        },
    },
    tracking::{RecordedStep, RecordingConfig},
};
use futures_lite::future::block_on;
use serde_json::json;
use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

const KAPPA: f64 = 0.7;
const DELTA: f64 = 1e-12;
const FORCE: [f64; 9] = [0.3, -1.2, 0.5, -0.7, 0.2, 1.1, 1.5, 0.6, -0.4];
const VELOCITY: [f64; 9] = [0.9, -0.4, 2.1, -1.3, 0.8, 0.25, 0.2, 5., -2.6];
const BARYON: C = C {
    re: -0.696727457892956,
    im: 0.0428697985366938,
};

fn close(z: C, re: f64, im: f64) {
    assert!(
        (z.re - re).abs() < 1e-12 && (z.im - im).abs() < 1e-12,
        "{z:?} != {re} + {im}i"
    );
}
fn near(a: C, b: C, t: f64) {
    assert!((a - b).abs() < t, "{a:?} != {b:?}, tolerance {t}");
}
fn bits(a: &[C]) -> Vec<[u64; 2]> {
    a.iter().map(|z| [z.re.to_bits(), z.im.to_bits()]).collect()
}
/// Colours of `[n, d]` rows through the single-walker map; every row valid.
fn encode(force: &[f64], velocity: &[f64], d: usize, k: f64) -> Vec<C> {
    let mut out = vec![C::ZERO; force.len()];
    for ((f, v), c) in force
        .chunks_exact(d)
        .zip(velocity.chunks_exact(d))
        .zip(out.chunks_exact_mut(d))
    {
        assert!(color_vector(f, v, k, DELTA, c));
    }
    out
}
fn triangle(c: &[C]) -> C {
    dot(&c[..3], &c[3..6]) * dot(&c[3..6], &c[6..]) * dot(&c[6..], &c[..3])
}
fn baryon(c: &[C]) -> C {
    det3(&c[..3], &c[3..6], &c[6..])
}
/// Deterministic values in `[-0.5, 0.5)`.
fn unit(k: u64) -> f64 {
    let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn kick(force: &[f64], velocity: &[f64], n: usize) -> StageKick {
    StageKick {
        force: force.to_vec(),
        velocity: velocity.to_vec(),
        available: vec![true; n],
        generation: vec![0; n],
    }
}
fn frame(step: u64, n: usize, d: usize) -> Frame {
    Frame {
        step,
        n,
        d,
        x: vec![0.; n * d],
        eligible: vec![true; n],
        generation: vec![0; n],
        cloned: vec![false; n],
        revived: vec![false; n],
        ..Frame::default()
    }
}
fn viscous(alignment: ColorAlignment) -> ColorSource {
    ColorSource::ViscousForce {
        alignment,
        threshold: DELTA,
    }
}
fn recorded(alignment: StepAlignment) -> ColorSource {
    ColorSource::RecordedField {
        stage: "B1".into(),
        amplitude: "total_force".into(),
        phase: "force_input_velocity".into(),
        alignment,
        threshold: DELTA,
    }
}
/// Every alignment of both sources at one threshold.
fn sources(threshold: f64) -> Vec<ColorSource> {
    let mut all: Vec<ColorSource> = [
        ColorAlignment::PrecedingKick,
        ColorAlignment::PrecedingForce,
        ColorAlignment::MatchedKick {
            stage: KickStage::B1,
        },
        ColorAlignment::MatchedKick {
            stage: KickStage::B2,
        },
        ColorAlignment::ReferenceOffset,
    ]
    .into_iter()
    .map(|alignment| ColorSource::ViscousForce {
        alignment,
        threshold,
    })
    .collect();
    for alignment in [StepAlignment::Preceding, StepAlignment::Matched] {
        all.push(ColorSource::RecordedField {
            stage: "B1".into(),
            amplitude: "total_force".into(),
            phase: "force_input_velocity".into(),
            alignment,
            threshold,
        });
    }
    all
}
/// Steps 4 and 5 with the same force and velocity in every record a source reads.
fn uniform_pair() -> (Frame, Frame) {
    let record = || Some(kick(&FORCE, &VELOCITY, 3));
    let mut previous = frame(4, 3, 3);
    previous.kick = Some(Kick {
        b1: None,
        b2: record(),
    });
    previous.recorded_color = record();
    let mut current = frame(5, 3, 3);
    current.v = Some(VELOCITY.to_vec());
    current.kick = Some(Kick {
        b1: record(),
        b2: record(),
    });
    current.recorded_color = record();
    (previous, current)
}
fn fill(
    source: &ColorSource,
    previous: Option<&Frame>,
    frame: &Frame,
) -> (Availability, FrameState) {
    let mut state = FrameState {
        step: frame.step,
        n: frame.n,
        d: frame.d,
        ..FrameState::default()
    };
    let availability = source.fill(previous, frame, KAPPA, &mut state);
    (availability, state)
}
fn all_invalid(state: &FrameState, n: usize, d: usize) -> bool {
    state.color == vec![C::ZERO; n * d]
        && state.color_valid == vec![false; n]
        && state.force_valid == vec![false; n]
        && state.force.is_none()
        && state.phase_velocity.is_none()
}

#[test]
fn direct_inputs_reproduce_the_colour_pair_determinant_and_triangle_vectors() {
    let c = encode(&FORCE, &VELOCITY, 3, KAPPA);
    let expected = [
        (0.181692666918097, 0.132474799645058),
        (-0.864409807074876, 0.248564779238168),
        (0.0377110879509771, 0.37286366415595),
        (-0.325695616359635, 0.418964868320025),
        (0.128460488414299, 0.0805382434839876),
        (0.82117118886446, 0.145190142328959),
        (0.892444694986737, 0.125764997599001),
        (-0.337597375059193, -0.126459128852409),
        (0.0592747972794803, 0.232912497001475),
    ];
    for (z, (re, im)) in c.iter().zip(expected) {
        close(*z, re, im);
    }
    for row in c.chunks_exact(3) {
        assert!((row.iter().map(|z| z.abs2()).sum::<f64>() - 1.).abs() < 1e-15);
        near(dot(row, row), C::ONE, 1e-15);
    }
    let (q01, q12, q20) = (
        dot(&c[..3], &c[3..6]),
        dot(&c[3..6], &c[6..]),
        dot(&c[6..], &c[..3]),
    );
    close(q01, -0.00959436605184832, -0.282989114815417);
    close(q12, -0.209035574798508, -0.22126466858169);
    close(q20, 0.528280470119553, -0.0845334393867648);
    near(dot(&c[3..6], &c[..3]), q01.conj(), 1e-15);
    // Exchange-odd: the two orientations of a pair cancel exactly.
    assert_eq!(q01.im + dot(&c[3..6], &c[..3]).im, 0.);
    let b = baryon(&c);
    close(b, BARYON.re, BARYON.im);
    assert!((b.abs2() - 0.487266970208558).abs() < 1e-12);
    let pi = triangle(&c);
    close(pi, -0.026839028124646, 0.0374953707822108);
    assert!((pi.abs() - 0.0461111294675267).abs() < 1e-12);
    assert!((pi.im.atan2(pi.re) - 2.19204500628267).abs() < 1e-12);
    let reversed = dot(&c[..3], &c[6..]) * dot(&c[6..], &c[3..6]) * dot(&c[3..6], &c[..3]);
    near(reversed, pi.conj(), 1e-15);

    let pythagorean = encode(&[3., -4., 12.], &[0.5, -1.25, 2.], 3, KAPPA);
    close(pythagorean[0], 0.216778318349395, 0.0791302632589503);
    close(pythagorean[1], -0.197229802511792, 0.236167231457239);
    close(pythagorean[2], 0.15689274729253, 0.909645904604732);
}

#[test]
fn vanishing_phase_factor_gives_the_real_geometry_of_the_unit_forces() {
    let c = encode(&FORCE, &VELOCITY, 3, 0.);
    assert!(c.iter().all(|z| z.im == 0.));
    let q01 = dot(&c[..3], &c[3..6]);
    assert_eq!(q01.im, 0.);
    assert!((q01.re - 0.0568218507028155).abs() < 1e-12);
    let b = baryon(&c);
    assert_eq!(b.im, 0.);
    assert!((b.re + 0.759977352102067).abs() < 1e-12);
}

#[test]
fn dense_kernel_normalisers_share_one_colour_and_differ_in_momentum() {
    let x = [0., 0., 0., 0.5, -0.3, 0.2, -0.4, 0.9, 0.1];
    let (nu, rho) = (1.5, 0.8);
    let weight = |i: usize, j: usize| {
        let r2 = (0..3)
            .map(|a| (x[i * 3 + a] - x[j * 3 + a]) * (x[i * 3 + a] - x[j * 3 + a]))
            .sum::<f64>();
        (-r2 / (2. * rho * rho)).exp()
    };
    assert!((weight(0, 1) - 0.743136898668758).abs() < 1e-12);
    assert!((weight(0, 2) - 0.465043188134056).abs() < 1e-12);
    assert!((weight(1, 2) - 0.171079828171343).abs() < 1e-12);
    let force = |normaliser: &dyn Fn(usize) -> f64| {
        let mut f = vec![0.; 9];
        for i in 0..3 {
            for j in (0..3).filter(|&j| j != i) {
                for a in 0..3 {
                    f[i * 3 + a] += nu * weight(i, j) / normaliser(i)
                        * (VELOCITY[j * 3 + a] - VELOCITY[i * 3 + a]);
                }
            }
        }
        f
    };
    let book = force(&|_| 1.);
    let counted = force(&|_| 3.);
    let rows = force(&|i| (0..3).filter(|&l| l != i).map(|l| weight(i, l)).sum());
    let expected_counted = [
        -0.980215704382554,
        1.70149874716321,
        -1.78025312338363,
        0.945760459664141,
        -0.0866145000414356,
        0.443612876124438,
        0.0344552447184127,
        -1.61488424712177,
        1.3366402472592,
    ];
    for (f, e) in counted.iter().zip(expected_counted) {
        assert!((f - e).abs() < 1e-12);
    }
    assert!((book[0] + 2.94064711314766).abs() < 1e-12);
    assert!((rows[7] + 7.61590544153401).abs() < 1e-12);
    let momentum = |f: &[f64], a: usize| (0..3).map(|i| f[i * 3 + a]).sum::<f64>();
    for a in 0..3 {
        assert!(momentum(&book, a).abs() < 1e-14 && momentum(&counted, a).abs() < 1e-14);
    }
    assert!((momentum(&rows, 1) + 3.67518420576957).abs() < 1e-12);

    let c = encode(&counted, &VELOCITY, 3, KAPPA);
    for other in [&book, &rows] {
        let o = encode(other, &VELOCITY, 3, KAPPA);
        for (a, b) in c.iter().zip(&o) {
            near(*a, *b, 1e-15);
        }
    }
    close(c[0], -0.298825583376558, -0.217878134313874);
    close(c[5], 0.416743561224495, 0.0736838527573434);
    close(c[7], 0.721303045681474, 0.270189763114939);
    close(
        dot(&c[..3], &c[3..6]),
        -0.123085455809543,
        0.567526761675932,
    );
    close(baryon(&c), 0.151176923860637, -0.223856549554325);
    close(triangle(&c), 0.102859810849207, -0.0887772312948183);
}

#[test]
fn threshold_is_strict_and_an_invalid_colour_is_an_exact_zero_vector() {
    let mut c = [C::ONE; 3];
    for (f, valid) in [(5e-13, false), (1e-12, false), (2e-12, true), (0., false)] {
        assert_eq!(
            color_vector(&[f, 0., 0.], &[0.; 3], KAPPA, DELTA, &mut c),
            valid
        );
        let expected = if valid { C::ONE } else { C::ZERO };
        assert_eq!(bits(&c), bits(&[expected, C::ZERO, C::ZERO]), "{f}");
    }
    for (f, v) in [
        ([f64::NAN, 1., 0.], [0.; 3]),
        ([f64::INFINITY, 1., 0.], [0.; 3]),
        ([1e200, 1., 0.], [0.; 3]),
        ([1., 1., 0.], [0., f64::NAN, 0.]),
        ([1., 1., 0.], [0., 0., f64::NEG_INFINITY]),
        ([1., 1., 0.], [0., 1e308, 0.]),
    ] {
        c = [C::ONE; 3];
        assert!(!color_vector(&f, &v, 10., DELTA, &mut c), "{f:?} {v:?}");
        assert_eq!(bits(&c), bits(&[C::ZERO; 3]));
    }
    // A zero force has no direction under any threshold, and no NaN colour.
    for threshold in [0., -1., f64::NAN] {
        c = [C::ONE; 3];
        assert!(!color_vector(&[0.; 3], &[0.; 3], KAPPA, threshold, &mut c));
        assert_eq!(bits(&c), bits(&[C::ZERO; 3]), "{threshold}");
    }
    assert!(color_vector(&[0., 1e-100, 0.], &[0.; 3], KAPPA, 0., &mut c));
    assert_eq!(bits(&c), bits(&[C::ZERO, C::ONE, C::ZERO]));
    // The mask is absolute in force units, the colour is scale free.
    let reference = encode(&FORCE, &VELOCITY, 3, KAPPA);
    for scale in [3.7, 1e-6, 2.5e4] {
        let scaled: Vec<f64> = FORCE.iter().map(|f| f * scale).collect();
        let c = encode(&scaled, &VELOCITY, 3, KAPPA);
        for (a, b) in c.iter().zip(&reference) {
            near(*a, *b, 1e-15);
        }
    }
    let tiny: Vec<f64> = FORCE[..3].iter().map(|f| f * 1e-13).collect();
    assert!(!color_vector(
        &tiny,
        &VELOCITY[..3],
        KAPPA,
        DELTA,
        &mut [C::ZERO; 3]
    ));
}

#[test]
fn colour_matches_the_recorded_colour_convention_bit_for_bit_on_its_fixture_inputs() {
    for i in 0..7 {
        let t = i as f64 * 0.8;
        let f = [1. + t.sin(), 0.4 + t.cos(), 0.2 + t.sin() * t.cos()];
        let v = [t, 0.3 * t, -0.4 * t];
        let norm = f.iter().map(|x| x * x).sum::<f64>().sqrt();
        let expected: [C; 3] =
            std::array::from_fn(|a| C::phase(KAPPA * v[a]) * (f[a] / norm.max(DELTA)));
        let mut c = [C::ZERO; 3];
        assert!(color_vector(&f, &v, KAPPA, DELTA, &mut c));
        assert_eq!(bits(&c), bits(&expected));
    }
}

#[test]
fn spatial_parity_maps_a_colour_to_minus_its_conjugate_exactly() {
    let c = encode(&FORCE, &VELOCITY, 3, KAPPA);
    let force: Vec<f64> = FORCE.iter().map(|f| -f).collect();
    let velocity: Vec<f64> = VELOCITY.iter().map(|v| -v).collect();
    let p = encode(&force, &velocity, 3, KAPPA);
    let mirrored: Vec<C> = c.iter().map(|z| -z.conj()).collect();
    assert_eq!(bits(&p), bits(&mirrored));
    let (q, qp) = (dot(&c[..3], &c[3..6]), dot(&p[..3], &p[3..6]));
    near(qp, q.conj(), 1e-15);
    near(baryon(&p), -baryon(&c).conj(), 1e-15);
    near(triangle(&p), triangle(&c).conj(), 1e-15);
}

#[test]
fn boosts_and_walker_rephasings_move_only_the_documented_phases() {
    let c = encode(&FORCE, &VELOCITY, 3, KAPPA);
    let (q, b, pi) = (dot(&c[..3], &c[3..6]), baryon(&c), triangle(&c));

    let u = [0.3, -1.1, 0.6];
    let boosted: Vec<f64> = (0..9).map(|k| VELOCITY[k] + u[k % 3]).collect();
    let g = encode(&FORCE, &boosted, 3, KAPPA);
    near(dot(&g[..3], &g[3..6]), q, 1e-15);
    near(triangle(&g), pi, 1e-15);
    close(baryon(&g), -0.683928488594191, 0.139673879797992);
    near(baryon(&g), C::phase(-0.14) * b, 1e-15);

    let shift = [0.4, -0.9, 1.7];
    let shifted: Vec<f64> = (0..9).map(|k| VELOCITY[k] + shift[k / 3]).collect();
    let l = encode(&FORCE, &shifted, 3, KAPPA);
    let alpha = shift.map(|s| KAPPA * s);
    let rephased: Vec<C> = (0..9).map(|k| C::phase(alpha[k / 3]) * c[k]).collect();
    for (a, b) in l.iter().zip(&rephased) {
        near(*a, *b, 1e-15);
    }
    let ql = dot(&l[..3], &l[3..6]);
    close(ql, -0.229309465821682, -0.166108578491675);
    assert!((ql.abs2() - q.abs2()).abs() < 1e-15);
    close(baryon(&l), -0.49696237842135, -0.490199311140227);
    near(triangle(&l), pi, 1e-15);

    for turns in [1., -3.] {
        let wound: Vec<f64> = VELOCITY.iter().map(|v| v + TAU * turns / KAPPA).collect();
        let w = encode(&FORCE, &wound, 3, KAPPA);
        for (a, b) in w.iter().zip(&c) {
            near(*a, *b, 1e-13);
        }
    }
}

#[test]
fn a_common_unitary_frame_changes_only_the_determinant_by_its_own() {
    let c = encode(&FORCE, &VELOCITY, 3, KAPPA);
    let (q, b, pi) = (dot(&c[..3], &c[3..6]), baryon(&c), triangle(&c));
    let (cz, sz, cx, sx) = (0.4f64.cos(), 0.4f64.sin(), 1.2f64.cos(), 1.2f64.sin());
    let about_z = [cz, -sz, 0., sz, cz, 0., 0., 0., 1.];
    let about_x = [1., 0., 0., 0., cx, -sx, 0., sx, cx];
    for (third, re, im) in [
        (0.8, BARYON.re, BARYON.im),
        (1.3, -0.631988743690376, -0.296407149133112),
    ] {
        let phases = [0.3, -1.1, third].map(C::phase);
        let mut u = [C::ZERO; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    u[i * 3 + j] =
                        u[i * 3 + j] + phases[k] * (about_z[i * 3 + k] * about_x[k * 3 + j]);
                }
            }
        }
        let moved: Vec<C> = (0..9)
            .map(|k| {
                let (walker, i) = (k / 3, k % 3);
                (0..3).fold(C::ZERO, |s, j| s + u[i * 3 + j] * c[walker * 3 + j])
            })
            .collect();
        near(dot(&moved[..3], &moved[3..6]), q, 1e-15);
        near(triangle(&moved), pi, 1e-15);
        close(baryon(&moved), re, im);
        near(baryon(&moved), C::phase(0.3 - 1.1 + third) * b, 1e-15);
    }
}

#[test]
fn component_permutations_sign_the_determinant_and_a_rotation_is_not_a_symmetry() {
    let c = encode(&FORCE, &VELOCITY, 3, KAPPA);
    for (order, sign) in [([1, 2, 0], 1.), ([1, 0, 2], -1.)] {
        let pick = |values: &[f64; 9]| -> Vec<f64> {
            (0..9).map(|k| values[k / 3 * 3 + order[k % 3]]).collect()
        };
        let p = encode(&pick(&FORCE), &pick(&VELOCITY), 3, KAPPA);
        near(dot(&p[..3], &p[3..6]), dot(&c[..3], &c[3..6]), 1e-15);
        near(triangle(&p), triangle(&c), 1e-15);
        near(baryon(&p), baryon(&c) * sign, 1e-15);
    }
    let h = FRAC_1_SQRT_2;
    let rotated_inputs = encode(&[h, h], &[2. * h, 2. * h], 2, KAPPA);
    close(rotated_inputs[0], 0.388012177999697, 0.591140042395989);
    close(rotated_inputs[1], 0.388012177999697, 0.591140042395989);
    let plain = encode(&[1., 0.], &[2., 0.], 2, KAPPA);
    let rotated_colour = [(plain[0] - plain[1]) * h, (plain[0] + plain[1]) * h];
    close(rotated_colour[0], 0.120184919323663, 0.696818186593292);
    let gap = rotated_inputs
        .iter()
        .zip(&rotated_colour)
        .map(|(a, b)| (*a - *b).abs2())
        .sum::<f64>()
        .sqrt();
    assert!((gap - 0.407183768465554).abs() < 1e-12);
}

#[test]
fn determinant_is_alternating_and_bounded_with_the_pair_and_triangle_contractions() {
    let c = encode(&FORCE, &VELOCITY, 3, KAPPA);
    let (x, y, z) = (&c[..3], &c[3..6], &c[6..]);
    let b = det3(x, y, z);
    near(det3(y, x, z), -b, 1e-15);
    near(det3(x, z, y), -b, 1e-15);
    near(det3(z, y, x), -b, 1e-15);
    near(det3(y, z, x), b, 1e-15);
    near(det3(z, x, y), b, 1e-15);
    assert!(det3(x, x, y).abs() < 1e-15 && det3(x, y, y).abs() < 1e-15);
    let unit_frame = [C::ONE, C::ZERO, C::ZERO, C::ZERO, C::ONE, C::ZERO];
    assert_eq!(
        det3(
            &unit_frame[..3],
            &unit_frame[3..],
            &[C::ZERO, C::ZERO, C::ONE]
        ),
        C::ONE
    );

    let h = FRAC_1_SQRT_2;
    let book = [
        C::ONE,
        C::ZERO,
        C::ZERO,
        C::new(h, 0.),
        C::new(h, 0.),
        C::ZERO,
        C::new(h, 0.),
        C::new(0., h),
        C::ZERO,
    ];
    close(triangle(&book), 0.25, 0.25);
    close(dot(&book[3..6], &book[6..]), 0.5, 0.5);
    assert!(baryon(&book).abs() < 1e-15);

    for sample in 0..64u64 {
        let draw = |offset: u64, scale: f64| -> Vec<f64> {
            (0..9)
                .map(|k| scale * unit(sample * 64 + offset + k))
                .collect()
        };
        let c = encode(&draw(0, 2.), &draw(16, 9.), 3, KAPPA);
        assert!(dot(&c[..3], &c[3..6]).abs() <= 1. + 1e-15);
        assert!(baryon(&c).abs() <= 1. + 1e-15);
        assert!(triangle(&c).abs() <= 1. + 1e-15);
        let same = [&c[..3], &c[..3], &c[..3]].concat();
        near(triangle(&same), C::ONE, 1e-14);
    }
}

#[test]
fn fixed_force_pair_contraction_is_the_weighted_sum_of_velocity_phases() {
    let (f, v, w) = (&FORCE[..3], &VELOCITY[..3], &VELOCITY[3..6]);
    let (mut a, mut b) = ([C::ZERO; 3], [C::ZERO; 3]);
    assert!(color_vector(f, v, KAPPA, DELTA, &mut a) && color_vector(f, w, KAPPA, DELTA, &mut b));
    let norm2 = f.iter().map(|x| x * x).sum::<f64>();
    let expected = (0..3).fold(C::ZERO, |s, k| {
        s + C::phase(KAPPA * (w[k] - v[k])) * (f[k] * f[k] / norm2)
    });
    near(dot(&a, &b), expected, 1e-15);
}

#[test]
fn preceding_kick_reads_the_b2_record_of_the_step_before_with_its_own_velocity() {
    let gradient = [0.2, 0.1, -0.3, 0., -0.5, 0.4, -0.6, 0.3, 0.1];
    let pre_clone: Vec<f64> = (0..9)
        .map(|k| VELOCITY[k] + 0.05 * (FORCE[k] - gradient[k]))
        .collect();
    let mut previous = frame(4, 3, 3);
    previous.kick = Some(Kick {
        b1: None,
        b2: Some(kick(&FORCE, &VELOCITY, 3)),
    });
    let mut current = frame(5, 3, 3);
    current.v = Some(pre_clone.clone());

    let (availability, kicked) = fill(
        &viscous(ColorAlignment::PrecedingKick),
        Some(&previous),
        &current,
    );
    assert!(availability.is_available());
    assert_eq!(
        bits(&kicked.color),
        bits(&encode(&FORCE, &VELOCITY, 3, KAPPA))
    );
    assert_eq!(kicked.color_valid, vec![true; 3]);
    assert_eq!(kicked.force_valid, vec![true; 3]);
    assert_eq!(kicked.force.as_deref(), Some(&FORCE[..]));
    assert_eq!(kicked.phase_velocity.as_deref(), Some(&VELOCITY[..]));
    let c = &kicked.color;
    close(
        dot(&c[..3], &c[3..6]),
        -0.00959436605184832,
        -0.282989114815417,
    );

    let (availability, forced) = fill(
        &viscous(ColorAlignment::PrecedingForce),
        Some(&previous),
        &current,
    );
    assert!(availability.is_available());
    assert_eq!(forced.force.as_deref(), Some(&FORCE[..]));
    assert_eq!(forced.phase_velocity.as_deref(), Some(&pre_clone[..]));
    let p = &forced.color;
    close(p[0], 0.181227893199533, 0.133109911273607);
    close(p[1], -0.852209393687125, 0.287624604921703);
    close(p[2], 0.0272574877042988, 0.373773283644373);
    close(dot(&p[..3], &p[3..6]), 1.8704443268e-05, -0.289346772312664);
    close(
        dot(&p[3..6], &p[6..]),
        -0.159547132787117,
        -0.245434300968034,
    );
    close(dot(&p[6..], &p[..3]), 0.522592206335487, -0.107506908509553);
    for k in 0..9 {
        let half_kick = C::phase(KAPPA * 0.05 * (FORCE[k] - gradient[k]));
        near(p[k], c[k] * half_kick, 1e-15);
    }
}

#[test]
fn a_preceding_record_is_masked_by_generation_and_a_matched_record_by_the_clone_flag() {
    let mut previous = frame(4, 3, 3);
    // Walker 2 cloned at the preceding step: its B2 record is post-clone.
    previous.generation = vec![0, 0, 0];
    let mut b2 = kick(&FORCE, &VELOCITY, 3);
    b2.generation = vec![0, 0, 1];
    previous.kick = Some(Kick {
        b1: None,
        b2: Some(b2),
    });
    let mut current = frame(5, 3, 3);
    current.generation = vec![0, 0, 1];
    current.cloned = vec![true, false, false];
    current.v = Some(VELOCITY.to_vec());
    current.kick = Some(Kick {
        b1: Some(kick(&FORCE, &VELOCITY, 3)),
        b2: Some(kick(&VELOCITY, &FORCE, 3)),
    });

    for alignment in [
        ColorAlignment::PrecedingKick,
        ColorAlignment::PrecedingForce,
    ] {
        let (_, state) = fill(&viscous(alignment), Some(&previous), &current);
        assert_eq!(state.color_valid, vec![true; 3], "{alignment:?}");
    }
    let mut stale = current.clone();
    stale.generation = vec![0, 2, 1];
    let (availability, state) = fill(
        &viscous(ColorAlignment::PrecedingKick),
        Some(&previous),
        &stale,
    );
    assert!(availability.is_available());
    assert_eq!(state.color_valid, vec![true, false, true]);
    assert_eq!(state.force_valid, vec![true, false, true]);
    assert_eq!(bits(&state.color[3..6]), bits(&[C::ZERO; 3]));
    assert_eq!(state.force.as_ref().unwrap()[3..6], [0.; 3]);
    assert_eq!(state.phase_velocity.as_ref().unwrap()[3..6], [0.; 3]);

    for (alignment, force, velocity) in [
        (
            ColorAlignment::MatchedKick {
                stage: KickStage::B1,
            },
            &FORCE,
            &VELOCITY,
        ),
        (
            ColorAlignment::MatchedKick {
                stage: KickStage::B2,
            },
            &VELOCITY,
            &FORCE,
        ),
        (ColorAlignment::ReferenceOffset, &FORCE, &VELOCITY),
    ] {
        let (availability, state) = fill(&viscous(alignment), None, &current);
        assert!(availability.is_available());
        assert_eq!(state.color_valid, vec![false, true, true], "{alignment:?}");
        assert_eq!(state.force_valid, vec![false, true, true]);
        let expected = encode(force, velocity, 3, KAPPA);
        assert_eq!(bits(&state.color[3..]), bits(&expected[3..]));
        assert_eq!(bits(&state.color[..3]), bits(&[C::ZERO; 3]));
    }
}

#[test]
fn reference_offset_pairs_the_b1_force_with_the_pre_clone_velocity() {
    let pre_clone: Vec<f64> = VELOCITY.iter().map(|v| 0.5 * v - 0.1).collect();
    let mut current = frame(5, 3, 3);
    current.v = Some(pre_clone.clone());
    current.kick = Some(Kick {
        b1: Some(kick(&FORCE, &VELOCITY, 3)),
        b2: None,
    });
    let (availability, state) = fill(&viscous(ColorAlignment::ReferenceOffset), None, &current);
    assert!(availability.is_available());
    assert_eq!(
        bits(&state.color),
        bits(&encode(&FORCE, &pre_clone, 3, KAPPA))
    );
    assert_eq!(state.phase_velocity.as_deref(), Some(&pre_clone[..]));
    assert_eq!(state.force.as_deref(), Some(&FORCE[..]));
}

#[test]
fn uncovered_ineligible_and_forceless_rows_are_invalid_without_a_fabricated_value() {
    let mut force = FORCE;
    force[6..].fill(0.);
    let mut b1 = kick(&force, &VELOCITY, 3);
    b1.available[0] = false;
    let mut current = frame(5, 3, 3);
    current.kick = Some(Kick {
        b1: Some(b1),
        b2: None,
    });
    let matched = viscous(ColorAlignment::MatchedKick {
        stage: KickStage::B1,
    });
    let (availability, state) = fill(&matched, None, &current);
    assert!(availability.is_available());
    assert_eq!(state.color_valid, vec![false, true, false]);
    // A zero force on a covered row is a valid force with no colour.
    assert_eq!(state.force_valid, vec![false, true, true]);
    assert_eq!(bits(&state.color[..3]), bits(&[C::ZERO; 3]));
    assert_eq!(bits(&state.color[6..]), bits(&[C::ZERO; 3]));
    assert_eq!(state.phase_velocity.as_ref().unwrap()[6..], VELOCITY[6..]);

    current.eligible[1] = false;
    let (_, state) = fill(&matched, None, &current);
    assert_eq!(state.color_valid, vec![false; 3]);
    assert_eq!(state.force_valid, vec![false, false, true]);
}

#[test]
fn nonfinite_records_are_masked_rows_and_never_reach_the_state() {
    let mut force = FORCE;
    force[4] = f64::NAN;
    let mut pre_clone = VELOCITY;
    pre_clone[8] = f64::INFINITY;
    let mut previous = frame(4, 3, 3);
    // An untrusted record may flag a nonfinite row as available.
    previous.kick = Some(Kick {
        b1: None,
        b2: Some(kick(&force, &VELOCITY, 3)),
    });
    let mut current = frame(5, 3, 3);
    current.v = Some(pre_clone.to_vec());
    let (availability, state) = fill(
        &viscous(ColorAlignment::PrecedingForce),
        Some(&previous),
        &current,
    );
    assert!(availability.is_available());
    assert_eq!(state.color_valid, vec![true, false, false]);
    // A finite force with a nonfinite phase velocity is a force without a colour.
    assert_eq!(state.force_valid, vec![true, false, true]);
    assert_eq!(state.force.as_ref().unwrap()[3..6], [0.; 3]);
    assert_eq!(state.force.as_ref().unwrap()[6..], FORCE[6..]);
    assert_eq!(state.phase_velocity.as_ref().unwrap()[..3], VELOCITY[..3]);
    assert_eq!(state.phase_velocity.as_ref().unwrap()[3..], [0.; 6]);
    assert_eq!(bits(&state.color[3..]), bits(&[C::ZERO; 6]));
    let every = state.force.iter().chain(&state.phase_velocity).flatten();
    assert!(every.into_iter().all(|x| x.is_finite()));
    assert!(
        state
            .color
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite())
    );
}

#[test]
fn a_fill_depends_on_the_frames_alone_and_not_on_what_the_state_held() {
    let mut previous = frame(4, 3, 3);
    previous.kick = Some(Kick {
        b1: None,
        b2: Some(kick(&FORCE, &VELOCITY, 3)),
    });
    let current = frame(5, 3, 3);
    let mut wide = frame(9, 4, 2);
    wide.kick = Some(Kick {
        b1: Some(kick(&FORCE[..8], &VELOCITY[..8], 4)),
        b2: None,
    });
    let source = viscous(ColorAlignment::PrecedingKick);
    let (_, fresh) = fill(&source, Some(&previous), &current);
    let (_, mut reused) = fill(
        &viscous(ColorAlignment::MatchedKick {
            stage: KickStage::B1,
        }),
        None,
        &wide,
    );
    assert_eq!(reused.color.len(), 8);
    (reused.step, reused.n, reused.d) = (fresh.step, fresh.n, fresh.d);
    assert!(
        source
            .fill(Some(&previous), &current, KAPPA, &mut reused)
            .is_available()
    );
    assert_eq!(reused, fresh);
    let (_, again) = fill(&source, Some(&previous), &current);
    assert_eq!(bits(&again.color), bits(&fresh.color));
}

#[test]
fn a_missing_record_is_explicitly_unavailable_and_leaves_the_state_all_invalid() {
    let mut previous = frame(4, 3, 3);
    previous.kick = Some(Kick {
        b1: None,
        b2: Some(kick(&FORCE, &VELOCITY, 3)),
    });
    let mut current = frame(5, 3, 3);
    current.kick = Some(Kick {
        b1: Some(kick(&FORCE, &VELOCITY, 3)),
        b2: None,
    });
    let preceding = viscous(ColorAlignment::PrecedingKick);
    let matched = |stage| viscous(ColorAlignment::MatchedKick { stage });
    let mut gap = previous.clone();
    gap.step = 3;
    let mut resized = previous.clone();
    resized.n = 2;
    let mut bare = previous.clone();
    bare.kick = None;
    let mut malformed = current.clone();
    malformed
        .kick
        .as_mut()
        .unwrap()
        .b1
        .as_mut()
        .unwrap()
        .force
        .pop();
    let mut misshapen = current.clone();
    misshapen.cloned.pop();
    let cases: [(&ColorSource, Option<&Frame>, &Frame); 10] = [
        (&preceding, None, &current),
        (&preceding, Some(&gap), &current),
        (&preceding, Some(&resized), &current),
        (&preceding, Some(&bare), &current),
        (&preceding, Some(&current), &current),
        (&matched(KickStage::B2), Some(&previous), &current),
        (
            &viscous(ColorAlignment::PrecedingForce),
            Some(&previous),
            &current,
        ),
        (&viscous(ColorAlignment::ReferenceOffset), None, &current),
        (&matched(KickStage::B1), None, &malformed),
        (&recorded(StepAlignment::Matched), None, &current),
    ];
    for (k, (source, before, at)) in cases.into_iter().enumerate() {
        let (mut availability, mut state) = fill(&matched(KickStage::B1), None, &current);
        assert!(availability.is_available() && state.color_valid == vec![true; 3]);
        availability = source.fill(before, at, KAPPA, &mut state);
        assert!(!availability.reason().unwrap().is_empty(), "case {k}");
        assert!(all_invalid(&state, 3, 3), "case {k}");
    }
    let (availability, state) = fill(&matched(KickStage::B1), None, &misshapen);
    assert!(!availability.is_available() && state.color.is_empty() && state.force.is_none());
    let mut state = FrameState::default();
    let availability = matched(KickStage::B1).fill(None, &current, f64::NAN, &mut state);
    assert!(!availability.is_available() && all_invalid(&state, 3, 3));
}

#[test]
fn colour_is_normalised_over_every_component_in_any_dimension() {
    for d in [1, 2, 5] {
        let n = 4;
        let force: Vec<f64> = (0..n * d).map(|k| 3. * unit(k as u64)).collect();
        let velocity: Vec<f64> = (0..n * d).map(|k| 8. * unit(100 + k as u64)).collect();
        let mut current = frame(1, n, d);
        current.kick = Some(Kick {
            b1: Some(kick(&force, &velocity, n)),
            b2: None,
        });
        let (availability, state) = fill(
            &viscous(ColorAlignment::MatchedKick {
                stage: KickStage::B1,
            }),
            None,
            &current,
        );
        assert!(availability.is_available());
        assert_eq!(state.color.len(), n * d);
        assert_eq!(state.color_valid, vec![true; n]);
        for ((c, f), v) in state
            .color
            .chunks_exact(d)
            .zip(force.chunks_exact(d))
            .zip(velocity.chunks_exact(d))
        {
            assert!((c.iter().map(|z| z.abs2()).sum::<f64>() - 1.).abs() < 1e-15);
            let norm = f.iter().map(|x| x * x).sum::<f64>().sqrt();
            for a in 0..d {
                near(c[a], C::phase(KAPPA * v[a]) * (f[a] / norm), 1e-15);
            }
        }
    }
}

#[test]
fn recorded_field_source_reads_the_selected_record_with_the_same_identity_rules() {
    let mut previous = frame(4, 3, 3);
    let mut record = kick(&FORCE, &VELOCITY, 3);
    record.generation = vec![0, 7, 0];
    previous.recorded_color = Some(record);
    let mut current = frame(5, 3, 3);
    current.cloned = vec![false, false, true];
    current.recorded_color = Some(kick(&VELOCITY, &FORCE, 3));

    let (availability, state) = fill(
        &recorded(StepAlignment::Preceding),
        Some(&previous),
        &current,
    );
    assert!(availability.is_available());
    assert_eq!(state.color_valid, vec![true, false, true]);
    let expected = encode(&FORCE, &VELOCITY, 3, KAPPA);
    assert_eq!(bits(&state.color[..3]), bits(&expected[..3]));
    assert_eq!(bits(&state.color[6..]), bits(&expected[6..]));

    let (availability, state) = fill(&recorded(StepAlignment::Matched), Some(&previous), &current);
    assert!(availability.is_available());
    assert_eq!(state.color_valid, vec![true, true, false]);
    assert_eq!(
        bits(&state.color[..6]),
        bits(&encode(&VELOCITY, &FORCE, 3, KAPPA)[..6])
    );
    assert_eq!(state.force.as_ref().unwrap()[..6], VELOCITY[..6]);

    let (availability, state) = fill(&recorded(StepAlignment::Preceding), None, &current);
    assert!(!availability.is_available() && all_invalid(&state, 3, 3));
    previous.recorded_color = None;
    let (availability, _) = fill(
        &recorded(StepAlignment::Preceding),
        Some(&previous),
        &current,
    );
    assert!(!availability.is_available());
}

#[test]
fn phase_factor_is_mass_times_length_over_action_with_every_length_arm() {
    let phase = PhaseScale {
        h_eff: 1.,
        mass: 1.,
        length: LengthScale::Fixed { value: 0.7 },
    };
    let fixed = phase_length(phase.length, &[]).unwrap();
    assert_eq!(kappa(&phase, fixed).unwrap().to_bits(), 0.7f64.to_bits());
    let scaled = PhaseScale {
        h_eff: 3.,
        mass: 1.7,
        ..phase
    };
    assert_eq!(
        kappa(&scaled, 0.9).unwrap().to_bits(),
        (1.7f64 * 0.9 / 3.).to_bits()
    );
    // `mass * length / h_eff` in that order: the other two groupings of these
    // inputs round to 0.8571428571428572.
    let ordered = PhaseScale {
        h_eff: 1.4,
        mass: 0.8,
        ..phase
    };
    assert_eq!(
        kappa(&ordered, 1.5).unwrap().to_bits(),
        0.8571428571428573f64.to_bits()
    );
    // Equal phase factors give equal colours whatever the three scales are.
    let halved = PhaseScale {
        h_eff: 0.5,
        mass: 0.5,
        ..phase
    };
    assert_eq!(
        kappa(&halved, 0.7).unwrap().to_bits(),
        kappa(&phase, 0.7).unwrap().to_bits()
    );
    for (bad, length) in [
        (PhaseScale { h_eff: 0., ..phase }, 1.),
        (
            PhaseScale {
                mass: f64::NAN,
                ..phase
            },
            1.,
        ),
        (phase, 0.),
        (phase, f64::INFINITY),
        (
            PhaseScale {
                h_eff: 1e-300,
                ..phase
            },
            1e300,
        ),
    ] {
        assert!(matches!(
            kappa(&bad, length),
            Err(GasError::Configuration(_))
        ));
    }

    let x = [0., 0., 0., 1., 0., 0., 1., 2., 0., 1., 2., 2.5];
    let companion = [1, 2, 3, 0];
    let distances: Vec<f64> = (0..4)
        .map(|i| {
            (0..3)
                .map(|a| {
                    (x[i * 3 + a] - x[companion[i] * 3 + a])
                        * (x[i * 3 + a] - x[companion[i] * 3 + a])
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    assert!((distances[3] - 3.35410196624968).abs() < 1e-12);
    let median = phase_length(LengthScale::WarmupCompanionMedian, &distances).unwrap();
    assert_eq!(median, 2.25);
    assert_eq!(
        phase_length(LengthScale::WarmupCompanionMedian, &distances[..3]).unwrap(),
        2.
    );
    let edges = [&distances[..], &[0., f64::INFINITY, f64::NAN]].concat();
    for geodesic in [false, true] {
        let mean = phase_length(LengthScale::WarmupEdgeMean { geodesic }, &edges).unwrap();
        assert!((mean - 2.21352549156242).abs() < 1e-14);
    }
    for (arm, samples) in [
        (LengthScale::WarmupCompanionMedian, &[][..]),
        (LengthScale::WarmupCompanionMedian, &[0., 0., 0.][..]),
        (
            LengthScale::WarmupEdgeMean { geodesic: true },
            &[0., f64::NAN][..],
        ),
    ] {
        assert!(matches!(
            phase_length(arm, samples),
            Err(GasError::Capability(_))
        ));
    }
    for value in [-1., 0., f64::NAN, f64::INFINITY] {
        assert!(matches!(
            phase_length(LengthScale::Fixed { value }, &[1.]),
            Err(GasError::Configuration(_))
        ));
    }
    // Pooled samples arrive in no order and may hold nonfinite entries; a zero
    // distance is a sample of the median and not an edge of the mean.
    let shuffled = [
        distances[3],
        f64::NAN,
        distances[0],
        f64::INFINITY,
        distances[2],
        f64::NEG_INFINITY,
        distances[1],
    ];
    assert_eq!(
        phase_length(LengthScale::WarmupCompanionMedian, &shuffled).unwrap(),
        2.25
    );
    assert_eq!(
        phase_length(LengthScale::WarmupCompanionMedian, &[5., 0., 4., 0., 3.]).unwrap(),
        3.
    );
    assert_eq!(
        phase_length(
            LengthScale::WarmupEdgeMean { geodesic: false },
            &[5., 0., 4., 0., 3.]
        )
        .unwrap(),
        4.
    );
}

#[test]
fn phase_wrapping_counts_the_velocity_components_beyond_half_a_turn() {
    assert_eq!(phase_wrapping(KAPPA, &VELOCITY), Some(1. / 9.));
    assert_eq!(phase_wrapping(0., &VELOCITY), Some(0.));
    assert_eq!(
        phase_wrapping(-KAPPA, &[5., -5., f64::NAN, 1.]),
        Some(2. / 3.)
    );
    // Half a turn itself does not alias: the bound is strict.
    let beyond = f64::from_bits(PI.to_bits() + 1);
    assert_eq!(phase_wrapping(1., &[PI, -PI, beyond]), Some(1. / 3.));
    assert_eq!(phase_wrapping(-2., &[0.5 * PI, -0.5 * beyond]), Some(0.5));
    assert_eq!(phase_wrapping(KAPPA, &[]), None);
    assert_eq!(phase_wrapping(KAPPA, &[f64::NAN]), None);
}

#[test]
fn the_threshold_of_the_source_masks_the_colour_in_force_units_and_keeps_the_force() {
    // |F| of the three rows is √1.78 ≈ 1.334, √1.74 ≈ 1.319 and √2.77 ≈ 1.664.
    let (previous, current) = uniform_pair();
    let reference = encode(&FORCE, &VELOCITY, 3, KAPPA);
    for (threshold, held) in [
        (1.32, [true, false, true]),
        (1.5, [false, false, true]),
        (2., [false; 3]),
    ] {
        for source in sources(threshold) {
            let (availability, state) = fill(&source, Some(&previous), &current);
            assert!(availability.is_available(), "{source:?}");
            assert_eq!(state.color_valid, held, "{source:?}");
            assert_eq!(state.force_valid, vec![true; 3]);
            assert_eq!(state.force.as_deref(), Some(&FORCE[..]));
            assert_eq!(state.phase_velocity.as_deref(), Some(&VELOCITY[..]));
            for (i, kept) in held.into_iter().enumerate() {
                let row = i * 3..(i + 1) * 3;
                let expected = if kept {
                    &reference[row.clone()]
                } else {
                    &[C::ZERO; 3][..]
                };
                assert_eq!(bits(&state.color[row]), bits(expected));
            }
        }
    }
}

#[test]
fn a_preceding_record_answers_to_the_eligibility_of_the_frame_and_to_any_generation_change() {
    let (mut previous, mut current) = uniform_pair();
    // Walker 0 was dead before the cloning of the step before and revived by
    // it, walker 2 cloned there: their B2 records are of the new incarnation.
    previous.eligible = vec![false, true, true];
    previous.cloned = vec![true, false, true];
    previous.revived = vec![true, false, false];
    for record in [
        previous.kick.as_mut().unwrap().b2.as_mut().unwrap(),
        previous.recorded_color.as_mut().unwrap(),
    ] {
        record.generation = vec![1, 0, 1];
    }
    current.generation = vec![1, 0, 1];
    let preceding = [
        viscous(ColorAlignment::PrecedingKick),
        viscous(ColorAlignment::PrecedingForce),
        recorded(StepAlignment::Preceding),
    ];
    for source in &preceding {
        let (_, state) = fill(source, Some(&previous), &current);
        assert_eq!(state.color_valid, vec![true; 3], "{source:?}");
        assert_eq!(state.force_valid, vec![true; 3]);
    }
    // A walker that is not eligible at the frame has no colour there.
    let mut dead = current.clone();
    dead.eligible[1] = false;
    // A record of another incarnation is masked whichever generation is larger.
    let mut older = current.clone();
    older.generation = vec![0, 0, 1];
    let mut newer = current.clone();
    newer.generation = vec![1, 0, 3];
    for (at, held) in [
        (&dead, [true, false, true]),
        (&older, [false, true, true]),
        (&newer, [true, true, false]),
    ] {
        for source in &preceding {
            let (availability, state) = fill(source, Some(&previous), at);
            assert!(availability.is_available());
            assert_eq!(state.color_valid, held, "{source:?}");
            assert_eq!(state.force_valid, held);
            for (i, kept) in held.into_iter().enumerate() {
                let row = i * 3..(i + 1) * 3;
                let expected = if kept { &FORCE[row.clone()] } else { &[0.; 3] };
                assert_eq!(&state.force.as_ref().unwrap()[row], expected);
            }
        }
    }
}

#[test]
fn malformed_records_a_step_overflow_and_an_infinite_phase_factor_are_unavailable() {
    let (previous, current) = uniform_pair();
    let every = sources(DELTA);
    let reads_previous = |source: &ColorSource| {
        matches!(
            source,
            ColorSource::ViscousForce {
                alignment: ColorAlignment::PrecedingKick | ColorAlignment::PrecedingForce,
                ..
            } | ColorSource::RecordedField {
                alignment: StepAlignment::Preceding,
                ..
            }
        )
    };
    let edits: [&dyn Fn(&mut StageKick); 4] = [
        &|r| r.available.truncate(2),
        &|r| r.generation.truncate(2),
        &|r| r.force.push(0.),
        &|r| {
            r.force.truncate(6);
            r.velocity.truncate(6);
        },
    ];
    for (k, edit) in edits.into_iter().enumerate() {
        let (mut before, mut at) = (previous.clone(), current.clone());
        for frame in [&mut before, &mut at] {
            let stages = frame.kick.as_mut().unwrap();
            for record in [&mut stages.b1, &mut stages.b2, &mut frame.recorded_color] {
                record.iter_mut().for_each(edit);
            }
        }
        for source in &every {
            let (availability, state) = fill(source, Some(&before), &current);
            assert_eq!(availability.is_available(), !reads_previous(source));
            assert!(availability.is_available() || all_invalid(&state, 3, 3));
            let (availability, state) = fill(source, Some(&previous), &at);
            assert_eq!(availability.is_available(), reads_previous(source), "{k}");
            assert!(availability.is_available() || all_invalid(&state, 3, 3));
        }
    }
    let mut short = current.clone();
    short.v.as_mut().unwrap().pop();
    for alignment in [
        ColorAlignment::PrecedingForce,
        ColorAlignment::ReferenceOffset,
    ] {
        let (availability, state) = fill(&viscous(alignment), Some(&previous), &short);
        assert!(!availability.is_available() && all_invalid(&state, 3, 3));
    }
    let edits: [&dyn Fn(&mut Frame); 3] = [
        &|f| f.generation.truncate(2),
        &|f| f.eligible.truncate(2),
        &|f| f.cloned.push(false),
    ];
    for edit in edits {
        let mut at = current.clone();
        edit(&mut at);
        for source in &every {
            let (availability, state) = fill(source, Some(&previous), &at);
            assert!(!availability.is_available());
            assert!(state.color.is_empty() && state.color_valid.is_empty());
            assert!(state.force.is_none() && state.phase_velocity.is_none());
        }
    }
    // The step after `u64::MAX` does not exist, so step 0 has no preceding one.
    let (mut last, mut zeroth) = (previous.clone(), current.clone());
    (last.step, zeroth.step) = (u64::MAX, 0);
    for source in every.iter().filter(|s| reads_previous(s)) {
        let (availability, state) = fill(source, Some(&last), &zeroth);
        assert!(!availability.is_available() && all_invalid(&state, 3, 3));
    }
    for factor in [f64::INFINITY, f64::NEG_INFINITY] {
        for source in &every {
            let mut state = FrameState::default();
            let availability = source.fill(Some(&previous), &current, factor, &mut state);
            assert!(!availability.is_available() && all_invalid(&state, 3, 3));
        }
    }
}

#[test]
fn gaussian_velocities_dephase_the_pair_contraction_by_the_thermal_factor() {
    // Independent `v ~ N(0, T)` per component under one force direction give
    // `E q = Σ_a r_a² E e^{iκ(v'_a − v_a)} = e^{−κ² T}`. The draws are a fixed
    // stream; the standard errors of the two means are 0.0037 and 0.0075.
    let (temperature, pairs) = (0.5f64, 4096u64);
    let normal = |k: u64| {
        let (radius, angle) = (unit(2 * k) + 0.5, unit(2 * k + 1) + 0.5);
        (-2. * temperature * (1. - radius).ln()).sqrt() * (TAU * angle).cos()
    };
    let force = [3., -4., 12.];
    let mut sum = C::ZERO;
    for pair in 0..pairs {
        let v: [f64; 3] = std::array::from_fn(|a| normal(6 * pair + a as u64));
        let w: [f64; 3] = std::array::from_fn(|a| normal(6 * pair + 3 + a as u64));
        let c = encode(&[force, force].concat(), &[v, w].concat(), 3, KAPPA);
        sum = sum + dot(&c[..3], &c[3..]);
    }
    let mean = sum * (1. / pairs as f64);
    assert!((mean.re - 0.782704538241868).abs() < 0.015, "{mean:?}");
    assert!(mean.im.abs() < 0.03, "{mean:?}");
}

/// Reward `stiffness · |x|² / 2` and its gradient; zero stiffness is a free gas.
#[derive(Clone)]
struct Quadratic {
    stiffness: f64,
}
impl RewardSource<f64> for Quadratic {
    fn id(&self) -> String {
        "test-quadratic/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            Ok(RewardBatch::new(
                x.values()
                    .chunks_exact(x.width())
                    .map(|r| 0.5 * self.stiffness * r.iter().map(|a| a * a).sum::<f64>())
                    .collect(),
                Provenance {
                    population_version: p.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}
impl GradientProvider<f64> for Quadratic {
    fn id(&self) -> String {
        "test-quadratic-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            let (d, eligible) = (x.width(), p.eligible(false));
            let mut g: Vec<f64> = x.values().iter().map(|a| self.stiffness * a).collect();
            for (row, _) in g.chunks_exact_mut(d).zip(&eligible).filter(|(_, e)| !**e) {
                row.fill(0.);
            }
            TensorBatch::vectors(p.len(), d, g)
        })
    }
}
/// Recorded steps of a three-dimensional gas from `[n, 3]` positions and velocities.
fn record(
    x: Vec<f64>,
    v: Vec<f64>,
    stiffness: f64,
    config: GasConfig,
    steps: usize,
) -> RunArchive<f64> {
    let n = x.len() / 3;
    let mut observations = ObservationBatch::positions(TensorBatch::vectors(n, 3, x).unwrap());
    observations
        .fields
        .insert("velocities".into(), TensorBatch::vectors(n, 3, v).unwrap());
    let reward = Quadratic { stiffness };
    block_on(async {
        let mut gas = GasBuilder::new(Population::new(observations).unwrap(), reward.clone())
            .gradient(reward)
            .config(config)
            .build()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..steps {
            gas.step().await.unwrap();
        }
        gas.stop_recording().unwrap()
    })
}
/// Six recorded steps of a 12-walker Euclidean Gas in three dimensions.
fn archive(viscosity: Option<ViscousForceConfig>) -> RunArchive<f64> {
    let x = (0..36).map(|k| unit(k as u64)).collect();
    let v = (0..36).map(|k| unit(1_000_003 + k as u64)).collect();
    let mut config = GasConfig::euclidean(3, 0.05).unwrap();
    config.seed = 7;
    config.qft.viscosity = viscosity;
    record(x, v, 1., config, 6)
}
fn dense() -> Option<ViscousForceConfig> {
    Some(ViscousForceConfig {
        coefficient: 1.,
        bandwidth: 1.,
        row_normalized: false,
    })
}
/// The records of a step the colour sources read, straight from the archive.
fn recorded_frame(gas: &GasConfig, s: &RecordedStep<f64>) -> Frame {
    let KineticKind::Baoab {
        positions,
        velocities,
        ..
    } = &gas.kinetic.integrator
    else {
        panic!("the Euclidean Gas integrates with BAOAB");
    };
    let x = s.before.observations.field(positions).unwrap();
    let v = s.before.observations.field(velocities).unwrap();
    let (n, d) = (s.before.len(), x.width());
    let stage = |label: &str| {
        let find = |name: &str| {
            s.field_evaluations
                .iter()
                .find(|f| f.stage == label && f.field == name)
        };
        let (f, w) = (find("viscous_force")?, find("force_input_velocity")?);
        let input = s
            .stages
            .iter()
            .find(|p| p.stage == format!("{label}_input") && p.version == f.version)?;
        (f.version == w.version && f.item_shape == [d] && w.item_shape == [d]).then(|| StageKick {
            force: f.values.clone(),
            velocity: w.values.clone(),
            available: (0..n).map(|i| f.available[i] && w.available[i]).collect(),
            generation: input.generations.clone(),
        })
    };
    let choices = &s.report.clone_plan.choices;
    Frame {
        step: s.report.step,
        epoch: s.epoch,
        n,
        d,
        x: x.values().to_vec(),
        v: Some(v.values().to_vec()),
        eligible: s.report.pre_clone_eligible.clone(),
        generation: s.before.generations.clone(),
        cloned: choices.iter().map(|c| c.accepted).collect(),
        revived: choices.iter().map(|c| c.revival).collect(),
        kick: Some(Kick {
            b1: stage("B1"),
            b2: stage("B2"),
        }),
        ..Frame::default()
    }
}

#[test]
fn matched_b1_colour_equals_the_recorded_colour_experiment_bit_for_bit() {
    let archive = archive(dense());
    let current = recorded_frame(&archive.gas_config, archive.steps.last().unwrap());
    current.validate().unwrap();
    let phase = PhaseScale {
        h_eff: 1.,
        mass: 1.,
        length: LengthScale::Fixed { value: 0.7 },
    };
    let k = kappa(&phase, phase_length(phase.length, &[]).unwrap()).unwrap();
    let mut state = FrameState::default();
    let matched = viscous(ColorAlignment::MatchedKick {
        stage: KickStage::B1,
    });
    assert!(matched.fill(None, &current, k, &mut state).is_available());

    let request = ExperimentRequest {
        experiment: 8,
        parameters: json!({}),
    };
    let details = analyze_archive(&request, Some(&archive)).unwrap().details;
    assert_eq!(details["kappa"].as_f64().unwrap().to_bits(), k.to_bits());
    let slots = details["source_slots"].as_array().unwrap();
    let mut compared = 0;
    for (row, slot) in slots.iter().enumerate() {
        let i = slot.as_u64().unwrap() as usize;
        let held = current.eligible[i] && !current.cloned[i];
        assert_eq!(
            state.color_valid[i],
            held && details["valid_mask"][row].as_bool().unwrap()
        );
        if !state.color_valid[i] {
            continue;
        }
        compared += 1;
        for a in 0..3 {
            let raw = &details["raw_color"][row][a];
            assert_eq!(
                [
                    state.color[i * 3 + a].re.to_bits(),
                    state.color[i * 3 + a].im.to_bits()
                ],
                [
                    raw["real"].as_f64().unwrap().to_bits(),
                    raw["imaginary"].as_f64().unwrap().to_bits()
                ]
            );
        }
    }
    assert!(compared >= 6, "{compared}");
}

#[test]
fn preceding_kick_is_the_matched_b2_colour_of_the_step_before_and_keeps_its_clones() {
    let archive = archive(dense());
    let frames: Vec<Frame> = archive
        .steps
        .iter()
        .map(|s| recorded_frame(&archive.gas_config, s))
        .collect();
    let (mut compared, mut cloned, mut kept_clones) = (0, 0, 0);
    for pair in frames.windows(2) {
        let (before, current) = (&pair[0], &pair[1]);
        let b2 = before.kick.as_ref().unwrap().b2.as_ref().unwrap();
        // The B2 population of a step is the pre-clone population of the next.
        assert_eq!(b2.generation, current.generation);
        let (availability, preceding) = fill(
            &viscous(ColorAlignment::PrecedingKick),
            Some(before),
            current,
        );
        assert!(availability.is_available());
        let (_, matched) = fill(
            &viscous(ColorAlignment::MatchedKick {
                stage: KickStage::B2,
            }),
            None,
            before,
        );
        let (_, first) = fill(
            &viscous(ColorAlignment::MatchedKick {
                stage: KickStage::B1,
            }),
            None,
            current,
        );
        for i in 0..current.n {
            cloned += usize::from(current.cloned[i]);
            let b1 = current.kick.as_ref().unwrap().b1.as_ref().unwrap();
            assert_eq!(
                first.color_valid[i],
                b1.available[i] && current.eligible[i] && !current.cloned[i]
            );
            kept_clones += usize::from(before.cloned[i] && preceding.color_valid[i]);
            if preceding.color_valid[i] && matched.color_valid[i] {
                compared += 1;
                let row = i * 3..(i + 1) * 3;
                assert_eq!(
                    bits(&preceding.color[row.clone()]),
                    bits(&matched.color[row])
                );
            }
        }
    }
    assert!(compared >= 30, "{compared}");
    assert!(cloned >= 1 && kept_clones >= 1, "{cloned} {kept_clones}");
}

#[test]
fn preceding_force_differs_from_the_preceding_kick_by_the_phase_of_the_velocity_change() {
    let archive = archive(dense());
    let frames: Vec<Frame> = archive
        .steps
        .iter()
        .map(|s| recorded_frame(&archive.gas_config, s))
        .collect();
    let (before, current) = (&frames[3], &frames[4]);
    let (_, kicked) = fill(
        &viscous(ColorAlignment::PrecedingKick),
        Some(before),
        current,
    );
    let (_, forced) = fill(
        &viscous(ColorAlignment::PrecedingForce),
        Some(before),
        current,
    );
    assert_eq!(kicked.color_valid, forced.color_valid);
    assert_eq!(kicked.force, forced.force);
    assert_eq!(forced.phase_velocity.as_ref(), current.v.as_ref());
    let input = &before.kick.as_ref().unwrap().b2.as_ref().unwrap().velocity;
    let mut moved = 0.;
    for k in (0..36).filter(|k| kicked.color_valid[k / 3]) {
        let change = current.v.as_ref().unwrap()[k] - input[k];
        moved = f64::max(moved, change.abs());
        near(
            forced.color[k],
            kicked.color[k] * C::phase(KAPPA * change),
            1e-13,
        );
    }
    assert!(moved > 1e-3);
}

#[test]
fn a_gas_without_viscosity_has_a_force_record_and_no_valid_colour() {
    let archive = archive(None);
    let frames: Vec<Frame> = archive
        .steps
        .iter()
        .map(|s| recorded_frame(&archive.gas_config, s))
        .collect();
    let (availability, state) = fill(
        &viscous(ColorAlignment::PrecedingKick),
        Some(&frames[1]),
        &frames[2],
    );
    assert!(availability.is_available());
    assert_eq!(state.color_valid, vec![false; 12]);
    assert_eq!(state.color, vec![C::ZERO; 36]);
    assert_eq!(state.force, Some(vec![0.; 36]));
    assert!(state.force_valid.iter().any(|&v| v));
    let (availability, state) = fill(&viscous(ColorAlignment::PrecedingKick), None, &frames[0]);
    assert!(!availability.is_available() && all_invalid(&state, 12, 3));
}

#[test]
fn two_walker_run_has_the_closed_form_colour_of_every_alignment() {
    // Noiseless frictionless BAOAB, h = 0.1, two walkers leaving the origin
    // back to back: every force is along the first axis, so `c_0 = −e^{iκv}`
    // and `c_1 = e^{−iκv}` with `v` the phase velocity of walker 0.
    let config = GasConfig {
        precision: Precision::F64,
        qft: QftExecutionConfig {
            viscosity: Some(ViscousForceConfig {
                coefficient: 2.,
                bandwidth: 1.,
                row_normalized: false,
            }),
            ..Default::default()
        },
        kinetic: KineticOperator {
            integrator: KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.1,
                friction: 0.,
            },
            noise: Noise {
                innovation: InnovationLaw::Gaussian,
                geometry: NoiseGeometry::Isotropic {
                    scale: FactorValues::Constant { values: vec![0.] },
                },
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let archive = record(vec![0.; 6], vec![1., 0., 0., -1., 0., 0.], 0., config, 2);
    let frames: Vec<Frame> = archive
        .steps
        .iter()
        .map(|s| recorded_frame(&archive.gas_config, s))
        .collect();
    let (first, second) = (&frames[0], &frames[1]);
    assert!(!first.cloned.iter().chain(&second.cloned).any(|&c| c));
    let b2 = first.kick.as_ref().unwrap().b2.as_ref().unwrap();
    assert_eq!((b2.force[0], b2.velocity[0]), (-1.771074925690515, 0.9));
    assert_eq!(second.v.as_ref().unwrap()[0], 0.8114462537154743);

    let matched = |stage| viscous(ColorAlignment::MatchedKick { stage });
    let cases = [
        (matched(KickStage::B1), None, first, 1.),
        (matched(KickStage::B2), None, first, 0.9),
        (
            viscous(ColorAlignment::PrecedingKick),
            Some(first),
            second,
            0.9,
        ),
        (
            viscous(ColorAlignment::PrecedingForce),
            Some(first),
            second,
            0.8114462537154743,
        ),
        (
            viscous(ColorAlignment::ReferenceOffset),
            None,
            second,
            0.8114462537154743,
        ),
        (matched(KickStage::B1), None, second, 0.8114462537154743),
        (matched(KickStage::B2), None, second, 0.7316055807431976),
    ];
    let expected = [
        (-0.764842187284488, -0.644217687237691),
        (-0.808027508312152, -0.58914475794227),
        (-0.808027508312152, -0.58914475794227),
        (-0.842971896178942, -0.537957602653294),
        (-0.842971896178942, -0.537957602653294),
        (-0.842971896178942, -0.537957602653294),
        (-0.871705697122301, -0.490029772161368),
    ];
    for ((source, previous, at, velocity), (re, im)) in cases.iter().zip(expected) {
        let (availability, state) = fill(source, *previous, at);
        assert!(availability.is_available());
        assert_eq!(state.color_valid, vec![true; 2]);
        assert_eq!(state.phase_velocity.as_ref().unwrap()[0], *velocity);
        let c = &state.color;
        close(c[0], re, im);
        close(c[3], -re, im);
        near(c[0], -C::phase(KAPPA * velocity), 1e-15);
        let (q01, q10) = (dot(&c[..3], &c[3..]), dot(&c[3..], &c[..3]));
        near(q01, -C::phase(-2. * KAPPA * velocity), 1e-15);
        assert_eq!(q01.im + q10.im, 0.);
    }
    let c = encode(&b2.force, &b2.velocity, 3, KAPPA);
    close(dot(&c[..3], &c[3..]), -0.305816908378289, 0.952090341590516);
    let (availability, state) = fill(&viscous(ColorAlignment::PrecedingKick), None, first);
    assert!(!availability.is_available() && all_invalid(&state, 2, 3));
}
