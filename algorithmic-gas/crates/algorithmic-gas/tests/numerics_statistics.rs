use algorithmic_gas::{
    GasError,
    physics::{
        numerics::{
            BlockMoments, BlockSize, LagMoments, ResampleKind, Resampling, Samples, SeriesView,
            Subtraction, TauInt, Whitener, auto_block, chi2_q, cross_moments, errors, gamma_q,
            moments::straddling_pairs, resample, resample::block_below_target, resample_blocks,
            sample_covariance, series_moments, special::ln_gamma, submatrix, tau_int,
            tau_int_of_correlator,
        },
        qft::math::Rng,
    },
    random::{RandomStream, Stream},
};
use std::cell::RefCell;

const X: [f64; 12] = [1., 3., 2., 5., 4., 6., 3., 7., 5., 8., 6., 9.];
const D: [f64; 8] = [1., 2., 4., 3., 0., 5., 2., 1.];
fn view<'a>(
    values: &'a [f64],
    weight: &'a [f64],
    segment: &'a [u32],
    components: usize,
) -> SeriesView<'a> {
    SeriesView {
        values,
        weight,
        segment,
        components,
    }
}
fn close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * e.abs().max(1.),
            "entry {i}: {a} vs {e}"
        );
    }
}
fn some(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().map(|x| x.unwrap()).collect()
}
fn jackknife(frames: usize) -> Resampling {
    Resampling::BlockJackknife {
        block: BlockSize::Fixed { frames },
    }
}
fn ar1(seed: u64, frames: usize, phi: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut x = vec![rng.normal() / (1. - phi * phi).sqrt()];
    for t in 1..frames {
        x.push(phi * x[t - 1] + rng.normal());
    }
    x
}
/// Per-origin sums written out from the definition, the oracle of the
/// blocked builder: `ab`, `n` are `[lags]`, `a`, `b` are `[lags, components]`.
fn origin(a: SeriesView<'_>, b: SeriesView<'_>, t: usize, lags: usize) -> LagMoments {
    let c = a.components;
    let mut sums = LagMoments {
        ab: vec![0.; lags],
        a: vec![0.; lags * c],
        b: vec![0.; lags * c],
        n: vec![0.; lags],
    };
    for lag in 0..lags {
        let s = t + lag;
        if s >= a.frames() || a.segment[s] != a.segment[t] {
            continue;
        }
        let w = a.weight[t] * b.weight[s];
        if w == 0. {
            continue;
        }
        sums.n[lag] = w;
        for k in 0..c {
            let (x, y) = (a.values[t * c + k], b.values[s * c + k]);
            sums.ab[lag] += w * x * y;
            sums.a[lag * c + k] = w * x;
            sums.b[lag * c + k] = w * y;
        }
    }
    sums
}

#[test]
fn direct_lag_sums_book_every_pair_into_the_block_of_its_origin() {
    let (w, g) = ([1.; 12], [0; 12]);
    let m = series_moments(view(&X, &w, &g, 1), 2, 3).unwrap();
    m.validate().unwrap();
    assert_eq!(
        (m.blocks, m.lags, m.origins_per_block, m.open),
        (4, 3, 3, 3)
    );
    assert_eq!(
        m.ab,
        [
            14., 19., 25., 77., 62., 84., 83., 96., 101., 181., 102., 72.
        ]
    );
    assert_eq!(
        m.a,
        [6., 6., 6., 15., 15., 15., 15., 15., 15., 23., 14., 8.]
    );
    assert_eq!(
        m.b,
        [6., 10., 11., 15., 13., 16., 15., 20., 19., 23., 15., 9.]
    );
    assert_eq!(m.n, [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2., 1.]);
    close(
        &some(&m.estimate(Subtraction::None)),
        &[29.5833333333333, 25.3636363636364, 28.2],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[5.40972222222222, 1.39669421487603, 4.],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[5.40972222222222, 1.2645202020202, 3.69861111111111],
        1e-13,
    );
    let pooled = m.pooled(Some(3));
    assert_eq!(pooled.ab, [174., 177., 210.]);
    assert_eq!(pooled.n, [9., 9., 9.]);
    assert_eq!(m.pooled(None), m.weighted(&[1.; 4]));
}

#[test]
fn masked_frames_and_segment_breaks_drop_exactly_their_pairs() {
    let mut x = X;
    x[3] = f64::NAN;
    let mut w = [1.; 12];
    w[3] = 0.;
    let g = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let m = series_moments(view(&x, &w, &g, 1), 2, 3).unwrap();
    assert_eq!(
        m.ab,
        [14., 9., 10., 52., 24., 0., 83., 96., 101., 181., 102., 72.]
    );
    assert_eq!(m.a, [6., 4., 3., 10., 4., 0., 15., 15., 15., 23., 14., 8.]);
    assert_eq!(m.b, [6., 5., 6., 10., 6., 0., 15., 20., 19., 23., 15., 9.]);
    assert_eq!(m.n, [3., 2., 2., 2., 1., 0., 3., 3., 3., 3., 2., 1.]);
    close(
        &some(&m.estimate(Subtraction::None)),
        &[30., 28.875, 30.5],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[5.90082644628099, 2.28125, 5.94444444444445],
        1e-13,
    );
    let s = resample(&m, Subtraction::LagMeans, &jackknife(3), None).unwrap();
    close(
        &s.values,
        &[
            3.5,
            -0.583333333333329,
            3.,
            6.98765432098765,
            2.63265306122449,
            5.94444444444445,
            7.109375,
            4.12,
            9.,
            3.609375,
            1.69444444444444,
            4.2,
        ],
        1e-12,
    );
    close(
        &sample_covariance(&s),
        &[
            9.16516322592485,
            7.55277949354896,
            10.3353467399691,
            7.55277949354896,
            8.74273886575036,
            10.9212407407407,
            10.3353467399691,
            10.9212407407407,
            15.2867361111111,
        ],
        1e-12,
    );
    let s = resample(&m, Subtraction::GlobalMean, &jackknife(3), None).unwrap();
    close(
        &s.central,
        &[5.90082644628099, 2.04235537190083, 5.50826446280992],
        1e-13,
    );
    close(
        &s.values,
        &[
            3.5,
            -1.,
            2.75,
            6.98765432098765,
            2.48853615520282,
            5.51234567901235,
            7.109375,
            3.965625,
            8.84895833333333,
            3.609375,
            1.640625,
            3.890625,
        ],
        1e-12,
    );
    close(
        &sample_covariance(&s),
        &[
            9.16516322592485,
            7.79253792528215,
            10.314603243018,
            7.79253792528215,
            9.76998493499041,
            11.3934973170845,
            10.314603243018,
            11.3934973170845,
            15.8394193505391,
        ],
        1e-12,
    );
}

#[test]
fn every_jackknife_resample_recomputes_its_own_disconnected_part() {
    let (w, g) = ([1.; 12], [0; 12]);
    let m = series_moments(view(&X, &w, &g, 1), 2, 3).unwrap();
    let s = resample(&m, Subtraction::LagMeans, &jackknife(3), None).unwrap();
    s.validate().unwrap();
    assert_eq!(
        (s.kind, s.count, s.blocks, s.effective_block, s.tau_int),
        (ResampleKind::Jackknife, 4, 4, 3, None)
    );
    assert_eq!(s.defined, [true; 3]);
    close(&s.central, &some(&m.estimate(Subtraction::LagMeans)), 1e-15);
    close(
        &s.values,
        &[
            3.20987654320987,
            -0.5,
            2.59183673469388,
            6.98765432098765,
            2.515625,
            5.20408163265306,
            6.32098765432098,
            2.09375,
            4.55102040816326,
            3.33333333333333,
            0.555555555555557,
            2.88888888888889,
        ],
        1e-12,
    );
    for block in 0..4 {
        let direct = m.pooled(Some(block)).estimate(Subtraction::LagMeans);
        close(s.sample(block), &some(&direct), 1e-13);
    }
    close(
        &sample_covariance(&s),
        &[
            8.75445816186558,
            5.9309413580247,
            5.59914336104812,
            5.9309413580247,
            4.37280442979601,
            3.87053792871314,
            5.59914336104812,
            3.87053792871314,
            3.61870343375444,
        ],
        1e-12,
    );
    let raw = resample(&m, Subtraction::None, &jackknife(3), None).unwrap();
    close(
        &sample_covariance(&raw),
        &[
            132.118055555556,
            88.78125,
            89.4107142857143,
            88.78125,
            69.4140625,
            70.6651785714286,
            89.4107142857143,
            70.6651785714286,
            75.8928571428572,
        ],
        1e-12,
    );
    let global = resample(&m, Subtraction::GlobalMean, &jackknife(3), None).unwrap();
    close(
        &global.values,
        &[
            3.20987654320987,
            -0.543209876543209,
            2.40917107583774,
            6.98765432098765,
            2.13734567901235,
            4.69488536155203,
            6.32098765432098,
            2.16512345679012,
            4.36155202821869,
            3.33333333333333,
            0.555555555555557,
            2.88888888888889,
        ],
        1e-12,
    );
    close(
        &sample_covariance(&global),
        &[
            8.75445816186558,
            5.48602537722909,
            4.87311385459534,
            5.48602537722909,
            3.90398876886145,
            3.21742112482853,
            4.87311385459534,
            3.21742112482853,
            2.77647777217883,
        ],
        1e-12,
    );
}

#[test]
fn a_zero_weight_frame_is_still_an_origin_of_its_block() {
    let g = [0; 8];
    let m = series_moments(view(&D, &[1.; 8], &g, 1), 3, 2).unwrap();
    assert_eq!(
        m.ab,
        [
            5., 10., 10., 3., 25., 12., 15., 26., 25., 10., 5., 0., 5., 2., 0., 0.
        ]
    );
    assert_eq!(
        m.b,
        [
            3., 6., 7., 3., 7., 3., 5., 7., 5., 7., 3., 1., 3., 1., 0., 0.
        ]
    );
    assert_eq!(
        m.n,
        [
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 1., 0., 0.
        ]
    );
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[2.4375, -1.04081632653061, -1.25, 1.4],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[2.4375, -1.00892857142857, -1.1875, 1.4125],
        1e-13,
    );
    let s = resample(&m, Subtraction::LagMeans, &jackknife(2), None).unwrap();
    close(
        &s.values,
        &[
            2.91666666666667,
            -1.36,
            -1.,
            2.44444444444444,
            2.47222222222222,
            -1.2,
            -1.25,
            -0.333333333333333,
            1.13888888888889,
            0.,
            -1.25,
            1.,
            2.91666666666667,
            -1.33333333333333,
            -1.25,
            1.4,
        ],
        1e-12,
    );
    close(
        &some(&errors(&s)),
        &[
            1.26197963240006,
            0.978979060041633,
            0.1875,
            1.72312700247738,
        ],
        1e-12,
    );
    assert!((sample_covariance(&s)[1] + 1.22222222222222).abs() < 1e-12);
    let s = resample(&m, Subtraction::GlobalMean, &jackknife(2), None).unwrap();
    close(
        &some(&errors(&s)),
        &[
            1.26197963240006,
            0.962042941022057,
            0.196908984427192,
            1.45559862637717,
        ],
        1e-12,
    );
    assert!((sample_covariance(&s)[1] + 1.21064814814815).abs() < 1e-12);
    let mut w = [1.; 8];
    w[3] = 0.;
    let m = series_moments(view(&D, &w, &g, 1), 3, 2).unwrap();
    assert_eq!(
        m.ab,
        [
            5., 10., 4., 0., 16., 0., 0., 20., 25., 10., 5., 0., 5., 2., 0., 0.
        ]
    );
    assert_eq!(
        m.n,
        [
            2., 2., 1., 1., 1., 0., 1., 1., 2., 2., 2., 1., 2., 1., 0., 0.
        ]
    );
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[
            2.69387755102041,
            -1.29387755102041,
            -2.26530612244898,
            2.68707482993197,
        ],
        1e-13,
    );
    let s = resample(&m, Subtraction::LagMeans, &jackknife(2), None).unwrap();
    close(
        &s.values,
        &[
            3.44,
            -2.22222222222222,
            -1.33333333333333,
            4.,
            2.47222222222222,
            -1.2,
            -1.66666666666667,
            -0.5,
            1.2,
            0.111111111111111,
            -3.,
            2.5,
            3.44,
            -1.5,
            -2.125,
            2.66666666666667,
        ],
        1e-12,
    );
    close(
        &some(&errors(&s)),
        &[
            1.59257674679886,
            1.46302420171876,
            1.08418436124428,
            2.85043856274785,
        ],
        1e-12,
    );
    let g = [0, 0, 0, 0, 0, 1, 1, 1];
    let m = series_moments(view(&D, &w, &g, 1), 3, 8).unwrap();
    assert_eq!(m.blocks, 2);
    let pooled = m.pooled(None);
    assert_eq!(pooled.ab, [51., 22., 9., 0.]);
    assert_eq!(pooled.a, [15., 10., 10., 2.]);
    assert_eq!(pooled.b, [15., 9., 5., 0.]);
    assert_eq!(pooled.n, [7., 4., 3., 1.]);
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[
            2.69387755102041,
            -0.0867346938775517,
            -3.12244897959184,
            0.306122448979592,
        ],
        1e-13,
    );
}

#[test]
fn a_lag_whose_pairs_live_in_one_block_is_undefined_under_delete_one() {
    let a = [11. / 3., 2., 10. / 3., 2. / 3., 11. / 3., 3.];
    let g = [0, 0, 0, 0, 1, 1];
    let m = series_moments(view(&a, &[1.; 6], &g, 1), 3, 2).unwrap();
    assert_eq!(m.pooled(None).n, [6., 4., 2., 1.]);
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[377. / 324., -23. / 72., 10. / 9., 0.],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[377. / 324., -343. / 648., 167. / 162., -629. / 324.],
        1e-13,
    );
    let s = resample(&m, Subtraction::LagMeans, &jackknife(2), None).unwrap();
    assert_eq!(s.defined, [true, true, false, false]);
    assert_eq!(&s.central[2..], [0., 0.]);
    assert!(s.values.chunks_exact(4).all(|row| row[2..] == [0., 0.]));
    let covariance = sample_covariance(&s);
    assert!((0..4).all(|i| covariance[2 * 4 + i] == 0. && covariance[i * 4 + 3] == 0.));
    assert!(covariance[0] > 0. && covariance[5] > 0.);
    let e = errors(&s);
    assert!(e[0].is_some() && e[1].is_some() && e[2].is_none() && e[3].is_none());
    // Fractional weights: the delete-one denominator is exactly zero, not the
    // rounding residue of a total minus a block.
    let thirds = series_moments(view(&a, &[1. / 3.; 6], &g, 1), 3, 2).unwrap();
    let s = resample(&thirds, Subtraction::GlobalMean, &jackknife(2), None).unwrap();
    assert_eq!(s.defined, [true, true, false, false]);
    for block in 0..3 {
        let direct = thirds.pooled(Some(block)).estimate(Subtraction::GlobalMean);
        close(&s.sample(block)[..2], &some(&direct[..2]), 1e-13);
    }
    let empty = series_moments(view(&a, &[0.; 6], &g, 1), 3, 2).unwrap();
    assert_eq!(empty.estimate(Subtraction::None), [None; 4]);
    assert_eq!(empty.estimate(Subtraction::GlobalMean), [None; 4]);
}

#[test]
fn contracted_vector_correlator_is_the_sum_of_component_correlators() {
    let y: Vec<f64> = (0..12).flat_map(|t| [X[t], 0.5 * X[11 - t]]).collect();
    let first: Vec<f64> = y.iter().step_by(2).copied().collect();
    let second: Vec<f64> = y.iter().skip(1).step_by(2).copied().collect();
    let (w, g) = ([1.; 12], [0; 12]);
    for subtraction in [
        Subtraction::None,
        Subtraction::LagMeans,
        Subtraction::GlobalMean,
    ] {
        let joint = some(
            &series_moments(view(&y, &w, &g, 2), 2, 3)
                .unwrap()
                .estimate(subtraction),
        );
        let parts: Vec<f64> = (0..3)
            .map(|lag| {
                [&first, &second]
                    .iter()
                    .map(|x| {
                        let m = series_moments(view(x, &w, &g, 1), 2, 3).unwrap();
                        m.estimate(subtraction)[lag].unwrap()
                    })
                    .sum()
            })
            .collect();
        close(&joint, &parts, 1e-13);
    }
    let joint = series_moments(view(&y, &w, &g, 2), 2, 3).unwrap();
    close(
        &some(&joint.estimate(Subtraction::LagMeans)),
        &[6.76215277777777, 1.74586776859504, 5.],
        1e-13,
    );
    let cross = |a: &[f64], b: &[f64]| {
        let m = cross_moments(view(a, &w, &g, 1), view(b, &w, &g, 1), 2, 3).unwrap();
        some(&m.estimate(Subtraction::LagMeans))
    };
    close(
        &cross(&first, &second),
        &[-2.58680555555556, -0.785123966942152, -2.08],
        1e-13,
    );
    close(
        &cross(&second, &first),
        &[-2.58680555555556, -0.673553719008265, -2.025],
        1e-13,
    );
    let y = [1., -1., 2., 0.5, 0., 1.5, 3., -2., 1., 1., 2., 0.];
    let m = series_moments(view(&y, &[1.; 6], &[0; 6], 2), 2, 6).unwrap();
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[2.33333333333333, -1.79, 0.53125],
        1e-13,
    );
}

#[test]
fn global_mean_of_a_cross_correlator_centres_each_leg_on_its_own_mean() {
    let a = [1., 4., 2., 8., 5., 7., 3., 6.];
    let b = [2., 0., 5., 1., 1., 6., 4., 3.];
    let mut wa = [1.; 8];
    let mut wb = [1.; 8];
    (wa[2], wb[6]) = (0., 0.);
    let g = [0; 8];
    let m = cross_moments(view(&a, &wa, &g, 1), view(&b, &wb, &g, 1), 3, 3).unwrap();
    let valid: Vec<usize> = (0..8).filter(|&t| wa[t] * wb[t] > 0.).collect();
    let mean_a = valid.iter().map(|&t| a[t]).sum::<f64>() / valid.len() as f64;
    let mean_b = valid.iter().map(|&t| b[t]).sum::<f64>() / valid.len() as f64;
    let expected: Vec<f64> = (0..4)
        .map(|lag| {
            let pairs: Vec<usize> = (0..8 - lag).filter(|&t| wa[t] * wb[t + lag] > 0.).collect();
            pairs
                .iter()
                .map(|&t| (a[t] - mean_a) * (b[t + lag] - mean_b))
                .sum::<f64>()
                / pairs.len() as f64
        })
        .collect();
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &expected,
        1e-13,
    );
}

#[test]
fn connected_estimates_ignore_a_shift_and_scale_with_the_square() {
    let (w, g) = ([1.; 12], [0; 12]);
    let estimate = |x: &[f64], subtraction| {
        some(
            &series_moments(view(x, &w, &g, 1), 4, 3)
                .unwrap()
                .estimate(subtraction),
        )
    };
    let shifted: Vec<f64> = X.iter().map(|x| x + 37.5).collect();
    let scaled: Vec<f64> = X.iter().map(|x| 4. * x).collect();
    for subtraction in [Subtraction::LagMeans, Subtraction::GlobalMean] {
        close(
            &estimate(&shifted, subtraction),
            &estimate(&X, subtraction),
            1e-11,
        );
        let quadrupled: Vec<f64> = estimate(&X, subtraction).iter().map(|c| 16. * c).collect();
        assert_eq!(estimate(&scaled, subtraction), quadrupled);
        assert_eq!(estimate(&[3.; 12], subtraction), [0.; 5]);
    }
    assert_eq!(estimate(&[3.; 12], Subtraction::None), [9.; 5]);
    let alternating: Vec<f64> = (0..12).map(|t| if t % 2 == 0 { 1. } else { -1. }).collect();
    assert_eq!(
        estimate(&alternating, Subtraction::None),
        [1., -1., 1., -1., 1.]
    );
}

#[test]
fn streamed_origins_with_adaptive_pair_merging_equal_the_coarsened_builder() {
    let mut w = [1.; 12];
    w[7] = 0.;
    let g = [0; 12];
    let series = view(&X, &w, &g, 1);
    let mut streamed = BlockMoments::new(3, 1, 1, 4).unwrap();
    for t in 0..12 {
        let sums = origin(series, series, t, 3);
        streamed
            .push_origin(&sums.ab, &sums.a, &sums.b, &sums.n)
            .unwrap();
    }
    assert_eq!(streamed, series_moments(series, 2, 4).unwrap());
    let fine = series_moments(series, 2, 1).unwrap();
    let coarse = fine.coarsen(4).unwrap();
    assert_eq!(
        (&coarse.ab, &coarse.a, &coarse.b, &coarse.n),
        (&streamed.ab, &streamed.a, &streamed.b, &streamed.n)
    );
    assert_eq!(
        (coarse.blocks, coarse.origins_per_block, coarse.open),
        (3, 4, 4)
    );
    assert_eq!(
        coarse.estimate(Subtraction::LagMeans),
        fine.estimate(Subtraction::LagMeans)
    );
    let odd = series_moments(view(&X[..11], &w[..11], &g[..11], 1), 2, 1).unwrap();
    let mut merged = odd.clone();
    merged.merge_pairs();
    assert_eq!(merged, odd.coarsen(2).unwrap());
    assert_eq!(
        (
            merged.blocks,
            merged.origins_per_block,
            merged.open,
            merged.origins()
        ),
        (6, 2, 1, 11)
    );
    merged.validate().unwrap();
    let sums = origin(series, series, 11, 3);
    merged
        .push_origin(&sums.ab, &sums.a, &sums.b, &sums.n)
        .unwrap();
    assert_eq!((merged.blocks, merged.open), (6, 2));
    assert!(odd.coarsen(0).is_err());
}

#[test]
fn a_gap_closes_the_open_block_and_runs_join_block_by_block() {
    let g = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1];
    let series = view(&X, &[1.; 12], &g, 1);
    let built = series_moments(series, 2, 4).unwrap();
    assert_eq!((built.blocks, built.open, built.origins()), (4, 3, 15));
    assert_eq!(&built.n[3..9], [1., 0., 0., 4., 4., 4.]);
    let mut streamed = BlockMoments::new(3, 1, 4, 4).unwrap();
    streamed.close_block();
    for t in 0..12 {
        if t == 5 {
            streamed.close_block();
        }
        let sums = origin(series, series, t, 3);
        streamed
            .push_origin(&sums.ab, &sums.a, &sums.b, &sums.n)
            .unwrap();
    }
    assert_eq!(streamed, built);
    let before = streamed.clone();
    assert!(matches!(
        streamed.push_origin(&[0.; 2], &[0.; 3], &[0.; 3], &[0.; 3]),
        Err(GasError::Configuration(_))
    ));
    assert!(
        streamed
            .push_origin(&[0.; 3], &[0.; 3], &[0.; 3], &[0., -1., 0.])
            .is_err()
    );
    assert!(
        streamed
            .push_origin(&[f64::NAN, 0., 0.], &[0.; 3], &[0.; 3], &[0.; 3])
            .is_err()
    );
    assert_eq!(streamed, before);
    let other = series_moments(view(&D, &[1.; 8], &[0; 8], 1), 2, 4).unwrap();
    let joint = BlockMoments::concat(&[&built, &other]).unwrap();
    joint.validate().unwrap();
    assert_eq!((joint.blocks, joint.open, joint.max_blocks), (6, 4, 6));
    let (p, q, r) = (built.pooled(None), other.pooled(None), joint.pooled(None));
    assert_eq!(
        r.ab,
        [p.ab[0] + q.ab[0], p.ab[1] + q.ab[1], p.ab[2] + q.ab[2]]
    );
    assert_eq!(r.n, [20., 17., 14.]);
    assert!(BlockMoments::concat(&[]).is_err());
    let mismatched = series_moments(series, 2, 2).unwrap();
    assert!(matches!(
        BlockMoments::concat(&[&built, &mismatched]),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn block_moments_reject_malformed_shapes_and_round_trip_through_json() {
    for (lags, components, origins, max_blocks) in [
        (0, 1, 1, 4),
        (1, 0, 1, 4),
        (1, 1, 0, 4),
        (1, 1, 1, 2),
        (1, 1, 1, 5),
        (4098, 1, 1, 4),
    ] {
        assert!(matches!(
            BlockMoments::new(lags, components, origins, max_blocks),
            Err(GasError::Configuration(_))
        ));
    }
    let m = series_moments(view(&X, &[1.; 12], &[0; 12], 1), 2, 3).unwrap();
    let text = serde_json::to_string(&m).unwrap();
    assert_eq!(serde_json::from_str::<BlockMoments>(&text).unwrap(), m);
    let unknown = text.replacen('{', r#"{"extra":1,"#, 1);
    assert!(serde_json::from_str::<BlockMoments>(&unknown).is_err());
    let tau = TauInt {
        tau: 2.5,
        window: 13,
    };
    let text = serde_json::to_string(&tau).unwrap();
    assert_eq!(text, r#"{"tau":2.5,"window":13}"#);
    assert_eq!(serde_json::from_str::<TauInt>(&text).unwrap(), tau);
    assert!(serde_json::from_str::<TauInt>(r#"{"tau":2.5,"window":13,"x":0}"#).is_err());
    let (w, g) = ([1.; 12], [0; 12]);
    let series = view(&X, &w, &g, 1);
    assert!(series_moments(series, 4097, 1).is_err());
    assert!(series_moments(series, 2, 0).is_err());
    assert!(cross_moments(series, view(&X, &w, &[1; 12], 1), 2, 1).is_err());
    assert!(cross_moments(series, view(&X, &w[..6], &g[..6], 2), 2, 1).is_err());
    assert!(series_moments(view(&[f64::NAN; 12], &w, &g, 1), 2, 1).is_err());
}

#[test]
fn jackknife_of_a_block_mean_is_the_variance_of_the_mean() {
    let values = [1., 2., 4., 7.];
    let mean = |m: &[f64]| {
        let n: f64 = m.iter().sum();
        vec![(n > 0.).then(|| m.iter().zip(&values).map(|(m, x)| m * x).sum::<f64>() / n)]
    };
    let s = resample_blocks(4, 1, 4, &Resampling::Uncorrelated, None, &mean).unwrap();
    assert_eq!(s.central, [3.5]);
    close(&s.values, &[13. / 3., 4., 10. / 3., 7. / 3.], 1e-15);
    assert!((sample_covariance(&s)[0] - 1.75).abs() < 1e-14);
    assert!((errors(&s)[0].unwrap() - 1.75f64.sqrt()).abs() < 1e-14);
    let table = |kind| Samples {
        kind,
        dimension: 2,
        count: 4,
        central: vec![3., 0.],
        values: vec![1., 5., 2., 5., 4., 5., 7., 5.],
        defined: vec![true, false],
        effective_block: 1,
        blocks: 4,
        tau_int: None,
    };
    assert_eq!(
        sample_covariance(&table(ResampleKind::Bootstrap)),
        [7., 0., 0., 0.]
    );
    assert_eq!(
        sample_covariance(&table(ResampleKind::Jackknife)),
        [15.75, 0., 0., 0.]
    );
    assert_eq!(
        errors(&table(ResampleKind::Bootstrap)),
        [Some(7f64.sqrt()), None]
    );
    let reordered = Samples {
        values: vec![7., 5., 1., 5., 4., 5., 2., 5.],
        ..table(ResampleKind::Jackknife)
    };
    assert_eq!(sample_covariance(&reordered), [15.75, 0., 0., 0.]);
}

#[test]
fn bootstrap_draws_whole_blocks_from_the_seed_and_reproduces_the_point_estimate() {
    let (w, g) = ([1.; 12], [0; 12]);
    let m = series_moments(view(&X, &w, &g, 1), 2, 1).unwrap();
    let bootstrap = |seed| Resampling::Bootstrap {
        block: BlockSize::Fixed { frames: 3 },
        samples: 64,
        seed,
    };
    let draws = |seed, scale: f64| {
        let seen = RefCell::new(Vec::new());
        let statistic = |multiplicity: &[f64]| {
            seen.borrow_mut().push(multiplicity.to_vec());
            vec![Some(scale * multiplicity[0])]
        };
        let s = resample_blocks(12, 1, 12, &bootstrap(seed), None, &statistic).unwrap();
        assert_eq!(
            (s.kind, s.count, s.blocks, s.effective_block),
            (ResampleKind::Bootstrap, 64, 4, 3)
        );
        seen.into_inner()
    };
    let seen = draws(9, 1.);
    assert_eq!(seen.len(), 65);
    assert_eq!(seen[0], [1.; 12]);
    assert_eq!(seen, draws(9, -2.));
    assert_ne!(seen, draws(10, 1.));
    for multiplicity in &seen {
        assert_eq!(multiplicity.iter().sum::<f64>(), 12.);
        assert!(
            multiplicity
                .chunks_exact(3)
                .all(|group| group[0] == group[1] && group[1] == group[2])
        );
    }
    let s = resample(&m, Subtraction::LagMeans, &bootstrap(9), None).unwrap();
    let coarse = m.coarsen(3).unwrap();
    let mut permutations = 0;
    for (r, multiplicity) in seen[1..].iter().enumerate() {
        let groups: Vec<f64> = multiplicity.iter().step_by(3).copied().collect();
        let expected = coarse.weighted(&groups).estimate(Subtraction::LagMeans);
        assert_eq!(s.sample(r), some(&expected));
        if groups == [1.; 4] {
            permutations += 1;
            assert_eq!(s.sample(r), s.central);
        }
    }
    assert!(permutations > 0);
    assert_eq!(s.central, some(&coarse.estimate(Subtraction::LagMeans)));
}

#[test]
fn resampling_needs_two_blocks_and_an_autocorrelation_time_for_automatic_blocks() {
    let (w, g) = ([1.; 12], [0; 12]);
    let m = series_moments(view(&X, &w, &g, 1), 2, 3).unwrap();
    assert!(matches!(
        resample(&m, Subtraction::LagMeans, &jackknife(12), None),
        Err(GasError::Numerical(_))
    ));
    let ones = |m: &[f64]| vec![Some(m.iter().sum())];
    let auto = Resampling::default();
    assert!(matches!(
        resample_blocks(12, 1, 12, &auto, None, &ones),
        Err(GasError::Configuration(_))
    ));
    assert!(resample_blocks(12, 1, 12, &auto, Some(0.2), &ones).is_err());
    assert!(resample_blocks(12, 0, 12, &jackknife(1), None, &ones).is_err());
    assert!(matches!(
        resample_blocks(1, 1, 12, &jackknife(1), None, &ones),
        Err(GasError::Numerical(_))
    ));
    let few = Resampling::Bootstrap {
        block: BlockSize::Auto,
        samples: 2,
        seed: 1,
    };
    assert!(matches!(
        resample(&m, Subtraction::LagMeans, &few, None),
        Err(GasError::Configuration(_))
    ));
    let ragged = |m: &[f64]| vec![Some(1.); if m[0] == 0. { 2 } else { 1 }];
    assert!(matches!(
        resample_blocks(12, 1, 12, &jackknife(1), None, &ragged),
        Err(GasError::Shape(_))
    ));
    let s = resample_blocks(12, 2, 24, &auto, Some(0.5), &ones).unwrap();
    assert_eq!((s.effective_block, s.blocks, s.tau_int), (2, 12, Some(0.5)));
    let s = resample(&m, Subtraction::LagMeans, &auto, None).unwrap();
    assert_eq!((s.effective_block, s.blocks), (3, 4));
    assert!(s.tau_int.is_some_and(|tau| tau >= 0.5));
}

#[test]
fn automatic_block_rounds_the_batch_size_rule_up_to_whole_stored_blocks() {
    for (tau, base, origins, block) in [
        (19.5, 1, 1000, 116),
        (19.5, 8, 1000, 120),
        (100., 1, 1000, 125),
        (0.5, 1, 2000, 13),
        (3.2, 4, 64, 8),
        (9.5, 1, 40, 5),
        (9.5, 1, 4000, 114),
        (19.5, 1, 4000, 183),
        (9.48203755648973, 1, 4000, 113),
        (11.7781737864465, 1, 1000, 83),
    ] {
        assert_eq!(
            auto_block(tau, base, origins),
            block,
            "{tau} {base} {origins}"
        );
    }
    // The cap that keeps eight blocks binds before the batch-size rule is
    // reached on the rows above whose block is origins / 8; the returned
    // block says nothing about it and `block_below_target` does.
    for (tau, base, origins) in [
        (100., 1, 1000),
        (9.5, 1, 40),
        (3.2, 4, 64),
        (f64::MAX, 1, 1000),
        (1e150, 3, 1000),
    ] {
        let block = auto_block(tau, base, origins);
        assert!(
            block_below_target(tau, block, origins),
            "{tau} {base} {origins} {block}"
        );
    }
    for (tau, base, origins) in [
        (19.5, 1, 1000),
        (19.5, 8, 1000),
        (0.5, 1, 2000),
        (9.5, 1, 4000),
        (19.5, 1, 4000),
        (9.48203755648973, 1, 4000),
        (11.7781737864465, 1, 1000),
        (0.5, 1, 1000),
        (4., 1, 8000),
        (13.5, 1, 1000),
    ] {
        let block = auto_block(tau, base, origins);
        assert!(
            !block_below_target(tau, block, origins),
            "{tau} {base} {origins} {block}"
        );
    }
    assert_eq!(auto_block(f64::NAN, 0, 0), 1);
    // Targets that are whole numbers: the block is decided by b^3 >= (2 tau)^2 T,
    // not by the last digit of a cube root.
    for (tau, base, origins, block) in [
        (0.5, 1, 1000, 10),
        (0.5, 1, 8000, 20),
        (0.5, 1, 1001, 11),
        (4., 1, 1000, 40),
        (4., 1, 8000, 80),
        (0.5, 7, 1000, 14),
        (13.5, 1, 1000, 90),
        (f64::MAX, 1, 1000, 125),
        (1e150, 3, 1000, 123),
    ] {
        assert_eq!(
            auto_block(tau, base, origins),
            block,
            "{tau} {base} {origins}"
        );
    }
}

#[test]
fn integrated_autocorrelation_time_of_seeded_autoregressive_series() {
    // Madras-Sokal error of the estimate: tau sqrt(2 (2 W + 1) / T) = 2.1 at
    // phi = 0.9, so 9.48 against the exact 9.5 is a fixed-seed regression value.
    for (seed, frames, phi, head, correlator, tau, window) in [
        (
            2024,
            4000,
            0.9,
            [1.4782211867701, 1.16291541978438, -0.430036482476639],
            [
                5.35717238309346,
                4.81839190934984,
                4.36019297725294,
                3.97328985691411,
            ],
            9.48203755648973,
            48,
        ),
        (
            7,
            1000,
            0.95,
            [1.27544662484993, 1.97761271522735, -0.720776870966221],
            [
                12.3666157178275,
                11.8408554268332,
                11.2796553162798,
                10.7788968204521,
            ],
            11.7781737864465,
            61,
        ),
        (
            3,
            2000,
            0.,
            [0.753249877418509, -1.19636878009446, 0.527846736446005],
            [
                1.01969268071294,
                -0.0305666055768644,
                0.0132398462357857,
                -0.0371016592698856,
            ],
            0.5,
            3,
        ),
    ] {
        let x = ar1(seed, frames, phi);
        let (w, g) = (vec![1.; frames], vec![0; frames]);
        let series = view(&x, &w, &g, 1);
        close(&x[..3], &head, 1e-12);
        let estimate = series_moments(series, 100, frames)
            .unwrap()
            .estimate(Subtraction::LagMeans);
        close(&some(&estimate[..4]), &correlator, 1e-9);
        let t = tau_int(series).unwrap();
        assert!((t.tau - tau).abs() < 1e-9 && t.window == window, "{t:?}");
        assert_eq!(tau_int_of_correlator(&estimate, frames), Some(t));
        let short = tau_int_of_correlator(&estimate[..11], frames).unwrap();
        assert!(short.window <= 10 && short.tau <= t.tau + 1e-12);
    }
    assert_eq!(tau_int(view(&[1., 2., 3.], &[1.; 3], &[0; 3], 1)), None);
    assert_eq!(tau_int(view(&[2.; 8], &[1.; 8], &[0; 8], 1)), None);
    assert_eq!(tau_int(view(&[1., 2.], &[1.; 3], &[0; 3], 1)), None);
    assert_eq!(tau_int_of_correlator(&[None, Some(1.)], 100), None);
    assert_eq!(tau_int_of_correlator(&[Some(0.), Some(1.)], 100), None);
    assert_eq!(
        tau_int_of_correlator(&[Some(2.), Some(1.), None, Some(1.)], 100),
        Some(TauInt { tau: 1., window: 1 })
    );
}

#[test]
fn autocorrelation_time_of_a_masked_segmented_vector_series_matches_its_lag_sums() {
    let frames = 600;
    let (x, y) = (ar1(11, frames, 0.8), ar1(12, frames, 0.5));
    let values: Vec<f64> = (0..frames).flat_map(|t| [x[t], y[t]]).collect();
    let weight: Vec<f64> = (0..frames)
        .map(|t| if t % 17 == 3 { 0. } else { 1. })
        .collect();
    let segment: Vec<u32> = (0..frames).map(|t| (t / 250) as u32).collect();
    let series = view(&values, &weight, &segment, 2);
    let estimate = series_moments(series, 120, 50)
        .unwrap()
        .estimate(Subtraction::LagMeans);
    let t = tau_int(series).unwrap();
    let from_lags = tau_int_of_correlator(&estimate, frames).unwrap();
    assert_eq!(t.window, from_lags.window);
    assert!((t.tau - from_lags.tau).abs() < 1e-12);
    // The contraction weights (1 + phi) / (2 (1 - phi)) = 4.5 and 1.5 by the
    // variances 1 / (1 - phi^2) = 2.78 and 1.33: tau = 3.5.
    assert!(t.tau > 1. && t.tau < 9. && t.window < 120);
}

#[test]
fn block_jackknife_errors_of_seeded_autoregressive_correlators() {
    let run = |seed, frames, phi, resampling: &Resampling| {
        let x = ar1(seed, frames, phi);
        let (w, g) = (vec![1.; frames], vec![0; frames]);
        let series = view(&x, &w, &g, 1);
        let tau = tau_int(series).unwrap().tau;
        let m = series_moments(series, 10, 1).unwrap();
        resample(&m, Subtraction::LagMeans, resampling, Some(tau)).unwrap()
    };
    let auto = Resampling::default();
    for (seed, frames, phi, resampling, block, blocks, error, covariance) in [
        (
            2024,
            4000,
            0.9,
            &auto,
            113,
            36,
            [
                0.390580503485537,
                0.388071636350906,
                0.381983578537157,
                0.374284458172954,
            ],
            0.151487838384831,
        ),
        (
            7,
            1000,
            0.95,
            &auto,
            83,
            13,
            [
                1.68297478846383,
                1.70653195872635,
                1.72144962958041,
                1.70535067644027,
            ],
            2.86943999747314,
        ),
        (
            3,
            2000,
            0.,
            &auto,
            13,
            154,
            [
                0.031487897689737,
                0.024062276124203,
                0.023897595600324,
                0.02116834971058,
            ],
            -0.000210695688156268,
        ),
        (
            2024,
            4000,
            0.9,
            &jackknife(19),
            19,
            211,
            [
                0.337323584631164,
                0.336859240842418,
                0.331566001635189,
                0.32568155997872,
            ],
            0.113310093844438,
        ),
        (
            2024,
            4000,
            0.9,
            &jackknife(38),
            38,
            106,
            [
                0.348866420200186,
                0.35101356514286,
                0.350058043706739,
                0.348295419926248,
            ],
            0.122253123596227,
        ),
        (
            7,
            1000,
            0.95,
            &jackknife(24),
            24,
            42,
            [
                1.42094699214553,
                1.40849926545561,
                1.38266800589282,
                1.35281607619002,
            ],
            1.99899446978702,
        ),
        (
            7,
            1000,
            0.95,
            &jackknife(48),
            48,
            21,
            [
                1.50390309241069,
                1.50912674752784,
                1.51612587932936,
                1.51720020557601,
            ],
            2.26806325496816,
        ),
        (
            3,
            2000,
            0.,
            &Resampling::Uncorrelated,
            1,
            2000,
            [
                0.0325858603772913,
                0.0234935807053714,
                0.0232389614631745,
                0.0221994509101926,
            ],
            -2.20104121394449e-05,
        ),
    ] {
        let s = run(seed, frames, phi, resampling);
        assert_eq!(
            (s.effective_block, s.blocks, s.count),
            (block, blocks, blocks)
        );
        close(&some(&errors(&s)[..4]), &error, 1e-8);
        assert!((sample_covariance(&s)[1] - covariance).abs() < 1e-8 * covariance.abs().max(1.));
    }
    let fixed = sample_covariance(&run(2024, 4000, 0.9, &jackknife(19)));
    assert!((fixed[2 * 11 + 5] - 0.0991817898435915).abs() < 1e-9);
    let s = run(2024, 4000, 0.9, &auto);
    assert!((sample_covariance(&s)[2 * 11 + 5] - 0.135188936908419).abs() < 1e-9);
    // Exact autocovariance phi^l / (1 - phi^2): the seeded estimate sits 0.24
    // jackknife errors above it at lag 0.
    let e = errors(&s);
    for (lag, e) in e.iter().enumerate() {
        let exact = 0.9f64.powi(lag as i32) / (1. - 0.81);
        assert!(
            (s.central[lag] - exact).abs() < 3. * e.unwrap(),
            "lag {lag}"
        );
    }
    // Short blocks cut the correlation between neighbours and shrink the error.
    let short = errors(&run(7, 1000, 0.95, &jackknife(10)));
    let long = errors(&run(7, 1000, 0.95, &auto));
    assert!(short[0].unwrap() < 0.85 * long[0].unwrap());
}

#[test]
fn block_bootstrap_errors_agree_with_the_block_jackknife() {
    let frames = 4000;
    let x = ar1(2024, frames, 0.9);
    let (w, g) = (vec![1.; frames], vec![0; frames]);
    let m = series_moments(view(&x, &w, &g, 1), 10, 1).unwrap();
    let bootstrap = Resampling::Bootstrap {
        block: BlockSize::Fixed { frames: 38 },
        samples: 200,
        seed: 17,
    };
    let s = resample(&m, Subtraction::LagMeans, &bootstrap, None).unwrap();
    assert_eq!(
        (s.kind, s.count, s.blocks, s.effective_block),
        (ResampleKind::Bootstrap, 200, 106, 38)
    );
    assert_eq!(
        s,
        resample(&m, Subtraction::LagMeans, &bootstrap, Some(3.)).unwrap()
    );
    let reference = errors(&resample(&m, Subtraction::LagMeans, &jackknife(38), None).unwrap());
    // The error of an error from 200 draws has relative sd 1 / sqrt(400) = 5 %;
    // the band is six of them.
    for (boot, jack) in errors(&s).iter().zip(&reference) {
        let ratio = boot.unwrap() / jack.unwrap();
        assert!((0.7..1.3).contains(&ratio), "{ratio}");
    }
    let covariance = sample_covariance(&s);
    assert!((0..11).all(|i| (0..11).all(|j| covariance[i * 11 + j] == covariance[j * 11 + i])));
}

#[test]
fn block_bootstrap_keeps_lag_products_at_their_separation_and_frame_draws_do_not() {
    let frames = 2000;
    let x = ar1(11, frames, 0.9);
    let (w, g) = (vec![1.; frames], vec![0; frames]);
    let ratio = |c: &[f64]| c[1] / c[0];
    let m = series_moments(view(&x, &w, &g, 1), 1, 1).unwrap();
    let central = ratio(&some(&m.estimate(Subtraction::LagMeans)));
    assert!((central - 0.881556411342408).abs() < 1e-9, "{central}");
    // The ratio estimate has sd sqrt((1 - 0.81) / 2000) = 0.01; the band is
    // five of them. A block of one origin still owns its lag products.
    for (block, blocks) in [(50, 40), (1, 2000)] {
        let bootstrap = Resampling::Bootstrap {
            block: BlockSize::Fixed { frames: block },
            samples: 64,
            seed: 5,
        };
        let s = resample(&m, Subtraction::LagMeans, &bootstrap, None).unwrap();
        assert_eq!(
            (s.count, s.blocks, s.defined.as_slice()),
            (64, blocks, &[true; 2][..])
        );
        for r in 0..s.count {
            let resampled = ratio(s.sample(r));
            assert!(
                (resampled - central).abs() < 0.05,
                "{block} {r} {resampled}"
            );
        }
    }
    // Frames drawn one by one before the products are formed lose the signal.
    let mut rng = Rng::new(5);
    let drawn: Vec<f64> = (0..frames)
        .map(|_| x[((rng.uniform() * frames as f64) as usize).min(frames - 1)])
        .collect();
    let shuffled = series_moments(view(&drawn, &w, &g, 1), 1, 1).unwrap();
    let lost = ratio(&some(&shuffled.estimate(Subtraction::LagMeans)));
    assert!((lost + 0.0142842561553619).abs() < 1e-9, "{lost}");
}

#[test]
fn global_mean_of_a_masked_propagator_subtracts_the_single_source_mean() {
    // Per source frame: n, ab, a, b over lags 0..=3; a gap follows frame 3.
    let origins: [[[f64; 4]; 4]; 6] = [
        [
            [3., 1., 3., 3.],
            [49., 4., 48., 12.],
            [11., 2., 11., 11.],
            [11., 2., 11., 2.],
        ],
        [
            [2., 2., 2., 0.],
            [8., 12., 0., 0.],
            [4., 4., 4., 0.],
            [4., 6., 0., 0.],
        ],
        [
            [3., 3., 0., 0.],
            [44., 12., 0., 0.],
            [10., 10., 0., 0.],
            [10., 2., 0., 0.],
        ],
        [
            [3., 0., 0., 0.],
            [4., 0., 0., 0.],
            [2., 0., 0., 0.],
            [2., 0., 0., 0.],
        ],
        [
            [3., 3., 0., 0.],
            [49., 35., 0., 0.],
            [11., 11., 0., 0.],
            [11., 9., 0., 0.],
        ],
        [
            [3., 0., 0., 0.],
            [33., 0., 0., 0.],
            [9., 0., 0., 0.],
            [9., 0., 0., 0.],
        ],
    ];
    let mut m = BlockMoments::new(4, 1, 2, 4).unwrap();
    for (t, [n, ab, a, b]) in origins.iter().enumerate() {
        if t == 4 {
            m.close_block();
        }
        m.push_origin(ab, a, b, n).unwrap();
    }
    let pooled = m.pooled(None);
    assert_eq!(pooled.n, [17., 9., 5., 3.]);
    assert_eq!(pooled.ab, [187., 63., 48., 12.]);
    assert_eq!(pooled.a, [47., 27., 15., 11.]);
    assert_eq!(pooled.b, [47., 19., 11., 2.]);
    close(
        &some(&m.estimate(Subtraction::None)),
        &[11., 7., 9.6, 4.],
        1e-14,
    );
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[970. / 289., 2. / 3., 3., 14. / 9.],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[970. / 289., 1334. / 2601., 4143. / 1445., -292. / 867.],
        1e-13,
    );
    // Lags 2 and 3 have sinks in the first block only.
    let s = resample(&m, Subtraction::GlobalMean, &jackknife(2), None).unwrap();
    assert_eq!(
        (s.blocks, s.defined.as_slice()),
        (3, &[true, true, false, false][..])
    );
    // Without frames 2 and 3: N = [11, 6], AB(1) = 51, A = [35, 17], B(1) = 17.
    let mean = 35. / 11.;
    let expected = 51. / 6. - mean * (17. / 6. + 17. / 6.) + mean * mean;
    assert!((s.sample(1)[1] - expected).abs() < 1e-13);
}

#[test]
fn whitener_floors_the_correlation_spectrum_and_reports_the_rank() {
    let sigma = [0.5, 0.2, 0.1, 0.05];
    let r = [0.3, -0.1, 0.05, 0.02];
    let covariance: Vec<f64> = (0..16)
        .map(|k| sigma[k / 4] * sigma[k % 4] * 0.95f64.powi((k / 4).abs_diff(k % 4) as i32))
        .collect();
    let spectrum = [
        0.0299717308135491,
        0.0506086841758458,
        0.162653269186451,
        3.75676631582415,
    ];
    for (cut, floor, rank, chi2, condition) in [
        (
            0.,
            3.75676631582415e-12,
            4,
            21.9102564102564,
            125.343655966837,
        ),
        (0.05, 0.187838315791208, 1, 4.19741649662648, 20.),
        (0.3, 1.12702989474725, 1, 0.75363895617296, 3.33333333333333),
    ] {
        let w = Whitener::new(&covariance, 4, cut).unwrap();
        close(w.eigenvalues(), &spectrum, 1e-12);
        assert!((w.floor() - floor).abs() < 1e-12 * floor);
        assert_eq!((w.n(), w.rank()), (4, rank));
        assert!((w.chi2(&r) - chi2).abs() < 1e-10 * chi2, "{}", w.chi2(&r));
        assert!((w.condition().unwrap() - condition).abs() < 1e-10 * condition);
    }
    let z: Vec<f64> = r.iter().zip(&sigma).map(|(r, s)| r / s).collect();
    let closed = (z[0] * z[0] + z[3] * z[3] + (1. + 0.95 * 0.95) * (z[1] * z[1] + z[2] * z[2])
        - 2. * 0.95 * (z[0] * z[1] + z[1] * z[2] + z[2] * z[3]))
        / (1. - 0.95 * 0.95);
    let w = Whitener::new(&covariance, 4, 0.).unwrap();
    assert!((w.chi2(&r) - closed).abs() < 1e-10 * closed);
    let identity: Vec<f64> = (0..16)
        .map(|k| f64::from(u8::from(k / 4 == k % 4)))
        .collect();
    let transform = w.apply_columns(&identity, 4);
    assert_eq!(w.apply(&r), w.apply_columns(&r, 1));
    for i in 0..4 {
        for j in 0..4 {
            let product: f64 = (0..4)
                .flat_map(|k| (0..4).map(move |l| (k, l)))
                .map(|(k, l)| transform[k * 4 + i] * transform[k * 4 + l] * covariance[l * 4 + j])
                .sum();
            assert!(
                (product - f64::from(u8::from(i == j))).abs() < 1e-10,
                "{i} {j}"
            );
        }
    }
    let scale = [1e-7, 3., 2e5, 1e-3];
    let scaled: Vec<f64> = (0..16)
        .map(|k| covariance[k] * scale[k / 4] * scale[k % 4])
        .collect();
    let residual: Vec<f64> = r.iter().zip(&scale).map(|(r, s)| r * s).collect();
    for cut in [0., 0.05] {
        let (a, b) = (
            Whitener::new(&covariance, 4, cut).unwrap(),
            Whitener::new(&scaled, 4, cut).unwrap(),
        );
        assert!((a.chi2(&r) - b.chi2(&residual)).abs() < 1e-9 * a.chi2(&r));
        close(a.eigenvalues(), b.eigenvalues(), 1e-12);
    }
    let sub = submatrix(&covariance, 4, &[3, 1]);
    assert_eq!(
        sub,
        [covariance[15], covariance[13], covariance[7], covariance[5]]
    );
    let pair = Whitener::new(&sub, 2, 0.).unwrap();
    let (x, y, rho) = (r[3] / sigma[3], r[1] / sigma[1], 0.95f64.powi(2));
    let expected = (x * x + y * y - 2. * rho * x * y) / (1. - rho * rho);
    assert!((pair.chi2(&[r[3], r[1]]) - expected).abs() < 1e-10 * expected);
}

#[test]
fn whitener_handles_tiny_scales_rank_deficiency_and_rejects_bad_covariances() {
    let sigma = [0.5, 0.2, 0.1, 0.05];
    let tiny: Vec<f64> = (0..16)
        .map(|k| 1e-14 * sigma[k / 4] * sigma[k % 4] * 0.95f64.powi((k / 4).abs_diff(k % 4) as i32))
        .collect();
    let w = Whitener::new(&tiny, 4, 0.).unwrap();
    close(
        w.eigenvalues(),
        &[
            0.0299717308135491,
            0.0506086841758458,
            0.162653269186451,
            3.75676631582415,
        ],
        1e-12,
    );
    let mut rng = Rng::new(7);
    let values: Vec<f64> = (0..40).map(|_| rng.normal()).collect();
    let s = Samples {
        kind: ResampleKind::Jackknife,
        dimension: 8,
        count: 5,
        central: vec![0.; 8],
        values,
        defined: vec![true; 8],
        effective_block: 1,
        blocks: 5,
        tau_int: None,
    };
    let deficient = Whitener::new(&sample_covariance(&s), 8, 1e-6).unwrap();
    assert_eq!(deficient.rank(), 4);
    assert!(deficient.eigenvalues()[..4].iter().all(|v| v.abs() < 1e-12));
    assert!((deficient.condition().unwrap() - 1e6).abs() < 1e-3);
    let w = Whitener::diagonal(&[0.5, 2.]).unwrap();
    assert_eq!(w.apply(&[1., 1.]), [2., 0.5]);
    assert_eq!((w.rank(), w.floor(), w.condition()), (2, 0., Some(1.)));
    assert_eq!(w.chi2(&[1., 1.]), 4.25);
    assert_eq!(w.eigenvalues(), [1., 1.]);
    for (covariance, n, cut) in [
        (vec![1., 0., 0., 1.], 2, 1.),
        (vec![1.; 3], 2, 0.),
        (vec![], 0, 0.),
    ] {
        assert!(matches!(
            Whitener::new(&covariance, n, cut),
            Err(GasError::Configuration(_))
        ));
    }
    for covariance in [
        [1., 0., 0., 0.],
        [1., 0.5, 0.2, 1.],
        [1., f64::NAN, f64::NAN, 1.],
        [-1., 0., 0., 1.],
    ] {
        assert!(matches!(
            Whitener::new(&covariance, 2, 0.),
            Err(GasError::Numerical(_))
        ));
    }
    assert!(matches!(
        Whitener::diagonal(&[1., 0.]),
        Err(GasError::Numerical(_))
    ));
    assert!(matches!(
        Whitener::diagonal(&[]),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn incomplete_gamma_function_matches_reference_values_and_closed_forms() {
    for (a, x, q) in [
        (0.5, 0.5, 0.317310507862911),
        (1., 1., 0.367879441171442),
        (2.5, 1., 0.84914503608461),
        (5., 5., 0.440493285065213),
        (10., 3., 0.998897511869885),
        (10., 30., 7.12175086281559e-06),
        (50., 40., 0.929664933340605),
        (0.5, 8., 6.33424836662399e-05),
        (3., 2.5, 0.54381311588333),
        (100., 150., 5.92454033548392e-06),
        (1.5, 0.01, 0.999252244660609),
        (4., 3.9, 0.453246760138729),
        (4., 5.1, 0.251268264577879),
        (0.25, 1e-08, 0.988967373508857),
        (30.5, 80., 8.11689833493877e-11),
        (3., 0., 1.),
        (2., 3., 0.199148273471456),
        (0.5, 2., 0.0455002638963584),
    ] {
        let value = gamma_q(a, x).unwrap();
        assert!((value - q).abs() < 1e-12 * q, "Q({a}, {x}) = {value}");
    }
    for x in [0.1f64, 0.9, 1.7, 4., 12.5, 60.] {
        let check = |a: f64, exact: f64| {
            let value = gamma_q(a, x).unwrap();
            assert!(
                (value - exact).abs() < 1e-12 * exact,
                "Q({a}, {x}) = {value}"
            );
        };
        check(1., (-x).exp());
        check(2., (1. + x) * (-x).exp());
        check(3., (1. + x + x * x / 2.) * (-x).exp());
        assert!((chi2_q(2. * x, 2).unwrap() - (-x).exp()).abs() < 1e-12 * (-x).exp());
    }
    for (x, value) in [
        (0.5, 0.5723649429247),
        (1., 0.),
        (1.5, -0.120782237635245),
        (3., std::f64::consts::LN_2),
        (10., 12.8018274800815),
        (100.5, 361.435540467778),
        (0.001, 6.90717888538385),
        (0.1, 2.25271265173421),
    ] {
        assert!(
            (ln_gamma(x).unwrap() - value).abs() < 1e-13 * value.abs().max(1.),
            "{x}"
        );
    }
    for (chi2, dof, q) in [
        (10., 10, 0.440493285065213),
        (3.2, 5, 0.669182902033243),
        (25., 8, 0.00155455784301107),
        (1., 1, 0.317310507862911),
        (7.3, 2, 0.0259911287787554),
        (120., 80, 0.00254819230361334),
    ] {
        assert!(
            (chi2_q(chi2, dof).unwrap() - q).abs() < 1e-12 * q,
            "{chi2} {dof}"
        );
    }
    for (a, x) in [
        (0., 1.),
        (-1., 1.),
        (1., -0.5),
        (f64::NAN, 1.),
        (1., f64::INFINITY),
    ] {
        assert_eq!(gamma_q(a, x), None);
    }
    assert_eq!(
        (ln_gamma(0.), ln_gamma(-2.), ln_gamma(f64::INFINITY)),
        (None, None, None)
    );
    assert_eq!(
        (chi2_q(1., 0), chi2_q(-1., 3), chi2_q(f64::NAN, 3)),
        (None, None, None)
    );
    assert!(gamma_q(1e6, 1e6).is_some_and(|q| (q - 0.5).abs() < 1e-3));
    assert_eq!((ln_gamma(1e306), gamma_q(1e306, 1.)), (None, None));
    assert!(ln_gamma(1e300).is_some_and(f64::is_finite));
}

#[test]
fn fractional_pair_weights_enter_every_sum_and_cancel_when_uniform() {
    // Exact fractions from the definition, every weight dyadic.
    let w = [0.5, 2., 1., 0.25, 1.5, 0., 1., 0.75];
    let g = [0; 8];
    let m = series_moments(view(&D, &w, &g, 1), 2, 3).unwrap();
    let pooled = m.pooled(None);
    assert_eq!(pooled.n, [9.125, 4.375, 4.]);
    assert_eq!(pooled.ab, [37.375, 22.5, 5.]);
    assert_eq!(pooled.a, [15., 8.625, 7.5]);
    assert_eq!(pooled.b, [15., 11.5, 6.5]);
    close(
        &some(&m.estimate(Subtraction::None)),
        &[299. / 73., 36. / 7., 1.25],
        1e-14,
    );
    close(
        &some(&m.estimate(Subtraction::LagMeans)),
        &[7427. / 5329., -48. / 1225., -115. / 64.],
        1e-13,
    );
    close(
        &some(&m.estimate(Subtraction::GlobalMean)),
        &[7427. / 5329., 10572. / 37303., -38395. / 21316.],
        1e-13,
    );
    // The origin carries the first series' weight, the partner the second's.
    let e = [2., 0., 5., 1., 1., 6., 4., 3.];
    let v = [1., 0.5, 0., 2., 0.25, 1., 3., 0.5];
    let cross = cross_moments(view(&D, &w, &g, 1), view(&e, &v, &g, 1), 2, 3).unwrap();
    let pooled = cross.pooled(None);
    assert_eq!(pooled.n, [5.75, 4.3125, 9.]);
    assert_eq!(pooled.ab, [27.625, 11.1875, 13.5]);
    assert_eq!(pooled.a, [10.375, 9.4375, 9.75]);
    assert_eq!(pooled.b, [15., 12.5625, 23.75]);
    close(
        &some(&cross.estimate(Subtraction::LagMeans)),
        &[103. / 1058., -2000. / 529., -587. / 432.],
        1e-13,
    );
    close(
        &some(&cross.estimate(Subtraction::GlobalMean)),
        &[103. / 1058., -11629. / 3174., -52583. / 38088.],
        1e-13,
    );
    // A uniform weight cancels between the sums and the pair count.
    let unit = series_moments(view(&D, &[1.; 8], &g, 1), 2, 3).unwrap();
    for subtraction in [
        Subtraction::None,
        Subtraction::LagMeans,
        Subtraction::GlobalMean,
    ] {
        let halves = series_moments(view(&D, &[0.5; 8], &g, 1), 2, 3).unwrap();
        assert_eq!(halves.estimate(subtraction), unit.estimate(subtraction));
        let thirds = series_moments(view(&D, &[1. / 3.; 8], &g, 1), 2, 3).unwrap();
        close(
            &some(&thirds.estimate(subtraction)),
            &some(&unit.estimate(subtraction)),
            1e-13,
        );
    }
}

#[test]
fn blocks_cut_by_misaligned_segment_breaks_are_all_held() {
    let g = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let m = series_moments(view(&X, &[1.; 12], &g, 1), 1, 2).unwrap();
    assert_eq!(
        (m.blocks, m.origins_per_block, m.max_blocks, m.open),
        (8, 2, 8, 1)
    );
    assert_eq!(
        m.n,
        [
            2., 2., 1., 0., 2., 2., 1., 0., 2., 2., 1., 0., 2., 2., 1., 0.
        ]
    );
    assert_eq!(
        m.ab,
        [
            10., 9., 4., 0., 41., 44., 36., 0., 58., 56., 25., 0., 100., 102., 81., 0.
        ]
    );
    let s = resample(&m, Subtraction::LagMeans, &jackknife(2), None).unwrap();
    assert_eq!((s.blocks, s.count, s.effective_block), (8, 8, 2));
    for block in 0..8 {
        let direct = m.pooled(Some(block)).estimate(Subtraction::LagMeans);
        close(s.sample(block), &some(&direct), 1e-13);
    }
}

#[test]
fn autocorrelation_window_closes_at_equality_and_stops_at_half_the_frames() {
    // rho = 1/4, 1/8, 1/8, 0, 0, 1/8: tau(5) = 1 and 5 = 5 tau closes the window.
    let correlator = [8., 2., 1., 1., 0., 0., 1.].map(Some);
    assert_eq!(
        tau_int_of_correlator(&correlator, 100),
        Some(TauInt { tau: 1., window: 5 })
    );
    assert_eq!(
        tau_int_of_correlator(&correlator, 6),
        Some(TauInt { tau: 1., window: 3 })
    );
    assert_eq!(
        tau_int_of_correlator(&correlator, 4),
        Some(TauInt {
            tau: 0.875,
            window: 2
        })
    );
    assert_eq!(tau_int_of_correlator(&correlator, 3), None);
    // A ramp of m frames has lag-wise connected variance (m^2 - 1) / 12, so
    // rho(l) = ((8 - l)^2 - 1) / 63 and the window never closes within the
    // 8 / 2 lags searched: tau = 1/2 + (48 + 35 + 24 + 15) / 63.
    let ramp: Vec<f64> = (0..8).map(f64::from).collect();
    let t = tau_int(view(&ramp, &[1.; 8], &[0; 8], 1)).unwrap();
    assert!(
        (t.tau - 307. / 126.).abs() < 1e-13 && t.window == 4,
        "{t:?}"
    );
    // Four valid frames are the least that give an estimate.
    let t = tau_int(view(&ramp[..4], &[1.; 4], &[0; 4], 1)).unwrap();
    assert!((t.tau - (0.5 + 8. / 15. + 3. / 15.)).abs() < 1e-13 && t.window == 2);
    assert_eq!(
        tau_int(view(&ramp[..5], &[1., 1., 0., 1., 0.], &[0; 5], 1)),
        None
    );
}

#[test]
fn a_fixed_block_rounds_up_to_whole_stored_blocks() {
    let m = series_moments(view(&X, &[1.; 12], &[0; 12], 1), 2, 3).unwrap();
    for (frames, block, groups, held) in [
        (1, 3, 4, [1., 1., 1., 0.]),
        (4, 6, 2, [1., 1., 0., 0.]),
        (5, 6, 2, [1., 1., 0., 0.]),
        (7, 9, 2, [1., 1., 1., 0.]),
    ] {
        let s = resample(&m, Subtraction::LagMeans, &jackknife(frames), None).unwrap();
        assert_eq!(
            (s.effective_block, s.blocks, s.count),
            (block, groups, groups)
        );
        let last = m.weighted(&held).estimate(Subtraction::LagMeans);
        close(s.sample(groups - 1), &some(&last), 1e-13);
    }
}

#[test]
fn bootstrap_resample_draws_its_blocks_from_the_addressed_stream() {
    let (seed, blocks, group) = (41, 10, 3);
    let bootstrap = Resampling::Bootstrap {
        block: BlockSize::Fixed { frames: group },
        samples: 16,
        seed,
    };
    let multiplicities = |m: &[f64]| m.iter().map(|m| Some(*m)).collect();
    let s = resample_blocks(blocks, 1, blocks, &bootstrap, None, &multiplicities).unwrap();
    assert_eq!((s.count, s.blocks, s.dimension), (16, 4, blocks));
    assert_eq!(s.central, [1.; 10]);
    for r in 0..s.count {
        let mut stream = RandomStream::new(seed, 0, Stream::Initialize, r as u64, 811);
        let mut drawn = [0.; 4];
        for _ in 0..4 {
            drawn[stream.index(4)] += 1.;
        }
        let expected: Vec<f64> = (0..blocks).map(|stored| drawn[stored / group]).collect();
        assert_eq!(s.sample(r), expected, "resample {r}");
    }
    assert!((1..s.count).any(|r| s.sample(r) != s.sample(0)));
}

#[test]
fn automatic_block_without_an_override_uses_the_pooled_autocorrelation_time() {
    let frames = 4000;
    let x = ar1(2024, frames, 0.9);
    let (w, g) = (vec![1.; frames], vec![0; frames]);
    let m = series_moments(view(&x, &w, &g, 1), 100, 1).unwrap();
    let s = resample(&m, Subtraction::None, &Resampling::default(), None).unwrap();
    assert_eq!((s.effective_block, s.blocks), (113, 36));
    assert!(
        s.tau_int
            .is_some_and(|tau| (tau - 9.48203755648973).abs() < 1e-9)
    );
    // No variance: tau = 1/2 and the block is 64^(1/3) = 4.
    let flat = series_moments(view(&[2.; 64], &[1.; 64], &[0; 64], 1), 3, 1).unwrap();
    let s = resample(&flat, Subtraction::LagMeans, &Resampling::default(), None).unwrap();
    assert_eq!((s.effective_block, s.blocks, s.tau_int), (4, 16, Some(0.5)));
    assert_eq!(errors(&s), [Some(0.); 4]);
}

#[test]
fn whitener_maps_each_column_of_a_tall_matrix_like_a_residual() {
    let w = Whitener::diagonal(&[0.5, 2., 4.]).unwrap();
    assert_eq!(
        w.apply_columns(&[1., 2., 3., 4., 5., 6.], 2),
        [2., 4., 1.5, 2., 1.25, 1.5]
    );
    let covariance = [4., 1.2, 1.2, 1.];
    let w = Whitener::new(&covariance, 2, 0.).unwrap();
    let design = [1., -2., 0.5, 3., 4., -1.];
    let whitened = w.apply_columns(&design, 3);
    // chi2 of each column against the closed-form inverse of a 2 x 2 matrix.
    let det = 4. * 1. - 1.2 * 1.2;
    for c in 0..3 {
        let (x, y) = (design[c], design[3 + c]);
        let expected = (x * x - 2. * 1.2 * x * y + 4. * y * y) / det;
        let chi2 = whitened[c].powi(2) + whitened[3 + c].powi(2);
        assert!((chi2 - expected).abs() < 1e-12 * expected, "{c}");
        assert!((w.chi2(&[x, y]) - expected).abs() < 1e-12 * expected);
    }
}

#[test]
fn bootstrap_multiplicities_scale_every_block_sum() {
    // Blocks of the first test: block 0 twice, block 1 never, blocks 2 and 3 once.
    let m = series_moments(view(&X, &[1.; 12], &[0; 12], 1), 2, 3).unwrap();
    let sums = m.weighted(&[2., 0., 1., 1.]);
    assert_eq!(sums.ab, [292., 236., 223.]);
    assert_eq!(sums.a, [50., 41., 35.]);
    assert_eq!(sums.b, [50., 55., 50.]);
    assert_eq!(sums.n, [12., 11., 10.]);
    close(
        &some(&sums.estimate(Subtraction::LagMeans)),
        &[
            292. / 12. - 2500. / 144.,
            236. / 11. - 2255. / 121.,
            22.3 - 17.5,
        ],
        1e-13,
    );
    // Two vector components keep their own leg sums under a multiplicity.
    let y: Vec<f64> = (0..12).flat_map(|t| [X[t], 0.5 * X[11 - t]]).collect();
    let joint = series_moments(view(&y, &[1.; 12], &[0; 12], 2), 0, 6).unwrap();
    let sums = joint.weighted(&[3., 1.]);
    assert_eq!(sums.n, [24.]);
    assert_eq!(sums.a, [3. * 21. + 38., 0.5 * (3. * 38. + 21.)]);
    assert_eq!(sums.ab, [3. * (91. + 0.25 * 264.) + 264. + 0.25 * 91.]);
}

#[test]
fn a_segment_id_that_returns_after_a_gap_starts_a_new_run() {
    let x = [1., 3., 2., 5., 4., 6.];
    let returning = series_moments(view(&x, &[1.; 6], &[7, 7, 3, 3, 7, 7], 1), 5, 2).unwrap();
    let pooled = returning.pooled(None);
    assert_eq!(pooled.n, [6., 3., 0., 0., 0., 0.]);
    assert_eq!(pooled.ab, [91., 37., 0., 0., 0., 0.]);
    assert_eq!(
        returning,
        series_moments(view(&x, &[1.; 6], &[0, 0, 1, 1, 2, 2], 1), 5, 2).unwrap()
    );
    // Runs of three frames around a lone frame: rho(1) = (15/16) / (40/7),
    // rho(2) = 0 from the pairs (0, 2) and (4, 6), and no pair at lag 3.
    let x = [1., 2., 4., 9., 3., 5., 4.];
    let t = tau_int(view(&x, &[1.; 7], &[0, 0, 0, 1, 0, 0, 0], 1)).unwrap();
    assert!((t.tau - 85. / 128.).abs() < 1e-14 && t.window == 2, "{t:?}");
    assert_eq!(
        tau_int(view(&x, &[1.; 7], &[0, 0, 0, 1, 2, 2, 2], 1)),
        Some(t)
    );
}

#[test]
fn undefined_entries_and_single_resamples_give_zero_covariance_and_no_error() {
    let table = |kind, count: usize| Samples {
        kind,
        dimension: 2,
        count,
        central: vec![3., 0.],
        values: vec![1., 5., 2., 9., 4., -3., 7., 40.][..2 * count].to_vec(),
        defined: vec![true, false],
        effective_block: 1,
        blocks: 4,
        tau_int: None,
    };
    assert_eq!(
        sample_covariance(&table(ResampleKind::Bootstrap, 4)),
        [7., 0., 0., 0.]
    );
    assert_eq!(
        sample_covariance(&table(ResampleKind::Jackknife, 4)),
        [15.75, 0., 0., 0.]
    );
    assert_eq!(
        errors(&table(ResampleKind::Jackknife, 4)),
        [Some(15.75f64.sqrt()), None]
    );
    for kind in [ResampleKind::Bootstrap, ResampleKind::Jackknife] {
        assert_eq!(sample_covariance(&table(kind, 1)), [0.; 4]);
        assert_eq!(errors(&table(kind, 1)), [None, None]);
    }
}

#[test]
fn global_mean_is_undefined_without_equal_time_pairs() {
    // The legs are never valid on the same frame: lag 0 is empty, lag 1 holds
    // the pairs (0, 1) and (2, 3).
    let (a, b) = ([1., 2., 3., 4.], [5., 6., 7., 8.]);
    let g = [0; 4];
    let m = cross_moments(
        view(&a, &[1., 0., 1., 0.], &g, 1),
        view(&b, &[0., 1., 0., 1.], &g, 1),
        1,
        2,
    )
    .unwrap();
    assert_eq!(m.pooled(None).n, [0., 2.]);
    assert_eq!(m.estimate(Subtraction::None), [None, Some(15.)]);
    assert_eq!(m.estimate(Subtraction::LagMeans), [None, Some(1.)]);
    assert_eq!(m.estimate(Subtraction::GlobalMean), [None, None]);
}

#[test]
fn joining_an_empty_run_changes_nothing() {
    let series = view(&X, &[1.; 12], &[0; 12], 1);
    let built = series_moments(series, 2, 5).unwrap();
    assert_eq!((built.blocks, built.open, built.origins()), (3, 2, 12));
    let empty = BlockMoments::new(3, 1, 5, 4).unwrap();
    assert_eq!(BlockMoments::concat(&[&built, &empty]).unwrap(), built);
    assert_eq!(BlockMoments::concat(&[&empty, &built]).unwrap(), built);
    let nothing = BlockMoments::concat(&[&empty]).unwrap();
    assert_eq!((nothing.blocks, nothing.origins()), (0, 0));
    assert_eq!(nothing.estimate(Subtraction::LagMeans), [None; 3]);
}

#[test]
fn corrupted_block_moments_are_rejected_before_any_sum_is_read() {
    let sound = series_moments(view(&X, &[1.; 12], &[0; 12], 1), 2, 3).unwrap();
    let corrupt: [&dyn Fn(&mut BlockMoments); 6] = [
        &|m| m.n[4] = -1.,
        &|m| m.ab[0] = f64::INFINITY,
        &|m| m.a[7] = f64::NAN,
        &|m| m.b.pop().map_or((), drop),
        &|m| m.open = m.origins_per_block + 1,
        &|m| m.max_blocks = m.blocks - 1,
    ];
    for damage in corrupt {
        let mut m = sound.clone();
        damage(&mut m);
        assert!(matches!(m.validate(), Err(GasError::Configuration(_))));
        assert!(resample(&m, Subtraction::LagMeans, &jackknife(3), None).is_err());
        assert!(BlockMoments::concat(&[&sound, &m]).is_err());
    }
    sound.validate().unwrap();
}

/// Study: fraction of one-sigma jackknife intervals that hold the exact
/// autocovariance `phi^l / (1 - phi^2)` of an AR(1) series with `phi = 0.95`.
#[test]
fn automatic_blocks_cover_a_slowly_mixing_series_and_blocks_of_ten_do_not() {
    let (frames, replicas, phi) = (4000, 500, 0.95f64);
    let lags = [0, 5, 10];
    let mut covered = [[0usize; 3]; 2];
    for replica in 0..replicas {
        let x = ar1(1000 + replica, frames, phi);
        let (w, g) = (vec![1.; frames], vec![0; frames]);
        let series = view(&x, &w, &g, 1);
        let tau = tau_int(series).unwrap().tau;
        let m = series_moments(series, 10, 1).unwrap();
        for (scheme, resampling) in [Resampling::default(), jackknife(10)].iter().enumerate() {
            let s = resample(&m, Subtraction::LagMeans, resampling, Some(tau)).unwrap();
            let e = errors(&s);
            for (slot, &lag) in lags.iter().enumerate() {
                let exact = phi.powi(lag as i32) / (1. - phi * phi);
                covered[scheme][slot] +=
                    usize::from((s.central[lag] - exact).abs() <= e[lag].unwrap());
            }
        }
    }
    let coverage = covered.map(|row| row.map(|count| count as f64 / replicas as f64));
    eprintln!(
        "coverage automatic {:?}, block of ten {:?}",
        coverage[0], coverage[1]
    );
    // Binomial sd of a coverage near 0.65 from 500 series is 0.021. The
    // seeded study gives 0.672, 0.656, 0.640 for the automatic block, below
    // the nominal 0.683 because the estimate is right-skewed and its error
    // correlates with it, and 0.458, 0.462, 0.422 for a block of ten, more
    // than eight sd lower.
    for slot in 0..3 {
        assert!((0.60..=0.76).contains(&coverage[0][slot]), "{coverage:?}");
        assert!(coverage[1][slot] < 0.52, "{coverage:?}");
    }
}

#[test]
fn a_slowly_mixing_series_keeps_the_automatic_block_far_above_ten() {
    // The cheap guard of the coverage study above: a block of ten is below
    // twice the integrated autocorrelation time of an AR(1) series with
    // phi = 0.95, where the automatic rule stands near 183.
    let (frames, phi) = (4000, 0.95f64);
    let x = ar1(1000, frames, phi);
    let (w, g) = (vec![1.; frames], vec![0; frames]);
    let tau = tau_int(view(&x, &w, &g, 1)).unwrap().tau;
    let block = auto_block(tau, 1, frames);
    assert!(block >= 40 && block as f64 >= 2. * tau, "{block} {tau}");
    assert!(10. < 2. * tau, "{tau}");
    assert!(block_below_target(tau, 10, frames), "{tau}");
    assert!(!block_below_target(tau, block, frames), "{block} {tau}");
}

#[test]
fn a_straddling_pair_is_the_one_a_deleted_block_keeps() {
    // Oracle: the pairs of a contiguous series counted one origin at a time,
    // a pair straddling when its two frames sit in different blocks.
    let counted = |lags: usize, block: usize, origins: usize| {
        (0..lags)
            .flat_map(|lag| (0..origins.saturating_sub(lag)).map(move |t| (lag, t)))
            .filter(|&(lag, t)| t / block != (t + lag) / block)
            .count() as u64
    };
    for (lags, block, origins) in [
        (3, 2, 7),
        (41, 24, 200),
        (11, 10, 40),
        (1, 5, 9),
        (5, 1, 6),
        (21, 8, 64),
        (13, 7, 5),
    ] {
        assert_eq!(
            straddling_pairs(lags, block, origins),
            counted(lags, block, origins),
            "{lags} {block} {origins}"
        );
    }
    // Beyond the block every pair of a lag straddles it, so a delete-one
    // resample removes none of the products that lag holds.
    assert_eq!(straddling_pairs(9, 8, 64) - straddling_pairs(8, 8, 64), 56);
    assert_eq!(
        straddling_pairs(21, 8, 64) - straddling_pairs(20, 8, 64),
        44
    );
    assert_eq!(straddling_pairs(1, 8, 64), 0);
    assert_eq!(straddling_pairs(3, 0, 7), straddling_pairs(3, 1, 7));
    // A lag of `origins` or more reaches no sink, so a lag range longer than
    // the series adds nothing and costs nothing: the sum stops at the shorter
    // of the two rather than walking a caller's range to its end.
    assert_eq!(straddling_pairs(usize::MAX, 8, 64), 1792);
    assert_eq!(straddling_pairs(64, 8, 64), 1792);
    assert_eq!(straddling_pairs(21, 8, 64), 846);
    assert_eq!(straddling_pairs(9, 8, 0), 0);
}

/// Study: quoted against true one-sigma jackknife errors of an AR(1)
/// correlator at lags beyond the resampling block. The deletion removes time
/// origins, so every straddling pair survives it and the quoted error is a
/// lower bound there.
#[test]
fn automatic_blocks_at_lags_beyond_the_block_are_a_stated_lower_bound() {
    let (frames, replicas, phi, max_lag) = (200, 600, 0.9f64, 40);
    let lags = [0, 10, 30, 40];
    let mut central = vec![vec![]; lags.len()];
    let mut quoted = vec![vec![vec![]; lags.len()]; 2];
    let mut blocks = 0;
    for replica in 0..replicas {
        let x = ar1(3000 + replica as u64, frames, phi);
        let (w, g) = (vec![1.; frames], vec![0; frames]);
        let series = view(&x, &w, &g, 1);
        let tau = tau_int(series).unwrap().tau;
        let m = series_moments(series, max_lag, 1).unwrap();
        blocks += auto_block(tau, 1, frames);
        for (scheme, resampling) in [Resampling::default(), jackknife(10)].iter().enumerate() {
            let s = resample(&m, Subtraction::LagMeans, resampling, Some(tau)).unwrap();
            let e = errors(&s);
            for (slot, &lag) in lags.iter().enumerate() {
                quoted[scheme][slot].push(e[lag].unwrap());
                if scheme == 0 {
                    central[slot].push(s.central[lag]);
                }
            }
        }
    }
    let scatter = |values: &[f64]| {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let spread: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();
        (spread / (values.len() - 1) as f64).sqrt()
    };
    let median = |values: &mut [f64]| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let block = blocks as f64 / replicas as f64;
    let ratio: Vec<[f64; 2]> = (0..lags.len())
        .map(|slot| {
            let true_error = scatter(&central[slot]);
            [0, 1].map(|scheme| median(&mut quoted[scheme][slot]) / true_error)
        })
        .collect();
    eprintln!("mean automatic block {block}, quoted/true per lag {ratio:?}");
    // The automatic block is 24.3 here, so lags 30 and 40 lie beyond it and no
    // product of theirs is ever deleted. The seeded study gives 0.756, 0.684,
    // 0.830, 0.834 for the automatic block, a lower bound that stays inside a
    // third of the truth, against 0.668, 0.611, 0.739, 0.718 for a fixed
    // block of ten, which is under 2 tau_int = 19 and worse at every lag.
    assert!((20. ..30.).contains(&block), "{block}");
    for slot in 0..lags.len() {
        assert!((0.60..=0.95).contains(&ratio[slot][0]), "{ratio:?}");
        assert!(
            ratio[slot][1] < ratio[slot][0] && ratio[slot][1] < 0.76,
            "{ratio:?}"
        );
    }
}
