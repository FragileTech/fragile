//! Agreement with the Python QFT operator pipeline (`src/fragile/physics`).
//! Fixtures come from `tools/export_qft_fixtures.py`; their README fixes the
//! schema, the index conventions and the quantities the audit excluded because
//! Rust pins the corrected behaviour there. Masks, counts and element sets are
//! discrete structure and match exactly; numbers carry the per-quantity
//! tolerance of that README, and every bound looser than 1e-8 carries its
//! reason at the call site.
//!
//! Out of scope, because the fixtures carry no data for them and other suites
//! own them: the GEVP bootstrap (D3), the block-10 full-sample-mean jackknife
//! (D6), `cloning_frames_only` (D7), the flow scale `w0` (D10), the
//! `gvar(0, 0)` effective-mass placeholders (D11c) and the PDG priors (D2).
//!
//! Each test ends with one line naming the largest deviation it saw per
//! compared quantity, so a regression shows how far it moved.
use algorithmic_gas::{
    GasConfig, GasError,
    cloning::CloneDecision,
    physics::{
        numerics::{
            ResampleKind, Samples, SeriesView, Subtraction, Whitener, linear_fit, series_moments,
        },
        qft::math::{C, dot},
        spectroscopy::{
            config::{
                AnalysisConfig, BaryonMode, ChannelSpec, CompanionChoice, Displacement,
                ElectroweakScales, FrameNormalization, GlueballObservable, MeasurementConfig,
                MesonMode, MesonQuantum, Momentum, MomentumPhase, PairDistance, PairSelection,
                Range, Su2Mode, TensorMode, TimeUnit, U1Mode, VectorProjection, VectorQuantum,
                WindowScanConfig,
            },
            contract::{
                Auxiliary, Capabilities, Companions, Element, ElementKind, ExchangeParity, Frame,
                FrameState, LocalOperator, NO_COMPANION, OperatorContext, Record, Topology,
            },
            estimators::Estimated,
            fits::window_scan::window_scan,
            report::{CorrelatorEstimate, EstimatorKind, SamplesMeta},
            topology,
        },
    },
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The three exported cases, in the order the fixture README lists them.
const CASES: [&str; 3] = ["non_involutive", "mutual_pairing_odd", "masked"];

// ---- fixture access: an untyped `Value` with small accessors, no structs ----

fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/tests/fixtures/qft/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let parsed: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    // A regenerated fixture with another schema must fail loudly, not silently pass.
    assert_eq!(parsed["schema_version"].as_u64(), Some(1), "{name}: schema");
    assert_eq!(parsed["shape"]["dimension"].as_u64(), Some(3), "{name}: d");
    // Every parameter the configurations below hold as a literal rather than
    // reading it from the file: `h_eff` (and `h_s = h_eff`) in
    // `measurement_config`, `flux_alpha` in the baryon specifications, the
    // pooled pair selection, the SU(2) operator mode, and the two floors whose
    // inertness `colour_overlaps_…` asserts. A regenerated fixture that moved
    // one of them would otherwise be compared against the old value.
    for (key, value) in [
        ("h_eff", 1.),
        ("eps", 1e-12),
        ("norm_floor", 1e-12),
        ("flux_exp_alpha", 1.),
    ] {
        assert_eq!(
            parsed["parameters"][key].as_f64(),
            Some(value),
            "{name}: {key}"
        );
    }
    for (key, value) in [
        ("pair_selection", "both"),
        ("su2_operator_mode", "standard"),
    ] {
        assert_eq!(
            parsed["parameters"][key].as_str(),
            Some(value),
            "{name}: {key}"
        );
    }
    parsed
}
/// Every leaf of a nested array, row-major; a `null` leaf becomes `NaN`.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(items) => items.iter().flat_map(flat).collect(),
        Value::Null => vec![f64::NAN],
        other => vec![other.as_f64().unwrap()],
    }
}
/// Every leaf of a nested integer array, row-major.
fn ints(v: &Value) -> Vec<i64> {
    match v {
        Value::Array(items) => items.iter().flat_map(ints).collect(),
        other => vec![other.as_i64().unwrap()],
    }
}
/// A nested `0`/`1` array as booleans, row-major.
fn mask(v: &Value) -> Vec<bool> {
    ints(v).into_iter().map(|x| x != 0).collect()
}
/// `(frames, walkers)` of a case.
fn shape(f: &Value) -> (usize, usize) {
    (
        f["shape"]["frames"].as_u64().unwrap() as usize,
        f["shape"]["walkers"].as_u64().unwrap() as usize,
    )
}
/// Width of one record: its own `components` array, or 1 where the key is
/// absent. The four U(1) records of `mutual_pairing_odd.json` are `[T][1]`
/// because the imaginary part is exchange odd (audit D1), while its two SU(2)
/// records stay `[T][2]`, so the width is never a constant.
fn width(record: &Value) -> usize {
    record["components"].as_array().map_or(1, Vec::len)
}
/// `[T]` or `[T][C]` record values; a frame the exporter wrote as `null`, or
/// one holding a `null`, is `None` and is never compared (audit D11b).
fn record(record: &Value, components: usize) -> Vec<Option<Vec<f64>>> {
    record["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| {
            let row = flat(frame);
            assert_eq!(row.len(), components, "record component width");
            row.iter().all(|x| x.is_finite()).then_some(row)
        })
        .collect()
}
/// Largest `|a − e| / scale(e)` of the vector, asserted to stay at `tolerance`.
#[track_caller]
fn deviation(
    label: &str,
    actual: &[f64],
    expected: &[f64],
    tolerance: f64,
    scale: fn(f64) -> f64,
) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let mut worst = 0f64;
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        let seen = (a - e).abs() / scale(*e);
        assert!(
            seen <= tolerance,
            "{label}[{k}]: {a} vs {e} ({seen:e} above {tolerance:e})"
        );
        worst = worst.max(seen);
    }
    worst
}
/// `|a − e| ≤ tolerance · max(1, |e|)`, the README's own rule: absolute for
/// the O(1) quantities of these fixtures and relative on a larger one.
#[track_caller]
fn close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) -> f64 {
    deviation(label, actual, expected, tolerance, |e| e.abs().max(1.))
}
/// `|a − e| ≤ tolerance`, with no relative relief anywhere.
#[track_caller]
fn close_absolute(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) -> f64 {
    deviation(label, actual, expected, tolerance, |_| 1.)
}
/// A series Python computed in float32 (audit D11i): `|a − e| ≤ 1e-6 max(1, |e|)`.
/// Purely relative would fail, because a float32 summation error of about 1e-7
/// absolute is up to 1.7e-5 relative on these small frame means (README).
#[track_caller]
fn close_f32(label: &str, actual: &[f64], expected: &[f64]) -> f64 {
    close(label, actual, expected, 1e-6)
}
#[track_caller]
fn same_mask(label: &str, actual: &[bool], expected: &[bool]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a, e, "{label}[{k}]: mask");
    }
}
/// Every lag of a `BlockMoments::estimate`, which is `Vec<Option<f64>>`. With
/// 16 frames and `max_lag = 6` no lag has zero weight, so a `None` is a failure.
#[track_caller]
fn defined(label: &str, estimate: &[Option<f64>]) -> Vec<f64> {
    estimate
        .iter()
        .enumerate()
        .map(|(lag, v)| v.unwrap_or_else(|| panic!("{label}: lag {lag} has no estimate")))
        .collect()
}
/// Largest deviation per compared quantity, reported once at the end of a test.
#[derive(Default)]
struct Deviations(BTreeMap<String, f64>);
impl Deviations {
    fn see(&mut self, label: &str, deviation: f64) {
        let worst = self.0.entry(label.to_string()).or_default();
        *worst = worst.max(deviation);
    }
    fn report(&self, test: &str) {
        let rows: Vec<String> = self.0.iter().map(|(k, v)| format!("{k}={v:.3e}")).collect();
        eprintln!("{test}: max deviation {}", rows.join(" "));
    }
}

// ---- inputs, hand-built from one fixture frame ----

fn gas_config(epsilon_clone: f64) -> GasConfig {
    GasConfig {
        // The SU(2) phase regulariser of the fixtures; never an interaction
        // range (audit D4). `CloneDecision::default().epsilon` is 1e-6.
        clone_decision: CloneDecision {
            epsilon: epsilon_clone,
            ..CloneDecision::default()
        },
        // `GasConfig::default().boundary` is unbounded, so `periodic_box` is
        // `None` and every displacement is the raw difference the fixture took.
        ..GasConfig::default()
    }
}
fn measurement_config(epsilon_d: f64, lambda: f64) -> MeasurementConfig {
    MeasurementConfig {
        pairs: PairSelection::Both,
        companions: CompanionChoice::First,
        electroweak: ElectroweakScales {
            h_eff: 1.,
            // The fixture's h_S = h_eff.
            h_s: None,
            epsilon_d: Range::Fixed { value: epsilon_d },
            // Unused: no dressed SU(2) arm is compared (audit D4). It is set
            // to a fixed value so that no capability decides the comparison.
            epsilon_c: Range::Fixed { value: epsilon_d },
            distance: PairDistance::Raw,
            // Explicit even at 0: `None` would fall back to the donor module's
            // own velocity weight instead of the fixture's `lambda_alg`.
            lambda: Some(lambda),
        },
        ..MeasurementConfig::default()
    }
}
/// `slot` holds absolute walker indices. A fixture index outside `[0, n)` is
/// an entry the engine never sampled, so it is `valid = false` with slot 0:
/// `Companions::validate` bounds every sampled slot by `n`. `mutual` carries
/// the fixture's `involutive` so the record is honest; `topology::build` never
/// reads it, which is what makes the `involutive` assertion of D1 independent.
fn companions(index: &[i64], n: usize, mutual: bool) -> Companions {
    let sampled = |j: i64| (0..n as i64).contains(&j);
    Companions {
        count: 1,
        slot: index
            .iter()
            .map(|&j| if sampled(j) { j as u32 } else { 0 })
            .collect(),
        generation: vec![0; n],
        valid: index.iter().map(|&j| sampled(j)).collect(),
        // The fixtures hold no historical companion.
        historical: vec![false; n],
        mutual,
    }
}
/// `Companions::first` resolved to the slot, `NO_COMPANION` otherwise: the
/// rule `frame::base_state` applies.
fn first_map(companions: Option<&Companions>, eligible: &[bool], n: usize) -> Vec<u32> {
    match companions {
        Some(record) => (0..n)
            .map(|i| {
                record
                    .first(i, eligible)
                    .map_or(NO_COMPANION, |k| record.slot[i * record.count + k])
            })
            .collect(),
        None => vec![NO_COMPANION; n],
    }
}
fn frame_of(f: &Value, t: usize) -> Frame {
    let (_, n) = shape(f);
    let involutive = f["involutive"].as_bool().unwrap();
    let inputs = &f["inputs"];
    let eligible = mask(&inputs["alive"][t]);
    let frame = Frame {
        step: t as u64,
        n,
        d: 3,
        x: flat(&inputs["positions"][t]),
        v: inputs["velocities"].as_array().map(|v| flat(&v[t])),
        eligible,
        generation: vec![0; n],
        fitness: Some(flat(&inputs["fitness"][t])),
        distance: Some(companions(
            &ints(&inputs["companions_distance"][t]),
            n,
            involutive,
        )),
        cloning: Some(companions(
            &ints(&inputs["companions_clone"][t]),
            n,
            involutive,
        )),
        cloned: mask(&inputs["will_clone"][t]),
        revived: vec![false; n],
        ..Frame::default()
    };
    frame.validate().unwrap();
    frame
}
/// The fixture supplies the cloning score directly, so `fields::score::fill`
/// is never called and `Frame::companion_fitness` stays absent.
/// `score_valid = alive`: Python applies no score mask at all and Rust masks
/// every reader of `score` on it, which is faithful because a dead walker
/// never enters an element (`topology::build` masks it as `Ineligible`).
/// `force`, `phase_velocity`, `score_gradient`, `role` and `euclidean_time`
/// stay absent: no compared channel reads them.
fn state_of(f: &Value, t: usize) -> FrameState {
    let frame = frame_of(f, t);
    let n = frame.n;
    let inputs = &f["inputs"];
    let re = flat(&inputs["color_re"][t]);
    let im = flat(&inputs["color_im"][t]);
    FrameState {
        step: frame.step,
        n,
        d: 3,
        x: frame.x.clone(),
        v: frame.v.clone(),
        color: re.iter().zip(&im).map(|(&a, &b)| C::new(a, b)).collect(),
        color_valid: mask(&inputs["color_valid"][t]),
        fitness: frame.fitness.clone(),
        score: Some(flat(&inputs["scores"][t])),
        score_valid: frame.eligible.clone(),
        cloned: frame.cloned.clone(),
        generation: vec![0; n],
        eligible: frame.eligible.clone(),
        distance_companion: Some(first_map(frame.distance.as_ref(), &frame.eligible, n)),
        cloning_companion: Some(first_map(frame.cloning.as_ref(), &frame.eligible, n)),
        ..FrameState::default()
    }
}
/// The three topologies of one frame, built only through `topology::build`:
/// hand-assembling elements would bypass the ineligible, self and unsampled
/// masks and let a dead walker reach `meson::evaluate`.
fn topologies(frame: &Frame, measurement: &MeasurementConfig) -> [Topology; 4] {
    let kinds = [
        ElementKind::DistancePair,
        ElementKind::CloningPair,
        ElementKind::Triplet,
        ElementKind::Site,
    ];
    kinds.map(|kind| {
        let built = topology::build(frame, kind, measurement);
        // One element per anchor under `CompanionChoice::First`: a later
        // change to `All` must be loud, because the fixture tables are keyed
        // by the anchor row alone.
        let anchors: BTreeSet<u32> = built.elements.iter().map(|e| e.walkers[0]).collect();
        assert_eq!(anchors.len(), built.elements.len(), "anchor uniqueness");
        built
    })
}

// ---- evaluation ----

/// One evaluated topology: the topology with the values and masks of one
/// operator on it, as the pooled frame mean of a record consumes them.
type Evaluated<'a> = (&'a Topology, (Vec<f64>, Vec<bool>));

/// `(values, valid)` of one operator on one topology, `components` values per
/// element: what the Python per-element table holds.
///
/// A score mode leaves the contraction in the sampled order and reports the
/// factor the score contributes in the auxiliary column beside it
/// (`Auxiliary::{ScoreOrientation, ScoreDispersion}`), which the measurement
/// freezes with the element and reapplies at every sink. Python folds the same
/// factor into the value — `conj(q)` on a downhill pair is `sgn(ΔS) Im q`, and
/// the weighted arm is `|ΔS| q` — so the two columns are multiplied back
/// together here. The factor is 0 on a tie, where Python applies no
/// conjugation and Rust returns 0; the fixture scores hold no tie, so the
/// divergence is inert on this data.
fn evaluate(
    spec: &ChannelSpec,
    topo: &Topology,
    state: &FrameState,
    context: &OperatorContext<'_>,
    components: usize,
) -> (Vec<f64>, Vec<bool>) {
    let width = spec.signature(topo.kind, context).unwrap().width();
    assert!(width >= components, "operator width below its components");
    let mut values = vec![0.; topo.elements.len() * width];
    let mut valid = vec![false; topo.elements.len()];
    spec.evaluate_all(&topo.elements, state, context, &mut values, &mut valid);
    if width == components {
        return (values, valid);
    }
    let folded = values
        .chunks_exact(width)
        .flat_map(|row| {
            let factor = row[components];
            row[..components].iter().map(move |v| v * factor)
        })
        .collect();
    (folded, valid)
}
/// Number of measured components of the operator on `kind`; the auxiliary
/// column `evaluate` also writes is not one of them.
fn components(spec: &ChannelSpec, kind: ElementKind, context: &OperatorContext<'_>) -> usize {
    spec.signature(kind, context).unwrap().components
}
/// Per-anchor validity and values of one evaluated topology, in the fixture's
/// `[N]` row order: an anchor without an element is invalid, which catches a
/// missing element that an "all produced elements agree" loop would hide.
fn by_anchor(
    topo: &Topology,
    values: &[f64],
    valid: &[bool],
    n: usize,
    components: usize,
) -> (Vec<bool>, Vec<f64>) {
    let mut rows = vec![false; n];
    let mut out = vec![0.; n * components];
    for (e, element) in topo.elements.iter().enumerate() {
        let i = element.walkers[0] as usize;
        rows[i] = valid[e];
        out[i * components..(i + 1) * components]
            .copy_from_slice(&values[e * components..(e + 1) * components]);
    }
    (rows, out)
}
/// Valid-count masked mean over element kinds pooled together, weighted by
/// `Element::weight`, with the denominator it used. `None` for a frame without
/// a valid element: the fixture's `count = 0` is weight 0, never the stored
/// `0.0` an `f32` record writes there (audit D11b).
fn masked_mean(
    parts: &[(&[Element], &[f64], &[bool])],
    components: usize,
) -> (Option<Vec<f64>>, f64) {
    let mut sum = vec![0.; components];
    let mut weight = 0.;
    for (elements, values, valid) in parts {
        for (e, element) in elements.iter().enumerate() {
            if !valid[e] {
                continue;
            }
            weight += element.weight;
            for (k, s) in sum.iter_mut().enumerate() {
                *s += element.weight * values[e * components + k];
            }
        }
    }
    let mean = (weight > 0.).then(|| sum.iter().map(|s| s / weight).collect());
    (mean, weight)
}
/// Anchors the fixture's own structural rule keeps for one companion role:
/// the companion is in range, is not the walker itself, and both walkers are
/// alive. The colour mask is applied by the operator, not here.
fn expected_anchors(f: &Value, t: usize, role: &str) -> BTreeSet<u32> {
    let (_, n) = shape(f);
    let index = ints(&f["inputs"][role][t]);
    let alive = mask(&f["inputs"]["alive"][t]);
    (0..n)
        .filter(|&i| {
            let j = index[i];
            (0..n as i64).contains(&j) && j != i as i64 && alive[i] && alive[j as usize]
        })
        .map(|i| i as u32)
        .collect()
}

#[test]
fn colour_overlaps_determinants_and_plaquettes_match_the_python_reference() {
    let mut worst = Deviations::default();
    for name in CASES {
        let f = fixture(name);
        let (frames, n) = shape(&f);
        let gas = gas_config(f["parameters"]["epsilon_clone"].as_f64().unwrap());
        let measurement = measurement_config(
            f["parameters"]["epsilon_d"].as_f64().unwrap(),
            f["parameters"]["lambda_alg"].as_f64().unwrap(),
        );
        let capabilities = Capabilities::nominal();
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        };
        let scalar = ChannelSpec::Meson {
            quantum: MesonQuantum::Scalar,
            mode: MesonMode::Standard,
        };
        let pseudoscalar = ChannelSpec::Meson {
            quantum: MesonQuantum::Pseudoscalar,
            mode: MesonMode::Standard,
        };
        let unit = ChannelSpec::Vector {
            quantum: VectorQuantum::Vector,
            projection: VectorProjection::Full,
            displacement: Displacement::Unit,
        };
        let baryon = ChannelSpec::Baryon {
            mode: BaryonMode::Complex,
            flux_alpha: 1.,
        };
        let re_plaquette = ChannelSpec::Glueball {
            observable: GlueballObservable::RePlaquette,
            momentum: None,
        };
        let one_minus_cos = ChannelSpec::Glueball {
            observable: GlueballObservable::OneMinusCos,
            momentum: None,
        };
        let sin2 = ChannelSpec::Glueball {
            observable: GlueballObservable::Sin2,
            momentum: None,
        };
        let q_re = flat(&f["pairs"]["q_re"]);
        let q_im = flat(&f["pairs"]["q_im"]);
        let q_valid = mask(&f["pairs"]["valid"]);
        // Absent when it equals `valid`: only `masked.json` has coincident positions.
        let unit_valid = f["pairs"]
            .get("valid_unit_displacement")
            .map(mask)
            .unwrap_or_else(|| q_valid.clone());
        let b_re = flat(&f["triplets"]["b_re"]);
        let b_im = flat(&f["triplets"]["b_im"]);
        let b_valid = mask(&f["triplets"]["b_valid"]);
        let pi_re = flat(&f["triplets"]["pi_re"]);
        let pi_im = flat(&f["triplets"]["pi_im"]);
        let pi_valid = mask(&f["triplets"]["pi_valid"]);
        // The masks below agree although Python adds an `|q| > eps` floor to
        // `pairs.valid` and an `|b| > eps` floor to `triplets.b_valid` that
        // Rust does not apply: the fixture colours are generic and every
        // exported value sits ten orders of magnitude above `eps = 1e-12`. The
        // agreement is therefore a structural one, and a future fixture with a
        // near-degenerate overlap fails here rather than in a mask comparison
        // whose message would not name the floor. `pi_valid` needs no such
        // argument: `glueball::plaquette_phase` applies exactly the same
        // three-link 1e-12 floor.
        let smallest = |modulus: &dyn Fn(usize) -> f64, valid: &[bool]| -> f64 {
            (0..valid.len())
                .filter(|&k| valid[k])
                .fold(f64::INFINITY, |m, k| m.min(modulus(k)))
        };
        assert!(
            smallest(&|k| q_re[k].hypot(q_im[k]), &q_valid) > 1e-3,
            "{name}: a valid overlap sits near the eps floor Rust does not apply"
        );
        assert!(
            smallest(&|k| b_re[k].hypot(b_im[k]), &b_valid) > 1e-3,
            "{name}: a valid determinant sits near the eps floor Rust does not apply"
        );
        for t in 0..frames {
            let frame = frame_of(&f, t);
            let state = state_of(&f, t);
            let [distance, cloning, triplets, _] = topologies(&frame, &measurement);
            for (p, topo) in [&distance, &cloning].into_iter().enumerate() {
                let (re, re_ok) = evaluate(&scalar, topo, &state, &context, 1);
                let (im, im_ok) = evaluate(&pseudoscalar, topo, &state, &context, 1);
                assert_eq!(
                    re_ok, im_ok,
                    "{name} t{t} p{p}: the two meson arms disagree"
                );
                let (rows, got_re) = by_anchor(topo, &re, &re_ok, n, 1);
                let (_, got_im) = by_anchor(topo, &im, &im_ok, n, 1);
                let want: Vec<bool> = (0..n).map(|i| q_valid[(t * n + i) * 2 + p]).collect();
                same_mask(&format!("{name} t{t} p{p}: pairs.valid"), &rows, &want);
                // A masked element writes a placeholder on both sides.
                let keep =
                    |v: &[f64]| -> Vec<f64> { (0..n).filter(|&i| want[i]).map(|i| v[i]).collect() };
                let expect = |v: &[f64]| -> Vec<f64> {
                    (0..n)
                        .filter(|&i| want[i])
                        .map(|i| v[(t * n + i) * 2 + p])
                        .collect()
                };
                worst.see(
                    "pairs.q_re",
                    close(
                        &format!("{name} t{t} p{p}: q_re"),
                        &keep(&got_re),
                        &expect(&q_re),
                        1e-12,
                    ),
                );
                worst.see(
                    "pairs.q_im",
                    close(
                        &format!("{name} t{t} p{p}: q_im"),
                        &keep(&got_im),
                        &expect(&q_im),
                        1e-12,
                    ),
                );
                // `Displacement::Unit` masks a zero separation, which is the
                // fixture's `|r| > norm_floor`: the inputs are rounded to five
                // decimals, so coincident means bitwise equal.
                let (_, unit_ok) = evaluate(&unit, topo, &state, &context, 3);
                let (unit_rows, _) = by_anchor(topo, &[], &unit_ok, n, 0);
                let want_unit: Vec<bool> =
                    (0..n).map(|i| unit_valid[(t * n + i) * 2 + p]).collect();
                same_mask(
                    &format!("{name} t{t} p{p}: pairs.valid_unit_displacement"),
                    &unit_rows,
                    &want_unit,
                );
            }
            // Column order is anchor, distance companion, cloning companion,
            // the fixture's `(i, companions_distance, companions_clone)`.
            // `glueball::colors` masks a degenerate triplet `j == k` itself,
            // which is the fixture's `j != k`; Rust applies no `|b| > eps`
            // floor and does require eligibility, both inert on this data.
            let (b, b_ok) = evaluate(&baryon, &triplets, &state, &context, 2);
            let (rows, got_b) = by_anchor(&triplets, &b, &b_ok, n, 2);
            let want: Vec<bool> = (0..n).map(|i| b_valid[t * n + i]).collect();
            same_mask(&format!("{name} t{t}: triplets.b_valid"), &rows, &want);
            let kept: Vec<usize> = (0..n).filter(|&i| want[i]).collect();
            worst.see(
                "triplets.b_re",
                close(
                    &format!("{name} t{t}: b_re"),
                    &kept.iter().map(|&i| got_b[2 * i]).collect::<Vec<_>>(),
                    &kept.iter().map(|&i| b_re[t * n + i]).collect::<Vec<_>>(),
                    1e-12,
                ),
            );
            worst.see(
                "triplets.b_im",
                close(
                    &format!("{name} t{t}: b_im"),
                    &kept.iter().map(|&i| got_b[2 * i + 1]).collect::<Vec<_>>(),
                    &kept.iter().map(|&i| b_im[t * n + i]).collect::<Vec<_>>(),
                    1e-12,
                ),
            );
            // `OneMinusCos` applies exactly the fixture's three-link 1e-12
            // floor; `RePlaquette` has no floor, so its own mask is weaker and
            // is not asserted against `pi_valid`.
            let (cos, cos_ok) = evaluate(&one_minus_cos, &triplets, &state, &context, 1);
            let (cos_rows, got_cos) = by_anchor(&triplets, &cos, &cos_ok, n, 1);
            let want_pi: Vec<bool> = (0..n).map(|i| pi_valid[t * n + i]).collect();
            same_mask(
                &format!("{name} t{t}: triplets.pi_valid"),
                &cos_rows,
                &want_pi,
            );
            let (re, re_ok) = evaluate(&re_plaquette, &triplets, &state, &context, 1);
            let (_, got_re) = by_anchor(&triplets, &re, &re_ok, n, 1);
            let (s2, s2_ok) = evaluate(&sin2, &triplets, &state, &context, 1);
            let (_, got_sin2) = by_anchor(&triplets, &s2, &s2_ok, n, 1);
            let pi_kept: Vec<usize> = (0..n).filter(|&i| want_pi[i]).collect();
            worst.see(
                "triplets.pi_re",
                close(
                    &format!("{name} t{t}: pi_re"),
                    &pi_kept.iter().map(|&i| got_re[i]).collect::<Vec<_>>(),
                    &pi_kept
                        .iter()
                        .map(|&i| pi_re[t * n + i])
                        .collect::<Vec<_>>(),
                    1e-12,
                ),
            );
            // No operator arm returns `Im Π`. `qft::math::dot` is `pub` and
            // `glueball::plaquette` is `dot(a,b) dot(b,c) dot(c,a)` on the
            // same three rows in the same order, so this reproduces the
            // operator's arithmetic operation for operation: a direct-algebra
            // cross-check standing in for a missing arm, not a second
            // implementation of the physics.
            let row = |w: u32| &state.color[w as usize * 3..(w as usize + 1) * 3];
            let mut algebra_re = vec![0.; n];
            let mut algebra_im = vec![0.; n];
            for element in &triplets.elements {
                let [i, j, k] = element.walkers;
                let pi = dot(row(i), row(j)) * dot(row(j), row(k)) * dot(row(k), row(i));
                algebra_re[i as usize] = pi.re;
                algebra_im[i as usize] = pi.im;
            }
            worst.see(
                "triplets.pi_re via dot",
                close(
                    &format!("{name} t{t}: pi_re (dot)"),
                    &pi_kept.iter().map(|&i| algebra_re[i]).collect::<Vec<_>>(),
                    &pi_kept
                        .iter()
                        .map(|&i| pi_re[t * n + i])
                        .collect::<Vec<_>>(),
                    1e-12,
                ),
            );
            worst.see(
                "triplets.pi_im via dot",
                close(
                    &format!("{name} t{t}: pi_im (dot)"),
                    &pi_kept.iter().map(|&i| algebra_im[i]).collect::<Vec<_>>(),
                    &pi_kept
                        .iter()
                        .map(|&i| pi_im[t * n + i])
                        .collect::<Vec<_>>(),
                    1e-12,
                ),
            );
            // The phase arms are a consistency check only, never a route to
            // `pi_im`: `OneMinusCos` is `1 − Re Π/|Π|` and the measured
            // minimum of `|Re Π/|Π||` is 2.15e-4, so recovering the cosine
            // from it cancels about four digits and leaves ~5e-13 relative.
            let unit_circle: Vec<f64> = pi_kept
                .iter()
                .map(|&i| {
                    let cosine = 1. - got_cos[i];
                    cosine * cosine + got_sin2[i]
                })
                .collect();
            worst.see(
                "glueball phase identity",
                close(
                    &format!("{name} t{t}: cos^2 + sin^2"),
                    &unit_circle,
                    &vec![1.; pi_kept.len()],
                    1e-11,
                ),
            );
            let from_algebra: Vec<f64> = pi_kept
                .iter()
                .map(|&i| {
                    let modulus = algebra_re[i].hypot(algebra_im[i]);
                    (algebra_im[i] / modulus).powi(2)
                })
                .collect();
            worst.see(
                "glueball sin2 against the algebra",
                close(
                    &format!("{name} t{t}: sin2"),
                    &pi_kept.iter().map(|&i| got_sin2[i]).collect::<Vec<_>>(),
                    &from_algebra,
                    1e-11,
                ),
            );
        }
    }
    worst.report("colour_overlaps_determinants_and_plaquettes");
}

/// Fixture record name, its specification, and the `counts` key that is the
/// denominator of its masked mean. The count key also names the topology:
/// `pairs*` pools the two roles, `triplets*` is the triplet topology.
fn colour_records() -> Vec<(&'static str, ChannelSpec, &'static str)> {
    let meson = |quantum, mode| ChannelSpec::Meson { quantum, mode };
    let vector = |quantum, displacement| ChannelSpec::Vector {
        quantum,
        projection: VectorProjection::Full,
        displacement,
    };
    // `flux_alpha` is read by `FluxWeighted` alone; 1 is the fixture's value.
    let baryon = |mode| ChannelSpec::Baryon {
        mode,
        flux_alpha: 1.,
    };
    let glueball = |observable| ChannelSpec::Glueball {
        observable,
        momentum: None,
    };
    use MesonQuantum::{Pseudoscalar, Scalar};
    use VectorQuantum::{Axial, Vector};
    vec![
        ("scalar", meson(Scalar, MesonMode::Standard), "pairs"),
        (
            "pseudoscalar",
            meson(Pseudoscalar, MesonMode::Standard),
            "pairs",
        ),
        (
            "scalar_score_directed",
            meson(Scalar, MesonMode::ScoreDirected),
            "pairs",
        ),
        (
            "pseudoscalar_score_directed",
            meson(Pseudoscalar, MesonMode::ScoreDirected),
            "pairs",
        ),
        (
            "scalar_score_weighted",
            meson(Scalar, MesonMode::ScoreWeighted),
            "pairs",
        ),
        (
            "pseudoscalar_score_weighted",
            meson(Pseudoscalar, MesonMode::ScoreWeighted),
            "pairs",
        ),
        ("vector", vector(Vector, Displacement::Raw), "pairs"),
        ("axial", vector(Axial, Displacement::Raw), "pairs"),
        // A zero displacement is a value under `Raw` and a mask under `Unit`,
        // so only the unit arms take the second count key.
        (
            "vector_unit",
            vector(Vector, Displacement::Unit),
            "pairs_unit_displacement",
        ),
        (
            "axial_unit",
            vector(Axial, Displacement::Unit),
            "pairs_unit_displacement",
        ),
        ("baryon_re_f64", baryon(BaryonMode::Real), "triplets_b"),
        ("baryon_im_f64", baryon(BaryonMode::Imag), "triplets_b"),
        ("baryon_abs2_f64", baryon(BaryonMode::Abs2), "triplets_b"),
        ("baryon_det_abs", baryon(BaryonMode::Abs), "triplets_b"),
        // Python orders the three colours by ascending score and takes
        // `Re det`; Rust returns `sgn(σ) Re b_ijk`. The determinant is
        // alternating, so the two agree.
        (
            "baryon_score_signed",
            baryon(BaryonMode::ScoreOrdered),
            "triplets_b",
        ),
        (
            "glueball_re_plaquette",
            glueball(GlueballObservable::RePlaquette),
            "triplets_pi",
        ),
        (
            "glueball_action_re_plaquette",
            glueball(GlueballObservable::OneMinusRe),
            "triplets_pi",
        ),
        (
            "glueball_phase_action",
            glueball(GlueballObservable::OneMinusCos),
            "triplets_pi",
        ),
        (
            "glueball_phase_sin2",
            glueball(GlueballObservable::Sin2),
            "triplets_pi",
        ),
    ]
}

#[test]
fn colour_channel_frame_series_match_the_python_reference() {
    let mut worst = Deviations::default();
    for name in CASES {
        let f = fixture(name);
        let (frames, _) = shape(&f);
        let gas = gas_config(f["parameters"]["epsilon_clone"].as_f64().unwrap());
        let measurement = measurement_config(
            f["parameters"]["epsilon_d"].as_f64().unwrap(),
            f["parameters"]["lambda_alg"].as_f64().unwrap(),
        );
        let capabilities = Capabilities::nominal();
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        };
        let counts = &f["series"]["counts"];
        let records = &f["series"]["records"];
        // Every exported record is compared but the three Python behaviours
        // that have no Rust arm at all, which are named here so that a record
        // is never dropped in silence: `ChannelSpec::Vector` carries no
        // score-direction axis (`MesonMode::ScoreDirected` is meson only and
        // `Displacement::ScoreGradient` is another observable), and
        // `BaryonMode::FluxWeighted` is `|b| exp(α (1 − cos arg Π))` where
        // Python takes `|b| (1 − cos arg Π)`, which no `flux_alpha` reaches.
        // Both are `status: extra` in the fixture README. A record that
        // appears here without being listed is an unported channel, not a
        // skipped one.
        let compared: BTreeSet<&str> = colour_records().iter().map(|(n, ..)| *n).collect();
        let exported: BTreeSet<&str> = records
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            exported.difference(&compared).copied().collect::<Vec<_>>(),
            [
                "axial_score_directed",
                "baryon_flux_action",
                "vector_score_directed"
            ]
            .into_iter()
            .filter(|record| exported.contains(record))
            .collect::<Vec<_>>(),
            "{name}: unported series records"
        );
        let states: Vec<FrameState> = (0..frames).map(|t| state_of(&f, t)).collect();
        let built: Vec<[Topology; 4]> = (0..frames)
            .map(|t| topologies(&frame_of(&f, t), &measurement))
            .collect();
        for (record_name, spec, key) in colour_records() {
            // `mutual_pairing_odd.json` omits the exchange-odd records (D1);
            // the surviving ones are compared here and in the D1 test.
            let Some(row) = records.get(record_name) else {
                continue;
            };
            let triplet = key.starts_with("triplets");
            let kind = if triplet {
                ElementKind::Triplet
            } else {
                ElementKind::DistancePair
            };
            let width = components(&spec, kind, &context);
            let values = record(row, width);
            let float32 = row["dtype"].as_str() == Some("f32");
            let count = ints(&counts[key]);
            for t in 0..frames {
                let state = &states[t];
                let [distance, cloning, triplets, _] = &built[t];
                let evaluated: Vec<Evaluated<'_>> = if triplet {
                    vec![(triplets, evaluate(&spec, triplets, state, &context, width))]
                } else {
                    vec![
                        (distance, evaluate(&spec, distance, state, &context, width)),
                        (cloning, evaluate(&spec, cloning, state, &context, width)),
                    ]
                };
                let parts: Vec<(&[Element], &[f64], &[bool])> = evaluated
                    .iter()
                    .map(|(topo, (v, ok))| (&topo.elements[..], &v[..], &ok[..]))
                    .collect();
                let (mean, weight) = masked_mean(&parts, width);
                // The count localises a failure to the mask before any value
                // is looked at; with `CompanionChoice::First` every element
                // weight is 1, so the weight sum is the fixture's integer.
                assert_eq!(
                    weight, count[t] as f64,
                    "{name} t{t}: {record_name} count ({key})"
                );
                match (&mean, &values[t]) {
                    (Some(got), Some(want)) => {
                        let label = format!("{name} t{t}: {record_name}");
                        let deviation = if float32 {
                            close_f32(&label, got, want)
                        } else {
                            close(&label, got, want, 1e-12)
                        };
                        worst.see(record_name, deviation);
                    }
                    // A frame with no valid element: the stored `0.0` of an
                    // `f32` record is a placeholder and is never compared
                    // (audit D11b). `masked.json` frame 5 is such a frame.
                    (None, _) => assert_eq!(count[t], 0, "{name} t{t}: {record_name} empty frame"),
                    (Some(_), None) => {
                        assert_eq!(count[t], 0, "{name} t{t}: {record_name} undefined frame");
                    }
                }
            }
        }
    }
    worst.report("colour_channel_frame_series");
}

#[test]
fn electroweak_phase_series_match_the_python_reference() {
    let mut worst = Deviations::default();
    let table: Vec<(&str, ChannelSpec, ElementKind, &str)> = vec![
        (
            "u1_phase",
            ChannelSpec::U1 {
                mode: U1Mode::Phase,
                charge: 1,
            },
            ElementKind::DistancePair,
            "u1",
        ),
        (
            "u1_phase_q2",
            ChannelSpec::U1 {
                mode: U1Mode::Phase,
                charge: 2,
            },
            ElementKind::DistancePair,
            "u1",
        ),
        (
            "u1_dressed",
            ChannelSpec::U1 {
                mode: U1Mode::Dressed,
                charge: 1,
            },
            ElementKind::DistancePair,
            "u1",
        ),
        (
            "u1_dressed_q2",
            ChannelSpec::U1 {
                mode: U1Mode::Dressed,
                charge: 2,
            },
            ElementKind::DistancePair,
            "u1",
        ),
        (
            "su2_phase",
            ChannelSpec::Su2 {
                mode: Su2Mode::Phase,
                directed: false,
            },
            ElementKind::CloningPair,
            "su2",
        ),
        (
            "su2_phase_directed",
            ChannelSpec::Su2 {
                mode: Su2Mode::Phase,
                directed: true,
            },
            ElementKind::CloningPair,
            "su2",
        ),
        (
            "fitness_phase",
            ChannelSpec::FitnessPhase,
            ElementKind::Site,
            "alive",
        ),
        (
            "clone_indicator",
            ChannelSpec::CloneIndicator,
            ElementKind::Site,
            "alive",
        ),
    ];
    for name in CASES {
        let f = fixture(name);
        let (frames, _) = shape(&f);
        let gas = gas_config(f["parameters"]["epsilon_clone"].as_f64().unwrap());
        let measurement = measurement_config(
            f["parameters"]["epsilon_d"].as_f64().unwrap(),
            f["parameters"]["lambda_alg"].as_f64().unwrap(),
        );
        let capabilities = Capabilities::nominal();
        let context = OperatorContext {
            gas: &gas,
            measurement: &measurement,
            capabilities: &capabilities,
        };
        // The fixture is a mean over `alive` while the Rust signature fixes
        // `1/N`; the parity comparison does the valid-count normalisation
        // itself, so the divergence is pinned here rather than hidden. It is
        // the chapter 04 against chapter 08 normalisation question, not a defect.
        assert_eq!(
            ChannelSpec::CloneIndicator
                .signature(ElementKind::Site, &context)
                .unwrap()
                .normalization,
            Some(FrameNormalization::FixedN),
            "{name}: clone indicator normalisation"
        );
        let counts = &f["electroweak"]["counts"];
        let records = &f["electroweak"]["records"];
        // The block holds no record outside the table: nothing is skipped.
        assert_eq!(
            records
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            table.iter().map(|(n, ..)| *n).collect::<BTreeSet<_>>(),
            "{name}: electroweak records"
        );
        for (record_name, spec, kind, key) in &table {
            let row = &records[record_name];
            let stored = width(row);
            let full = components(spec, *kind, &context);
            let values = record(row, stored);
            let count = ints(&counts[key]);
            for t in 0..frames {
                let frame = frame_of(&f, t);
                let state = state_of(&f, t);
                let [distance, cloning, _, sites] = topologies(&frame, &measurement);
                let topo = match kind {
                    ElementKind::DistancePair => &distance,
                    ElementKind::CloningPair => &cloning,
                    _ => &sites,
                };
                let (v, ok) = evaluate(spec, topo, &state, &context, full);
                let (mean, weight) = masked_mean(&[(&topo.elements, &v, &ok)], full);
                assert_eq!(
                    weight, count[t] as f64,
                    "{name} t{t}: {record_name} count ({key})"
                );
                let (Some(got), Some(want)) = (&mean, &values[t]) else {
                    assert_eq!(count[t], 0, "{name} t{t}: {record_name} empty frame");
                    continue;
                };
                // `mutual_pairing_odd` exports Re alone for the U(1) records:
                // their imaginary part is exchange odd (D1). The SU(2) phase
                // is only weight antisymmetric, so its imaginary part survives
                // the mutual pairing and is present and compared.
                // Python evaluates the phasors in complex64, so the README
                // fixes an absolute 1e-6 here. `fitness_phase` is the one
                // record of the block whose values leave the unit disc (up to
                // 1.2), and it takes the same absolute bound rather than the
                // relative relief `close` would grant it.
                worst.see(
                    record_name,
                    close_absolute(
                        &format!("{name} t{t}: {record_name}"),
                        &got[..stored],
                        want,
                        1e-6,
                    ),
                );
                // A `Su2 { mode: Phase }` value is an undressed unit-modulus
                // phasor although `epsilon_c` is configured to a fixed 0.9:
                // the phase arm carries no amplitude (audit D4).
                if *record_name == "su2_phase" {
                    let moduli: Vec<f64> = (0..topo.elements.len())
                        .filter(|&e| ok[e])
                        .map(|e| v[2 * e] * v[2 * e] + v[2 * e + 1] * v[2 * e + 1])
                        .collect();
                    worst.see(
                        "su2 unit modulus (D4)",
                        close(
                            &format!("{name} t{t}: su2 phase modulus"),
                            &moduli,
                            &vec![1.; moduli.len()],
                            1e-15,
                        ),
                    );
                }
            }
        }
    }
    worst.report("electroweak_phase_series");
}

#[test]
fn exchange_odd_frame_means_cancel_on_a_mutual_pairing_and_rust_reports_none() {
    let mut worst = Deviations::default();
    // Audit D1(a): `involutive` is computed from the elements by
    // `is_involutive`, never read from `Companions::mutual`, so the claim does
    // not depend on the flag the fixture record carries.
    for name in CASES {
        let f = fixture(name);
        let (frames, _) = shape(&f);
        let expected = f["involutive"].as_bool().unwrap();
        let measurement = measurement_config(
            f["parameters"]["epsilon_d"].as_f64().unwrap(),
            f["parameters"]["lambda_alg"].as_f64().unwrap(),
        );
        for t in 0..frames {
            let frame = frame_of(&f, t);
            for kind in [ElementKind::DistancePair, ElementKind::CloningPair] {
                let topo = topology::build(&frame, kind, &measurement);
                assert_eq!(
                    topo.involutive, expected,
                    "{name} t{t} {kind:?}: involutive"
                );
            }
        }
    }
    let name = "mutual_pairing_odd";
    let f = fixture(name);
    let (frames, n) = shape(&f);
    let gas = gas_config(f["parameters"]["epsilon_clone"].as_f64().unwrap());
    let measurement = measurement_config(
        f["parameters"]["epsilon_d"].as_f64().unwrap(),
        f["parameters"]["lambda_alg"].as_f64().unwrap(),
    );
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let cancelled: Vec<(&str, ChannelSpec, usize)> = vec![
        (
            "pseudoscalar",
            ChannelSpec::Meson {
                quantum: MesonQuantum::Pseudoscalar,
                mode: MesonMode::Standard,
            },
            1,
        ),
        (
            "vector",
            ChannelSpec::Vector {
                quantum: VectorQuantum::Vector,
                projection: VectorProjection::Full,
                displacement: Displacement::Raw,
            },
            3,
        ),
        (
            "vector_unit",
            ChannelSpec::Vector {
                quantum: VectorQuantum::Vector,
                projection: VectorProjection::Full,
                displacement: Displacement::Unit,
            },
            3,
        ),
    ];
    // The oracle of the identity test: the fixture's own float64 frame sum.
    let oracle = f["exchange_cancellation"]["max_abs_frame_sum_im_q_f64"]
        .as_f64()
        .unwrap();
    assert!(oracle < 1e-14 * n as f64, "{name}: fixture oracle");
    for t in 0..frames {
        let frame = frame_of(&f, t);
        let state = state_of(&f, t);
        for kind in [ElementKind::DistancePair, ElementKind::CloningPair] {
            let topo = topology::build(&frame, kind, &measurement);
            // Audit D1(b): every pair is mirrored, so the accumulator's Odd
            // frame mean runs over "valid elements whose mirror is not valid"
            // and has no contributing element: weight 0, an unavailable frame
            // mean rather than a roundoff number. `topology::mirrors` is a
            // helper the owner added outside the frozen contract, so the
            // load-bearing claim stays `Topology::involutive` above.
            let mirrors = topology::mirrors(&topo.elements);
            assert!(
                mirrors.iter().all(Option::is_some),
                "{name} t{t} {kind:?}: an unmirrored pair"
            );
            // Audit D1(d): the per-element table the source-frozen propagator
            // consumes is emphatically not zero.
            for (label, spec, width) in &cancelled {
                let (v, ok) = evaluate(spec, &topo, &state, &context, *width);
                let (mean, weight) = masked_mean(&[(&topo.elements, &v, &ok)], *width);
                assert!(weight > 0., "{name} t{t}: {label} has no element");
                let mean = mean.unwrap();
                // Audit D1(c): each role is separately involutive, so the
                // weighted frame sum of an exchange-odd operator cancels to
                // float64 roundoff on it alone.
                let sums: Vec<f64> = mean.iter().map(|m| m * weight).collect();
                worst.see(
                    &format!("{label} frame sum"),
                    close(
                        &format!("{name} t{t} {kind:?}: {label} frame sum"),
                        &sums,
                        &vec![0.; sums.len()],
                        1e-14 * n as f64,
                    ),
                );
                let largest = (0..topo.elements.len())
                    .filter(|&e| ok[e])
                    .flat_map(|e| v[e * width..(e + 1) * width].iter())
                    .fold(0f64, |m, x| m.max(x.abs()));
                assert!(
                    largest > 1e-3,
                    "{name} t{t} {kind:?}: {label} per-element table is degenerate ({largest:e})"
                );
            }
        }
    }
    // Audit D1(e): the six surviving channels reproduce their series through
    // the normal frame-mean path, and their magnitudes are the listed ones.
    let surviving = &f["exchange_cancellation"]["surviving"];
    let records = &f["series"]["records"];
    for (record_name, spec, key) in colour_records() {
        let Some(listed) = surviving.get(record_name).and_then(Value::as_f64) else {
            continue;
        };
        let row = &records[record_name];
        let width = components(&spec, ElementKind::DistancePair, &context);
        let values = record(row, width);
        let counts = ints(&f["series"]["counts"][key]);
        let mut peak = 0f64;
        for (t, want) in values.iter().enumerate() {
            let frame = frame_of(&f, t);
            let state = state_of(&f, t);
            let [distance, cloning, ..] = topologies(&frame, &measurement);
            let (a, a_ok) = evaluate(&spec, &distance, &state, &context, width);
            let (b, b_ok) = evaluate(&spec, &cloning, &state, &context, width);
            let (mean, weight) = masked_mean(
                &[
                    (&distance.elements, &a, &a_ok),
                    (&cloning.elements, &b, &b_ok),
                ],
                width,
            );
            assert_eq!(weight, counts[t] as f64, "{name} t{t}: {record_name} count");
            let got = mean.unwrap();
            worst.see(
                record_name,
                close_f32(
                    &format!("{name} t{t}: {record_name}"),
                    &got,
                    want.as_ref().unwrap(),
                ),
            );
            peak = got.iter().fold(peak, |m, x| m.max(x.abs()));
        }
        // `exchange_cancellation.surviving` is the peak of the same float32
        // series the loop above just matched frame by frame, so this is the
        // f32 bound of the README once more and the listed magnitude is at
        // most 0.31: the number is a legibility check on the README table, not
        // a second tolerance.
        assert!(
            listed > 1e-3,
            "{name}: {record_name} is listed as cancelled"
        );
        worst.see(
            "surviving magnitude",
            close_absolute(
                &format!("{name}: {record_name} magnitude"),
                &[peak],
                &[listed],
                1e-6,
            ),
        );
    }
    worst.report("exchange_odd_frame_means_cancel");
}

#[test]
fn masked_companion_records_reproduce_the_python_element_sets() {
    for name in CASES {
        let f = fixture(name);
        let (frames, n) = shape(&f);
        let measurement = measurement_config(
            f["parameters"]["epsilon_d"].as_f64().unwrap(),
            f["parameters"]["lambda_alg"].as_f64().unwrap(),
        );
        for t in 0..frames {
            let frame = frame_of(&f, t);
            for (role, kind) in [
                ("companions_distance", ElementKind::DistancePair),
                ("companions_clone", ElementKind::CloningPair),
            ] {
                // Out-of-range indices (`-1`, `N`, `N+3`), self companions and
                // dead walkers each build no element, and every other row does.
                let topo = topology::build(&frame, kind, &measurement);
                let got: BTreeSet<u32> = topo.elements.iter().map(|e| e.walkers[0]).collect();
                assert_eq!(
                    got,
                    expected_anchors(&f, t, role),
                    "{name} t{t} {kind:?}: element set"
                );
                assert_eq!(
                    topo.coverage.valid as usize,
                    topo.elements.len(),
                    "{name} t{t} {kind:?}: coverage"
                );
                // Audit D11d: Rust masks an electroweak self companion and the
                // fixtures were exported with the same masking; Python does
                // not. `Companions::mask` tests eligibility before the self
                // companion, so a dead walker pointing at itself is counted
                // as ineligible and only a living one as a self companion.
                let index = ints(&f["inputs"][role][t]);
                let alive = mask(&f["inputs"]["alive"][t]);
                let selves = (0..n).filter(|&i| index[i] == i as i64 && alive[i]).count();
                assert_eq!(
                    topo.coverage.masked_self as usize, selves,
                    "{name} t{t} {kind:?}: masked self companions"
                );
            }
        }
        // Audit D11d, second half: the `*_python_unmasked` counts document
        // what Python would have used and must differ wherever a self
        // companion was removed. They are never a comparison target.
        let counts = &f["electroweak"]["counts"];
        for (key, unmasked, role) in [
            ("u1", "u1_python_unmasked", "companions_distance"),
            ("su2", "su2_python_unmasked", "companions_clone"),
        ] {
            let kept = ints(&counts[key]);
            let python = ints(&counts[unmasked]);
            for t in 0..frames {
                let index = ints(&f["inputs"][role][t]);
                let alive = mask(&f["inputs"]["alive"][t]);
                let selves = (0..n).filter(|&i| index[i] == i as i64 && alive[i]).count() as i64;
                assert_eq!(
                    python[t] - kept[t],
                    selves,
                    "{name} t{t}: {unmasked} against {key}"
                );
            }
        }
    }
    // Audit D11b: `masked.json` frame 5 has no valid colour, so every colour
    // channel has weight 0 and no frame mean, while the electroweak channels
    // are compared normally. The topology is not empty there: it is
    // colour-blind, and every element is masked by `evaluate`.
    let f = fixture("masked");
    let gas = gas_config(f["parameters"]["epsilon_clone"].as_f64().unwrap());
    let measurement = measurement_config(
        f["parameters"]["epsilon_d"].as_f64().unwrap(),
        f["parameters"]["lambda_alg"].as_f64().unwrap(),
    );
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let frame = frame_of(&f, 5);
    let state = state_of(&f, 5);
    let [distance, cloning, triplets, _] = topologies(&frame, &measurement);
    assert!(
        !distance.elements.is_empty() && !triplets.elements.is_empty(),
        "masked t5: the colour-blind topology is empty"
    );
    assert_eq!(
        ints(&f["series"]["counts"]["pairs"])[5],
        0,
        "masked t5: pairs"
    );
    assert_eq!(
        ints(&f["electroweak"]["counts"]["u1"])[5],
        5,
        "masked t5: u1"
    );
    for (record_name, spec, key) in colour_records() {
        let triplet = key.starts_with("triplets");
        let kind = if triplet {
            ElementKind::Triplet
        } else {
            ElementKind::DistancePair
        };
        let width = components(&spec, kind, &context);
        let evaluated: Vec<Evaluated<'_>> = if triplet {
            vec![(
                &triplets,
                evaluate(&spec, &triplets, &state, &context, width),
            )]
        } else {
            vec![
                (
                    &distance,
                    evaluate(&spec, &distance, &state, &context, width),
                ),
                (&cloning, evaluate(&spec, &cloning, &state, &context, width)),
            ]
        };
        let parts: Vec<(&[Element], &[f64], &[bool])> = evaluated
            .iter()
            .map(|(topo, (v, ok))| (&topo.elements[..], &v[..], &ok[..]))
            .collect();
        let (mean, weight) = masked_mean(&parts, width);
        assert!(
            mean.is_none() && weight == 0.,
            "masked t5: {record_name} should have no frame mean"
        );
    }
    // Every alive walker of that frame still carries a fitness phase.
    let sites = topology::build(&frame, ElementKind::Site, &measurement);
    assert_eq!(
        sites.elements.len(),
        frame.eligible.iter().filter(|e| **e).count(),
        "masked t5: site topology"
    );
}

#[test]
fn operator_signatures_pin_the_audited_exchange_parities_and_component_counts() {
    let gas = gas_config(0.01);
    let measurement = measurement_config(0.9, 0.);
    let capabilities = Capabilities::nominal();
    let context = OperatorContext {
        gas: &gas,
        measurement: &measurement,
        capabilities: &capabilities,
    };
    let signature = |spec: &ChannelSpec, kind| spec.signature(kind, &context).unwrap();
    let pair = ElementKind::DistancePair;
    let meson = |quantum, mode| ChannelSpec::Meson { quantum, mode };
    let vector = |quantum, projection, displacement| ChannelSpec::Vector {
        quantum,
        projection,
        displacement,
    };
    // Audit D1: the exchange parity is what makes the estimator choice, so it
    // is pinned for the cancelled and the surviving sets alike.
    assert_eq!(
        signature(&meson(MesonQuantum::Scalar, MesonMode::Standard), pair).exchange,
        ExchangeParity::Even
    );
    assert_eq!(
        signature(
            &meson(MesonQuantum::Pseudoscalar, MesonMode::Standard),
            pair
        )
        .exchange,
        ExchangeParity::Odd
    );
    // Score direction repairs the pseudoscalar: the orientation sign and the
    // imaginary part both change under exchange.
    assert_eq!(
        signature(
            &meson(MesonQuantum::Pseudoscalar, MesonMode::ScoreDirected),
            pair
        )
        .exchange,
        ExchangeParity::Even
    );
    let raw_vector = vector(
        VectorQuantum::Vector,
        VectorProjection::Full,
        Displacement::Raw,
    );
    let raw_axial = vector(
        VectorQuantum::Axial,
        VectorProjection::Full,
        Displacement::Raw,
    );
    assert_eq!(signature(&raw_vector, pair).exchange, ExchangeParity::Odd);
    assert_eq!(signature(&raw_axial, pair).exchange, ExchangeParity::Even);
    assert_eq!(
        signature(
            &ChannelSpec::Baryon {
                mode: BaryonMode::Complex,
                flux_alpha: 1.
            },
            ElementKind::Triplet
        )
        .exchange,
        ExchangeParity::Odd
    );
    for observable in [
        GlueballObservable::RePlaquette,
        GlueballObservable::OneMinusRe,
        GlueballObservable::OneMinusCos,
        GlueballObservable::Sin2,
    ] {
        let spec = ChannelSpec::Glueball {
            observable,
            momentum: None,
        };
        let plaquette = signature(&spec, ElementKind::Triplet);
        assert_eq!(plaquette.exchange, ExchangeParity::Even, "{observable:?}");
        assert!(!plaquette.degenerate, "{observable:?}");
    }
    assert_eq!(
        signature(
            &ChannelSpec::U1 {
                mode: U1Mode::Phase,
                charge: 1
            },
            pair
        )
        .exchange,
        ExchangeParity::Mixed
    );
    // The score arms leave the contraction in the sampled order and report the
    // factor the score contributes in an auxiliary column, which the
    // measurement freezes with the element. Python folds that factor into the
    // value, and so does the comparison of the series tests; the column is
    // pinned here so the folding never becomes silent.
    assert_eq!(
        signature(&meson(MesonQuantum::Scalar, MesonMode::ScoreDirected), pair).auxiliary,
        Some(Auxiliary::ScoreOrientation)
    );
    assert_eq!(
        signature(
            &meson(MesonQuantum::Pseudoscalar, MesonMode::ScoreWeighted),
            pair
        )
        .auxiliary,
        Some(Auxiliary::ScoreDispersion)
    );
    assert_eq!(
        signature(
            &ChannelSpec::Baryon {
                mode: BaryonMode::ScoreOrdered,
                flux_alpha: 1.
            },
            ElementKind::Triplet
        )
        .auxiliary,
        Some(Auxiliary::ScoreOrientation)
    );
    assert_eq!(
        signature(&meson(MesonQuantum::Scalar, MesonMode::Standard), pair).auxiliary,
        None
    );
    // Audit D8: components are kept and contracted in the correlator, never
    // averaged. `Vector::signature` answers `capabilities.dimension`, which is
    // 3 for `nominal()`; the family is not hard-coded to three components.
    assert_eq!(signature(&raw_vector, pair).components, 3);
    assert_eq!(signature(&raw_axial, pair).components, 3);
    // Audit D11a: a diagnostic envelope never enters a correlator.
    assert!(
        !signature(
            &ChannelSpec::Tensor {
                mode: TensorMode::Envelope
            },
            pair
        )
        .correlatable
    );
    // The mixed triplet is the only degenerate-tolerant arm of the set.
    assert!(signature(&ChannelSpec::ElectroweakMixed, ElementKind::Triplet).degenerate);
    assert!(
        !signature(
            &ChannelSpec::Baryon {
                mode: BaryonMode::Complex,
                flux_alpha: 1.
            },
            ElementKind::Triplet
        )
        .degenerate
    );
    // `pseudoscalar/abs2` duplicates the scalar `|q|²` and does not exist.
    assert!(matches!(
        meson(MesonQuantum::Pseudoscalar, MesonMode::Abs2).signature(pair, &context),
        Err(GasError::Capability(_))
    ));
    // Audit D11e: only `Full` carries fixture data; the other projections read
    // the score gradient of the anchor alone, so they are a different
    // observable and not a silently ignored option.
    let longitudinal = signature(
        &vector(
            VectorQuantum::Vector,
            VectorProjection::Longitudinal,
            Displacement::Raw,
        ),
        pair,
    );
    assert_eq!(longitudinal.exchange, ExchangeParity::Mixed);
    assert!(
        longitudinal.requires.records.contains(&Record::Fitness)
            && longitudinal
                .requires
                .records
                .contains(&Record::CloningCompanions)
    );
    assert!(
        !signature(&raw_vector, pair)
            .requires
            .records
            .contains(&Record::Fitness)
    );
    // Audit D11g: a momentum projection requires a periodic box. `signature`
    // never reads `gas.boundary`; it is `Capabilities::of` that turns an
    // unbounded boundary into a missing record, and `nominal()` misses none,
    // so the claim is about the requirement, not about availability. Mode 0
    // with a sine phase is rejected by `ChannelSpec::validate`, hence mode 1.
    let projected = ChannelSpec::Glueball {
        observable: GlueballObservable::RePlaquette,
        momentum: Some(Momentum {
            axis: 0,
            mode: 1,
            phase: MomentumPhase::Cos,
        }),
    };
    assert!(
        signature(&projected, ElementKind::Triplet)
            .requires
            .records
            .contains(&Record::PeriodicBox)
    );
    assert!(
        !signature(
            &ChannelSpec::Glueball {
                observable: GlueballObservable::RePlaquette,
                momentum: None
            },
            ElementKind::Triplet
        )
        .requires
        .records
        .contains(&Record::PeriodicBox)
    );
    // Audit D4: `epsilon_c` is a real input and not a synonym of the clone
    // regulariser. `nominal()` carries a cloning kernel width, so
    // `Range::FromKernel` succeeds on it and fails without one: the capability
    // decided, not the specification.
    let from_kernel = MeasurementConfig {
        electroweak: ElectroweakScales {
            epsilon_c: Range::FromKernel,
            ..measurement.electroweak
        },
        ..measurement.clone()
    };
    let component = ChannelSpec::Su2 {
        mode: Su2Mode::Component,
        directed: false,
    };
    let no_kernel = Capabilities {
        cloning_kernel_width: None,
        ..Capabilities::nominal()
    };
    let without = OperatorContext {
        gas: &gas,
        measurement: &from_kernel,
        capabilities: &no_kernel,
    };
    match component.signature(ElementKind::CloningPair, &without) {
        Err(GasError::Capability(message)) => {
            assert!(message.contains("epsilon_c"), "{message}");
        }
        other => panic!("expected a capability error naming epsilon_c, got {other:?}"),
    }
    let with = OperatorContext {
        gas: &gas,
        measurement: &from_kernel,
        capabilities: &capabilities,
    };
    assert!(component.signature(ElementKind::CloningPair, &with).is_ok());
}

#[test]
fn lagged_moments_match_the_python_correlator_statistics() {
    let mut worst = Deviations::default();
    for name in CASES {
        let f = fixture(name);
        let statistics = &f["statistics"];
        let max_lag = statistics["max_lag"].as_u64().unwrap() as usize;
        let lags = max_lag + 1;
        let scalar = flat(&statistics["scalar_series"]);
        let vector = flat(&statistics["vector_series"]);
        let frames = scalar.len();
        let weight = vec![1.; frames];
        let segment = vec![0u32; frames];
        // Python's `connected` is the series minus its full-series mean per
        // component, formed before any product, then `Subtraction::None`.
        // `Subtraction::GlobalMean` is a different estimator: its means are
        // the lag-0 moments, whose denominators differ per lag.
        let centred = |values: &[f64], components: usize| -> Vec<f64> {
            let rows = values.len() / components;
            let mean: Vec<f64> = (0..components)
                .map(|k| (0..rows).map(|t| values[t * components + k]).sum::<f64>() / rows as f64)
                .collect();
            values
                .iter()
                .enumerate()
                .map(|(i, x)| x - mean[i % components])
                .collect()
        };
        for block in ["connected", "raw"] {
            let scalar_series = if block == "connected" {
                centred(&scalar, 1)
            } else {
                scalar.clone()
            };
            let vector_series = if block == "connected" {
                centred(&vector, 3)
            } else {
                vector.clone()
            };
            let moments = series_moments(
                SeriesView {
                    values: &scalar_series,
                    weight: &weight,
                    segment: &segment,
                    components: 1,
                },
                max_lag,
                1,
            )
            .unwrap();
            // One block per origin, so the block index is the origin index; a
            // change to `BlockMoments::new` would otherwise break the indexing.
            assert_eq!(moments.blocks, frames, "{name} {block}: blocks");
            if block == "connected" {
                // `scalar_sums` and `scalar_counts` exist in this block only.
                worst.see(
                    "statistics.scalar_sums",
                    close(
                        &format!("{name} {block}: scalar_sums"),
                        &moments.ab,
                        &flat(&statistics[block]["scalar_sums"]),
                        1e-10,
                    ),
                );
                worst.see(
                    "statistics.scalar_counts",
                    close(
                        &format!("{name} {block}: scalar_counts"),
                        &moments.n,
                        &flat(&statistics[block]["scalar_counts"]),
                        1e-10,
                    ),
                );
            }
            let scalar_mean = defined(
                &format!("{name} {block}: scalar_mean"),
                &moments.estimate(Subtraction::None),
            );
            assert_eq!(scalar_mean.len(), lags);
            let expected = flat(&statistics[block]["scalar_mean"]);
            worst.see(
                "statistics.scalar_mean",
                close(
                    &format!("{name} {block}: scalar_mean"),
                    &scalar_mean,
                    &expected,
                    1e-10,
                ),
            );
            // Rust has no FFT correlator; the fixture states the two Python
            // estimators agree to 1e-12 and the measured spread is 1e-15, so
            // both arrays are compared against the single Rust estimate.
            worst.see(
                "statistics.scalar_fft",
                close(
                    &format!("{name} {block}: scalar_fft"),
                    &scalar_mean,
                    &flat(&statistics[block]["scalar_fft"]),
                    1e-10,
                ),
            );
            // `cross_moments` contracts the components inside `ab`, so the
            // estimate is already `C(τ) = Σ_k C_kk(τ)` (audit D8).
            let contracted = defined(
                &format!("{name} {block}: vector"),
                &series_moments(
                    SeriesView {
                        values: &vector_series,
                        weight: &weight,
                        segment: &segment,
                        components: 3,
                    },
                    max_lag,
                    1,
                )
                .unwrap()
                .estimate(Subtraction::None),
            );
            worst.see(
                "statistics.vector_contracted_mean",
                close(
                    &format!("{name} {block}: vector_contracted_mean"),
                    &contracted,
                    &flat(&statistics[block]["vector_contracted_mean"]),
                    1e-10,
                ),
            );
            worst.see(
                "statistics.vector_contracted_fft",
                close(
                    &format!("{name} {block}: vector_contracted_fft"),
                    &contracted,
                    &flat(&statistics[block]["vector_contracted_fft"]),
                    1e-10,
                ),
            );
        }
    }
    worst.report("lagged_moments");
}

/// Log-space window fit of the exported correlator: `y = ln C`,
/// `σ = error/|C|`, design rows `[1, t]` on the absolute lag.
/// Returns `(mass, variance, chi2, dof)`.
fn log_window(
    correlator: &[f64],
    error: &[f64],
    t0: usize,
    width: usize,
) -> (f64, f64, f64, usize) {
    let sigma: Vec<f64> = (t0..t0 + width)
        .map(|t| error[t] / correlator[t].abs())
        .collect();
    let y: Vec<f64> = (t0..t0 + width).map(|t| correlator[t].ln()).collect();
    let design: Vec<f64> = (t0..t0 + width).flat_map(|t| [1., t as f64]).collect();
    let fit = linear_fit(&design, &y, &Whitener::diagonal(&sigma).unwrap(), 2).unwrap();
    (-fit.beta[1], fit.covariance[3], fit.chi2, fit.dof)
}

#[test]
fn window_scan_fits_and_aic_weights_match_the_python_table() {
    let mut worst = Deviations::default();
    for name in CASES {
        let f = fixture(name);
        let scan = &f["window_scan"];
        // The fit is quoted per unit time; the exported tables were made at
        // `dt = 1`, so the absolute lag is the time.
        assert_eq!(scan["dt"].as_f64(), Some(1.), "{name}: dt");
        let correlator = flat(&scan["correlator"]);
        let error = flat(&scan["error"]);
        let lags = correlator.len();
        let max_log_error = scan["max_log_error"].as_f64().unwrap();
        let widths: Vec<usize> = ints(&scan["window_widths"])
            .into_iter()
            .map(|w| w as usize)
            .collect();
        let point_valid: Vec<bool> = (0..lags)
            .map(|t| {
                let sigma = error[t] / correlator[t].abs();
                correlator[t].is_finite()
                    && correlator[t] > 0.
                    && sigma.is_finite()
                    && sigma > 0.
                    && sigma <= max_log_error
            })
            .collect();
        same_mask(
            &format!("{name}: point_valid"),
            &point_valid,
            &mask(&scan["point_valid"]),
        );
        let points = point_valid.iter().filter(|v| **v).count();
        let mass = &scan["window_mass"];
        let variance = &scan["window_mass_variance"];
        let chi2 = &scan["window_chi2"];
        let aic = &scan["window_aic"];
        // `(width, t0, mass, variance, aic)` of every exported window.
        let mut table: Vec<(usize, usize, f64, f64, f64)> = vec![];
        for (wi, &width) in widths.iter().enumerate() {
            for t0 in 0..lags {
                let usable = t0 + width <= lags && (t0..t0 + width).all(|t| point_valid[t]);
                if mass[wi][t0].is_null() {
                    assert!(!usable, "{name}: window ({width}, {t0}) is null but usable");
                    continue;
                }
                assert!(
                    usable,
                    "{name}: window ({width}, {t0}) is exported but unusable"
                );
                let (m, v, c, dof) = log_window(&correlator, &error, t0, width);
                assert_eq!(dof, width - 2, "{name}: window ({width}, {t0}) dof");
                worst.see(
                    "window_mass",
                    close(
                        &format!("{name}: window_mass[{width}][{t0}]"),
                        &[m],
                        &[mass[wi][t0].as_f64().unwrap()],
                        1e-9,
                    ),
                );
                worst.see(
                    "window_mass_variance",
                    close(
                        &format!("{name}: window_mass_variance[{width}][{t0}]"),
                        &[v],
                        &[variance[wi][t0].as_f64().unwrap()],
                        1e-9,
                    ),
                );
                // Python evaluates χ² through the normal-equation identity and
                // loses digits when it is small (8e-8 relative, 2e-11
                // absolute); `linear_fit` sums explicit residuals, so the
                // comparison is absolute.
                let expected_chi2 = chi2[wi][t0].as_f64().unwrap();
                worst.see(
                    "window_chi2",
                    close(
                        &format!("{name}: window_chi2[{width}][{t0}]"),
                        &[c],
                        &[expected_chi2],
                        1e-9,
                    ),
                );
                // Audit D11j: no positivity filter and no `min_mass` enters
                // the average, and the fixture still reproduces, which is the
                // proof that Python's filter was inert on this correlator.
                let criterion = c + 4. + 2. * (points - width) as f64;
                worst.see(
                    "window_aic",
                    close(
                        &format!("{name}: window_aic[{width}][{t0}]"),
                        &[criterion],
                        &[aic[wi][t0].as_f64().unwrap()],
                        1e-9,
                    ),
                );
                table.push((width, t0, m, v, criterion));
            }
        }
        assert_eq!(
            table.len(),
            scan["n_valid_windows"].as_u64().unwrap() as usize,
            "{name}: valid window count"
        );
        let best_criterion = table.iter().map(|w| w.4).fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = table
            .iter()
            .map(|w| (-0.5 * (w.4 - best_criterion)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.into_iter().map(|w| w / total).collect();
        let averaged: f64 = weights.iter().zip(&table).map(|(w, r)| w * r.2).sum();
        let statistical: f64 = weights
            .iter()
            .zip(&table)
            .map(|(w, r)| w * r.3)
            .sum::<f64>()
            .sqrt();
        let spread: f64 = weights
            .iter()
            .zip(&table)
            .map(|(w, r)| w * (r.2 - averaged) * (r.2 - averaged))
            .sum::<f64>()
            .sqrt();
        for (label, got, want) in [
            ("mass", averaged, scan["mass"].as_f64().unwrap()),
            (
                "statistical_error",
                statistical,
                scan["statistical_error"].as_f64().unwrap(),
            ),
            (
                "window_spread",
                spread,
                scan["window_spread"].as_f64().unwrap(),
            ),
            (
                "mass_error",
                statistical.hypot(spread),
                scan["mass_error"].as_f64().unwrap(),
            ),
        ] {
            worst.see(
                label,
                close(&format!("{name}: {label}"), &[got], &[want], 1e-9),
            );
        }
        let best = (0..table.len()).fold(0, |b, i| if table[i].4 < table[b].4 { i } else { b });
        let expected = &scan["best_window"];
        assert_eq!(
            [table[best].0, table[best].1],
            [
                expected["width"].as_u64().unwrap() as usize,
                expected["t_start"].as_u64().unwrap() as usize
            ],
            "{name}: best window"
        );
        worst.see(
            "best_window.mass",
            close(
                &format!("{name}: best_window.mass"),
                &[table[best].2, table[best].3.sqrt(), table[best].4],
                &[
                    expected["mass"].as_f64().unwrap(),
                    expected["mass_error"].as_f64().unwrap(),
                    expected["aic"].as_f64().unwrap(),
                ],
                1e-9,
            ),
        );
    }
    worst.report("window_scan_fits_and_aic_weights");
}

#[test]
fn the_delivered_window_scan_reproduces_every_exported_window_row() {
    let mut worst = Deviations::default();
    for name in CASES {
        let f = fixture(name);
        let scan = &f["window_scan"];
        let correlator = flat(&scan["correlator"]);
        let error = flat(&scan["error"]);
        let lags = correlator.len();
        let covariance: Vec<f64> = (0..lags * lags)
            .map(|c| {
                if c / lags == c % lags {
                    error[c / lags] * error[c / lags]
                } else {
                    0.
                }
            })
            .collect();
        let estimated = Estimated {
            channel: "fixture/window_scan".into(),
            kind: EstimatorKind::FrameMean,
            estimate: CorrelatorEstimate {
                lags: (0..lags).collect(),
                time_unit: TimeUnit::Frames,
                time_step: scan["dt"].as_f64().unwrap(),
                value: correlator.iter().map(|&c| Some(c)).collect(),
                error: error.iter().map(|&e| Some(e)).collect(),
                covariance: Some(covariance),
                samples_meta: SamplesMeta {
                    resampling: ResampleKind::Jackknife,
                    effective_block: 1,
                    blocks: 2,
                    tau_int: None,
                    covariance_rank: lags,
                    replicas: 1,
                    sampling_unit: "fixture".into(),
                },
                connected: false,
                connected_bias: None,
            },
            // Only `dimension` is read: the covariance is supplied above, so
            // the resample table only has to be shape-valid.
            samples: Samples {
                kind: ResampleKind::Jackknife,
                dimension: lags,
                count: 2,
                central: correlator.clone(),
                values: [correlator.clone(), correlator.clone()].concat(),
                defined: vec![true; lags],
                effective_block: 1,
                blocks: 2,
                tau_int: None,
            },
        };
        let analysis = AnalysisConfig {
            window_scan: WindowScanConfig {
                t_min: 0,
                t_max: Some(lags - 1),
                min_points: 3,
                // `min_point_snr` stays at its default 2.0: that is exactly
                // Python's `max_log_error = 0.5` filter, so the usable range
                // and therefore the AIC of every window agree.
                correlated: false,
                ..WindowScanConfig::default()
            },
            ..AnalysisConfig::default()
        };
        let outcome = window_scan(&estimated, &analysis).unwrap();
        let points = mask(&scan["point_valid"]).iter().filter(|v| **v).count();
        let widths: Vec<usize> = ints(&scan["window_widths"])
            .into_iter()
            .map(|w| w as usize)
            .collect();
        let mut matched = 0;
        for (wi, &width) in widths.iter().enumerate() {
            for t0 in 0..lags {
                if scan["window_mass"][wi][t0].is_null() {
                    continue;
                }
                let row = outcome
                    .windows
                    .iter()
                    .find(|w| w.t_min == t0 && w.t_max == t0 + width - 1)
                    .unwrap_or_else(|| panic!("{name}: no window row ({width}, {t0})"));
                matched += 1;
                assert_eq!(row.dof, width - 2, "{name}: ({width}, {t0}) dof");
                // The design is `[1, −t]`, so `value` is the mass itself, and
                // with `correlated: false` its squared error is the sandwich
                // over a diagonal block, algebraically `covariance[1][1]`.
                worst.see(
                    "delivered window mass",
                    close(
                        &format!("{name}: delivered mass ({width}, {t0})"),
                        &[row.value, row.error * row.error],
                        &[
                            scan["window_mass"][wi][t0].as_f64().unwrap(),
                            scan["window_mass_variance"][wi][t0].as_f64().unwrap(),
                        ],
                        1e-9,
                    ),
                );
                worst.see(
                    "delivered window chi2",
                    close(
                        &format!("{name}: delivered chi2 ({width}, {t0})"),
                        &[row.chi2, row.chi2 + 4. + 2. * (points - width) as f64],
                        &[
                            scan["window_chi2"][wi][t0].as_f64().unwrap(),
                            scan["window_aic"][wi][t0].as_f64().unwrap(),
                        ],
                        1e-9,
                    ),
                );
                // `WindowFit::aic` is the delivered criterion; it must be the
                // one recomputed above, which pins `usable == points`.
                worst.see(
                    "delivered window aic",
                    close(
                        &format!("{name}: delivered aic ({width}, {t0})"),
                        &[row.aic],
                        &[row.chi2 + 4. + 2. * (points - width) as f64],
                        1e-12,
                    ),
                );
            }
        }
        assert_eq!(
            matched,
            scan["n_valid_windows"].as_u64().unwrap() as usize,
            "{name}: matched window rows"
        );
        // `FitOutcome::mass` averages every width from `min_points` upward,
        // Python only the three fixed widths, so the model averages cannot
        // agree by construction and are not compared. The weights are
        // normalised over the wider Rust set for the same reason.
        assert!(
            outcome.windows.len() > matched,
            "{name}: the Rust scan should enumerate more windows than Python"
        );
    }
    worst.report("delivered_window_scan");
}
