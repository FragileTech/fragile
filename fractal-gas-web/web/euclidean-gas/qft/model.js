// Session adapter and pure Rust-JSON -> view-descriptor mappings of the QFT
// Simulator. Nothing here samples, fits, estimates or arranges numbers: every
// plotted point, the lag axis in the reported time unit and both edges of every
// error band are produced by `spectroscopy/presentation.rs` and copied here.
// The tables copy the fields of the report as they are.
import { COLORS, format } from "../lecture/plots.js";

// Wire names of `spectroscopy/config.rs` enums with their UI labels. A value
// Rust does not accept comes back as a Rust error and is shown verbatim.
export const VOCABULARY = Object.freeze({
  colorSource: [
    ["viscous_force", "Viscous force (book default)"],
    ["recorded_field", "Recorded field (explicit opt-in)"],
  ],
  alignment: [
    [
      "preceding_kick",
      "Preceding kick · B2 force of step t−1 with its input velocity",
    ],
    [
      "matched_kick",
      "Matched kick · a B stage of step t, cloned walkers masked",
    ],
    [
      "reference_offset",
      "Reference offset · pre-clone velocity with the B2 force of step t−1",
    ],
  ],
  stage: [
    ["b1", "B1"],
    ["b2", "B2"],
  ],
  timeAxis: [
    ["monte_carlo", "Algorithm (Monte Carlo) time"],
    ["euclidean", "Also Euclidean-time slabs of one coordinate"],
  ],
  estimator: [
    ["auto", "Automatic"],
    ["frame_mean", "Frame average"],
    ["source_frozen", "Source-frozen propagator"],
    ["euclidean_time", "Euclidean time"],
  ],
  resampling: [
    ["block_jackknife", "Block jackknife"],
    ["bootstrap", "Block bootstrap"],
    ["uncorrelated", "Uncorrelated origins"],
  ],
  combine: [
    ["pooled_blocks", "Pool the blocks of all replicas"],
    ["runs_as_samples", "Replicas as samples (needs at least 8)"],
  ],
  effectiveRate: [
    ["log_ratio", "Log ratio"],
    ["cosh", "Cosh"],
  ],
  fit: [
    ["window_scan", "Window scan"],
    ["multi_exponential", "Multi-exponential"],
    ["both", "Both"],
  ],
  timeUnit: [
    ["frames", "Frames"],
    ["step_dt", "stride · dt"],
  ],
});

// `GevpBasis::validate` of `spectroscopy/config.rs` accepts 2..=16 channels
// ("a GEVP basis needs an id, 2..=16 channels, a cut in [0, 1) and a reference
// lag beyond t0"). The page never sends a basis Rust would reject.
export const GEVP_BASIS = Object.freeze({ min: 2, max: 16 });
export const gevpReady = (count) =>
  count >= GEVP_BASIS.min && count <= GEVP_BASIS.max;

const ESTIMATOR_LABEL = {
  frame_mean: "frame average",
  source_frozen: "source-frozen propagator",
  euclidean_time: "Euclidean time",
};
const METHOD_LABEL = {
  window_scan: "Window scan",
  multi_exponential: "Multi-exponential",
  gevp: "GEVP",
  stability: "Stability scan",
};
// `ChannelReport.normalization` is the denominator the frame averages used;
// only the fixed-N arm has a transfer-matrix reading (09_qft_calibration).
const NORMALIZATION_LABEL = {
  valid_count: "sum of valid element weights",
  fixed_n: "population size N",
};
const clone = (value) =>
  value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const orNull = (value) => (finite(value) ? value : null);
const unitLabel = (unit) => (unit === "step_dt" ? "stride · dt" : "frames");

function stable(value) {
  if (Array.isArray(value)) return "[" + value.map(stable).join(",") + "]";
  if (value && typeof value === "object")
    return (
      "{" +
      Object.keys(value)
        .sort()
        .map((key) => JSON.stringify(key) + ":" + stable(value[key]))
        .join(",") +
      "}"
    );
  return JSON.stringify(value);
}

// ---------------------------------------------------------------- setup

export function catalogFamilies(catalog = []) {
  const families = new Map();
  for (const entry of catalog) {
    const family = entry.family || String(entry.id).split("/")[0];
    if (!families.has(family)) families.set(family, []);
    families.get(family).push(entry);
  }
  return [...families].map(([family, channels]) => ({ family, channels }));
}

// The request Rust resolved for a variant (`variants[i].request`); the default
// request when the variant carries none.
export function variantRequest(defaults, name) {
  const variant = (defaults.variants || []).find((v) => v.name === name);
  return variant?.request ?? defaults.request;
}

// Form state of a Rust request. The selected catalog rows are the rows of the
// specifications that request measures; a row is one (specification, element
// kind) pair, so one specification can own several rows.
export function formFromRequest(defaults, request) {
  const measurement = request.spectroscopy.measurement;
  const requested = new Set((measurement.channels || []).map(stable));
  const implemented = (defaults.variants || []).find((v) => v.implemented);
  return {
    variant: request.variant ?? implemented?.name ?? null,
    walkers: request.run.walkers,
    dimensions: request.run.dimensions,
    steps: request.steps,
    replicas: request.replicas,
    seed: request.seed,
    color: clone(measurement.color),
    time: clone(measurement.time),
    channels: (defaults.catalog || [])
      .filter((entry) => requested.has(stable(entry.spec)))
      .map((entry) => entry.id),
  };
}
export const formFromDefaults = (defaults) =>
  formFromRequest(defaults, defaults.request);

// `availability` is the Map of availabilityIndex(); with `probe` every catalog
// channel is requested so that Rust reports on all of them.
export function buildRequest(
  defaults,
  form,
  availability = new Map(),
  { probe = false } = {},
) {
  const request = clone(variantRequest(defaults, form.variant));
  request.variant = form.variant;
  request.run.walkers = form.walkers;
  request.run.dimensions = form.dimensions;
  request.steps = form.steps;
  request.replicas = form.replicas;
  request.seed = form.seed;
  const measurement = request.spectroscopy.measurement;
  measurement.color = clone(form.color);
  measurement.time = clone(form.time);
  const selected = new Set(form.channels);
  // Rust measures specifications; rows of one specification are sent once.
  const specs = new Map();
  for (const entry of defaults.catalog || [])
    if (
      probe ||
      (selected.has(entry.id) &&
        availability.get(entry.id)?.available !== false)
    )
      specs.set(stable(entry.spec), clone(entry.spec));
  measurement.channels = [...specs.values()];
  return request;
}

export function colorSourceOfKind(kind, previous = {}) {
  const threshold = previous.threshold;
  return kind === "recorded_field"
    ? {
        kind,
        stage: previous.stage ?? "",
        amplitude: previous.amplitude ?? "",
        phase: previous.phase ?? "",
        ...(threshold === undefined ? {} : { threshold }),
      }
    : {
        kind: "viscous_force",
        alignment: previous.alignment ?? { kind: "preceding_kick" },
        ...(threshold === undefined ? {} : { threshold }),
      };
}

// `spectroscopy_capabilities(request).channels`, or the catalog rows of
// `spectroscopy_defaults()` (both are `{id, availability}` rows keyed by channel
// id) -> Map id -> {available, reason}. The reason string is Rust's and is
// never rewritten.
export function availabilityIndex(source) {
  const index = new Map();
  const rows = Array.isArray(source) ? source : source?.channels || [];
  for (const { id, availability } of rows)
    index.set(id, {
      available: availability?.status !== "unavailable",
      reason:
        availability?.status === "unavailable"
          ? String(availability.reason ?? "")
          : null,
    });
  return index;
}

export function missingRecords(response) {
  return Object.entries(response?.capabilities?.missing || {}).map(
    ([record, reason]) => [record, reason],
  );
}

// ------------------------------------------------------------- analysis

export function controlsFromAnalysis(analysis, seed = 0) {
  const resampling = analysis.resampling || {};
  return {
    estimator: analysis.estimator,
    resampling: resampling.kind,
    block: resampling.block?.kind === "fixed" ? "fixed" : "auto",
    blockFrames: resampling.block?.frames ?? 16,
    samples: resampling.samples ?? 200,
    resampleSeed: resampling.seed ?? seed,
    connected: analysis.connected,
    combine: analysis.combine,
    effectiveRate: analysis.effective_mass,
    fit: analysis.fit,
    timeUnit: analysis.time_unit,
    stability: analysis.stability != null,
    gevp: (analysis.gevp || []).length > 0,
    assignments: { ...(analysis.assignments || {}) },
    anchors: [...(analysis.anchors || [])],
    // Channel or specification ids Rust analyses; empty selects every one.
    channels: [...(analysis.channels || [])],
  };
}

// Overrides the user-facing fields of a Rust `AnalysisConfig`; every other
// field (fit windows, priors, SVD cut, reference table) stays as Rust sent it.
export function buildAnalysis(base, controls, gevpChannels = []) {
  const analysis = clone(base);
  const block =
    controls.block === "fixed"
      ? { kind: "fixed", frames: controls.blockFrames }
      : { kind: "auto" };
  analysis.estimator = controls.estimator;
  analysis.resampling =
    controls.resampling === "uncorrelated"
      ? { kind: "uncorrelated" }
      : controls.resampling === "bootstrap"
        ? {
            kind: "bootstrap",
            block,
            samples: controls.samples,
            seed: controls.resampleSeed,
          }
        : { kind: "block_jackknife", block };
  analysis.connected = controls.connected;
  analysis.combine = controls.combine;
  analysis.effective_mass = controls.effectiveRate;
  analysis.fit = controls.fit;
  analysis.time_unit = controls.timeUnit;
  // `{}` asks Rust for its own default scan; the grid is never chosen here.
  analysis.stability = controls.stability ? (base.stability ?? {}) : null;
  // A synthesised basis is sent only inside the bounds Rust validates; outside
  // them the fit control is disabled and no basis is requested.
  analysis.gevp =
    controls.gevp && (base.gevp?.length || gevpReady(gevpChannels.length))
      ? base.gevp?.length
        ? clone(base.gevp)
        : [{ id: "selected", channels: [...gevpChannels] }]
      : [];
  analysis.assignments = Object.fromEntries(
    Object.entries(controls.assignments).filter(([, name]) => name),
  );
  analysis.anchors = [...controls.anchors];
  analysis.channels = [...(controls.channels || [])];
  return analysis;
}

// --------------------------------------------------------------- charts

// Undefined values stay `null`: chartSVG breaks the line there and draws no
// marker. They are never replaced by zero.
export function gapPoints(xs, ys) {
  return xs.map((x, i) => [x, orNull(ys?.[i])]);
}

// `SessionSnapshot.channels[]` is a `LiveChannel`: `correlator` and
// `effective_mass` are plain `Option<f64>` arrays indexed by the lag in frames
// (no errors before the analysis), empty for an unavailable channel.
const liveSeries = (channels, key) =>
  channels
    .filter((c) => c[key]?.length)
    .map((c) => ({
      name: c.id,
      style: "line",
      points: c[key].map((value, lag) => [lag, orNull(value)]),
    }));
export function measuredLive(snapshot) {
  return (snapshot?.channels || []).filter((c) => c.correlator?.length);
}
// One line per selected channel, used while the run is still accumulating.
export function liveCharts(snapshot, selected = []) {
  const wanted = new Set(selected);
  const channels = measuredLive(snapshot).filter(
    (c) => !wanted.size || wanted.has(c.id),
  );
  if (!channels.length) return [];
  const charts = [
    {
      title: "Live C(τ)",
      xLabel: "Lag τ (frames)",
      yLabel: "C(τ)",
      series: liveSeries(channels, "correlator"),
    },
  ];
  const rates = liveSeries(channels, "effective_mass");
  if (rates.length)
    charts.push({
      title: "Live effective decay rate",
      xLabel: "Lag τ (frames)",
      yLabel: "Effective rate (1/frames)",
      series: rates,
    });
  return charts;
}

// `SessionSnapshot.walkers` = {dimension, positions: flat [N·d], eligible: [N]}.
export function cloudChart(snapshot) {
  const cloud = snapshot?.walkers,
    d = cloud?.dimension;
  if (!d || !cloud.positions?.length) return null;
  const points = [];
  for (let i = 0, slot = 0; i + d <= cloud.positions.length; i += d, slot++)
    if (cloud.eligible?.[slot] !== false)
      points.push([cloud.positions[i], d > 1 ? cloud.positions[i + 1] : slot]);
  if (!points.length) return null;
  return {
    title: "Executed walker positions",
    xLabel: "Position 1",
    yLabel: d > 1 ? "Position 2" : "Walker slot",
    series: [
      {
        name: "Eligible walkers",
        style: "points",
        radius: 2.4,
        points,
      },
    ],
  };
}

// `SessionSnapshot.replicas[]` = {seed, step, frames, segments, terminal}.
export function progressOf(snapshot) {
  return {
    step: snapshot?.step ?? 0,
    required: snapshot?.steps ?? 0,
    done: Boolean(snapshot?.done),
    chunk: snapshot?.chunk ?? null,
    table: {
      columns: [
        "Replica",
        "Seed",
        "Completed steps",
        "Measured frames",
        "Segments",
        "Terminal",
      ],
      rows: (snapshot?.replicas || []).map((r, i) => [
        i,
        r.seed,
        r.step,
        r.frames,
        r.segments,
        r.terminal ?? "—",
      ]),
    },
  };
}

const COVERAGE_COLUMNS = [
  ["frames", "Frames"],
  ["empty_frames", "Empty frames"],
  ["valid", "Valid"],
  ["masked_historical", "Masked: historical"],
  ["masked_ineligible", "Masked: ineligible"],
  ["masked_color", "Masked: colour"],
  ["masked_identity", "Masked: identity"],
  ["masked_self", "Masked: self"],
  ["masked_scale", "Masked: scale"],
];
// `LiveChannel.estimator` is null when the live estimator reports no
// correlator for a measured channel; `LiveChannel.note` then says why.
export function coverageTable(channels = []) {
  return {
    columns: [
      "Channel",
      "Estimator",
      ...COVERAGE_COLUMNS.map(([, label]) => label),
    ],
    rows: channels.map((c) => [
      c.id,
      ESTIMATOR_LABEL[c.estimator] ?? "—",
      ...COVERAGE_COLUMNS.map(([key]) => c.coverage?.[key] ?? "—"),
    ]),
  };
}
// Notes of a live session: the session's own notes, then the sentence Rust
// attached to a channel the live charts cannot draw.
export function liveNotes(snapshot) {
  return [
    ...(snapshot?.notes || []),
    ...(snapshot?.channels || [])
      .filter((c) => c.note)
      .map((c) => ({ owner: c.id, text: c.note })),
  ];
}

// `CorrelatorEstimate.lags` are lag indices; the axis in the reported time
// unit belongs to the Rust presentation plots, so no lag is rescaled here.
export function correlatorTable(channel) {
  const c = channel.correlator;
  if (!c) return { columns: [], rows: [] };
  return {
    columns: [
      "Lag",
      "Time unit",
      "Time step",
      "C(τ)",
      "Error",
      "Effective rate",
      "Rate error",
    ],
    rows: c.lags.map((lag, i) => [
      lag,
      unitLabel(c.time_unit),
      orNull(c.time_step),
      orNull(c.value[i]),
      orNull(c.error?.[i]),
      orNull(channel.effective_mass?.[i]?.[0]),
      orNull(channel.effective_mass?.[i]?.[1]),
    ]),
  };
}

export function samplesTable(channels = []) {
  return {
    columns: [
      "Channel",
      "Estimator",
      "Frame normalisation",
      "Resampling",
      "Effective block",
      "Blocks",
      "τ_int",
      "Covariance rank",
      "Replicas",
      "Sampling unit",
      "Connected",
      "Connected bias",
    ],
    rows: channels
      .filter((c) => c.correlator)
      .map((c) => {
        const meta = c.correlator.samples_meta || {};
        return [
          c.id,
          ESTIMATOR_LABEL[c.estimator] ?? "—",
          NORMALIZATION_LABEL[c.normalization] ?? "—",
          meta.resampling ?? "—",
          meta.effective_block ?? "—",
          meta.blocks ?? "—",
          orNull(meta.tau_int),
          meta.covariance_rank ?? "—",
          meta.replicas ?? "—",
          meta.sampling_unit ?? "—",
          c.correlator.connected ? "yes" : "no",
          orNull(c.correlator.connected_bias),
        ];
      }),
  };
}

export function reportAvailabilityTable(report) {
  return {
    columns: [
      "Channel",
      "Elements",
      "Book label",
      "Definition",
      "Exchange",
      "Spatial parity",
      "Availability",
      "Reason",
    ],
    rows: (report?.channels || []).map((c) => [
      c.id,
      c.kind ?? "—",
      c.book_label || "—",
      c.definition || "—",
      c.exchange ?? "—",
      c.spatial_parity ?? "—",
      c.availability?.status ?? "—",
      c.availability?.reason ?? "—",
    ]),
  };
}

// The provenance Rust stamps on every report: `analyze` refuses to build one
// from anything but an executed run, and `present` repeats it on every result.
export function provenanceTable(report) {
  if (!report) return null;
  return {
    columns: ["Provenance", "Value"],
    rows: [
      ["Calculation origin", report.calculation_origin ?? "—"],
      ["Precision", report.precision ?? "—"],
      ["Schema version", report.schema_version ?? "—"],
      ["Measurement fingerprint", report.measurement_fingerprint ?? "—"],
      ["Replicas", report.replicas ?? "—"],
      ["Measured frames", report.frames ?? "—"],
    ],
  };
}

// --------------------------------------------------------- rates & fits

function diagnosticCells(d = {}, meta = {}) {
  return [
    orNull(d.chi2),
    d.dof ?? "—",
    orNull(d.q),
    d.window ? d.window[0] + "–" + d.window[1] : "—",
    d.n_windows ?? "—",
    d.correlated === undefined ? "—" : d.correlated ? "yes" : "diagonal",
    orNull(d.svd_cut),
    d.covariance_rank ?? "—",
    meta.effective_block ?? "—",
    orNull(meta.tau_int),
    orNull(d.prior_dominance?.width_ratio),
    orNull(d.prior_dominance?.shift_sigma),
    d.prior_dominance ? (d.prior_dominance.dominated ? "yes" : "no") : "—",
    d.model_rejected ?? "—",
    d.no_signal ?? "—",
  ];
}
const DIAGNOSTIC_COLUMNS = [
  "χ²",
  "dof",
  "Q",
  "Window (frames)",
  "Windows",
  "Correlated χ²",
  "SVD cut",
  "Covariance rank",
  "Effective block",
  "τ_int",
  "Prior width ratio",
  "Prior shift (σ)",
  "Prior dominated",
  "Model rejected",
  "No signal",
];
const rateCells = (rate) => [
  rate?.quantity ?? "—",
  orNull(rate?.value),
  orNull(rate?.error),
  orNull(rate?.statistical),
  orNull(rate?.systematic),
  rate ? "1/" + unitLabel(rate.time_unit) : "—",
];
const RATE_COLUMNS = [
  "Quantity",
  "Rate",
  "Error",
  "Statistical",
  "Systematic",
  "Unit",
];

// A missing rate is an empty cell (null), never 0; the Rust reasons
// (`model_rejected`, `no_signal`, unavailability) are copied verbatim.
export function fitTable(report) {
  const rows = [];
  for (const channel of report?.channels || []) {
    const meta = channel.correlator?.samples_meta;
    if (channel.availability?.status === "unavailable") {
      rows.push([
        channel.id,
        "Unavailable: " + channel.availability.reason,
        "—",
        ...rateCells(null),
        ...diagnosticCells(),
      ]);
      continue;
    }
    if (!channel.fits?.length)
      rows.push([
        channel.id,
        "No fit reported",
        "—",
        ...rateCells(null),
        ...diagnosticCells(undefined, meta),
      ]);
    for (const fit of channel.fits || []) {
      rows.push([
        channel.id,
        METHOD_LABEL[fit.method] ?? fit.method,
        "ground",
        ...rateCells(fit.mass),
        ...diagnosticCells(fit.diagnostics, meta),
      ]);
      (fit.excited || []).forEach((level, n) =>
        rows.push([
          channel.id,
          METHOD_LABEL[fit.method] ?? fit.method,
          "excited " + (n + 1),
          ...rateCells(level),
          ...diagnosticCells(fit.diagnostics, meta),
        ]),
      );
    }
  }
  return {
    columns: [
      "Channel",
      "Method",
      "Level",
      ...RATE_COLUMNS,
      ...DIAGNOSTIC_COLUMNS,
    ],
    rows,
  };
}

export function windowTable(channel) {
  return {
    columns: [
      "Method",
      "Window",
      "t_min",
      "t_max",
      "Rate",
      "Error",
      "χ²",
      "dof",
      "Weight",
    ],
    rows: (channel.fits || []).flatMap((fit) =>
      (fit.windows || []).map((w, i) => [
        METHOD_LABEL[fit.method] ?? fit.method,
        i,
        w.t_min,
        w.t_max,
        orNull(w.value),
        orNull(w.error),
        orNull(w.chi2),
        w.dof,
        orNull(w.weight),
      ]),
    ),
  };
}

export function groupTable(report) {
  return {
    columns: [
      "Group",
      "Channels",
      "Level",
      ...RATE_COLUMNS,
      ...DIAGNOSTIC_COLUMNS,
    ],
    rows: (report?.groups || []).flatMap((group) =>
      group.availability?.status === "unavailable"
        ? [
            [
              group.id,
              group.channels.join(", "),
              "Unavailable: " + group.availability.reason,
              ...rateCells(null),
              ...diagnosticCells(),
            ],
          ]
        : (group.levels.length ? group.levels : [null]).map((level, n) => [
            group.id,
            group.channels.join(", "),
            level ? "E" + n : "—",
            ...rateCells(level),
            ...diagnosticCells(group.diagnostics),
          ]),
    ),
  };
}

export function gevpTable(report) {
  return {
    columns: [
      "Basis",
      "Channels",
      "t0",
      "Rank",
      "Antisymmetric norm",
      "State",
      ...RATE_COLUMNS,
    ],
    rows: (report?.gevp || []).flatMap((g) =>
      g.availability?.status === "unavailable"
        ? [
            [
              g.id,
              g.channels.join(", "),
              g.t0,
              "—",
              "—",
              "Unavailable: " + g.availability.reason,
              ...rateCells(null),
            ],
          ]
        : (g.levels.length ? g.levels : [null]).map((level, n) => [
            g.id,
            g.channels.join(", "),
            g.t0,
            g.rank,
            orNull(g.antisymmetric_norm),
            n,
            ...rateCells(level),
          ]),
    ),
  };
}

// ----------------------------------------------------------- comparison

export const pair = (value) =>
  Array.isArray(value) && finite(value[0])
    ? format(value[0]) + " ± " + format(orNull(value[1]))
    : "—";

export function comparisonTables(report) {
  const comparison = report?.comparison;
  if (!comparison) return null;
  return {
    label: comparison.label,
    notes: comparison.notes || [],
    reference: {
      columns: [
        "Reference",
        "Assigned channel",
        "Reference value",
        "Reference error",
        "Unit",
        "Measured rate (lattice units)",
        "Estimator",
        "Frame normalisation",
      ],
      rows: comparison.reference.map((r) => [
        r.name,
        r.channel || "—",
        r.reference,
        r.reference_error,
        r.unit,
        pair(r.measured),
        ESTIMATOR_LABEL[r.estimator] ?? "—",
        NORMALIZATION_LABEL[r.normalization] ?? "—",
      ]),
    },
    anchors: comparison.anchors.map((a) => ({
      anchor: a.anchor,
      scale: pair(a.scale),
      table: {
        columns: ["Reference", "Predicted", "Reference value", "Tension (σ)"],
        rows: a.predictions.map((p) => [
          p.name,
          pair(p.predicted),
          p.reference,
          orNull(p.tension_sigma),
        ]),
      },
    })),
    ratios: {
      columns: [
        "Numerator",
        "Denominator",
        "Measured ratio",
        "Reference ratio",
        "Tension (σ)",
      ],
      rows: comparison.ratios.map((r) => [
        r.numerator,
        r.denominator,
        pair(r.measured),
        r.reference,
        orNull(r.tension_sigma),
      ]),
    },
    spread: {
      columns: ["Reference", "Relative spread over anchors"],
      rows: comparison.anchor_spread.map((s) => [s.name, orNull(s.spread)]),
    },
  };
}

const quantityTable = (quantities = []) => ({
  columns: [
    "Name",
    "Symbol",
    "Value",
    "Error",
    "Unit",
    "Definition",
    "Book label",
  ],
  rows: quantities.map((q) => [
    q.name,
    q.symbol,
    orNull(q.value),
    orNull(q.error),
    q.unit,
    q.definition,
    q.book_label,
  ]),
});
export function couplingTables(report) {
  const couplings = report?.couplings;
  if (!couplings) return null;
  return {
    notes: couplings.notes || [],
    scales: quantityTable(couplings.scales),
    couplings: quantityTable(couplings.couplings),
    inversion: quantityTable(couplings.inversion),
  };
}

export function calibrationTable(report) {
  const c = report?.calibration;
  if (!c) return null;
  return {
    columns: ["Calibrated scale", "Value"],
    rows: [
      ["Warm-up frames", c.warmup_frames],
      ["Colour phase length ℓ₀", c.length],
      ["ℓ₀ source", c.length_source],
      ["Phase factor κ = m ℓ₀ / ħ_eff", c.kappa],
      ["Phase wrapping fraction", orNull(c.phase_wrapping)],
      ["Mass m", c.mass],
      ["ħ_eff (colour phase)", c.h_eff],
      ["ħ_eff (electroweak)", c.electroweak_h_eff],
      ["ħ_s", c.h_s],
      ["ε_d", orNull(c.epsilon_d)],
      ["ε_c", orNull(c.epsilon_c)],
      ["ε_clone", c.epsilon_clone],
      ["Integrator dt", orNull(c.dt)],
      [
        "Euclidean-time range",
        c.euclidean_range ? c.euclidean_range.map(format).join(" … ") : null,
      ],
      [
        "Multiscale geodesic scales",
        c.scales?.length ? c.scales.map(format).join(", ") : null,
      ],
      ["Pair weight N₁ = E exp(−D²/ε_d²)", orNull(c.pair_weight_n1)],
      [
        "Viscous kernel second moment ⟨K²⟩",
        orNull(c.viscous_kernel_second_moment),
      ],
    ],
  };
}

// `FlowDiagnostic` = {frames, steps[], roughness[]}: the mean neighbour colour
// mismatch 1 − Re q_ij after each graph smoothing step. No length scale.
export function flowTable(report) {
  const flow = report?.flow;
  if (!flow) return null;
  return {
    frames: flow.frames,
    columns: ["Smoothing steps", "Mean neighbour colour mismatch 1 − Re q"],
    rows: (flow.steps || []).map((steps, i) => [
      steps,
      orNull(flow.roughness?.[i]),
    ]),
  };
}

// ------------------------------------------------------- presentation
//
// `presentation()` returns `Vec<partvi::ExperimentResult>`:
// `{experiment, title, model, metrics: [{label, value, unit}], plots, notes,
// details}`. A `Plot` is `{title, x_label, y_label, series}` and a `Series` is
// `{name, points: [[x, y]], kind}`. Rust splits a curve at every undefined
// point into consecutive series of the SAME name and emits the two edges of a
// band as `"<name> + error"` and `"<name> - error"`, so every drawn number,
// the lag axis included, is already the one Rust reported.
//
// One result per available channel (`title` = the channel id), preceded by the
// overview "Spectroscopy report" and followed by `Group <id>`, `GEVP <id>`,
// `Reference comparison (<label>)`, the three coupling tables and
// "Graph smoothing". An unavailable channel, group or basis has no result: its
// reason is a note of the overview.
const BAND = / [+-] error$/;
const GAP = [null, null];

const presentationResults = (results) =>
  Array.isArray(results) ? results : results ? [results] : [];
export function presentationNotes(results) {
  return presentationResults(results).flatMap((r) =>
    (r.notes || []).map((text) => ({ owner: r.title, text })),
  );
}
export function presentationMetrics(results) {
  return {
    columns: ["Result", "Quantity", "Value", "Unit"],
    rows: presentationResults(results).flatMap((r) =>
      (r.metrics || []).map((m) => [
        r.title,
        m.label,
        orNull(m.value),
        m.unit || "—",
      ]),
    ),
  };
}

// The consecutive runs Rust emitted under one name become one series whose
// points are joined by a `[null, null]` gap: chartSVG restarts the line there
// and chartTable skips it. Nothing is interpolated across the gap.
const chartOfPlot = (plot, { logY = false, owner = "" } = {}) => {
  const order = [];
  const merged = new Map();
  for (const s of plot.series || []) {
    if (!merged.has(s.name)) {
      order.push(s.name);
      merged.set(s.name, { name: s.name, kind: s.kind, points: [] });
    }
    const series = merged.get(s.name);
    if (series.points.length) series.points.push(GAP);
    series.points.push(...(s.points || []));
  }
  const bases = [...new Set(order.map((name) => name.replace(BAND, "")))];
  return {
    // Rust titles a plot inside its result ("Correlator"); the card names the
    // result it belongs to so several channels can be shown side by side.
    title: owner ? owner + " · " + plot.title : plot.title,
    xLabel: plot.x_label,
    yLabel: plot.y_label,
    ...(logY ? { yScale: "log" } : {}),
    series: order.map((name) => {
      const series = merged.get(name);
      const band = BAND.test(name);
      return {
        name,
        points: series.points,
        style:
          series.kind === "bars"
            ? "bars"
            : ["scatter", "points"].includes(series.kind)
              ? "points"
              : "line",
        color: COLORS[bases.indexOf(name.replace(BAND, "")) % COLORS.length],
        // A band edge is drawing geometry: dashed, and left out of the
        // numeric fallback, which lists the reported value and error instead.
        ...(band ? { dashed: true, geometry: true } : {}),
      };
    }),
  };
};

const resultOf = (results, title) =>
  presentationResults(results).find((r) => r.title === title) ?? null;
// A plot whose points are all undefined carries no number and is left out.
const charts = (results, options = {}) =>
  results.flatMap((result) =>
    (result.plots || [])
      .map((plot) =>
        chartOfPlot(plot, {
          ...options,
          owner: options.owned ? result.title : "",
          // `logY` applies to the correlator only: an effective rate, a
          // weight and a residual all cross zero.
          logY: Boolean(options.logY) && plot.title === CORRELATOR,
        }),
      )
      .filter((chart) => chart.series.length),
  );

// Every plot of every result, in Rust's order.
export function presentationCharts(results, options) {
  return charts(presentationResults(results), { owned: true, ...options });
}
const CORRELATOR = "Correlator";
const CURVE_PLOTS = [CORRELATOR, "Effective rate"];
const curves = (result, wanted) => ({
  ...result,
  plots: (result.plots || []).filter(
    (plot) => CURVE_PLOTS.includes(plot.title) === wanted,
  ),
});
// The correlator and effective-rate plots of one channel, and separately its
// fit-window plots: both live in the result Rust titled with the channel id.
export function channelCharts(results, id, options) {
  const result = resultOf(results, id);
  return result
    ? charts([curves(result, true)], { owned: true, ...options })
    : [];
}
export function windowCharts(results, id) {
  const result = resultOf(results, id);
  return result ? charts([curves(result, false)], { owned: true }) : [];
}
// One result per basis, titled `GEVP <id>`; an unavailable basis has none.
export function gevpCharts(results) {
  return charts(
    presentationResults(results).filter((r) => r.title.startsWith("GEVP ")),
    { owned: true },
  );
}

// Every `notes` array of a report, in report order, tagged with its owner.
export function collectNotes(report, scope = "all") {
  if (!report) return [];
  const notes = (report.notes || []).map((text) => ({ owner: "Report", text }));
  const channelNotes = () =>
    (report.channels || []).flatMap((c) =>
      (c.notes || []).map((text) => ({ owner: c.id, text })),
    );
  const fitNotes = () => [
    ...(report.channels || []).flatMap((c) =>
      (c.fits || []).flatMap((fit) =>
        (fit.notes || []).map((text) => ({
          owner: c.id + " · " + (METHOD_LABEL[fit.method] ?? fit.method),
          text,
        })),
      ),
    ),
    ...(report.groups || []).flatMap((g) =>
      (g.notes || []).map((text) => ({ owner: "Group " + g.id, text })),
    ),
    ...(report.gevp || []).flatMap((g) =>
      (g.notes || []).map((text) => ({ owner: "GEVP " + g.id, text })),
    ),
  ];
  const physicsNotes = () => [
    ...(report.comparison?.notes || []).map((text) => ({
      owner: report.comparison.label,
      text,
    })),
    ...(report.couplings?.notes || []).map((text) => ({
      owner: "Couplings",
      text,
    })),
  ];
  if (scope === "correlators") return [...notes, ...channelNotes()];
  if (scope === "fits") return [...notes, ...fitNotes()];
  if (scope === "physics") return [...notes, ...physicsNotes()];
  if (scope === "report") return notes;
  return [...notes, ...channelNotes(), ...fitNotes(), ...physicsNotes()];
}

// Numeric fallback of any chart descriptor. Band edges are drawing geometry
// (Rust's `value ± error` curves), not reported numbers: the tables next to
// the chart list the value and the error. The `[null, null]` that separates
// two runs of one curve is a gap, not a point.
export function chartTable(chart) {
  const series = (chart.series || []).filter((s) => !s.geometry);
  const errors = series.some((s) => s.errors);
  return {
    columns: ["Series", "x", "y", ...(errors ? ["Error"] : [])],
    rows: series.flatMap((s) =>
      (s.points || [])
        .map((p, i) => [
          s.name,
          orNull(p?.[0]),
          orNull(p?.[1]),
          ...(errors ? [orNull(s.errors?.[i])] : []),
        ])
        .filter((row) => row[1] !== null),
    ),
  };
}

// -------------------------------------------------------------- session

// `engine` is anything with the async methods of worker.js MESSAGE_TYPES: the
// worker client in the browser, a canned fake in the tests.
export function createSession(
  engine,
  { budget = 16, breathe = () => new Promise((r) => setTimeout(r, 0)) } = {},
) {
  let snapshot = null,
    running = false,
    imported = null,
    epoch = 0;
  const adopt = (next) => {
    snapshot = next;
    imported = null;
    return next;
  };
  return {
    get snapshot() {
      return snapshot;
    },
    get running() {
      return running;
    },
    get imported() {
      return imported;
    },
    get live() {
      return snapshot !== null;
    },
    async create(request) {
      running = false;
      epoch++;
      return adopt(await engine.create(request));
    },
    async step() {
      if (!snapshot || snapshot.done) return snapshot;
      snapshot = await engine.advance(budget);
      return snapshot;
    },
    // Advances `budget` updates per tick until done, paused or replaced.
    async run(onSnapshot) {
      if (running || !snapshot) return snapshot;
      running = true;
      const mine = epoch;
      try {
        while (running && mine === epoch && !snapshot.done) {
          const next = await engine.advance(budget);
          if (mine !== epoch) break;
          snapshot = next;
          onSnapshot?.(snapshot);
          await breathe();
        }
      } finally {
        if (mine === epoch) running = false;
      }
      return snapshot;
    },
    pause() {
      running = false;
    },
    async refresh() {
      snapshot = await engine.snapshot();
      return snapshot;
    },
    // The request Rust resolved for the live session, so a restored one shows
    // and restarts its own configuration instead of the page's last form.
    request() {
      return engine.request();
    },
    // Re-analysis never advances the gas: it reads the accumulated measurement.
    analyze(analysis) {
      return engine.analyze(analysis);
    },
    presentation(analysis) {
      return engine.presentation(analysis);
    },
    evidence() {
      return engine.evidence();
    },
    checkpoint() {
      return engine.checkpoint();
    },
    async restore(bytes) {
      running = false;
      epoch++;
      return adopt(await engine.restore(bytes));
    },
    async importEvidence(bytes, analysis) {
      running = false;
      const report = await engine.import_evidence({ bytes, analysis });
      epoch++;
      snapshot = null;
      imported = "evidence";
      return report;
    },
    async importArchive(bytes, config) {
      running = false;
      const report = await engine.import_archive({ bytes, config });
      epoch++;
      snapshot = null;
      imported = "archive";
      return report;
    },
    async dispose() {
      running = false;
      epoch++;
      snapshot = null;
      imported = null;
      await engine.dispose();
    },
  };
}
