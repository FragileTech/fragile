// Session adapter and pure Rust-JSON -> view-descriptor mappings of the QFT
// Simulator. Nothing here samples, fits or estimates: values and errors are
// copied from the Rust report; the only arithmetic is drawing geometry (lag
// axis in the reported time unit, the two edges of an error band).
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

const ESTIMATOR_LABEL = {
  frame_mean: "frame average",
  source_frozen: "source-frozen propagator",
  euclidean_time: "Euclidean time",
};
const METHOD_LABEL = {
  window_scan: "Window scan",
  multi_exponential: "Multi-exponential",
  gevp: "GEVP",
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
  analysis.gevp = controls.gevp
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
export function lagAxis(correlator) {
  const step = finite(correlator?.time_step) ? correlator.time_step : 1;
  return (correlator?.lags || []).map((lag) => lag * step);
}
function bandSeries(xs, values, errors, name, color) {
  const edge = (sign) =>
    xs.map((x, i) => [
      x,
      finite(values[i]) && finite(errors[i])
        ? values[i] + sign * errors[i]
        : null,
    ]);
  return [
    {
      name: name + " + error",
      style: "line",
      dashed: true,
      geometry: true,
      color,
      points: edge(1),
    },
    {
      name: name + " − error",
      style: "line",
      dashed: true,
      geometry: true,
      color,
      points: edge(-1),
    },
  ];
}
function channelTitle(channel) {
  const estimator = ESTIMATOR_LABEL[channel.estimator];
  return channel.id + (estimator ? " · " + estimator : "");
}

export function correlatorChart(channel, { logY = false } = {}) {
  const c = channel.correlator;
  if (!c) return null;
  const xs = lagAxis(c),
    points = gapPoints(xs, c.value);
  return {
    title: "C(τ) · " + channelTitle(channel),
    xLabel: "Lag τ (" + unitLabel(c.time_unit) + ")",
    yLabel: c.connected ? "Connected C(τ)" : "C(τ)",
    yScale: logY ? "log" : "linear",
    series: [
      {
        name: "C(τ)",
        style: "line",
        color: COLORS[0],
        points,
        errors: (c.error || []).map(orNull),
      },
      {
        name: "Measured lags",
        style: "points",
        color: COLORS[0],
        points,
        geometry: true,
      },
      ...bandSeries(xs, c.value, c.error || [], "C(τ)", COLORS[2]),
    ],
  };
}

const pairValues = (pairs) => (pairs || []).map((p) => orNull(p?.[0]));
const pairErrors = (pairs) => (pairs || []).map((p) => orNull(p?.[1]));

export function effectiveRateChart(channel) {
  const c = channel.correlator;
  if (!c || !channel.effective_mass) return null;
  const xs = lagAxis(c),
    values = pairValues(channel.effective_mass),
    points = gapPoints(xs, values);
  return {
    title: "Effective decay rate · " + channelTitle(channel),
    xLabel: "Lag τ (" + unitLabel(c.time_unit) + ")",
    yLabel: "Effective rate (1/" + unitLabel(c.time_unit) + ")",
    series: [
      {
        name: "Effective rate",
        style: "line",
        color: COLORS[1],
        points,
        errors: pairErrors(channel.effective_mass),
      },
      {
        name: "Defined lags",
        style: "points",
        color: COLORS[1],
        points,
        geometry: true,
      },
      ...bandSeries(
        xs,
        values,
        pairErrors(channel.effective_mass),
        "Effective rate",
        COLORS[3],
      ),
    ],
  };
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
export function coverageTable(channels = []) {
  return {
    columns: ["Channel", ...COVERAGE_COLUMNS.map(([, label]) => label)],
    rows: channels.map((c) => [
      c.id,
      ...COVERAGE_COLUMNS.map(([key]) => c.coverage?.[key] ?? "—"),
    ]),
  };
}

export function correlatorTable(channel) {
  const c = channel.correlator;
  if (!c) return { columns: [], rows: [] };
  return {
    columns: [
      "Lag (frames)",
      "τ (" + unitLabel(c.time_unit) + ")",
      "C(τ)",
      "Error",
      "Effective rate",
      "Rate error",
    ],
    rows: c.lags.map((lag, i) => [
      lag,
      lagAxis(c)[i],
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
      "Resampling",
      "Effective block",
      "Blocks",
      "τ_int",
      "Covariance rank",
      "Replicas",
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
          meta.resampling ?? "—",
          meta.effective_block ?? "—",
          meta.blocks ?? "—",
          orNull(meta.tau_int),
          meta.covariance_rank ?? "—",
          meta.replicas ?? "—",
          c.correlator.connected ? "yes" : "no",
          orNull(c.correlator.connected_bias),
        ];
      }),
  };
}

export function reportAvailabilityTable(report) {
  return {
    columns: ["Channel", "Book label", "Definition", "Availability", "Reason"],
    rows: (report?.channels || []).map((c) => [
      c.id,
      c.book_label || "—",
      c.definition || "—",
      c.availability?.status ?? "—",
      c.availability?.reason ?? "—",
    ]),
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

export function windowScanCharts(channel) {
  const charts = [];
  for (const fit of channel.fits || []) {
    if (!fit.windows?.length) continue;
    const label = METHOD_LABEL[fit.method] ?? fit.method;
    charts.push(
      {
        title: label + " rates by window start · " + channel.id,
        xLabel: "Window start t_min (frames)",
        yLabel: "Fitted decay rate",
        series: [
          {
            name: "Window fits",
            style: "points",
            points: fit.windows.map((w) => [w.t_min, orNull(w.value)]),
          },
        ],
      },
      {
        title: label + " model weights · " + channel.id,
        xLabel: "Window index (see table)",
        yLabel: "Normalized weight exp(−AIC/2)",
        series: [
          {
            name: "Weight",
            style: "bars",
            points: fit.windows.map((w, i) => [i, orNull(w.weight)]),
          },
        ],
      },
    );
  }
  return charts;
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

export function gevpCharts(report) {
  return (report?.gevp || [])
    .filter((g) => g.availability?.status !== "unavailable")
    .map((g) => ({
      title: "GEVP effective decay rates · " + g.id + " (t0 = " + g.t0 + ")",
      xLabel: "Lag τ (frames)",
      yLabel: "Effective rate",
      series: (g.effective_mass || []).map((state, n) => ({
        name: "State " + n,
        style: "line",
        points: gapPoints(g.lags, pairValues(state)),
      })),
    }));
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
      ],
      rows: comparison.reference.map((r) => [
        r.name,
        r.channel || "—",
        r.reference,
        r.reference_error,
        r.unit,
        pair(r.measured),
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
      ["ħ_eff", c.h_eff],
      ["ħ_s", c.h_s],
      ["ε_d", orNull(c.epsilon_d)],
      ["ε_c", orNull(c.epsilon_c)],
      ["ε_clone", c.epsilon_clone],
      ["Integrator dt", orNull(c.dt)],
      [
        "Euclidean-time range",
        c.euclidean_range ? c.euclidean_range.map(format).join(" … ") : null,
      ],
      ["Multiscale geodesic scales", (c.scales || []).map(format).join(", ")],
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

// `presentation()` returns `Vec<partvi::ExperimentResult>`: one lecture-shaped
// result (`title`, `plots`, `metrics`, `notes`) per report section.
const presentationResults = (results) =>
  Array.isArray(results) ? results : results ? [results] : [];
export function presentationNotes(results) {
  return presentationResults(results).flatMap((r) =>
    (r.notes || []).map((text) => ({ owner: r.title, text })),
  );
}
export function presentationCharts(results) {
  return presentationResults(results)
    .flatMap((r) => r.plots || [])
    .map((p) => ({
      title: p.title,
      xLabel: p.x_label,
      yLabel: p.y_label,
      series: p.series.map((s) => ({
        name: s.name,
        points: s.points,
        style:
          s.kind === "bars"
            ? "bars"
            : ["scatter", "points"].includes(s.kind)
              ? "points"
              : "line",
      })),
    }));
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
// (value ± error), not reported numbers: the tables list value and error.
export function chartTable(chart) {
  const series = (chart.series || []).filter((s) => !s.geometry);
  const errors = series.some((s) => s.errors);
  return {
    columns: ["Series", "x", "y", ...(errors ? ["Error"] : [])],
    rows: series.flatMap((s) =>
      (s.points || []).map((p, i) => [
        s.name,
        p[0],
        orNull(p[1]),
        ...(errors ? [orNull(s.errors?.[i])] : []),
      ]),
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
