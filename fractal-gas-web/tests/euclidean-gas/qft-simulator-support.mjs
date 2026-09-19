// Canned Rust-shaped JSON and a fake engine for the QFT Simulator host tests.
// The numbers are arbitrary fixtures: they test routing and rendering, not
// physics. The shapes follow the frozen Phase-0 contracts: `spectroscopy/
// {config,contract,report}.rs`, `operators::CatalogEntry`, and the
// `SessionSnapshot` / `defaults()` / `capabilities()` payloads of
// `benchmarks/src/spectroscopy.rs`.
const available = { status: "available" };
const unavailable = (reason) => ({ status: "unavailable", reason });

export const COLOR_REASON =
  "viscous force is identically zero: configure qft.viscosity or qft.graph_viscosity, or select an explicit RecordedField colour source";
export const ODD_REASON = "exchange-odd operator cancels on a mutual pairing";
export const RATE_QUANTITY = "decay rate of the algorithm-time autocorrelation";
export const REPORT_NOTE =
  "Rates are the " +
  RATE_QUANTITY +
  "; they are masses only under the positive transfer representation <cor-effective-twistor-positive-transfer>.";

const analysis = {
  connected: true,
  resampling: { kind: "block_jackknife", block: { kind: "auto" } },
  svd_cut: 1e-6,
  combine: "pooled_blocks",
  estimator: "auto",
  effective_mass: "log_ratio",
  fit: "window_scan",
  window_scan: {
    t_min: 1,
    t_max: null,
    min_points: 4,
    min_point_snr: 2,
    min_rate_snr: 2,
    correlated: true,
  },
  multi_exponential: {
    nexp: 1,
    t_min: 1,
    t_max: null,
    log_gap_mean: -2.302585092994046,
    log_gap_sigma: 3,
    log_amplitude_sigma: 5,
    dominance_ratio: 0.7,
  },
  groups: [],
  gevp: [],
  stability: null,
  reference: {
    unit: "MeV",
    entries: [
      { name: "pion", value: 139.57039, error: 0.00018, source: "PDG 2024" },
      { name: "nucleon", value: 938.272088, error: 3e-7, source: "PDG 2024" },
    ],
  },
  assignments: {
    "meson/pseudoscalar/standard": "pion",
    "baryon/real": "nucleon",
  },
  anchors: ["nucleon"],
  time_unit: "frames",
  channels: [],
  report_covariance: false,
};

// `operators::CatalogEntry`: one row per (specification, element kind).
const entry = (id, kind, spec, extra = {}) => ({
  id,
  spec,
  kind,
  family: spec.kind,
  standard: true,
  availability: available,
  descriptor: {
    definition: "\\operatorname{Re}\\, c_i^\\dagger c_j",
    book_label: "def-sm-meson-operators",
    spatial_parity: null,
    note: "",
  },
  exchange: "even",
  requirements: { records: ["color", "distance_companions"], dimension: null },
  components: 1,
  correlatable: true,
  assignment: null,
  ...extra,
});
const SCALAR = { kind: "meson", quantum: "scalar", mode: "standard" };
const PSEUDOSCALAR = {
  kind: "meson",
  quantum: "pseudoscalar",
  mode: "standard",
};

const request = {
  variant: "viscous_euclidean_gas",
  run: { walkers: 64, dimensions: 3, initial_lower: -1, initial_upper: 1 },
  steps: 256,
  replicas: 1,
  seed: 7,
  chunk: 16,
  spectroscopy: {
    measurement: {
      warmup: 16,
      max_lag: 8,
      stride: 1,
      color: {
        kind: "viscous_force",
        alignment: { kind: "preceding_kick" },
        threshold: 1e-12,
      },
      time: { kind: "monte_carlo" },
      channels: [SCALAR, PSEUDOSCALAR],
    },
    analysis,
  },
};

// `defaults()` = {schema_version, request, variants: [{name, title,
// implemented, request}], catalog: [CatalogEntry], reference}.
export const defaults = {
  schema_version: 1,
  request,
  variants: [
    {
      name: "euclidean_gas",
      title: "Euclidean Gas",
      implemented: true,
      request: {
        ...request,
        variant: "euclidean_gas",
        run: {
          walkers: 32,
          dimensions: 2,
          initial_lower: -2,
          initial_upper: 2,
        },
        steps: 128,
      },
    },
    {
      name: "viscous_euclidean_gas",
      title: "Viscous Euclidean Gas",
      implemented: true,
      request,
    },
    {
      name: "latent_fractal_gas",
      title: "Latent Fractal Gas",
      implemented: false,
      request: null,
    },
  ],
  catalog: [
    entry("meson/scalar/standard/distance", "distance_pair", SCALAR, {
      assignment: "f0_500",
    }),
    entry("meson/scalar/standard/cloning", "cloning_pair", SCALAR, {
      requirements: {
        records: ["color", "cloning_companions"],
        dimension: null,
      },
    }),
    entry(
      "meson/pseudoscalar/standard/distance",
      "distance_pair",
      PSEUDOSCALAR,
      { assignment: "pion" },
    ),
    entry(
      "baryon/real/triplet",
      "triplet",
      { kind: "baryon", mode: "real", flux_alpha: 1 },
      {
        standard: false,
        requirements: {
          records: ["color", "distance_companions", "cloning_companions"],
          dimension: 3,
        },
      },
    ),
    entry(
      "u1/phase/q1/distance",
      "distance_pair",
      { kind: "u1", mode: "phase", charge: 1 },
      {
        standard: false,
        exchange: "odd",
        correlatable: false,
        descriptor: {
          definition: "e^{i q \\varphi_{ij}}",
          book_label: "",
          spatial_parity: "odd",
          note: "",
        },
      },
    ),
  ],
  reference: analysis.reference,
};

// `capabilities(request)` = {capabilities, channels: [{id, availability}], chunk}.
export const capabilities = {
  capabilities: {
    dimension: 3,
    missing: { graph: "no geometry stage" },
    mutual_distance: true,
    mutual_cloning: false,
    euclidean_axis: null,
    distance_kernel_width: null,
    cloning_kernel_width: null,
    dense_viscosity: true,
  },
  channels: [
    { id: "meson/scalar/standard/distance", availability: available },
    { id: "meson/scalar/standard/cloning", availability: available },
    { id: "meson/pseudoscalar/standard/distance", availability: available },
    { id: "baryon/real/triplet", availability: unavailable(COLOR_REASON) },
    { id: "u1/phase/q1/distance", availability: unavailable(ODD_REASON) },
  ],
  chunk: 4,
};

const coverage = {
  frames: 40,
  empty_frames: 0,
  valid: 2400,
  masked_historical: 12,
  masked_ineligible: 3,
  masked_color: 0,
  masked_identity: 0,
  masked_self: 1,
  masked_scale: 0,
};
// Lag 3 is undefined (`None` in Rust): it must stay a gap in every view.
const correlator = {
  lags: [0, 1, 2, 3, 4],
  time_unit: "frames",
  time_step: 1,
  value: [1.0, 0.61, 0.37, null, 0.14],
  error: [0.02, 0.03, 0.04, null, 0.06],
  covariance: null,
  samples_meta: {
    resampling: "jackknife",
    effective_block: 8,
    blocks: 5,
    tau_int: 1.7,
    covariance_rank: 4,
    replicas: 1,
  },
  connected: true,
  connected_bias: -0.001,
};
const effectiveMass = [[0.49, 0.05], [0.5, 0.08], null, null, null];

const noCoverage = Object.fromEntries(
  Object.keys(coverage).map((key) => [key, 0]),
);
// `SessionSnapshot`: live channels carry plain `Option<f64>` arrays per lag;
// walker positions are one flat row-major array. The middle walker is not
// eligible.
export function snapshot(step, steps = 256) {
  return {
    schema_version: 1,
    step,
    steps,
    done: step >= steps,
    chunk: 16,
    replicas: [
      {
        seed: 7,
        step,
        frames: step > 16 ? step - 16 : 0,
        segments: 1,
        terminal: null,
      },
    ],
    capabilities: capabilities.capabilities,
    calibration: null,
    walkers: {
      dimension: 3,
      positions: [0.1, -0.2, 0.9, 0.4, 0.3, 0.8, -0.5, 0.2, 0.7],
      eligible: [true, false, true],
    },
    channels: [
      {
        id: "meson/scalar/standard/distance",
        availability: available,
        coverage,
        correlator: [1.0, 0.61, 0.37, null, 0.14],
        effective_mass: [0.49, 0.5, null, null, null],
      },
      {
        id: "u1/phase/q1/distance",
        availability: unavailable(ODD_REASON),
        coverage: noCoverage,
        correlator: [],
        effective_mass: [],
      },
    ],
    notes: ["Live correlators use the frame average without resampling."],
  };
}

const rate = {
  quantity: RATE_QUANTITY,
  value: 0.495,
  error: 0.04,
  statistical: 0.03,
  systematic: 0.026,
  method: "window_scan",
  time_unit: "frames",
};
const diagnostics = {
  chi2: 2.4,
  dof: 3,
  q: 0.49,
  window: [1, 4],
  n_windows: 2,
  correlated: true,
  svd_cut: 1e-6,
  covariance_rank: 4,
  prior_dominance: null,
  model_rejected: null,
  no_signal: null,
};
const channelReport = (id, spec, extra) => ({
  id,
  spec,
  kind: "distance_pair",
  scale: null,
  definition: "\\operatorname{Re}\\, c_i^\\dagger c_j",
  book_label: "def-sm-meson-operators",
  exchange: "even",
  spatial_parity: null,
  availability: available,
  coverage,
  estimator: "frame_mean",
  correlator,
  effective_mass: effectiveMass,
  mass: null,
  fits: [],
  notes: [],
  ...extra,
});

export const REJECTED =
  "correlator changes sign at lag 2: a single decaying exponential is rejected";

export function report(analysisConfig = analysis) {
  return {
    schema_version: 1,
    measurement_fingerprint: "fixture",
    analysis: analysisConfig,
    capabilities: capabilities.capabilities,
    calibration: {
      warmup_frames: 16,
      length: 0.31,
      length_source: "warmup_companion_median",
      kappa: 0.31,
      phase_wrapping: 0,
      h_eff: 1,
      h_s: 1,
      epsilon_d: null,
      epsilon_c: null,
      epsilon_clone: 1e-8,
      dt: 0.05,
      euclidean_range: null,
      scales: [],
    },
    replicas: 1,
    frames: 240,
    channels: [
      channelReport(
        "meson/scalar/standard/distance",
        defaults.catalog[0].spec,
        {
          mass: rate,
          fits: [
            {
              method: "window_scan",
              mass: rate,
              excited: [],
              diagnostics,
              windows: [
                {
                  t_min: 1,
                  t_max: 4,
                  value: 0.49,
                  error: 0.04,
                  chi2: 2.4,
                  dof: 3,
                  weight: 0.7,
                },
                {
                  t_min: 2,
                  t_max: 4,
                  value: 0.51,
                  error: 0.07,
                  chi2: 1.1,
                  dof: 2,
                  weight: 0.3,
                },
              ],
              notes: ["Window average with exp(-AIC/2) weights."],
            },
          ],
        },
      ),
      channelReport(
        "meson/pseudoscalar/standard/distance",
        defaults.catalog[2].spec,
        {
          fits: [
            {
              method: "window_scan",
              mass: null,
              excited: [],
              diagnostics: {
                ...diagnostics,
                chi2: null,
                dof: null,
                q: null,
                window: null,
                model_rejected: REJECTED,
              },
              windows: [],
              notes: [],
            },
          ],
        },
      ),
      channelReport("u1/phase/q1/distance", defaults.catalog[4].spec, {
        availability: unavailable(ODD_REASON),
        estimator: null,
        correlator: null,
        effective_mass: null,
      }),
    ],
    groups: [],
    gevp: [],
    comparison: {
      label: "hypothesis mapping",
      reference: [
        {
          name: "pion",
          channel: "meson/pseudoscalar/standard",
          reference: 139.57039,
          reference_error: 0.00018,
          unit: "MeV",
          measured: null,
        },
        {
          name: "nucleon",
          channel: "baryon/real",
          reference: 938.272088,
          reference_error: 3e-7,
          unit: "MeV",
          measured: null,
        },
      ],
      anchors: [
        {
          anchor: "nucleon",
          scale: null,
          predictions: [
            {
              name: "pion",
              predicted: null,
              reference: 139.57039,
              tension_sigma: null,
            },
          ],
        },
      ],
      ratios: [
        {
          numerator: "pion",
          denominator: "nucleon",
          measured: null,
          reference: 0.14875,
          tension_sigma: null,
        },
      ],
      anchor_spread: [{ name: "pion", spread: null }],
      notes: ["The anchor channel reports no rate; no scale is set."],
    },
    couplings: null,
    flow: null,
    notes: [REPORT_NOTE],
  };
}

// `presentation::present` returns `Vec<partvi::ExperimentResult>`.
export function presentation() {
  return [
    {
      experiment: 0,
      title: "Spectroscopy",
      model: "",
      metrics: [],
      plots: [
        {
          title: "C(τ)",
          x_label: "Lag",
          y_label: "C",
          series: [{ name: "scalar", kind: "line", points: [[0, 1]] }],
        },
      ],
      notes: [REPORT_NOTE],
      details: null,
    },
  ];
}

// Records every call; `advance` moves a counter, nothing else does.
export function createFakeEngine({ steps = 64, delays = {} } = {}) {
  const calls = [];
  let step = null,
    imported = false;
  const wait = (type) =>
    new Promise((resolve) => setTimeout(resolve, delays[type] ?? 0));
  const record = async (type, payload) => {
    calls.push({ type, payload });
    await wait(type);
  };
  const live = () => {
    if (step === null) throw new Error("no live session");
  };
  return {
    calls,
    count: (type) => calls.filter((c) => c.type === type).length,
    async defaults() {
      await record("defaults");
      return defaults;
    },
    async capabilities(request) {
      await record("capabilities", request);
      if (request?.run?.walkers < 2)
        throw new Error("configuration error: at least two walkers");
      return capabilities;
    },
    async create(request) {
      await record("create", request);
      step = 0;
      imported = false;
      return snapshot(step, steps);
    },
    async advance(n) {
      await record("advance", n);
      live();
      step = Math.min(steps, step + n);
      return snapshot(step, steps);
    },
    async snapshot() {
      await record("snapshot");
      live();
      return snapshot(step, steps);
    },
    async analyze(analysisConfig) {
      await record("analyze", analysisConfig);
      if (step === null && !imported) throw new Error("nothing to analyse");
      return report(analysisConfig);
    },
    async presentation(analysisConfig) {
      await record("presentation", analysisConfig);
      live();
      return presentation();
    },
    async evidence() {
      await record("evidence");
      live();
      return new Uint8Array([0xa1, 0x01, 0x02]);
    },
    async checkpoint() {
      await record("checkpoint");
      live();
      return new Uint8Array([0xa1, 0x03, step]);
    },
    async restore(bytes) {
      await record("restore", bytes);
      step = bytes[2];
      imported = false;
      return snapshot(step, steps);
    },
    async import_evidence(payload) {
      await record("import_evidence", payload);
      step = null;
      imported = true;
      return report(payload.analysis);
    },
    async import_archive(payload) {
      await record("import_archive", payload);
      step = null;
      imported = true;
      return report(payload.config.analysis);
    },
    async dispose() {
      await record("dispose");
      step = null;
      imported = false;
      return null;
    },
  };
}

// A module shaped like the wasm bundle, for `createWasmEngine(load)`.
export function createFakeWasm() {
  const log = [];
  let freed = 0;
  class SpectroscopyExperiment {
    static async create(json) {
      log.push(["create", json]);
      return new SpectroscopyExperiment();
    }
    static async restore(bytes) {
      log.push(["restore", bytes]);
      return new SpectroscopyExperiment();
    }
    async advance(steps) {
      log.push(["advance", steps]);
      return snapshot(steps);
    }
    snapshot() {
      return snapshot(0);
    }
    analyze(json) {
      log.push(["analyze", json]);
      return report(JSON.parse(json));
    }
    presentation(json) {
      log.push(["presentation", json]);
      return [];
    }
    evidence() {
      return new Uint8Array([1, 2, 3]);
    }
    checkpoint() {
      return new Uint8Array([4, 5, 6]);
    }
    free() {
      freed++;
    }
  }
  return {
    log,
    freed: () => freed,
    module: {
      SpectroscopyExperiment,
      spectroscopy_defaults: () => defaults,
      spectroscopy_capabilities: (json) => {
        log.push(["capabilities", json]);
        return capabilities;
      },
      spectroscopy_analyze: (bytes, json) => {
        log.push(["spectroscopy_analyze", bytes, json]);
        return report(JSON.parse(json));
      },
      spectroscopy_archive: (json, bytes) => {
        log.push(["spectroscopy_archive", json, bytes]);
        return report(JSON.parse(json).analysis);
      },
    },
  };
}
