import { GasClient } from "./client.js";
import {
  alive,
  benchmarkId,
  benchmarkInfo,
  benchmarkParameters,
  catalogEntry,
  dimensionFor,
  dimensionLocked,
  resolveConfig,
  frameMetrics,
} from "./config.js";
import { drawConvergence } from "./renderer.js";
import { MoleculeRenderer } from "./renderer3d.js";
import { StageController } from "./stage.js";
import { TrailBuffer } from "./trails.js";
import { createView } from "./view.js";
import { saveCheckpoint, loadCheckpoint } from "./storage.js";

const $ = (id) => document.getElementById(id);
const client = new GasClient();
let config = null,
  catalog = null,
  info = null,
  frame = null,
  stage = null,
  view = null,
  molecule = null,
  selected = 0;
let running = false,
  operation = false,
  pendingStep = null,
  records = [],
  bestRecorded = null,
  lastMetrics = null,
  eligible = new Uint8Array(0),
  drawPending = false,
  stepMs = null;
const trails = new TrailBuffer();
const format = (x) =>
  x == null || !Number.isFinite(Number(x))
    ? "—"
    : Math.abs(x) >= 10000 || (Math.abs(x) < 0.001 && x !== 0)
      ? Number(x).toExponential(3)
      : Number(x).toLocaleString(undefined, { maximumFractionDigits: 5 });
const pause = async () => {
  running = false;
  controls();
  if (pendingStep) await pendingStep;
  controls();
};
function controls() {
  const ready = !!frame && !operation;
  const extinct = !!frame && lastMetrics?.alive === 0 && !frame.elite_count;
  $("run").disabled = !ready || extinct;
  $("run").textContent = running ? "Pause" : "Run";
  $("step").disabled = !ready || extinct || running || !!pendingStep;
  $("reset").disabled = !ready;
  $("apply").disabled = operation;
  for (const id of [
    "save-checkpoint",
    "restore-checkpoint",
    "save-local",
    "load-local",
    "export-config",
    "import-config",
    "export-results",
  ])
    $(id).disabled = !ready;
  $("run-status").textContent = operation
    ? "Preparing run…"
    : extinct
      ? "Extinct · reset required"
      : running
        ? "Running"
        : pendingStep
          ? "Pausing at step boundary…"
          : frame
            ? "Paused"
            : "Preparing engine";
}
function showError(error) {
  running = false;
  $("error").hidden = false;
  $("error").textContent = error.message || String(error);
  controls();
}
function clearError() {
  $("error").hidden = true;
  $("error").textContent = "";
}
async function safe(action) {
  try {
    await action();
  } catch (error) {
    showError(error);
  }
}
function rows(target, entries) {
  target.replaceChildren();
  for (const [label, value] of entries) {
    const dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    target.append(dt, dd);
  }
}
function inspector() {
  if (!frame) return;
  const p = frame.population,
    report = frame.report;
  selected = Math.max(0, Math.min(p.validity.length - 1, selected));
  $("walker").value = selected;
  $("walker").max = p.validity.length - 1;
  $("walker-tag").textContent = `#${selected}`;
  const flags = Object.entries(p.validity[selected])
    .filter(([, value]) => value)
    .map(([key]) => key.replaceAll("_", " "));
  let distanceDonors = "—",
    cloneDonor = "—",
    decision = "—";
  if (report) {
    const companions = report.distance_companions,
      sources = report.distance_sources;
    distanceDonors = Array.from({ length: companions.count }, (_, a) => {
      const index = selected * companions.count + a;
      if (!companions.valid[index]) return "unmatched";
      const source = sources[companions.indices[index]];
      return `#${source.slot} @ ${source.frame}`;
    }).join(", ");
    const choice = report.clone_plan.choices[selected];
    cloneDonor =
      choice.donors
        .map((d) => {
          const source = report.clone_plan.sources[d.pool_index];
          return `#${source.slot} @ ${source.frame}`;
        })
        .join(", ") || "unmatched";
    decision = choice.revival ? "revived" : choice.accepted ? "cloned" : "kept";
  }
  const position = p.observations.fields.positions;
  const width = position.item_shape.reduce((a, b) => a * b, 1);
  const coordinates = Array.from(
    position.values.slice(selected * width, (selected + 1) * width),
  )
    .slice(0, 6)
    .map(format)
    .join(", ");
  rows($("walker-details"), [
    ["Raw reward · result", format(p.rewards.raw[selected])],
    [
      "Fitness · pre-clone",
      format(report?.pre_clone_fitness.fitness[selected]),
    ],
    [
      "Separation · pre-clone",
      format(report?.pre_clone_fitness.separation[selected]),
    ],
    ["Diversity donors · slot @ step", distanceDonors],
    ["Clone donor · slot @ step", cloneDonor],
    ["Last decision", decision],
    ["Validity", flags.join(", ") || "eligible"],
    ["Generation", String(p.generations[selected])],
    ["Coordinates", `[${coordinates}${width > 6 ? ", …" : ""}]`],
  ]);
  const stats = frame.execution;
  rows($("execution-details"), [
    [
      "Computation",
      `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · ${config.gas.precision}`,
    ],
    ["Synchronization points", String(stats.synchronizations)],
    ["Uploaded", `${format(stats.uploaded_bytes / 1024)} KiB`],
    ["Downloaded", `${format(stats.downloaded_bytes / 1024)} KiB`],
    ["Largest batch", `${format(stats.peak_batch_elements)} elements`],
    [
      "Objective",
      info.objectiveExecution === "host"
        ? `host f64${info.problemId ? ` · ${info.problemId}` : ""}`
        : "tensor graph",
    ],
    [
      "Gradient",
      {
        graph: "analytic tensor graph",
        zero: "zero",
        host_central_difference: "host central differences (2d evaluations)",
      }[info.gradientExecution],
    ],
    ...(info.objectiveExecution === "host"
      ? [
          [
            "Host evaluations",
            `${format(stats.host_reward_evaluations || 0)} reward · ${format(stats.host_gradient_evaluations || 0)} gradient`,
          ],
        ]
      : []),
    ["Population version", String(p.version)],
  ]);
}
function moleculeView() {
  const field = frame.population.observations.fields.positions,
    d = config.dimensions,
    active = !!info?.molecule && d % 3 === 0 && !!field;
  $("molecule").hidden = !active;
  if (!active) {
    molecule?.dispose();
    molecule = null;
    return;
  }
  molecule ??= new MoleculeRenderer($("molecule-canvas"));
  molecule.update(field.values.subarray(selected * d, (selected + 1) * d));
}
function viewNotes(settings) {
  const flat = settings.view === "2d",
    [x, y, z] = settings.axes,
    d = config.dimensions;
  $("z-axis-label").hidden = settings.view !== "spatial";
  $("reset-camera").hidden = flat;
  $("legend-best").hidden = flat;
  $("legend-high-item").hidden = settings.color === "uniform";
  const metric = settings.color === "fitness" ? "fitness" : "reward";
  $("legend-low").textContent =
    settings.color === "uniform" ? "Walker" : `Lower ${metric}`;
  $("legend-high").textContent = `Higher ${metric}`;
  $("stage-help").textContent = flat
    ? "Click a walker to inspect · scroll to zoom"
    : "Drag to orbit · right-drag to pan · scroll to zoom · click a walker";
  $("view-caption").textContent =
    `${info.label} · ${d}D state · ` +
    (flat
      ? `x${x + 1}, x${y + 1}`
      : settings.view === "landscape"
        ? `x${x + 1}, x${y + 1}, objective height (asinh scale)`
        : `x${x + 1}, x${y + 1}${z < d ? `, x${z + 1}` : ", plane"}`);
  $("slice-note").textContent = info.stochastic
    ? "The surface shows the expected objective; walker rewards carry fresh observation noise at every evaluation."
    : d > 2
      ? "The surface fixes the other coordinates to the slice values. Walker heights and colors use their actual full-dimensional reward and may lie away from the slice."
      : "The surface and walkers use the same objective and height transform.";
  const links = stage.linkInfo;
  $("links-note").hidden = flat || settings.links === "none";
  if (links && !$("links-note").hidden)
    $("links-note").textContent =
      `Links show companions drawn during the last step at current slot positions · ${links.drawn} drawn` +
      (links.skippedHistorical
        ? ` · ${links.skippedHistorical} historical donors skipped`
        : "") +
      (links.capped ? " · large population: selected walker only" : "");
  const render = flat || stage.renderMs == null ? null : stage.renderMs;
  $("timing").textContent = [
    stepMs == null ? null : `${stepMs.toFixed(1)} ms / step`,
    render == null ? null : `${render.toFixed(1)} ms / render`,
  ]
    .filter(Boolean)
    .join(" · ");
}
function render() {
  drawPending = false;
  if (!frame || !stage) return;
  const settings = view.settings();
  stage.update(frame, settings, selected, trails);
  inspector();
  drawConvergence($("convergence"), records, $("chart-axis").value);
  viewNotes(settings);
  moleculeView();
}
// Coalesces step, selection and control changes into one frame.
function draw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(render);
}
function select(slot) {
  selected = slot;
  trails.track(selected);
  draw();
}
function acceptFrame(next, reset = false) {
  frame = next;
  const metrics = (lastMetrics = frameMetrics(frame, config));
  if (reset) {
    records = [];
    bestRecorded = null;
    trails.clear();
    trails.track(selected);
  }
  const validity = frame.population.validity;
  if (eligible.length !== validity.length)
    eligible = new Uint8Array(validity.length);
  for (let i = 0; i < validity.length; i++)
    eligible[i] = alive(validity[i], config.gas.include_truncated) ? 1 : 0;
  trails.push(frame, eligible, {
    periodic: config.gas.boundary.kind === "periodic_box",
    low: info.low,
    high: info.high,
  });
  if (metrics.best !== null)
    bestRecorded =
      bestRecorded === null
        ? metrics.best
        : config.gas.fitness.direction === "minimize"
          ? Math.min(bestRecorded, metrics.best)
          : Math.max(bestRecorded, metrics.best);
  if (!records.length || records.at(-1).step !== metrics.step)
    records.push({ ...metrics, best: bestRecorded });
  if (records.length > 5000) records.splice(1, 1);
  $("metric-step").textContent = format(frame.step);
  $("metric-best").textContent = format(metrics.best);
  $("metric-alive").textContent = `${metrics.alive} / ${config.walkers}`;
  $("metric-evals").textContent = format(frame.reward_evaluations);
  const gap =
    info.minimum != null &&
    !info.stochastic &&
    config.gas.fitness.direction === "minimize";
  $("metric-gap-tile").hidden = !gap;
  if (gap)
    $("metric-gap").textContent = format(
      bestRecorded === null ? null : bestRecorded - info.minimum,
    );
  $("stage-loading").hidden = true;
  if (!metrics.alive && !frame.elite_count) {
    running = false;
    $("run-status").textContent = "Extinct";
  }
  draw();
  controls();
}
function fillForm(c) {
  const g = c.gas;
  const law = (module) =>
    module.law === "independent"
      ? module.kernel.kind === "uniform"
        ? "uniform"
        : "gaussian"
      : module.law;
  const geometry = g.kinetic.noise.geometry;
  const entry = catalogEntry(catalog, c.benchmark);
  $("benchmark").value = benchmarkId(c.benchmark);
  benchmarkFields(benchmarkParameters(entry, c.benchmark));
  const [low, high] = entry.bounds;
  const box = [c.initial_lower, c.initial_upper];
  const values = {
    "initial-box":
      box[0] === low && box[1] === high
        ? "domain"
        : box[0] === Math.max(low, -1) && box[1] === Math.min(high, 1)
          ? "unit"
          : "custom",
    dimensions: c.dimensions,
    walkers: c.walkers,
    "n-elite": g.n_elite ?? 0,
    seed: g.seed,
    direction: g.fitness.direction,
    backend: g.backend,
    precision: g.precision,
    kinetic: g.kinetic.integrator.kind,
    amplitude: g.kinetic.integrator.amplitude ?? 0.05,
    dt: g.kinetic.integrator.dt ?? 0.01,
    friction: g.kinetic.integrator.friction ?? 1,
    "distance-law": law(g.distance_donors),
    "clone-law": law(g.cloning_donors),
    "distance-count": g.distance_donors.count,
    "kernel-width":
      g.distance_donors.kernel.width ??
      g.distance_donors.kernel.temperature ??
      1,
    reducer: g.reducer.kind,
    distance: g.distance_donors.distance.kind,
    boundary: g.boundary.kind,
    alpha: g.fitness.reward_exponent,
    beta: g.fitness.diversity_exponent,
    "positive-map": g.fitness.reward_map.kind,
    standardizer: g.fitness.reward_standardizer.kind,
    "sigma-min": g.fitness.reward_standardizer.sigma_min ?? 0.001,
    innovation: g.kinetic.noise.innovation,
    "noise-geometry": geometry.kind,
    "noise-scale":
      geometry.scale?.values?.[0] ?? geometry.factor?.values?.[0] ?? 1,
  };
  for (const [id, value] of Object.entries(values))
    if ($(id)) $(id).value = value;
  $("n-elite").max = c.walkers;
  $("config-note").textContent =
    "Configuration applied. Changes require reset.";
}
// Benchmark-specific inputs, generated from the engine catalog.
function benchmarkFields(current) {
  const entry = catalogEntry(catalog, $("benchmark").value);
  const parameters = { ...benchmarkParameters(entry), ...current };
  $("benchmark-parameters").replaceChildren(
    ...entry.parameterFields.map((field) => {
      const label = document.createElement("label"),
        input = document.createElement("input");
      Object.assign(input, {
        id: `param-${field.key}`,
        type: "number",
        min: field.min,
        max: field.max,
        step: field.kind === "integer" ? 1 : "any",
        required: true,
        value: parameters[field.key],
      });
      input.dataset.parameter = field.key;
      label.append(field.label, input);
      return label;
    }),
  );
  $("benchmark-parameters").hidden = !entry.parameterFields.length;
  const dimensions = $("dimensions");
  dimensions.readOnly = dimensionLocked(entry);
  dimensions.value = dimensionFor(
    entry,
    formParameters(),
    Number(dimensions.value) || 2,
  );
  $("benchmark-note").textContent = [
    `Domain [${entry.bounds.join(", ")}]`,
    entry.dimension
      ? `${entry.dimension}D only`
      : entry.dimensionRule
        ? `dimension = ${entry.dimensionRule}`
        : entry.dimensions
          ? `dimensions ${entry.dimensions.join(", ")}`
          : "",
    entry.objective_execution === "host"
      ? "objective evaluated on the host in f64 on every backend"
      : "",
    entry.gradient_cost ? `gradient costs ${entry.gradient_cost}` : "",
    entry.reference || "",
  ]
    .filter(Boolean)
    .join(" · ");
}
const formParameters = () =>
  Object.fromEntries(
    Array.from(
      $("benchmark-parameters").querySelectorAll("[data-parameter]"),
      (input) => [input.dataset.parameter, Number(input.value)],
    ),
  );
function fillBenchmarks() {
  const groups = new Map();
  for (const entry of catalog.benchmarks) {
    const name =
      entry.suite === "bbob" ? `COCO BBOB · ${entry.group}` : entry.group;
    if (!groups.has(name)) {
      const group = document.createElement("optgroup");
      group.label = name;
      groups.set(name, group);
    }
    groups.get(name).append(new Option(entry.name, entry.id));
  }
  $("benchmark").replaceChildren(...groups.values());
}
// Adopts an initialized or restored run: domain, axes, slices, first frame.
function adopt(result, home) {
  config = result.config;
  info = benchmarkInfo(config, catalog, result.objective);
  frame = result.frame;
  selected = 0;
  fillForm(config);
  stage.setDomain({
    low: info.low,
    high: info.high,
    dimensions: config.dimensions,
    minimum: info.minimum,
  });
  view.configureAxes();
  acceptFrame(result.frame, true);
  render();
  if (home) stage.home();
}
async function initialize(nextConfig) {
  await pause();
  clearError();
  operation = true;
  controls();
  $("engine-status").textContent = "Initializing Rust engine…";
  try {
    const result = await client.request("initialize", nextConfig);
    adopt(result, true);
    $("chart-note").textContent =
      "Teal: best recorded. Lavender: current mean. Landscape queries do not count as reward evaluations.";
    $("engine-status").textContent =
      `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · ${config.gas.precision}`;
    $("engine-status").title = result.device;
    await view.refreshSurface(true);
  } finally {
    operation = false;
    if (frame && config)
      $("engine-status").textContent =
        `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · ${config.gas.precision}`;
    controls();
  }
}
async function stepOnce() {
  if (pendingStep) return pendingStep;
  const start = performance.now();
  pendingStep = client
    .request("step", { count: 1 })
    .then((next) => {
      stepMs = performance.now() - start;
      acceptFrame(next);
    })
    .finally(() => {
      pendingStep = null;
      controls();
    });
  controls();
  return pendingStep;
}
async function loop() {
  try {
    while (running) {
      await stepOnce();
      await new Promise((resolve) => setTimeout(resolve, 16));
    }
  } catch (error) {
    showError(error);
  } finally {
    running = false;
    controls();
  }
}
function download(name, data, type) {
  const blob = new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function restore(bytes) {
  await pause();
  clearError();
  operation = true;
  controls();
  try {
    const copy = new Uint8Array(bytes);
    const result = await client.request("restore", copy.buffer, [copy.buffer]);
    adopt(result, false);
    $("engine-status").textContent =
      `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · ${config.gas.precision}`;
    await view.refreshSurface(true);
    $("chart-note").textContent =
      "Trace starts at the restored checkpoint. Teal: best recorded. Lavender: current mean.";
  } finally {
    operation = false;
    controls();
  }
}
$("configuration").addEventListener("submit", (event) => {
  event.preventDefault();
  safe(async () => {
    if (!$("configuration").reportValidity()) return;
    const values = { parameters: formParameters() };
    for (const input of $("configuration").querySelectorAll("input,select"))
      values[input.id] = input.value;
    await initialize(resolveConfig(config, values, catalog));
  });
});
$("walkers").addEventListener("input", () => {
  $("n-elite").max = $("walkers").value;
});
$("configuration").addEventListener("input", () => {
  $("config-note").textContent = "Pending changes — Apply & reset to use them.";
});
$("benchmark").addEventListener("change", () => {
  benchmarkFields();
  // Wide domains are explored from a uniform start, as in the Optimization Lab.
  const [low, high] = catalogEntry(catalog, $("benchmark").value).bounds;
  $("initial-box").value = high - low > 25 || low > -1 ? "domain" : "unit";
});
$("benchmark-parameters").addEventListener("input", () => {
  const entry = catalogEntry(catalog, $("benchmark").value);
  if (dimensionLocked(entry))
    $("dimensions").value = dimensionFor(entry, formParameters(), 2);
});
$("dimensions").addEventListener("change", () => {
  const entry = catalogEntry(catalog, $("benchmark").value);
  $("dimensions").value = dimensionFor(
    entry,
    formParameters(),
    Number($("dimensions").value) || 2,
  );
});
$("kinetic").addEventListener("change", () => {
  if ($("kinetic").value === "baoab") {
    $("distance").value = "phase_space";
    $("distance-law").value = "fisher_yates";
    $("clone-law").value = "fisher_yates";
    $("noise-geometry").value =
      Number($("dimensions").value) === 2 ? "full" : "diagonal";
    $("noise-scale").value = 0.2;
  } else {
    $("distance").value = "euclidean";
    $("noise-geometry").value = "isotropic";
    $("noise-scale").value = 1;
  }
});
$("run").addEventListener("click", () => {
  if (running) safe(pause);
  else {
    clearError();
    running = true;
    controls();
    void loop();
  }
});
$("step").addEventListener("click", () =>
  safe(async () => {
    clearError();
    await stepOnce();
  }),
);
$("reset").addEventListener("click", () => safe(() => initialize(config)));
$("walker").addEventListener("input", () =>
  select(Number($("walker").value) || 0),
);
for (const id of ["x-axis", "y-axis", "z-axis"])
  $(id).addEventListener("change", () =>
    safe(async () => {
      if (!config) return;
      view.distinctAxes(id);
      view.sliceControls();
      draw();
      if (id !== "z-axis") await view.refreshSurface();
    }),
  );
$("view").addEventListener("change", () =>
  safe(async () => {
    if (!frame) return;
    render();
    await view.refreshSurface();
  }),
);
for (const id of [
  "color-mode",
  "point-size",
  "surface-opacity",
  "height-scale",
  "links",
  "trails",
  "landscape-toggle",
])
  $(id).addEventListener("input", draw);
$("resolution").addEventListener("change", () => view.refreshSurface());
$("fit-view").addEventListener("click", () => stage.fit());
$("reset-camera").addEventListener("click", () => stage.resetCamera());
$("slice-from-best").addEventListener("click", () => {
  if (frame) view.sliceFrom(view.best());
});
$("slice-from-selected").addEventListener("click", () => {
  if (frame) view.sliceFrom(selected);
});
$("slice-reset").addEventListener("click", () => {
  if (!frame) return;
  view.resetSlice();
  view.refreshSurface();
});
$("chart-axis").addEventListener("change", () =>
  drawConvergence($("convergence"), records, $("chart-axis").value),
);
$("save-checkpoint").addEventListener("click", () =>
  safe(async () => {
    await pause();
    const bytes = await client.request("checkpoint");
    download(
      `algorithmic-gas-step-${frame.step}.agc`,
      bytes,
      "application/cbor",
    );
  }),
);
$("restore-checkpoint").addEventListener("click", () =>
  $("checkpoint-file").click(),
);
$("checkpoint-file").addEventListener("change", () =>
  safe(async () => {
    const file = $("checkpoint-file").files[0];
    if (!file) return;
    if (file.size > 256 * 1024 * 1024)
      throw new Error("Checkpoint exceeds 256 MiB.");
    await restore(await file.arrayBuffer());
    $("checkpoint-file").value = "";
  }),
);
$("save-local").addEventListener("click", () =>
  safe(async () => {
    await pause();
    await saveCheckpoint(await client.request("checkpoint"));
    $("run-status").textContent = "Checkpoint saved in this browser";
  }),
);
$("load-local").addEventListener("click", () =>
  safe(async () => restore(await loadCheckpoint())),
);
$("export-config").addEventListener("click", () =>
  download(
    "algorithmic-gas-config.json",
    JSON.stringify(config, null, 2),
    "application/json",
  ),
);
$("import-config").addEventListener("click", () => $("config-file").click());
$("config-file").addEventListener("change", () =>
  safe(async () => {
    const file = $("config-file").files[0];
    if (!file) return;
    if (file.size > 1024 * 1024)
      throw new Error("Configuration exceeds 1 MiB.");
    await initialize(JSON.parse(await file.text()));
    $("config-file").value = "";
  }),
);
$("export-results").addEventListener("click", () =>
  download(
    `algorithmic-gas-results-${frame.step}.json`,
    JSON.stringify(
      {
        version: 1,
        config,
        measurement_stage: "post_kinetic",
        trace_limit: 5000,
        records,
        frame,
      },
      (_, value) => (ArrayBuffer.isView(value) ? Array.from(value) : value),
      2,
    ),
    "application/json",
  ),
);
new ResizeObserver(() =>
  drawConvergence($("convergence"), records, $("chart-axis").value),
).observe($("convergence"));
window.addEventListener("pagehide", () => {
  client.dispose();
  stage?.dispose();
  molecule?.dispose();
});

await safe(async () => {
  const requested = new URLSearchParams(location.search).get("view");
  if ([...$("view").options].some((option) => option.value === requested))
    $("view").value = requested;
  stage = new StageController($("stage"), select);
  view = createView({
    $,
    client,
    stage,
    state: {
      get config() {
        return config;
      },
      get frame() {
        return frame;
      },
      get info() {
        return info;
      },
    },
    redraw: draw,
    fail: showError,
  });
  const defaults = await client.request("defaults");
  catalog = defaults.catalog;
  fillBenchmarks();
  config = defaults.config;
  config.gas.n_elite = Math.min(2, config.walkers);
  await initialize(config);
  window.euclideanGasLab = {
    ready: true,
    stage,
    view,
    trails,
    get selected() {
      return selected;
    },
  };
});
