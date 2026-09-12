import { GasClient } from "./client.js";
import { BENCHMARKS, resolveConfig, frameMetrics } from "./config.js";
import { PopulationRenderer, drawConvergence } from "./renderer.js";
import { saveCheckpoint, loadCheckpoint } from "./storage.js";

const $ = (id) => document.getElementById(id);
const client = new GasClient();
let config = null,
  frame = null,
  renderer = null,
  selected = 0;
let running = false,
  operation = false,
  pendingStep = null,
  records = [],
  bestRecorded = null;
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
  const extinct = !!frame && frameMetrics(frame, config).alive === 0;
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
    ["Population version", String(p.version)],
  ]);
}
function draw() {
  if (!frame || !renderer) return;
  const axes = [Number($("x-axis").value), Number($("y-axis").value)];
  renderer.update(
    frame,
    BENCHMARKS[config.benchmark].bounds,
    axes,
    selected,
    config.gas.include_truncated,
  );
  inspector();
  drawConvergence($("convergence"), records, $("chart-axis").value);
}
function acceptFrame(next, reset = false) {
  frame = next;
  const metrics = frameMetrics(frame, config);
  if (reset) {
    records = [];
    bestRecorded = null;
  }
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
  $("stage-loading").hidden = true;
  if (!metrics.alive) {
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
  const values = {
    benchmark: c.benchmark,
    dimensions: c.dimensions,
    walkers: c.walkers,
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
  $("x-axis").replaceChildren();
  $("y-axis").replaceChildren();
  for (let i = 0; i < c.dimensions; i++)
    for (const id of ["x-axis", "y-axis"]) {
      const option = document.createElement("option");
      option.value = i;
      option.textContent = `x${i + 1}`;
      $(id).append(option);
    }
  $("x-axis").value = 0;
  $("y-axis").value = 1;
  $("view-caption").textContent =
    `${BENCHMARKS[c.benchmark]?.label || c.benchmark} · ${c.dimensions}D state`;
  $("config-note").textContent =
    "Configuration applied. Changes require reset.";
}
async function landscape() {
  if (!config || !frame) return;
  const x = Number($("x-axis").value),
    y = Number($("y-axis").value);
  if (x === y) throw new Error("Choose different projection axes.");
  const data = await client.request("landscape", {
    x,
    y,
    resolution: 72,
    center: Array(config.dimensions).fill(0),
  });
  renderer.landscape(data);
  renderer.setLandscapeVisible($("landscape-toggle").checked);
}
async function initialize(nextConfig) {
  await pause();
  clearError();
  operation = true;
  controls();
  $("engine-status").textContent = "Initializing Rust engine…";
  try {
    const result = await client.request("initialize", nextConfig);
    config = result.config;
    selected = 0;
    fillForm(config);
    acceptFrame(result.frame, true);
    renderer.fit();
    $("chart-note").textContent =
      "Teal: best recorded. Lavender: current mean. Landscape queries do not count as reward evaluations.";
    $("engine-status").textContent =
      `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · ${config.gas.precision}`;
    $("engine-status").title = result.device;
    await landscape();
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
      acceptFrame(next);
      $("timing").textContent =
        `${(performance.now() - start).toFixed(1)} ms / step`;
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
    config = result.config;
    selected = 0;
    fillForm(config);
    acceptFrame(result.frame, true);
    $("engine-status").textContent =
      `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · ${config.gas.precision}`;
    await landscape();
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
    const values = {};
    for (const input of $("configuration").querySelectorAll("input,select"))
      values[input.id] = input.value;
    await initialize(resolveConfig(config, values));
  });
});
$("configuration").addEventListener("input", () => {
  $("config-note").textContent = "Pending changes — Apply & reset to use them.";
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
$("walker").addEventListener("input", () => {
  selected = Number($("walker").value) || 0;
  inspector();
  draw();
});
for (const id of ["x-axis", "y-axis"])
  $(id).addEventListener("change", () =>
    safe(async () => {
      if ($("x-axis").value === $("y-axis").value)
        $(id === "x-axis" ? "y-axis" : "x-axis").value =
          (Number($(id).value) + 1) % config.dimensions;
      draw();
      await landscape();
    }),
  );
$("landscape-toggle").addEventListener("change", () =>
  renderer.setLandscapeVisible($("landscape-toggle").checked),
);
$("fit-view").addEventListener("click", () => renderer.fit());
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
  renderer?.dispose();
});

await safe(async () => {
  renderer = new PopulationRenderer($("stage"), (slot) => {
    selected = slot;
    draw();
  });
  const defaults = await client.request("defaults");
  config = defaults.config;
  await initialize(config);
});
