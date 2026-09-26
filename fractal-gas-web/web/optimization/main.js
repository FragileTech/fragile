import { EngineClient } from "./client.js";
import { frameInfo, row } from "./native.js";
import { Recording, importRecording, RECORDING_LIMIT } from "./recording.js";
import { geometryNames, geometryColors } from "./geometry.js";
import { exportFixedBudgetCSV } from "./fixed-budget.js";
import { SwarmRenderer, MoleculeRenderer } from "./renderer.js";
const $ = (id) => document.getElementById(id),
  form = $("configuration");
const DEFAULT_ALGORITHM = "gas",
  DEFAULT_GAS_PERTURBATION = "local_covariance";
const client = new EngineClient();
let catalog,
  config,
  recording,
  recordingEnabled = false,
  index = 0,
  selected = -1,
  running = false,
  busy = false,
  budgetStopped = false,
  creating = false,
  imported = false,
  epoch = 0,
  surfaceRevision = 0,
  replayTimer,
  simulationMs = 0,
  slice = [];
const renderer = new SwarmRenderer($("world"), (i) => {
  selected = i;
  $("walker-index").value = i;
  renderFrame();
});
let molecule;
const number = (value) =>
  Number.isFinite(value)
    ? new Intl.NumberFormat("en", {
        maximumSignificantDigits: 6,
        notation:
          value !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 1e6)
            ? "scientific"
            : "standard",
      }).format(value)
    : "—";
const status = (message, error = false) => {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
};
function numeric(key, label, value, min = -1e12, max = 1e12) {
  const wrap = document.createElement("label");
  wrap.textContent = label;
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.name = key;
  input.value = value;
  input.min = min;
  input.max = max;
  input.required = true;
  wrap.append(input);
  return wrap;
}
function selectInput(key, label, value, options) {
  const wrap = document.createElement("label");
  wrap.textContent = label;
  const input = document.createElement("select");
  input.name = key;
  for (const [id, name] of options) {
    const option = new Option(name, id);
    input.add(option);
  }
  input.value = value;
  wrap.append(input);
  return wrap;
}
function checkInput(key, label, value) {
  const wrap = document.createElement("label");
  wrap.className = "check";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.name = key;
  input.checked = value;
  wrap.append(input, document.createTextNode(label));
  return wrap;
}
const defaults = {
  horizon: 32,
  max_horizon: 0,
  max_walkers: 10000,
  freeze_prefix_after: 0,
  dt_min: 1,
  dt_max: 1,
  elites: 0,
  gamma: 1,
  beta: 1,
  delta_t: 0.002,
  substeps: 1,
  clone_every: 1,
  reward_coef: 1,
  distance_coef: 1,
  distance_metric: "l2",
  epsilon: 0.1,
  clone_epsilon: 0.1,
  lambda_alg: 0,
  eta: 0.1,
  sigma_min: 1e-8,
  amplitude: 2,
  epsilon_dist: 1e-8,
  rho: 0,
  p_max: 1,
  epsilon_clone: 1e-6,
  sigma_x: 1e-6,
  restitution: 0.5,
};
function parameterFields(panel, parameters, values = {}) {
  for (const parameter of parameters) {
    const value = values[parameter.id] ?? parameter.default;
    const control =
      parameter.type === "boolean"
        ? checkInput(parameter.id, parameter.label, value)
        : parameter.type === "enum"
          ? selectInput(parameter.id, parameter.label, value, parameter.options)
          : numeric(
              parameter.id,
              parameter.label,
              value,
              parameter.min,
              parameter.max,
            );
    if (parameter.type === "integer") control.querySelector("input").step = "1";
    panel.append(control);
  }
}
function perturbationFields(values = {}) {
  const panel = $("perturbation-parameters");
  panel.replaceChildren();
  const descriptor = catalog.perturbations.find(
    (p) => p.id === $("perturbation").value,
  );
  parameterFields(
    panel,
    (descriptor?.parameters || []).filter(
      (p) =>
        p.id !== "adaptive_euclidean_mode" ||
        $("algorithm").value === "euclidean",
    ),
    values,
  );
  perturbationNote();
}
function perturbationOptions(values = {}) {
  const algorithm = $("algorithm").value;
  const current = $("perturbation").value;
  const savedParameters = readConfig();
  const selected =
    values.perturbation ??
    (algorithm === "gas"
      ? DEFAULT_GAS_PERTURBATION
      : catalog.perturbations.some(
            (entry) =>
              entry.id === current &&
              (!entry.algorithms || entry.algorithms.includes(algorithm)),
          )
        ? current
        : "gaussian");
  $("perturbation").replaceChildren();
  for (const entry of catalog.perturbations)
    if (!entry.algorithms || entry.algorithms.includes(algorithm))
      $("perturbation").add(new Option(entry.name, entry.id));
  $("perturbation").value = selected;
  perturbationFields({ ...savedParameters, ...values });
}
function gasLocalSearchNote() {
  const local = form.elements.namedItem("gas_local_search");
  if (!local) return;
  const stochastic = catalog.benchmarks.find(
    (b) => b.id === $("benchmark").value,
  )?.stochastic;
  local.disabled = Boolean(stochastic);
  if (stochastic) local.checked = false;
  $("gas-local-note").textContent = stochastic
    ? "L-BFGS-B is disabled for stochastic objectives. Tabu memory and GAS jumps remain available."
    : "Each local search counts toward the evaluation budget.";
}
function perturbationNote() {
  const direct =
    ["adaptive_fractal", "cloning_guided"].includes($("perturbation").value) &&
    form.elements.namedItem("adaptive_euclidean_mode")?.value === "position";
  for (const key of ["gamma", "beta", "delta_t", "substeps", "kinetic"]) {
    const input = form.elements.namedItem(key);
    if (input) input.disabled = direct;
  }
  const force = form.elements.namedItem("potential_force");
  if (force)
    force.disabled =
      direct ||
      Boolean(
        catalog.benchmarks.find((b) => b.id === $("benchmark").value)
          ?.stochastic,
      );
  const mode = form.elements.namedItem("adaptive_euclidean_mode");
  if (mode) mode.onchange = perturbationNote;

  $("perturbation-note").textContent =
    $("perturbation").value === "cloning_guided"
      ? `Uses cloning-selected population geometry and bounded fitness-directed movement. Scale limits bound the noise coefficient, not displacement. ${direct ? "Kinetic controls are inactive." : $("algorithm").value === "euclidean" ? "Scales apply to velocity kicks before kinetic integration." : "Scales use coordinate units per proposal draw."} Experimental optimization strategy; no Gibbs sampling guarantee.`
      : $("perturbation").value === "adaptive_fractal"
        ? `Better walkers use smaller scales within the chosen limits; covariance learns direction only. Limits describe statistical scale, not maximum travel distance. ${$("algorithm").value === "euclidean" && !direct ? "Scales apply to velocity kicks before thermal integration." : "Scales use coordinate units per proposal draw."} ${direct ? "Kinetic controls are inactive." : "Paired alternatives count toward the evaluation budget."}`
        : $("perturbation").value === "local_covariance"
          ? "Learns local proposal directions from search outcomes. Standard deviation sets average coordinate variance; planner geometry stays fixed through search and execution."
          : $("perturbation").value === "gas_adaptive"
            ? "Objective-dependent Gaussian jumps: standard deviation ranges from 0.00001 to 0.1 of the domain width."
            : $("algorithm").value === "euclidean"
              ? "Scales the random velocity kick. Standard deviation 1 keeps the configured temperature."
              : "Standard deviation is measured in coordinate units for each proposal step.";
}
function objectiveNote() {
  $("objective-note").textContent =
    $("objective").value === "maximize"
      ? "Maximize the function value. Best values increase."
      : "Minimize the function by maximizing its negative value.";
}
function isCma(algorithm = config?.algorithm) {
  return ["cmaes_active", "cmaes_bipop"].includes(algorithm);
}
function cmaControls() {
  const cma = isCma($("algorithm").value);
  $("controller-panel").hidden = cma;
  for (const input of $("controller-panel").querySelectorAll("input[name]"))
    input.disabled = cma;
  for (const name of ["walkers", "boundary", "perturbation"])
    form.elements.namedItem(name).disabled = cma;
  if (cma) {
    form.elements.namedItem("boundary").value = "cma";
    form.elements.namedItem("controller_enabled").checked = false;
  }
  $("perturbation-parameters").hidden = cma;
  $("perturbation-note").hidden = cma;
  $("edges").disabled = cma;
  $("trails").disabled = cma;
}
function algorithmFields(values = {}) {
  cmaControls();
  $("walkers-label").textContent =
    $("algorithm").value === "graph" ? "Target leaves" : "Walkers";
  perturbationNote();
  const panel = $("algorithm-parameters");
  panel.replaceChildren();
  const algorithm = $("algorithm").value,
    gas = algorithm === "euclidean";
  if (!isCma(algorithm) && algorithm !== "graph") {
    panel.append(
      numeric(
        "max_walkers",
        "Maximum walkers",
        values.max_walkers ??
          (["gas", "euclidean"].includes(algorithm)
            ? Math.max(
                defaults.max_walkers,
                Number(form.elements.namedItem("walkers").value),
              )
            : Number(form.elements.namedItem("walkers").value)),
        2,
        100000,
      ),
    );
    panel.append(
      selectInput(
        "removal_policy",
        "Remove worst by",
        values.removal_policy ?? "virtual_reward",
        [
          ["virtual_reward", "Virtual fitness"],
          ["cumulative_reward", "Cumulative reward"],
        ],
      ),
    );
  }
  const descriptor = catalog.algorithms.find((a) => a.id === algorithm);
  if (Array.isArray(descriptor?.parameters)) {
    parameterFields(panel, descriptor.parameters, values);
    if (isCma(algorithm)) {
      const note = document.createElement("p");
      note.className = "hint";
      note.textContent =
        "CMA-ES initialization settings require Start / reset run. Only its evaluation budget applies live.";
      panel.prepend(note);
    }
    if (algorithm === "gas") {
      const note = document.createElement("p");
      note.id = "gas-local-note";
      panel.append(note);
      gasLocalSearchNote();
    }
    return;
  }
  if (!["euclidean", "wave", "graph", "fmc", "wave_jump"].includes(algorithm))
    return;
  const primary = gas
    ? [
        ["delta_t", "Time step", 1e-9, 1],
        ["gamma", "Friction", 1e-9, 1e6],
        ["beta", "Inverse temperature", 1e-9, 1e12],
      ]
    : ["fmc", "wave_jump"].includes(algorithm)
      ? [["horizon", "Search horizon", 1, 4096]]
      : [];
  for (const [key, label, min, max] of primary)
    panel.append(numeric(key, label, values[key] ?? defaults[key], min, max));
  const details = document.createElement("details");
  details.className = "advanced";
  const summary = document.createElement("summary");
  summary.textContent = "Algorithm parameters";
  const content = document.createElement("div");
  details.append(summary, content);
  panel.append(details);
  if (["wave", "graph", "fmc", "wave_jump"].includes(algorithm))
    content.append(
      selectInput(
        "distance_metric",
        "Observation distance",
        values.distance_metric ?? "l2",
        [
          ["l2", "L2"],
          ["cosine", "Cosine"],
        ],
      ),
    );
  const common = [
    ["reward_coef", "Reward exponent", 0, 10],
    ["distance_coef", "Distance exponent", 0, 10],
  ];
  const fields = gas
    ? [
        ...common,
        ["substeps", "Kinetic substeps", 1, 100],
        ["clone_every", "Clone every N iterations", 1, 100000],
        ["epsilon", "Distance companion range", 1e-9, 1e6],
        ["clone_epsilon", "Clone companion range", 1e-9, 1e6],
        ["lambda_alg", "Velocity distance weight", 0, 1e6],
        ["eta", "Fitness positivity floor", 0, 100],
        ["sigma_min", "Statistics regularization", 1e-12, 1e6],
        ["amplitude", "Logistic amplitude", 1e-9, 100],
        ["epsilon_dist", "Distance regularization", 0, 100],
        ["rho", "Localization radius (0 = global)", 0, 1e6],
        ["p_max", "Cloning probability scale", 1e-9, 1],
        ["epsilon_clone", "Cloning regularization", 1e-12, 100],
        ["sigma_x", "Clone position jitter", 0, 1e6],
        ["restitution", "Velocity restitution", 0, 1],
      ]
    : [
        ...common,
        ["dt_min", "Minimum proposal steps", 1, 100],
        ["dt_max", "Maximum proposal steps", 1, 100],
        algorithm === "graph"
          ? ["max_walkers", "Maximum tree population", 2, 1000000]
          : ["elites", "Elite walkers", 0, 100000],
        ...(algorithm === "graph"
          ? [
              [
                "freeze_prefix_after",
                "Freeze shared prefix after N nodes (0 = off)",
                0,
                1000000,
              ],
            ]
          : []),
      ];
  for (const [key, label, min, max] of fields) {
    const control = numeric(key, label, values[key] ?? defaults[key], min, max);
    if (key === "freeze_prefix_after")
      control.querySelector("input").step = "1";
    content.append(control);
  }
  if (algorithm === "wave_jump") {
    content.append(
      numeric(
        "max_horizon",
        "Maximum horizon (0 = automatic)",
        values.max_horizon ?? 0,
        0,
        4096,
      ),
    );
    content.append(
      checkInput(
        "consensus_prefix",
        "Execute shared ancestry prefix",
        values.consensus_prefix ?? true,
      ),
    );
  }
  if (gas) {
    const choices = [
      ["cloning", "Cloning (softmax + revival)"],
      ["softmax", "Softmax"],
      ["uniform", "Uniform"],
      ["random_pairing", "Random mutual pairs"],
      ["greedy_pairing", "Greedy mutual pairs"],
    ];
    for (const [key, label] of [
      ["companion", "Distance companions"],
      ["clone_companion", "Clone companions"],
    ])
      content.append(
        selectInput(key, label, values[key] ?? "cloning", choices),
      );
    for (const [key, label] of [
      ["potential_force", "Potential force"],
      ["cloning", "Enable cloning"],
      ["kinetic", "Enable kinetics"],
    ])
      content.append(
        checkInput(
          key,
          label,
          values[key] ??
            (key !== "potential_force" ||
              !$("benchmark").value.startsWith("bbob_")),
        ),
      );
  }
}
function benchmarkFields(values = {}) {
  const entry = catalog.benchmarks.find((b) => b.id === $("benchmark").value);
  $("benchmark-note").textContent =
    entry.reference ||
    (entry.minimum !== undefined ? `Reference minimum: ${entry.minimum}` : "");
  const panel = $("benchmark-parameters");
  panel.replaceChildren();
  for (const [key, value] of Object.entries(entry.parameters || {})) {
    const labels = {
      n_atoms: "Atoms per walker",
      n_gaussians: "Mixture components",
      benchmark_seed: "Mixture seed",
      std: "Noise standard deviation",
      alpha: "Well curvature",
      lambda_h: "Quartic coupling",
      vev: "Vacuum expectation value",
      field_scale: "Field scale",
      tilt: "Tilt",
      coco_instance: "COCO instance",
    };
    panel.append(
      numeric(
        key,
        labels[key] || key,
        values[key] ?? value,
        key === "tilt" ? -1e6 : key === "coco_instance" ? 1 : 0,
        key === "coco_instance" ? 1000 : 1e12,
      ),
    );
  }
  $("dimensions").disabled = !!entry.dimension || entry.id === "lennard_jones";
  $("dimensions").value =
    entry.dimension ||
    (entry.id === "lennard_jones"
      ? 3 * (values.n_atoms ?? 10)
      : Math.max(entry.minDimension || 1, values.dimensions ?? 3));
  $("dimensions").min = entry.minDimension || 1;
  $("dimensions").max = entry.maxDimension || 4096;
  if (
    entry.dimensions &&
    !entry.dimensions.includes(Number($("dimensions").value))
  )
    $("dimensions").value = 3;
  if (entry.suite === "bbob" && values.reference_minimum !== undefined)
    $("benchmark-note").textContent =
      `${entry.group} · Instance ${values.coco_instance} · Reference minimum ${number(values.reference_minimum)}. Dimensions: ${entry.dimensions.join(", ")}.`;
  const instance = panel.querySelector('[name="coco_instance"]');
  if (instance) instance.step = "1";
  $("low").value = values.low ?? entry.bounds[0];
  $("high").value = values.high ?? entry.bounds[1];
}
function readConfig() {
  const data = {};
  for (const input of form.elements) {
    if (!input.name) continue;
    data[input.name] =
      input.type === "checkbox"
        ? input.checked
        : input.type === "number"
          ? Number(input.value)
          : input.value;
  }
  data.periodic = data.boundary === "periodic";
  if (data.benchmark === "lennard_jones") data.dimensions = 3 * data.n_atoms;
  // Loaded mixture parameters survive a fresh rerun; edited mixtures regenerate.
  return data;
}
function populateForm(values) {
  values = {
    ...values,
    boundary:
      values.boundary ??
      (isCma(values.algorithm) ? "cma" : values.periodic ? "periodic" : "none"),
  };
  for (const input of $("live-tuning").querySelectorAll("input, select"))
    input.disabled = false;
  $("benchmark").value = values.benchmark;
  $("algorithm").value = values.algorithm;
  $("objective").value = values.objective ?? "minimize";
  perturbationOptions(values);
  benchmarkFields(values);
  algorithmFields(values);
  perturbationFields(values);
  for (const [key, value] of Object.entries(values)) {
    const input = form.elements.namedItem(key);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = value;
    else input.value = value;
  }
  gasLocalSearchNote();
  cmaControls();
}
for (const [id, name] of Object.entries(geometryNames)) {
  const label = document.createElement("label");
  label.className = "check";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = true;
  input.value = id;
  input.addEventListener("change", renderFrame);
  const swatch = document.createElement("span");
  swatch.textContent = "● ";
  swatch.style.color = `#${geometryColors[id].toString(16).padStart(6, "0")}`;
  label.append(input, swatch, name);
  $("geometry-methods").append(label);
}
function viewSettings() {
  return {
    geometryVisible: $("geometry-visible").checked,
    geometryMethods: [
      ...$("geometry-methods").querySelectorAll("input:checked"),
    ].map((e) => e.value),
    geometryScale: $("geometry-scale").value,
    geometryScope: $("geometry-scope").value,
    geometryOpacity: Number($("geometry-opacity").value),
    geometryVectorScale: Math.max(
      0.01,
      Math.min(100, Number($("geometry-vector-scale").value) || 1),
    ),
    geometryField: $("geometry-field").checked,
    geometryDrift: $("geometry-drift").checked,
    geometryJumps: $("geometry-jumps").checked,
    geometryClones: $("geometry-clones").checked,
    view: $("view").value,
    axes: [$("axis-x"), $("axis-y"), $("axis-z")].map((e) => Number(e.value)),
    color: $("color").value,
    pointSize: Number($("point-size").value),
    opacity: Number($("opacity").value),
    height: Number($("height").value),
    edges: isCma() ? "none" : $("edges").value,
    trails: !isCma() && $("trails").checked,
    showSlice: $("show-slice").checked,
    slice,
    planning: ["fmc", "wave_jump"].includes(config?.algorithm),
  };
}
function configureAxes() {
  for (const [id, axis] of [
    ["axis-x", 0],
    ["axis-y", 1],
    ["axis-z", 2],
  ]) {
    const select = $(id);
    select.replaceChildren();
    for (let d = 0; d < config.dimensions; d++)
      select.add(new Option(`x${d + 1}`, d));
    if (axis === 2 && config.dimensions < 3)
      select.add(new Option("Plane", config.dimensions));
    select.value = Math.min(
      axis,
      axis === 2 ? config.dimensions : config.dimensions - 1,
    );
  }
  slice = Array(config.dimensions).fill(
    Math.max(config.low, Math.min(config.high, 0)),
  );
  if (config.benchmark === "lennard_jones") {
    const best = frameInfo(recording.frames[0]).bestIndex;
    if (best >= 0) slice = Array.from(row(recording.frames[0], best).x);
  }
  sliceControls();
}
function sliceControls() {
  const axes = viewSettings().axes,
    panel = $("slice-controls");
  panel.replaceChildren();
  for (let d = 0; d < config.dimensions; d++) {
    if (d === axes[0] || d === axes[1]) continue;
    const label = numeric(
      `slice-${d}`,
      `Slice x${d + 1}`,
      slice[d],
      config.low,
      config.high,
    );
    label.querySelector("input").addEventListener("change", (e) => {
      const value = Number(e.target.value);
      if (
        !Number.isFinite(value) ||
        value < config.low ||
        value > config.high
      ) {
        e.target.value = slice[d];
        return;
      }
      slice[d] = value;
      refreshSurface();
    });
    panel.append(label);
  }
}
let surfaceTimer;
function refreshSurface() {
  clearTimeout(surfaceTimer);
  const revision = ++surfaceRevision;
  surfaceTimer = setTimeout(async () => {
    if (!config) return;
    try {
      const settings = viewSettings(),
        resolution = Number($("resolution").value);
      const result = await client.request("surface", {
        axes: settings.axes,
        slice: [...slice],
        resolution,
      });
      if (revision !== surfaceRevision) return;
      renderer.setSurface(result.values, resolution, settings);
      renderFrame();
    } catch (error) {
      status(error.message, true);
    }
  }, 80);
}
function controls() {
  const ready = !!recording;
  $("geometry-collect").disabled = imported || busy || creating;
  $("run").disabled =
    !ready ||
    creating ||
    budgetStopped ||
    running ||
    imported ||
    index !== recording.frames.length - 1;
  $("pause").disabled = !running;
  $("step").disabled =
    !ready ||
    creating ||
    budgetStopped ||
    running ||
    busy ||
    imported ||
    index !== recording.frames.length - 1;
  $("apply-live").disabled =
    creating ||
    imported ||
    !recording ||
    $("algorithm").value !== config.algorithm;
  for (const id of ["restart-round", "export-basins", "import-basins"])
    $(id).disabled = !ready || creating || imported || isCma();
  $("reset").disabled = !ready || creating;
  $("save").disabled = !ready || !recordingEnabled;
  $("export-csv").disabled = !ready || !recordingEnabled;
  $("load").disabled = !catalog;
  $("replay").disabled = !ready || recording.frames.length < 2;
  $("latest").disabled = !ready || !recordingEnabled;
  $("timeline").disabled = !ready || !recordingEnabled;
  $("reset").textContent = imported ? "Rerun settings" : "Reset";
}
function stopReplay() {
  clearInterval(replayTimer);
  replayTimer = null;
  $("replay").textContent = "Play replay";
}
function pause() {
  running = false;
  stopReplay();
  controls();
}
function convergence() {
  const canvas = $("convergence"),
    rect = canvas.getBoundingClientRect(),
    ratio = Math.min(devicePixelRatio, 2);
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const w = rect.width,
    h = rect.height;
  ctx.clearRect(0, 0, w, h);
  if (!recording?.frames.length) return;
  const frames = recording.frames;
  const evaluationAxis = $("chart-axis").value === "evaluations";
  const columnX = evaluationAxis ? 5 : 4;
  const lastX = Math.max(1, frames.at(-1)[columnX]);
  const chartX = (frame) => 12 + (frame[columnX] / lastX) * (w - 24);
  const finite = frames.flatMap((f) => [f[9], f[10]]).filter(Number.isFinite);
  if (!finite.length) return;
  let min = finite.reduce((a, b) => Math.min(a, b), Infinity),
    max = finite.reduce((a, b) => Math.max(a, b), -Infinity);
  const transform = (v) =>
    Math.asinh((v - min) / Math.max(1e-9, (max - min) / 5));
  const scale = transform(max) || 1;
  ctx.strokeStyle = "#3b3048";
  ctx.beginPath();
  ctx.moveTo(0, h - 18);
  ctx.lineTo(w, h - 18);
  ctx.stroke();
  for (const [column, color, label] of [
    [9, "#efc87b", "Best"],
    [10, "#7ef5df", "Mean"],
  ]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let connected = false;
    frames.forEach((f, i) => {
      if (!Number.isFinite(f[column])) {
        connected = false;
        return;
      }
      const x = chartX(f),
        y = h - 20 - (transform(f[column]) / scale) * (h - 40);
      connected ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      connected = true;
    });
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "11px system-ui";
    ctx.fillText(label, column === 9 ? 12 : 65, 12);
  }
  ctx.strokeStyle = "#d0a9e2";
  ctx.beginPath();
  const marker = chartX(frames[index]);
  ctx.moveTo(marker, 18);
  ctx.lineTo(marker, h - 18);
  ctx.stroke();
  ctx.fillStyle = "#b5a6c0";
  ctx.fillText(
    `Objective (asinh scale) · ${evaluationAxis ? "Evaluations" : "Iteration"}`,
    Math.max(120, w - 200),
    h - 3,
  );
}
function inspect(frame) {
  const info = frameInfo(frame);
  $("walker-index").max = info.n - 1;
  if (selected < 0 || selected >= info.n) {
    $("walker-info").textContent = "Click a walker to inspect its state.";
    $("molecule").hidden = true;
    return;
  }
  const w = row(frame, selected),
    div = $("walker-info");
  div.replaceChildren();
  const dl = document.createElement("dl");
  for (const [key, value] of [
    [
      "Role",
      viewSettings().planning
        ? selected === info.n - 1
          ? "Committed position"
          : "Search walker"
        : "Walker",
    ],
    ["Objective", number(w.value)],
    ["Pre-step fitness", number(w.fitness)],
    ["Status", w.alive ? "Alive" : "Invalid"],
    ["Leaf", w.leaf ? "Yes" : "No"],
    ["Cloned", w.cloned ? "Yes" : "No"],
    ["Parent", w.parent],
  ]) {
    if (
      isCma() &&
      ["Pre-step fitness", "Leaf", "Cloned", "Parent"].includes(key)
    )
      continue;
    const dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    dl.append(dt, dd);
  }
  div.append(dl);
  const coordinates = document.createElement("div");
  coordinates.className = "coordinates";
  for (let d = 0; d < info.d; d++) {
    const line = document.createElement("div");
    line.className = "coordinate";
    const label = document.createElement("span"),
      value = document.createElement("span");
    label.textContent = `x${d + 1}`;
    value.textContent = number(w.x[d]);
    line.append(label, value);
    coordinates.append(line);
    if (info.velocity) {
      const velocity = document.createElement("div");
      velocity.className = "coordinate";
      velocity.textContent = `v${d + 1}: ${number(w.v[d])}`;
      coordinates.append(velocity);
    }
  }
  div.append(coordinates);
  $("molecule").hidden = config.benchmark !== "lennard_jones" || !w.alive;
  if (!$("molecule").hidden) {
    molecule ??= new MoleculeRenderer($("molecule-canvas"));
    molecule.update(w.x);
  }
}
let activeStatus = null,
  pendingSettings = null,
  applyingSettings = false,
  liveError = "";
function liveValues() {
  const data = readConfig(),
    values = {};
  if (data.algorithm !== config?.algorithm) return values;
  for (const input of $("live-tuning").querySelectorAll("[name]")) {
    if (input.disabled || (isCma() && input.name !== "max_evaluations"))
      continue;
    values[input.name] = data[input.name];
  }
  return values;
}
function livePatch(values = liveValues()) {
  const baseline = activeStatus?.pending_settings || config || {};
  const patch = Object.fromEntries(
    Object.entries(values).filter(([key, value]) => value !== baseline[key]),
  );
  if ("walkers" in patch && "population_auto" in values)
    patch.population_auto = values.population_auto;
  if (
    ("perturbation_std" in patch ||
      "adaptive_min_scale" in patch ||
      "adaptive_max_scale" in patch) &&
    "scale_auto" in values
  )
    patch.scale_auto = values.scale_auto;
  return patch;
}
function showLive(state) {
  activeStatus = state;
  const controller = state.controller;
  $("controller-status").textContent = controller
    ? `Round ${controller.round} · ${controller.regime} · ${controller.round_evaluations} round / ${controller.global_evaluations} total evaluations · ${controller.restart_pending ? "restart queued" : controller.restart_reason || "active"}`
    : "";
  $("basin-rows").replaceChildren();
  for (const basin of controller?.basins || []) {
    const row = document.createElement("tr");
    for (const value of [
      basin.objective?.toPrecision(6) ?? "invalid",
      basin.visits,
      basin.validated ? basin.confidence.toFixed(2) : "reference",
      basin.discovery_round,
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    $("basin-rows").append(row);
  }
  const exploration = state.exploration;
  $("exploration-status").textContent = exploration
    ? exploration.strategy === "cloning_guided"
      ? `${exploration.model_count} local models · scale ${exploration.scale_min.toPrecision(3)}–${exploration.scale_max.toPrecision(3)} · signed clone scores · ${exploration.selection_comparisons.toFixed(1)} comparisons/model · condition ≤ ${exploration.condition_number.toPrecision(3)} · drift/noise ${exploration.drift_noise_ratio.toPrecision(3)} · ${exploration.fallback_count} fallbacks (${exploration.fallback_reason})`
      : `${exploration.model_count} local models · scale ${exploration.scale_min.toPrecision(3)}–${exploration.scale_max.toPrecision(3)} · ${exploration.effective_parents.toFixed(1)} effective parents · proposals ${exploration.proposal_probabilities.map((p) => (100 * p).toFixed(0) + "%").join(" / ")} · ${exploration.paired_outcomes} paired outcomes`
    : "";
  for (const id of ["restart-round", "export-basins", "import-basins"])
    $(id).disabled = imported || !controller || creating;

  const n = frameInfo(recording.frames.at(-1)).n;
  $("population-status").textContent =
    config.algorithm === "graph"
      ? `${n} stored nodes · ${config.walkers} target leaves · ${config.max_walkers} maximum`
      : isCma()
        ? `${n} candidates · CMA-ES: only the evaluation budget can be applied live.`
        : `${state.population?.active ?? n} active / ${config.max_walkers} maximum`;
  const dirty = Object.keys(livePatch()).length > 0;
  $("live-status").textContent =
    (imported
      ? "Recording is read-only. Edited settings apply only to a new run."
      : liveError) ||
    (state.pending_settings
      ? `Changes queued for the next search.${dirty ? " Additional unapplied edits." : ""}`
      : dirty
        ? "Unapplied changes."
        : "Live settings are up to date.");
}
$("live-tuning").addEventListener("input", (event) => {
  if (event.target.name === "walkers")
    form.elements.namedItem("population_auto").checked = false;
  if (
    ["perturbation_std", "adaptive_min_scale", "adaptive_max_scale"].includes(
      event.target.name,
    )
  )
    form.elements.namedItem("scale_auto").checked = false;
  liveError = "";
  if (activeStatus) showLive(activeStatus);
});
$("live-tuning").addEventListener("change", () => {
  liveError = "";
  if (activeStatus) showLive(activeStatus);
});
$("apply-live").onclick = () => {
  if (
    creating ||
    imported ||
    !recording ||
    $("algorithm").value !== config.algorithm
  )
    return;
  for (const input of $("live-tuning").querySelectorAll("input, select"))
    if (!input.reportValidity()) return;
  const patch = livePatch();
  if (!Object.keys(patch).length && !applyingSettings && !pendingSettings) {
    showLive(activeStatus);
    return;
  }
  liveError = "";
  // Capture the whole intended form, then diff against the acknowledged engine
  // state when dispatched. This also handles reverting an in-flight edit.
  pendingSettings = { values: liveValues() };
  $("live-status").textContent = "Waiting for the current step…";
  if (!busy) applySettings();
};
$("restart-round").onclick = () => {
  if (imported || creating || !recording) return;
  pendingSettings = {
    values: {
      ...liveValues(),
      restart_token: (activeStatus?.effective_settings?.restart_token ?? 0) + 1,
    },
  };
  $("live-status").textContent =
    "Round restart requested; committed actions finish first.";
  if (!busy) applySettings();
};
$("export-basins").onclick = async () => {
  try {
    const result = await client.request("exportBasins");
    const url = URL.createObjectURL(
      new Blob([result.text], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `${config.benchmark}.basins`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    status(error.message, true);
  }
};
$("import-basins").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file || imported || creating) return;
  if (file.size > 8 * 1024 * 1024) {
    status("Basin archive exceeds 8 MiB", true);
    return;
  }
  pause();
  const token = epoch;
  try {
    const text = await file.text();
    if (token !== epoch) return;
    const result = await client.request("importBasins", {
      text,
      remaining: recordingEnabled
        ? RECORDING_LIMIT - recording.bytes
        : Infinity,
    });
    if (token !== epoch) return;
    recording.append(result.frame, {
      ...result.status,
      settings_update: true,
      archive_update: true,
    });
    showLive(result.status);
    status(
      "Imported basin values remain references until validation during a restart.",
    );
  } catch (error) {
    status(error.message, true);
  }
  event.target.value = "";
};
async function applySettings() {
  const request = pendingSettings;
  pendingSettings = null;
  busy = true;
  applyingSettings = true;
  const token = epoch;
  try {
    const before = frameInfo(recording.frames.at(-1));
    const result = await client.request("updateSettings", {
      patch: livePatch(request.values),
      remaining: recordingEnabled
        ? RECORDING_LIMIT - recording.bytes
        : Infinity,
    });
    if (token !== epoch) return;
    config = result.config;
    if (!result.changed) {
      showLive(result.status);
      return;
    }
    recording.append(result.frame, {
      ...result.status,
      settings_update: true,
      population_changed: before.n !== frameInfo(result.frame).n,
    });
    index = recording.frames.length - 1;
    selected = -1;
    $("walker-index").value = "";
    budgetStopped =
      !!result.status.budget_exhausted || !!result.status.finished;
    showLive(result.status);
    renderFrame();
    status(
      result.status.pending_settings
        ? "Changes queued for the next search."
        : "Live settings applied.",
    );
  } catch (error) {
    if (token === epoch) {
      liveError = error.message;
      $("live-status").textContent = liveError;
      status(error.message, true);
    }
  } finally {
    busy = false;
    applyingSettings = false;
    controls();
    if (token === epoch) {
      if (pendingSettings) applySettings();
      else if (running) step();
    }
  }
}
function renderFrame() {
  if (!recording) return;
  index = Math.min(index, recording.frames.length - 1);
  const frame = recording.frames[index],
    info = frameInfo(frame),
    settings = viewSettings();
  renderer.update(
    frame,
    settings,
    recording.frames.slice(
      Math.max(
        0,
        recording.metadata
          .slice(0, index + 1)
          .findLastIndex((m) => m?.population_changed),
      ),
      index + 1,
    ),
    selected,
    recording.metadata[index],
  );
  $("planner-note").hidden = !settings.planning;
  $("geometry-status").textContent = renderer.geometrySummary;
  $("geometry-collect").disabled = imported || busy || creating;
  const available = new Set(
    recording.metadata[index]?.geometry?.methods?.map((m) => m.id) || [],
  );
  for (const input of $("geometry-methods").querySelectorAll("input"))
    input.parentElement.hidden = !available.has(input.value);
  const cmaMeta = recording.metadata[index];
  $("cma-note").hidden = !isCma();
  $("cma-note").textContent = cmaMeta
    ? `Generation ${cmaMeta.generation} · Population ${info.n} · Restarts ${cmaMeta.restarts} · σ ${number(cmaMeta.sigma)}${cmaMeta.stop_reason ? ` · ${cmaMeta.stop_reason}` : ""}`
    : "CMA-ES candidates";
  $("iteration").textContent = info.iteration;
  $("best").textContent = number(info.best);
  $("mean").textContent = number(info.mean);
  $("alive").textContent = `${info.alive} / ${info.n}`;
  $("evaluations").textContent = config.max_evaluations
    ? `${info.evaluations} / ${config.max_evaluations}`
    : info.evaluations;
  $("reference-metric").hidden =
    !Number.isFinite(config.reference_minimum) ||
    config.objective === "maximize";
  $("optimality-gap").textContent = number(
    info.best - config.reference_minimum,
  );
  $("timings").textContent =
    `Step ${simulationMs.toFixed(1)} ms · Draw ${renderer.renderMs.toFixed(1)} ms`;
  $("timeline").max = recording.frames.length - 1;
  $("timeline").value = index;
  $("frame-label").textContent = recordingEnabled
    ? `Frame ${index + 1} / ${recording.frames.length}`
    : "Live · recording off";
  $("scene-title").textContent =
    catalog.benchmarks.find((b) => b.id === config.benchmark)?.name ||
    config.benchmark;
  const [x, y, z] = settings.axes;
  $("axis-caption").textContent =
    settings.view === "landscape"
      ? `x${x + 1}, x${y + 1}, objective height (asinh scale)`
      : `x${x + 1}, x${y + 1}${z < info.d ? `, x${z + 1}` : ", plane"}`;
  $("axis-z-label").hidden = settings.view === "landscape";
  $("slice-note").textContent =
    config.benchmark === "stochastic_gaussian"
      ? "The surface shows expected objective 0. Walker colors show their recorded random samples."
      : info.d > 2
        ? "The surface fixes the other coordinates to the slice values. Walker heights use their actual full-dimensional objective and may lie away from the slice."
        : "The surface and walkers use the same objective and height transform.";
  inspect(frame);
  convergence();
  controls();
}
let renderPending = false;
function scheduleFrame() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    renderFrame();
  });
}
async function createSession(next, loaded = null) {
  if (creating) return;
  pendingSettings = null;
  liveError = "";
  creating = true;
  pause();
  const token = ++epoch;
  ++surfaceRevision;
  $("apply").disabled = true;
  status("Initializing swarm…");
  try {
    const result = await client.request("create", {
      config: {
        ...next,
        geometry_diagnostics: !loaded && $("geometry-collect").checked,
      },
    });
    if (token !== epoch) return;
    config = result.config;
    recordingEnabled = !!loaded || $("record-history").checked;
    recording = loaded || new Recording(config, undefined, recordingEnabled);
    if (!loaded) recording.append(result.frame, result.status);
    imported = !!loaded;
    index = 0;
    selected = -1;
    $("walker-index").value = "";
    simulationMs = 0;
    budgetStopped = !!result.status?.finished;
    renderer.setConfig(config);
    populateForm(config);
    showLive(result.status);
    objectiveNote();
    configureAxes();
    renderFrame();
    refreshSurface();
    status(
      imported
        ? "Recording loaded. Scrub or play it; Rerun settings starts a new simulation."
        : "Ready. Run the swarm or advance one iteration.",
    );
    $("engine-status").textContent = "C++ / WebAssembly";
    window.optimizationReady = true;
  } catch (error) {
    status(error.message, true);
  } finally {
    creating = false;
    $("apply").disabled = false;
    controls();
  }
}
async function step() {
  if (busy || !recording || imported) return;
  busy = true;
  controls();
  const token = epoch;
  try {
    const result = await client.request("step", {
      remaining: recordingEnabled
        ? RECORDING_LIMIT - recording.bytes
        : Infinity,
    });
    if (token !== epoch) return;
    const atLatest = index === recording.frames.length - 1;
    const previous = frameInfo(recording.frames.at(-1));
    for (const key of ["walkers", "perturbation_std"]) {
      const input = form.elements.namedItem(key);
      if (input && Number(input.value) === config[key])
        input.value = result.config[key];
    }
    config = result.config;
    recording.append(result.frame, {
      ...result.status,
      round_changed: result.roundChanged,
      population_changed:
        result.roundChanged || previous.n !== frameInfo(result.frame).n,
    });
    showLive(result.status);
    simulationMs = result.simulationMs;
    if (atLatest) index = recording.frames.length - 1;
    scheduleFrame();
    if (result.status?.finished) {
      pause();
      budgetStopped = true;
      status(`Optimizer finished: ${result.status.stop_reason || "converged"}`);
    } else if (!frameInfo(result.frame).alive && !config.controller_enabled) {
      pause();
      status(
        "All walkers left the domain or became invalid. Save the run, then reset or change settings.",
        true,
      );
    }
  } catch (error) {
    if (token === epoch) {
      pause();
      budgetStopped = error.message.startsWith("Evaluation budget reached");
      status(error.message, !budgetStopped);
    }
  } finally {
    busy = false;
    controls();
    if (pendingSettings && token === epoch) applySettings();
    else if (running && token === epoch)
      setTimeout(() => {
        if (running && token === epoch) step();
      }, 0);
  }
}
form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (form.reportValidity()) createSession(readConfig());
});
$("benchmark").addEventListener("change", () => {
  benchmarkFields();
  gasLocalSearchNote();
  const force = form.elements.namedItem("potential_force");
  if (force) force.checked = !$("benchmark").value.startsWith("bbob_");
});
$("chart-axis").addEventListener("change", convergence);
$("algorithm").addEventListener("change", () => {
  perturbationOptions();
  algorithmFields();
  controls();
  if (config && $("algorithm").value !== config.algorithm)
    $("live-status").textContent =
      "Use Start / reset run to change algorithms.";
  else if (activeStatus) showLive(activeStatus);
});
$("perturbation").addEventListener("change", () => {
  perturbationFields(readConfig());
  if (activeStatus) showLive(activeStatus);
});
$("objective").addEventListener("change", objectiveNote);
$("run").onclick = () => {
  stopReplay();
  running = true;
  status("Running. View changes do not alter the simulation.");
  controls();
  step();
};
$("pause").onclick = () => {
  pause();
  status("Paused.");
};
$("step").onclick = () => step();
$("reset").onclick = () => createSession(config);
for (const id of ["view", "axis-x", "axis-y", "axis-z"]) {
  $(id).addEventListener("change", () => {
    if (!config) return;
    const settings = viewSettings();
    if (
      id !== "axis-z" &&
      config.dimensions > 1 &&
      settings.axes[0] === settings.axes[1]
    )
      $(id === "axis-x" ? "axis-y" : "axis-x").value =
        (Number($(id).value) + 1) % config.dimensions;
    sliceControls();
    renderFrame();
    refreshSurface();
  });
}
for (const id of [
  "color",
  "point-size",
  "opacity",
  "edges",
  "trails",
  "show-slice",
  "geometry-visible",
  "geometry-scale",
  "geometry-scope",
  "geometry-opacity",
  "geometry-vector-scale",
  "geometry-field",
  "geometry-drift",
  "geometry-jumps",
  "geometry-clones",
])
  $(id).addEventListener("input", renderFrame);
$("geometry-collect").addEventListener("change", async () => {
  if (!recording || imported || busy || creating) return;
  const enabled = $("geometry-collect").checked;
  pause();
  busy = true;
  $("geometry-collect").disabled = true;
  try {
    const result = await client.request("geometry", { enabled });
    // Keep historical frames intact; a newly enabled collector has no past geometry.
    config.geometry_diagnostics = enabled;
    renderFrame();
    status(
      enabled
        ? "Geometry collection enabled; new evidence appears on the next step."
        : "Geometry collection stopped; recorded frames retain their overlays.",
    );
  } catch (error) {
    $("geometry-collect").checked = !enabled;
    status(error.message, true);
  } finally {
    busy = false;
    $("geometry-collect").disabled = false;
    controls();
  }
});
$("height").addEventListener("input", () => {
  if (renderer.surfaceValues)
    renderer.setSurface(
      renderer.surfaceValues,
      renderer.resolution,
      viewSettings(),
    );
  renderFrame();
});
$("resolution").addEventListener("change", refreshSurface);
$("camera").onclick = () => renderer.resetCamera();
$("walker-index").addEventListener("input", () => {
  const value = $("walker-index").value;
  selected = value === "" ? -1 : Number(value);
  if (!Number.isInteger(selected)) selected = -1;
  renderFrame();
});
$("timeline").oninput = () => {
  pause();
  index = Number($("timeline").value);
  renderFrame();
  status("Viewing recorded state. Choose Latest to return to the live swarm.");
};
$("latest").onclick = () => {
  pause();
  index = recording.frames.length - 1;
  renderFrame();
  status(
    imported
      ? "End of imported recording."
      : "Latest state. The live swarm can continue.",
  );
};
$("replay").onclick = () => {
  if (replayTimer) {
    stopReplay();
    return;
  }
  pause();
  if (index === recording.frames.length - 1) index = 0;
  $("replay").textContent = "Stop replay";
  replayTimer = setInterval(
    () => {
      if (index >= recording.frames.length - 1) {
        stopReplay();
        return;
      }
      index++;
      renderFrame();
    },
    1000 / Number($("replay-speed").value),
  );
};
$("save").onclick = () => {
  const blob = new Blob([recording.export()], { type: "application/json" }),
    url = URL.createObjectURL(blob),
    anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${config.benchmark}-${config.algorithm}-${config.seed}.fgopt`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("export-csv").onclick = () => {
  try {
    const csv = exportFixedBudgetCSV(recording);
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${config.benchmark}-${config.algorithm}-${config.seed}.csv`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status(
      "CSV exported. In IOHanalyzer, choose custom CSV and map evaluations, best, function, algorithm, dimension, and run. Use this run’s objective direction.",
    );
  } catch (error) {
    status(error.message, true);
  }
};
$("load").onclick = () => $("file").click();
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (!file) return;
  try {
    if (file.size > 96 * 1024 * 1024)
      throw new Error("Recording file is too large");
    const loaded = importRecording(await file.text());
    await createSession(loaded.config, loaded);
  } catch (error) {
    status(error.message, true);
  } finally {
    $("file").value = "";
  }
};
window.addEventListener("resize", convergence);
window.addEventListener("pagehide", () => {
  pause();
  client.dispose();
  renderer.dispose();
  molecule?.dispose();
});
try {
  catalog = await client.request("catalog");
  const groups = new Map();
  for (const entry of catalog.benchmarks) {
    const name =
      entry.suite === "bbob"
        ? `COCO BBOB · ${entry.group}`
        : "Classic benchmarks";
    if (!groups.has(name)) {
      const group = document.createElement("optgroup");
      group.label = name;
      groups.set(name, group);
      $("benchmark").append(group);
    }
    groups.get(name).append(new Option(entry.name, entry.id));
  }
  for (const entry of catalog.algorithms)
    $("algorithm").add(new Option(entry.name, entry.id));
  $("benchmark").value = "rastrigin";
  $("algorithm").value = DEFAULT_ALGORITHM;
  perturbationOptions();
  benchmarkFields();
  algorithmFields();
  await createSession(readConfig());
} catch (error) {
  status(error.message, true);
  $("engine-status").textContent = "Engine unavailable";
}
