import {
  FractalPopulation,
  encodePopulationRecording,
  decodePopulationRecording,
} from "./populations.js";
import { frameInfo, row } from "./native.js";
const $ = (id) => document.getElementById(id),
  number = (id) => Number($(id).value);
const colors = [
  "#7ef5df",
  "#efc87b",
  "#d0a9e2",
  "#ff729b",
  "#86b9fa",
  "#a6dc87",
  "#eda579",
];
let population,
  history = [],
  current,
  config,
  running = false,
  busy = false,
  playback = false,
  bytes = 0,
  heatmap;
const overrides = [
  ["walkers", "Walkers", "256", "1"],
  ["elites", "Elites", "5", "1"],
  ["perturbation_std", "Movement scale", "0.1", "any"],
  ["reward_coef", "Reward coefficient", "1", "any"],
  ["distance_coef", "Distance coefficient", "1", "any"],
  ["dt_min", "Minimum duration", "1", "1"],
  ["dt_max", "Maximum duration", "1", "1"],
  ["seed", "Seed", "Derived", "1"],
];
function error(e) {
  $("message").textContent = e.message;
  $("message").classList.add("error");
}
function message(text) {
  $("message").textContent = text;
  $("message").classList.remove("error");
}
function members() {
  const old = Array.from($("members").children).map((el) =>
    Object.fromEntries(
      Array.from(el.querySelectorAll("input,select")).map((v) => [
        v.name,
        v.value,
      ]),
    ),
  );
  const n = number("swarm-count");
  if (!Number.isInteger(n) || n < 1 || n > 1000) {
    error(
      new Error(
        "Choose a positive swarm count; resource limits are checked on creation.",
      ),
    );
    return;
  }
  $("members").replaceChildren();
  $("concurrency").max = String(n);
  if (number("concurrency") > n) $("concurrency").value = n;
  for (let i = 0; i < n; i++) {
    const details = document.createElement("details");
    details.className = "swarm-config";
    details.style.setProperty("--swarm-color", colors[i % colors.length]);
    details.open = i === 0;
    const summary = document.createElement("summary");
    summary.textContent = `Swarm ${i + 1}`;
    details.append(summary);
    const fields = document.createElement("div");
    fields.className = "fields";
    for (const [key, label, placeholder, step] of overrides) {
      const wrap = document.createElement("label");
      wrap.textContent = label;
      const input = document.createElement("input");
      input.name = key;
      input.type = "number";
      input.step = step;
      input.min =
        key === "walkers"
          ? "2"
          : ["dt_min", "dt_max"].includes(key)
            ? "1"
            : "0";
      input.placeholder =
        key === "walkers"
          ? $("walkers").value
          : key === "elites"
            ? $("elites").value
            : key === "perturbation_std"
              ? $("scale").value
              : placeholder;
      input.value = old[i]?.[key] ?? "";
      wrap.append(input);
      fields.append(wrap);
    }
    const wrap = document.createElement("label");
    wrap.textContent = "Perturbation";
    const select = document.createElement("select");
    select.name = "perturbation";
    for (const [value, label] of [
      ["", "Shared default"],
      ["gaussian", "Gaussian"],
      ["uniform", "Uniform"],
      ["local_covariance", "Local covariance"],
      ["cloning_guided", "Cloning guided"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.append(option);
    }
    select.value = old[i]?.perturbation ?? "";
    wrap.append(select);
    fields.append(wrap);
    details.append(fields);
    $("members").append(details);
  }
}
function readConfig() {
  const defaults = {
    algorithm: "wave",
    benchmark: $("benchmark").value,
    objective: $("objective").value,
    dimensions: number("dimensions"),
    walkers: number("walkers"),
    elites: number("elites"),
    perturbation_std: number("scale"),
    boundary: "periodic",
    controller_enabled: $("restarts").checked,
    population_auto: false,
  };
  const members = Array.from($("members").children).map((el, i) => {
    const settings = {};
    for (const input of el.querySelectorAll("input,select"))
      if (input.value !== "")
        settings[input.name] =
          input.tagName === "SELECT" ? input.value : Number(input.value);
    return { id: `swarm-${i + 1}`, settings };
  });
  return {
    seed: number("seed"),
    exchange_every: number("exchange-every"),
    global_elites: number("global-elites"),
    max_evaluations: number("budget"),
    concurrency: number("concurrency"),
    defaults,
    members,
  };
}
function controls() {
  $("run").disabled = !current || (busy && !running);
  $("run").textContent = running ? "Pause" : "Run";
  $("step").disabled = !current || busy || running || playback;
  $("save").disabled = !history.length;
  $("reset").disabled = busy;
  $("load").disabled = busy;
  $("live-settings").disabled = !current || busy || running || playback;
  $("timeline").disabled = !history.length || running || busy;
}
function remember(result) {
  current = result;
  config = result.config;
  const entry = {
    status: structuredClone(result.status),
    frames: result.frames.map((f) => f.slice()),
  };
  bytes +=
    entry.frames.reduce((n, f) => n + f.byteLength, 0) +
    JSON.stringify(entry.status).length * 2;
  history.push(entry);
  $("timeline").max = history.length - 1;
  $("timeline").value = history.length - 1;
  render(entry);
  controls();
}
async function surface() {
  const first = config.members[0].settings,
    d = first.dimensions,
    resolution = 96;
  const positions = new Float32Array(resolution * resolution * d);
  for (let y = 0; y < resolution; y++)
    for (let x = 0; x < resolution; x++) {
      const i = (y * resolution + x) * d;
      positions[i] =
        first.low + (x / (resolution - 1)) * (first.high - first.low);
      positions[i + 1] =
        first.low + (y / (resolution - 1)) * (first.high - first.low);
    }
  const { values } = await population.sample(positions, d);
  const finite = Array.from(values)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!finite.length) return;
  const low = finite[0],
    high = finite[Math.floor((finite.length - 1) * 0.95)];
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = resolution;
  const ctx = canvas.getContext("2d"),
    pixels = ctx.createImageData(resolution, resolution);
  for (let y = 0; y < resolution; y++)
    for (let x = 0; x < resolution; x++) {
      const t = Math.max(
          0,
          Math.min(1, (values[y * resolution + x] - low) / (high - low || 1)),
        ),
        i = ((resolution - 1 - y) * resolution + x) * 4;
      pixels.data.set([15 + 35 * t, 20 + 15 * t, 32 + 40 * t, 255], i);
    }
  ctx.putImageData(pixels, 0, 0);
  heatmap = canvas;
  draw();
}
function draw() {
  const canvas = $("population-view"),
    rect = canvas.getBoundingClientRect(),
    ratio = devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const w = rect.width,
    h = rect.height,
    pad = 32;
  if (heatmap) ctx.drawImage(heatmap, pad, pad, w - 2 * pad, h - 2 * pad);
  if (!current) return;
  const entry = history[Number($("timeline").value)] ?? current,
    settings = config.members[0].settings,
    low = settings.low,
    high = settings.high;
  const x = (v) => pad + ((v - low) / (high - low)) * (w - 2 * pad),
    y = (v) => h - pad - ((v - low) / (high - low)) * (h - 2 * pad);
  ctx.strokeStyle = "#3b3048";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  for (let k = 0; k < entry.frames.length; k++) {
    const frame = entry.frames[k],
      meta = entry.status.members?.[k],
      exports = new Set(meta?.exports ?? []),
      imports = new Set((meta?.imports ?? []).map((v) => v.destination));
    ctx.fillStyle = ctx.strokeStyle = colors[k % colors.length];
    for (let i = 0; i < frameInfo(frame).n; i++) {
      const r = row(frame, i);
      if (!r.alive || !Number.isFinite(r.x[0]) || !Number.isFinite(r.x[1]))
        continue;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(x(r.x[0]), y(r.x[1]), 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (exports.has(i)) {
        ctx.beginPath();
        ctx.arc(x(r.x[0]), y(r.x[1]), 5, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (imports.has(i)) ctx.strokeRect(x(r.x[0]) - 6, y(r.x[1]) - 6, 12, 12);
    }
  }
  ctx.fillStyle = "#b5a6c0";
  ctx.font = "11px system-ui";
  ctx.fillText(low.toFixed(2), pad, h - 14);
  ctx.fillText(high.toFixed(2), w - pad - 36, h - 14);
}
function render(entry) {
  $("round-label").textContent = entry.status.round;
  $("totals").textContent =
    `Round ${entry.status.round} · ${entry.status.evaluations.toLocaleString()} evaluations · ${entry.status.exchanges} exchanges · ${entry.status.basins.length} basins`;
  $("statistics").replaceChildren();
  $("provenance").replaceChildren();
  entry.frames.forEach((frame, i) => {
    const info = frameInfo(frame),
      m = entry.status.members[i],
      tr = document.createElement("tr");
    const values = [
      m.id,
      info.n,
      Number.isFinite(info.best) ? info.best.toPrecision(6) : "—",
      m.evaluations,
      m.exports.length,
      m.shortfall ? `Skipped (${m.shortfall} needed)` : m.imports.length,
    ];
    values.forEach((value, k) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (k === 0) {
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.setProperty("--swarm-color", colors[i % colors.length]);
        td.prepend(swatch);
      }
      tr.append(td);
    });
    $("statistics").append(tr);
    for (const imp of m.imports) {
      const li = document.createElement("li");
      li.textContent = `${m.id} walker ${imp.destination} ← ${imp.source} walker ${imp.row}`;
      $("provenance").append(li);
    }
  });
  $("elite-list").replaceChildren();
  for (const e of entry.status.global_elites) {
    const li = document.createElement("li");
    li.textContent = `${e.source}: score ${e.score.toPrecision(7)}`;
    $("elite-list").append(li);
  }
  $("basin-list").replaceChildren();
  for (const b of entry.status.basins) {
    const li = document.createElement("li");
    li.textContent = `Basin ${b.id}: objective ${b.objective.toPrecision(7)}, ${b.visits} visits`;
    $("basin-list").append(li);
  }
  draw();
}
async function tick() {
  if (playback) {
    let index = Number($("timeline").value) + 1;
    if (index >= history.length) {
      running = false;
      controls();
      return;
    }
    $("timeline").value = index;
    render(history[index]);
    if (running)
      setTimeout(() => {
        if (running) tick();
      }, 150);
    return;
  }
  busy = true;
  controls();
  try {
    const reserve =
      current.frames.reduce((n, f) => n + f.byteLength * 8, 0) + 1024 * 1024;
    if (bytes + reserve > 64 * 1024 * 1024)
      throw new Error(
        "Recording reached its 64 MiB limit. Save and reset to continue.",
      );
    remember(await population.step());
    message(`Completed round ${current.status.round}.`);
    if (current.status.budget_exhausted) {
      running = false;
      message(
        "Shared budget reached. Save this run or reset with a larger budget.",
      );
    }
  } catch (e) {
    running = false;
    error(e);
  } finally {
    busy = false;
    controls();
  }
  if (running)
    setTimeout(() => {
      if (running) tick();
    }, 0);
}
$("swarm-count").addEventListener("change", members);
for (const [shared, key] of [
  ["walkers", "walkers"],
  ["elites", "elites"],
  ["scale", "perturbation_std"],
])
  $(shared).addEventListener("input", () => {
    for (const input of $("members").querySelectorAll(`[name="${key}"]`))
      input.placeholder = $(shared).value;
  });
members();
$("population-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  running = false;
  busy = true;
  controls();
  message("Creating independent swarm workers…");
  population?.dispose();
  population = new FractalPopulation();
  history = [];
  bytes = 0;
  heatmap = null;
  playback = false;
  current = null;
  try {
    remember(await population.create(readConfig()));
    $("live-member").replaceChildren();
    for (const m of config.members) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.id;
      $("live-member").append(o);
    }
    await surface();
    message(
      `${config.members.length} Wave swarms ready. Each has independent parameters and random draws.`,
    );
  } catch (e) {
    error(e);
  } finally {
    busy = false;
    controls();
  }
});
$("step").onclick = () => tick();
$("run").onclick = () => {
  running = !running;
  controls();
  if (running) tick();
};
$("timeline").oninput = () => render(history[Number($("timeline").value)]);
$("save").onclick = () => {
  const blob = new Blob([encodePopulationRecording(config, history)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "fractal-populations.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("load").onchange = async () => {
  try {
    const file = $("load").files[0];
    if (!file) return;
    if (file.size > 64 * 1024 * 1024)
      throw new Error("Recording exceeds 64 MiB");
    const recording = decodePopulationRecording(await file.text());
    if (!recording.frames.length) throw new Error("Recording has no frames");
    population?.dispose();
    population = null;
    running = false;
    playback = true;
    config = recording.config;
    history = recording.frames;
    current = { config, ...history[0] };
    heatmap = null;
    $("timeline").max = history.length - 1;
    $("timeline").value = 0;
    render(history[0]);
    message("Recording loaded. Use the timeline or Run to play it.");
    controls();
  } catch (e) {
    error(e);
  }
};
async function live(restart = false) {
  busy = true;
  controls();
  try {
    const i = config.members.findIndex((m) => m.id === $("live-member").value),
      patch = restart
        ? {
            restart_token:
              (population.reports[i].settings.restart_token ?? 0) + 1,
          }
        : { perturbation_std: number("live-scale") };
    remember(await population.updateMember(i, patch));
    message(
      restart
        ? "Restart requested for the next round."
        : "Swarm scale updated.",
    );
  } catch (e) {
    error(e);
  } finally {
    busy = false;
    controls();
  }
}
$("apply").onclick = () => live();
$("restart").onclick = () => live(true);
new ResizeObserver(draw).observe($("population-view"));
