import { chartSVG, legendHTML, escapeXML as esc, format } from "./plots.js";
import { LectureScene } from "./scene.js";

const $ = (selector) => document.querySelector(selector);
const sceneHost = document.createElement("section");
sceneHost.className = "chart scene";
sceneHost.hidden = true;
document.querySelector("#charts").before(sceneHost);
const sceneView = new LectureScene(sceneHost);
const archiveButton = document.createElement("button");
archiveButton.id = "archive";
archiveButton.textContent = "Save trajectory archive";
archiveButton.hidden = true;
document.querySelector(".exports").append(archiveButton);
const archiveImport = document.createElement("label");
archiveImport.className = "file-button";
archiveImport.innerHTML =
  'Open trajectory archive<input id="archive-import" type="file" accept="application/json,.json">';
document.querySelector(".exports").append(archiveImport);
const nativeImport = document.createElement("label");
nativeImport.className = "file-button";
nativeImport.innerHTML =
  'Open native sweep<input id="native-results" type="file" accept="application/json,.json">';
document.querySelector(".exports").append(nativeImport);
const calculationDetails = document.createElement("details");
calculationDetails.className = "calculation-details";
calculationDetails.hidden = true;
calculationDetails.innerHTML =
  "<summary>Calculation inputs and results</summary><pre></pre>";
document.querySelector("#charts").after(calculationDetails);
const formulaIndex = document.createElement("details");
formulaIndex.className = "formula-index";
formulaIndex.hidden = true;
formulaIndex.innerHTML =
  '<summary>Find a Part VI formula</summary><label>Formula, theorem, or experiment<input type="search" placeholder="For example: reflection, Ricci, VI-39"></label><div class="formula-matches"></div>';
document.querySelector("#controls").after(formulaIndex);
let formulaEntries;
async function findFormulas() {
  try {
    if (!formulaEntries) {
      const response = await fetch(
        new URL("./partvi-formulas.json", import.meta.url),
      );
      if (!response.ok) throw new Error("Formula index is unavailable");
      formulaEntries = await response.json();
    }
    const q = formulaIndex.querySelector("input").value.trim().toLowerCase();
    const matches = formulaEntries.filter((e) =>
      q
        ? [e.label, e.title, e.expression, e.experiment]
            .join(" ")
            .toLowerCase()
            .includes(q)
        : e.experiment === demo?.id,
    );
    formulaIndex.querySelector(".formula-matches").innerHTML =
      "<p>" +
      matches.length +
      " linked formulas and statements</p>" +
      matches
        .slice(0, 60)
        .map(
          (e) =>
            '<button type="button" data-formula-demo="' +
            esc(e.experiment) +
            '"><strong>' +
            esc(e.experiment) +
            " · " +
            esc(e.title || e.label) +
            "</strong><span>" +
            esc(e.expression || e.section || e.kind) +
            "</span></button>",
        )
        .join("");
  } catch (error) {
    formulaIndex.querySelector(".formula-matches").textContent =
      error.message || String(error);
  }
}
formulaIndex.addEventListener("toggle", () => {
  if (formulaIndex.open) findFormulas();
});
formulaIndex.querySelector("input").addEventListener("input", findFormulas);
formulaIndex.addEventListener("click", (e) => {
  const b = e.target.closest("[data-formula-demo]");
  if (b) initialize(b.dataset.formulaDemo);
});
const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});
let serial = 0,
  pending = new Map(),
  catalog,
  demo,
  params,
  seed = 7,
  ticks = 0;
let busy = false,
  running = false,
  frames = [],
  selected = 0,
  generation = 0;
const query = new URLSearchParams(location.search);
if (query.get("embed") === "1") document.body.classList.add("embedded");
if (window.parent !== window) {
  let lastHeight = 0;
  new ResizeObserver(() => {
    const height = Math.ceil(
      $(".layout").getBoundingClientRect().bottom + window.scrollY + 8,
    );
    if (height !== lastHeight) {
      lastHeight = height;
      window.parent.postMessage(
        { type: "fragile-lecture-height", height },
        location.origin,
      );
    }
  }).observe($(".layout"));
}
function request(type, payload) {
  return new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload });
  });
}
worker.onmessage = ({ data }) => {
  const waiter = pending.get(data.id);
  if (!waiter) return;
  pending.delete(data.id);
  data.error
    ? waiter.reject(new Error(data.error))
    : waiter.resolve(data.result);
};
worker.onerror = (event) => {
  const error = new Error(
    event.message || "Could not load the experiment worker. Reload this page.",
  );
  pending.forEach((waiter) => waiter.reject(error));
  pending.clear();
  fail(error);
};
function fail(error) {
  pause();
  $("#error").hidden = false;
  $("#error").textContent = error.message || String(error);
  $("#status").textContent = "Experiment needs attention";
}
function pause() {
  running = false;
  $("#run").textContent = "Run";
}
function lock(value) {
  busy = value;
  for (const name of ["step", "reset"]) $("#" + name).disabled = value;
  $("#run").disabled = value && !running;
  $("#controls").inert = value;
  $("#seed").disabled = value;
}
function values() {
  return Object.fromEntries(
    [...$("#controls").elements]
      .filter((e) => e.name)
      .map((e) => [e.name, e.value]),
  );
}
function controlHTML(control) {
  const id = "control-" + control.key;
  let input;
  if (control.type === "select")
    input =
      '<select id="' +
      id +
      '" name="' +
      esc(control.key) +
      '">' +
      control.options
        .map(
          (option) =>
            '<option value="' +
            esc(option.value) +
            '">' +
            esc(option.label) +
            "</option>",
        )
        .join("") +
      "</select>";
  else
    input =
      '<input id="' +
      id +
      '" name="' +
      esc(control.key) +
      '" type="' +
      esc(control.type) +
      '" min="' +
      control.min +
      '" max="' +
      control.max +
      '" step="' +
      (control.step ?? "any") +
      '"><output></output>';
  return (
    '<label class="control" for="' +
    id +
    '"><span>' +
    esc(control.label) +
    "</span>" +
    input +
    "</label>"
  );
}
function controls() {
  $("#controls").innerHTML =
    demo.controls.slice(0, 3).map(controlHTML).join("") +
    (demo.controls.length > 3
      ? '<details class="advanced"><summary>More controls (' +
        (demo.controls.length - 3) +
        ')</summary><div class="control-grid">' +
        demo.controls.slice(3).map(controlHTML).join("") +
        "</div></details>"
      : "");
  for (const control of demo.controls) {
    const element = $("#controls").elements.namedItem(control.key);
    element.value = params[control.key] ?? control.value;
    if (element.nextElementSibling?.tagName === "OUTPUT")
      element.nextElementSibling.textContent = element.value;
  }
}
async function initialize(id, overrides, replaySeed) {
  const reuse = Boolean(demo && demo.id !== id && !overrides);
  pause();
  const token = ++generation;
  lock(true);
  $("#error").hidden = true;
  $("#status").textContent = "Preparing experiment…";
  demo = catalog.find((item) => item.id === id) || catalog[0];
  formulaIndex.hidden = demo.part !== "VI";
  if (formulaIndex.open && !formulaIndex.hidden) findFormulas();
  params =
    overrides ||
    Object.fromEntries(
      demo.controls.map((control) => [control.key, control.value]),
    );
  seed = replaySeed ?? Number($("#seed").value);
  $("#seed").value = seed;
  controls();
  $("#title").textContent = demo.title;
  $("#question").textContent = demo.question;
  $("#prediction").textContent = demo.prediction;
  $("#explanation").textContent = demo.explanation;
  $("#kind").textContent =
    "PART " + demo.part + " / " + demo.id + " · " + demo.kind;
  const url = new URL(location.href);
  url.searchParams.set("demo", demo.id);
  history.replaceState(null, "", url);
  url.searchParams.delete("embed");
  $("#standalone").href = url.href;
  document.title = demo.id + " · " + demo.title;
  document
    .querySelectorAll("[data-demo]")
    .forEach((link) =>
      link.setAttribute(
        "aria-current",
        link.dataset.demo === demo.id ? "page" : "false",
      ),
    );
  const selectedLink = document.querySelector('[data-demo="' + demo.id + '"]');
  if (selectedLink) {
    selectedLink.closest("details").open = true;
    selectedLink.scrollIntoView({ block: "nearest" });
  }
  try {
    const result = await request("initialize", {
      id: demo.id,
      params,
      seed,
      reuse,
    });
    if (token !== generation) return;
    params = result.params;
    $("#checkpoint").hidden = !result.checkpoint;
    archiveButton.hidden = !result.archive;
    ticks = result.initialTick || 0;
    $("#save").disabled = false;
    frames = [];
    selected = 0;
    record(result.snapshot);
    $("#status").textContent =
      "Ready · initialization " + format(result.initializationMs) + " ms";
  } catch (error) {
    if (token === generation) fail(error);
  } finally {
    if (token === generation) lock(false);
  }
}
function record(snapshot) {
  frames.push({ snapshot, tick: ticks });
  // Keep a bounded window; full scalar histories live in the model snapshot.
  const frameLimit = snapshot.scene?.faces?.length > 500 ? 8 : 80;
  while (frames.length > frameLimit) frames.shift();
  selected = frames.length - 1;
  render();
}
function render() {
  const record = frames[selected];
  if (!record) return;
  const snapshot = record.snapshot;
  if (snapshot.result?.details?.calculation_origin) {
    const sources = {
      executed_algorithm_archive: "Recorded algorithm measurements",
      independent_algorithm_continuations:
        "Independent complete-state continuations",
      constructed_finite_model: "Specified finite-model calculation",
    };
    $("#kind").textContent =
      "PART VI / VI-" +
      String(snapshot.result.experiment).padStart(2, "0") +
      " · Rust/WASM · " +
      sources[snapshot.result.details.calculation_origin];
  }
  if (snapshot.imported && snapshot.result)
    $("#title").textContent =
      "Imported native sweep · " + snapshot.result.title;
  calculationDetails.hidden = !snapshot.result;
  if (snapshot.result)
    calculationDetails.querySelector("pre").textContent = JSON.stringify(
      snapshot.result,
      null,
      2,
    );
  const columns = getComputedStyle($("#charts")).gridTemplateColumns.split(
    " ",
  ).length;
  const plotWidth = Math.max(
    320,
    Math.min(640, Math.round($("#charts").clientWidth / columns - 22)),
  );
  $("#metrics").innerHTML = (snapshot.metrics || [])
    .map(
      (metric) =>
        "<div><span>" +
        esc(metric.label) +
        "</span><strong>" +
        esc(format(metric.value)) +
        "<small>" +
        esc(metric.unit || "") +
        "</small></strong></div>",
    )
    .join("");
  $("#charts").innerHTML = (snapshot.charts || [])
    .map(
      (chart, index) =>
        '<section class="chart"><div class="chart-heading"><h2>' +
        esc(chart.title) +
        '</h2><button class="svg-export" data-chart="' +
        index +
        '" aria-label="Save ' +
        esc(chart.title) +
        ' as SVG">SVG ↓</button></div>' +
        chartSVG(chart, { width: plotWidth }) +
        '<div class="legend">' +
        legendHTML(chart) +
        "</div></section>",
    )
    .join("");
  $("#message").textContent = snapshot.message || "";
  sceneView.update(snapshot.scene);
  $("#scrub").max = Math.max(0, frames.length - 1);
  $("#scrub").value = selected;
  $("#frame").textContent =
    record.tick +
    (selected < frames.length - 1 ? " · replay view" : " · latest");
  const table = snapshot.table || {
    columns: ["Plot / series", "x", "y"],
    rows: (snapshot.charts || [])
      .flatMap((chart) =>
        chart.matrix
          ? chart.matrix.flatMap((row, i) =>
              row.map((value, j) => [
                chart.title + " / row " + (i + 1),
                j + 1,
                value,
              ]),
            )
          : (chart.series || []).flatMap((series) =>
              (series.points || [])
                .slice(-20)
                .map((point) => [
                  chart.title + " / " + series.name,
                  ...point.slice(0, 2),
                ]),
            ),
      )
      .slice(0, 200),
  };
  $("#table").innerHTML =
    '<p class="footnote">Numerical view (up to 200 rows; save JSON for full plotted data).</p><div class="table-scroll"><table><thead><tr>' +
    table.columns
      .map((column) => '<th scope="col">' + esc(column) + "</th>")
      .join("") +
    "</tr></thead><tbody>" +
    table.rows
      .slice(0, 200)
      .map(
        (row) =>
          "<tr>" +
          row.map((value) => "<td>" + esc(format(value)) + "</td>").join("") +
          "</tr>",
      )
      .join("") +
    "</tbody></table></div>";
}
async function advance() {
  if (busy || !frames.length || frames.at(-1).snapshot.done) {
    pause();
    return;
  }
  const token = generation;
  lock(true);
  const start = performance.now();
  try {
    const snapshot = await request("step");
    if (token !== generation) return;
    ticks++;
    record(snapshot);
    $("#status").textContent =
      (snapshot.done ? "Complete" : "Step " + ticks) +
      " · " +
      format(performance.now() - start) +
      " ms including transfer";
    if (snapshot.done) pause();
  } catch (error) {
    fail(error);
  } finally {
    if (token === generation) lock(false);
  }
}
async function loop() {
  if (!running) return;
  await advance();
  if (running) setTimeout(loop, 70);
}
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#controls").addEventListener("submit", (event) => event.preventDefault());
$("#controls").addEventListener("input", (event) => {
  if (event.target.nextElementSibling?.tagName === "OUTPUT")
    event.target.nextElementSibling.textContent = event.target.value;
});
$("#controls").addEventListener("change", () =>
  initialize(demo.id, values(), seed),
);
$("#seed").addEventListener("change", () => initialize(demo.id, params));
$("#reset").onclick = () => initialize(demo.id, params, seed);
$("#step").onclick = () => {
  pause();
  advance();
};
$("#run").onclick = () => {
  running = !running;
  $("#run").textContent = running ? "Pause" : "Run";
  if (running) loop();
};
$("#scrub").oninput = (event) => {
  pause();
  selected = Number(event.target.value);
  render();
};
$("#live").onclick = () => {
  selected = frames.length - 1;
  render();
};
$("#charts").onclick = (event) => {
  const button = event.target.closest("[data-chart]");
  if (!button) return;
  const chart = frames[selected].snapshot.charts[Number(button.dataset.chart)];
  download(
    demo.id + "-plot-" + button.dataset.chart + ".svg",
    chartSVG(chart),
    "image/svg+xml",
  );
};
$("#save").onclick = () => {
  if (!frames.length) return;
  const view = frames[selected];
  download(
    demo.id + "-seed-" + seed + ".json",
    JSON.stringify(
      {
        schema: view.snapshot.result
          ? "fragile-partvi-results-v1"
          : "fragile-lecture-experiment-v1",
        ...(view.snapshot.result
          ? {
              results: [view.snapshot.result],
              replayable: !view.snapshot.imported,
            }
          : {}),
        id: demo.id,
        seed,
        params,
        ticks: view.tick,
        kind: demo.kind,
        snapshot: view.snapshot,
        savedAt: new Date().toISOString(),
      },
      null,
      2,
    ),
    "application/json",
  );
};
archiveButton.onclick = async () => {
  try {
    download(
      demo.id + "-archive.json",
      JSON.stringify(await request("archive")),
      "application/json",
    );
  } catch (error) {
    fail(error);
  }
};
$("#archive-import").onchange = async (event) => {
  pause();
  lock(true);
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 128 * 1024 * 1024)
      throw new Error("Archive must be smaller than 128 MiB.");
    const archive = JSON.parse(await file.text());
    const qft = demo.part === "VI";
    const snapshot = await request(
      qft ? "qft_archive_import" : "archive_import",
      qft ? { archive, id: demo.id, params, seed } : archive,
    );
    frames = [];
    ticks = snapshot.step;
    selected = 0;
    record(snapshot);
    $("#title").textContent = qft
      ? demo.title + " · imported archive"
      : "Imported Fractal Set archive";
    $("#status").textContent = "Archive validated";
    $("#checkpoint").hidden = true;
    archiveButton.hidden = false;
    $("#save").disabled = true;
  } catch (error) {
    fail(error);
  } finally {
    event.target.value = "";
    lock(false);
  }
};
async function showNativeResults(data) {
  const snapshot = await request("qft_results_import", data);
  frames = [];
  ticks = 0;
  selected = 0;
  record(snapshot);
  $("#title").textContent = "Imported native sweep · " + snapshot.result.title;
  $("#status").textContent =
    data.results.length + " native calculations loaded. Step through results.";
  $("#checkpoint").hidden = true;
  archiveButton.hidden = true;
  $("#save").disabled = false;
}
$("#native-results").onchange = async (event) => {
  pause();
  lock(true);
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 32 * 1024 * 1024)
      throw new Error("Native results must be smaller than 32 MiB.");
    const data = JSON.parse(await file.text());
    await showNativeResults(data);
  } catch (error) {
    fail(error);
  } finally {
    event.target.value = "";
    lock(false);
  }
};
$("#load").onchange = async (event) => {
  pause();
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 20 * 1024 * 1024)
      throw new Error("Replay JSON must be smaller than 20 MB");
    const data = JSON.parse(await file.text());
    if (
      data.schema === "fragile-partvi-results-v1" &&
      data.replayable !== true
    ) {
      lock(true);
      try {
        await showNativeResults(data);
      } finally {
        lock(false);
      }
      event.target.value = "";
      return;
    }
    const expectedSchema = data.id?.startsWith("VI-")
      ? "fragile-partvi-results-v1"
      : "fragile-lecture-experiment-v1";
    if (
      data.schema !== expectedSchema ||
      !catalog.some((d) => d.id === data.id) ||
      !Number.isInteger(data.ticks) ||
      data.ticks < 0 ||
      data.ticks > 2000
    )
      throw new Error(
        "Choose an exported lecture experiment with at most 2000 replay steps.",
      );
    await initialize(data.id, data.params, data.seed);
    const token = generation;
    for (
      let i = 0;
      i < data.ticks && token === generation && $("#error").hidden;
      i++
    )
      await advance();
  } catch (error) {
    fail(error);
  }
  event.target.value = "";
};
$("#checkpoint").onclick = async () => {
  try {
    download(
      demo.id + ".agc",
      await request("checkpoint"),
      "application/octet-stream",
    );
  } catch (error) {
    $("#status").textContent = error.message || String(error);
  }
};
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pause();
});
window.addEventListener("message", (event) => {
  if (
    event.origin === location.origin &&
    event.data?.type === "fragile-lecture-pause"
  )
    pause();
});
window.addEventListener("pagehide", () => {
  pause();
  worker.terminate();
});
window.addEventListener("resize", render);
try {
  catalog = await request("catalog");
  const captionsResponse = await fetch(
    new URL("./captions.json", import.meta.url),
  );
  if (!captionsResponse.ok)
    throw new Error("Could not load the lecture captions. Reload this page.");
  const captions = await captionsResponse.json();
  catalog = catalog.map((entry) => ({ ...entry, ...captions[entry.id] }));
  const names = {
    I: "Foundations",
    II: "Convergence",
    III: "Mean-field limits",
    IV: "Entropy & regularity",
    V: "Fractal Set & continuum",
    VI: "Fields & algorithmic QFT",
  };
  $("#catalog").innerHTML = Object.entries(names)
    .map(
      ([part, title]) =>
        "<details open><summary>PART " +
        part +
        " · " +
        title +
        "</summary>" +
        catalog
          .filter((d) => d.part === part)
          .map(
            (d) =>
              '<a href="?demo=' +
              d.id +
              '" data-demo="' +
              d.id +
              '"><span>' +
              d.id +
              "</span>" +
              esc(d.title) +
              "</a>",
          )
          .join("") +
        "</details>",
    )
    .join("");
  $("#catalog").onclick = (event) => {
    const link = event.target.closest("[data-demo]");
    if (!link) return;
    event.preventDefault();
    initialize(link.dataset.demo);
  };
  await initialize(query.get("demo") || "I-01");
} catch (error) {
  fail(error);
}
