// Host of the QFT Simulator workbench: wiring between the controls, the worker
// client and the view builders. It computes no scientific quantity.
import { chartSVG, escapeXML as esc, format } from "../lecture/plots.js";
import { createClient } from "./worker.js";
import {
  VOCABULARY,
  availabilityIndex,
  buildAnalysis,
  buildRequest,
  calibrationTable,
  channelCharts,
  cloudChart,
  collectNotes,
  colorSourceOfKind,
  comparisonTables,
  controlsFromAnalysis,
  correlatorTable,
  couplingTables,
  coverageTable,
  createSession,
  fitTable,
  flowTable,
  formFromDefaults,
  formFromRequest,
  gevpCharts,
  gevpTable,
  groupTable,
  liveCharts,
  liveNotes,
  measuredLive,
  missingRecords,
  presentationCharts,
  presentationMetrics,
  presentationNotes,
  progressOf,
  provenanceTable,
  reportAvailabilityTable,
  samplesTable,
  variantRequest,
  windowCharts,
  windowTable,
} from "./model.js";
import {
  analysisControlsHTML,
  assignmentsHTML,
  capabilitySummaryHTML,
  channelListHTML,
  chartCardsHTML,
  colorFieldsHTML,
  comparisonHTML,
  couplingsHTML,
  fitControlsHTML,
  initTabs,
  notesHTML,
  optionsHTML,
  tableHTML,
  variantListHTML,
  variantOptionsHTML,
} from "./views.js";

const $ = (selector) => document.querySelector(selector);
const MAX_IMPORT = 256 * 1024 * 1024;
const ALL = "__all__";

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});
const client = createClient(worker, fail);
const session = createSession(client, { budget: 16 });
const tabs = initTabs($("#tabs"), (name) => {
  activeTab = name;
  renderActive();
});
const charts = {};
let activeTab = "setup",
  defaults,
  form,
  capabilities = null,
  availability = new Map(),
  request = null,
  requestChannels = [],
  baseAnalysis = null,
  controls = null,
  report = null,
  presentation = null,
  liveSelected = [],
  // The Rust presentation holds a plot for every result of the analysis; the
  // gallery of all of them is built only when it is asked for.
  showPresentation = false,
  busy = false,
  capabilityToken = 0,
  capabilityTimer = 0,
  analysisToken = 0;

function fail(error) {
  session.pause();
  $("#error").hidden = false;
  $("#error").textContent = error?.message || String(error);
  syncButtons();
}
function clearError() {
  $("#error").hidden = true;
}
function plotWidth(container) {
  const columns =
    getComputedStyle(container).gridTemplateColumns.split(" ").length;
  return Math.max(
    300,
    Math.min(640, Math.round(container.clientWidth / columns - 22)),
  );
}
function showCharts(selector, group, list) {
  const host = $(selector);
  charts[group] = list;
  host.innerHTML = chartCardsHTML(list, { width: plotWidth(host), group });
}
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
// Seed of the first replica as the Rust snapshot reports it.
const sessionSeed = () => session.snapshot?.replicas?.[0]?.seed ?? "unknown";
async function fileBytes(input, what) {
  const file = input.files[0];
  if (!file) return null;
  if (file.size > MAX_IMPORT)
    throw new Error(what + " must be smaller than 256 MiB.");
  return new Uint8Array(await file.arrayBuffer());
}

// ---------------------------------------------------------------- setup

function renderSetup() {
  const fields = $("#setup").elements;
  $("#variant").innerHTML = variantOptionsHTML(defaults.variants, form.variant);
  $("#variant-list").innerHTML = variantListHTML(defaults.variants);
  for (const key of ["walkers", "dimensions", "steps", "replicas", "seed"])
    fields[key].value = form[key];
  fields.colorSource.innerHTML = optionsHTML(
    VOCABULARY.colorSource,
    form.color.kind,
  );
  fields.timeAxis.innerHTML = optionsHTML(VOCABULARY.timeAxis, form.time.kind);
  $("#color-fields").innerHTML = colorFieldsHTML(form.color);
  renderChannels();
}
function selectedAvailable() {
  return form.channels.filter(
    (id) => availability.get(id)?.available !== false,
  );
}
function renderChannels() {
  $("#channels").innerHTML = channelListHTML(
    defaults.catalog || [],
    form.channels,
    availability,
  );
  $("#capabilities").innerHTML = capabilitySummaryHTML(
    capabilities,
    missingRecords(capabilities),
  );
  $("#setup-notes").innerHTML = notesHTML(collectNotes(report, "report"));
  syncButtons();
}
function readSetup(target) {
  const fields = $("#setup").elements;
  if (target.name === "channel") {
    form.channels = [...$("#setup").querySelectorAll('[name="channel"]')]
      .filter((box) => box.checked)
      .map((box) => box.value);
    syncButtons();
    return false;
  }
  if (target.name === "variant") {
    // Rust resolves every variant into a complete request; the form restarts
    // from it, keeping nothing of the previous variant's run configuration.
    form = formFromRequest(defaults, {
      ...variantRequest(defaults, target.value),
      variant: target.value,
    });
    renderSetup();
    return true;
  }
  for (const key of ["walkers", "dimensions", "steps", "replicas", "seed"])
    form[key] = Number(fields[key].value);
  if (target.name === "colorSource")
    form.color = colorSourceOfKind(target.value, form.color);
  else if (form.color.kind === "recorded_field")
    for (const key of ["stage", "amplitude", "phase"])
      form.color[key] = fields["color-" + key].value;
  else {
    const kind = fields.alignment.value;
    form.color.alignment =
      kind === "matched_kick"
        ? { kind, stage: fields.stage?.value ?? "b1" }
        : { kind };
  }
  form.time =
    fields.timeAxis.value === form.time.kind
      ? form.time
      : { kind: fields.timeAxis.value };
  if (["colorSource", "alignment"].includes(target.name))
    $("#color-fields").innerHTML = colorFieldsHTML(form.color);
  return true;
}
function scheduleCapabilities() {
  clearTimeout(capabilityTimer);
  capabilityTimer = setTimeout(refreshCapabilities, 200);
}
async function refreshCapabilities() {
  const token = ++capabilityToken;
  $("#setup-status").textContent = "Asking Rust which channels are available…";
  try {
    const response = await client.capabilities(
      buildRequest(defaults, form, new Map(), { probe: true }),
    );
    if (token !== capabilityToken) return;
    capabilities = response;
    availability = availabilityIndex(response);
    $("#capability-error").hidden = true;
    $("#setup-status").textContent = "Availability computed by Rust";
  } catch (error) {
    if (token !== capabilityToken) return;
    capabilities = null;
    availability = new Map();
    $("#capability-error").hidden = false;
    $("#capability-error").textContent = error.message || String(error);
    $("#setup-status").textContent = "Rust rejected this configuration";
  }
  renderChannels();
}

// ------------------------------------------------------------------ run

function syncButtons() {
  const snapshot = session.snapshot;
  $("#create").disabled =
    busy ||
    !defaults ||
    !selectedAvailable().length ||
    !$("#capability-error").hidden;
  $("#run").disabled = busy || !snapshot || snapshot.done;
  $("#run").textContent = session.running ? "Pause" : "Run";
  $("#step").disabled = busy || session.running || !snapshot || snapshot.done;
  $("#restart").disabled = busy || !request;
  $("#export-evidence").disabled = !session.live;
  $("#export-checkpoint").disabled = !session.live;
  // Every analysis brings its plots back with the report, so the button only
  // opens and closes the gallery of all of them.
  $("#presentation").disabled = !presentation;
  $("#presentation").textContent = showPresentation
    ? "Hide the Rust plot gallery"
    : "Every Rust presentation plot";
}
function renderRun() {
  const snapshot = session.snapshot;
  const progress = progressOf(snapshot);
  $("#progress").max = Math.max(1, progress.required);
  $("#progress").value = Math.min(progress.step, $("#progress").max);
  $("#progress-label").textContent = snapshot
    ? "Completed " + progress.step + " of " + progress.required + " steps"
    : session.imported
      ? "Imported " + session.imported + " · no live session"
      : "No session yet";
  $("#run-metrics").innerHTML = snapshot
    ? [
        ["Completed steps", progress.step],
        ["Required steps", progress.required],
        ["Updates per chunk", progress.chunk ?? "—"],
        ["Replicas", progress.table.rows.length || "—"],
      ]
        .map(
          ([label, value]) =>
            "<div><span>" +
            esc(label) +
            "</span><strong>" +
            esc(typeof value === "number" ? format(value) : value) +
            "</strong></div>",
        )
        .join("")
    : "";
  const measured = measuredLive(snapshot);
  $("#live-select").innerHTML = measured
    .map(
      (c) =>
        '<label class="check"><input type="checkbox" name="live" value="' +
        esc(c.id) +
        '"' +
        (liveSelected.includes(c.id) ? " checked" : "") +
        "> <code>" +
        esc(c.id) +
        "</code></label>",
    )
    .join("");
  const cloud = cloudChart(snapshot);
  showCharts("#run-charts", "run", [
    ...(cloud ? [cloud] : []),
    ...(liveSelected.length ? liveCharts(snapshot, liveSelected) : []),
  ]);
  $("#coverage").innerHTML = tableHTML(coverageTable(snapshot?.channels), {
    caption: "Element counts per channel, accumulated over frames",
  });
  $("#replicas").innerHTML = tableHTML(progress.table, {
    caption: "Per-replica progress",
  });
  $("#run-notes").innerHTML =
    notesHTML(liveNotes(snapshot), "Notes from the Rust session") +
    notesHTML(collectNotes(report, "report"));
  syncButtons();
}
function onSnapshot(snapshot) {
  if (!liveSelected.length)
    liveSelected = measuredLive(snapshot)
      .slice(0, 3)
      .map((c) => c.id);
  $("#status").textContent = snapshot.done
    ? "Complete at step " + snapshot.step
    : "Step " + snapshot.step;
  if (activeTab === "run") renderActive();
}
async function runLoop() {
  clearError();
  syncButtons();
  try {
    const last = await session.run(onSnapshot);
    syncButtons();
    if (last?.done) await analyze();
  } catch (error) {
    fail(error);
  }
}
// `channels` are the ticked catalog rows the request was built from.
async function create(nextRequest, channels) {
  busy = true;
  clearError();
  syncButtons();
  $("#status").textContent = "Creating the Rust session…";
  try {
    const snapshot = await session.create(nextRequest);
    request = nextRequest;
    requestChannels = channels;
    baseAnalysis = request.spectroscopy.analysis;
    controls = controlsFromAnalysis(baseAnalysis, request.seed);
    // Rust measures whole specifications; the analysis is restricted to the
    // catalog rows that were actually ticked.
    controls.channels = [...channels];
    report = null;
    presentation = null;
    liveSelected = [];
    tabs.select("run");
    onSnapshot(snapshot);
  } catch (error) {
    fail(error);
    return;
  } finally {
    busy = false;
    syncButtons();
  }
  runLoop();
}

// ------------------------------------------------------------- analysis

function analysedChannels() {
  return (report?.channels || []).filter((c) => c.correlator);
}
function ensureControls() {
  if (!baseAnalysis) {
    baseAnalysis = defaults.request.spectroscopy.analysis;
    controls = controlsFromAnalysis(baseAnalysis, defaults.request.seed);
  }
}
function currentAnalysis() {
  ensureControls();
  return buildAnalysis(
    baseAnalysis,
    controls,
    analysedChannels().map((c) => c.id),
  );
}
// The tables read the report and the charts read the presentation, so both
// are requested for the same `AnalysisConfig`: every plotted point, the lag
// axis and both edges of every error band are produced by Rust.
async function analyze() {
  if (!session.live && !session.imported) {
    $("#analysis-status").textContent = "Nothing to analyse yet";
    return;
  }
  const token = ++analysisToken;
  const analysis = currentAnalysis();
  $("#analysis-status").textContent = "Analysing in Rust…";
  try {
    const next = await session.analyze(analysis);
    if (token !== analysisToken) return;
    const plots = await session.presentation(analysis);
    if (token !== analysisToken) return;
    adoptReport(next, plots);
    $("#analysis-status").textContent =
      "Analysed " + format(next.frames) + " measured frames";
  } catch (error) {
    if (token !== analysisToken) return;
    $("#analysis-status").textContent = "Rust rejected the analysis";
    fail(error);
  }
}
function adoptReport(next, plots = null) {
  report = next;
  presentation = plots;
  clearError();
  renderActive();
}
function channelSelectHTML(current) {
  return optionsHTML(
    [
      [ALL, "All analysed channels"],
      ...analysedChannels().map((c) => [c.id, c.id]),
    ],
    current,
  );
}
function chosenChannels(select) {
  const all = analysedChannels();
  if (!all.length) return [];
  if (select.value === ALL) return all.slice(0, 12);
  return [all.find((c) => c.id === select.value) ?? all[0]];
}
function renderCorrelators() {
  ensureControls();
  $("#analysis").innerHTML = analysisControlsHTML(controls);
  const select = $("#correlator-channel");
  const previous = select.value || analysedChannels()[0]?.id;
  select.innerHTML = channelSelectHTML(previous);
  const shown = chosenChannels(select);
  showCharts(
    "#correlator-charts",
    "correlators",
    shown.flatMap((c) =>
      channelCharts(presentation, c.id, { logY: $("#log-y").checked }),
    ),
  );
  $("#correlator-table").innerHTML =
    (shown.length === 1
      ? tableHTML(correlatorTable(shown[0]), {
          caption: "C(τ) and effective decay rate · " + shown[0].id,
        })
      : "") +
    (report
      ? tableHTML(provenanceTable(report), {
          caption: "Provenance stamped on the report by Rust",
        }) +
        tableHTML(reportAvailabilityTable(report), {
          caption: "Channel availability reported by Rust",
        })
      : '<p class="observation">Run or import a measurement to see its correlators.</p>');
  $("#samples-table").innerHTML = tableHTML(samplesTable(report?.channels), {
    caption: "Resampling and autocorrelation per channel",
  });
  const flow = flowTable(report);
  $("#flow-table").innerHTML = flow
    ? tableHTML(flow, {
        caption:
          "Graph smoothing diagnostic over " +
          format(flow.frames) +
          " frames · defines no length scale",
      })
    : "";
  showCharts(
    "#presentation-charts",
    "presentation",
    showPresentation ? presentationCharts(presentation) : [],
  );
  if (showPresentation)
    $("#presentation-charts").insertAdjacentHTML(
      "beforeend",
      tableHTML(presentationMetrics(presentation), {
        caption: "Every quantity the Rust presentation reports",
      }),
    );
  $("#correlator-notes").innerHTML =
    notesHTML(
      presentationNotes(presentation),
      "Notes of the Rust presentation",
    ) + notesHTML(collectNotes(report, "correlators"));
  syncButtons();
}
function renderFits() {
  ensureControls();
  $("#fit-controls").innerHTML = fitControlsHTML(
    controls,
    analysedChannels().length,
  );
  $("#fit-table").innerHTML = tableHTML(fitTable(report), {
    caption: "Decay rates of the algorithm-time autocorrelation",
  });
  const select = $("#fit-channel");
  const previous = select.value || analysedChannels()[0]?.id;
  select.innerHTML = optionsHTML(
    analysedChannels().map((c) => [c.id, c.id]),
    previous,
  );
  const channel = analysedChannels().find((c) => c.id === select.value);
  showCharts(
    "#window-charts",
    "windows",
    channel ? windowCharts(presentation, channel.id) : [],
  );
  $("#window-table").innerHTML = channel
    ? tableHTML(windowTable(channel), {
        caption: "Fit windows · " + channel.id,
      })
    : "";
  $("#group-table").innerHTML = tableHTML(groupTable(report), {
    caption: "Joint fits with shared gaps",
  });
  showCharts("#gevp-charts", "gevp", gevpCharts(presentation));
  $("#gevp-table").innerHTML = tableHTML(gevpTable(report), {
    caption: "Generalized eigenvalue levels",
  });
  $("#fit-notes").innerHTML = notesHTML(collectNotes(report, "fits"));
}
function renderPhysics() {
  ensureControls();
  const references = (baseAnalysis.reference?.entries || []).map((e) => e.name);
  // Rust accepts a specification id or a channel id as an assignment key.
  const keys = [
    ...new Set([
      ...Object.keys(controls.assignments),
      ...(report ? analysedChannels().map((c) => c.id) : selectedAvailable()),
    ]),
  ];
  $("#mapping").innerHTML = assignmentsHTML(keys, references, controls);
  $("#comparison").innerHTML = comparisonHTML(comparisonTables(report));
  $("#calibration").innerHTML = tableHTML(calibrationTable(report), {
    caption: "Scales fixed during the warm-up frames",
  });
  $("#couplings").innerHTML = couplingsHTML(couplingTables(report));
  $("#physics-notes").innerHTML = notesHTML(collectNotes(report, "physics"));
}
// Panels are rebuilt from state; the focused control is found again by form,
// name and value so keyboard users keep their place.
function keepFocus(render) {
  const active = document.activeElement;
  const key = active?.form
    ? [active.form.id, active.name, active.dataset.channel]
    : null;
  const value = active?.type === "checkbox" ? active.value : null;
  render();
  if (!active || active.isConnected) return;
  const next = active.id
    ? document.getElementById(active.id)
    : key &&
      [...(document.getElementById(key[0])?.elements || [])].find(
        (e) =>
          e.name === key[1] &&
          e.dataset.channel === key[2] &&
          (value === null || e.value === value),
      );
  next?.focus();
}
function renderActive() {
  if (!defaults) return;
  keepFocus(
    {
      setup: renderChannels,
      run: renderRun,
      correlators: renderCorrelators,
      fits: renderFits,
      physics: renderPhysics,
    }[activeTab],
  );
}
function readControls(formElement) {
  for (const element of formElement.elements) {
    if (!element.name || ["assign", "anchor"].includes(element.name)) continue;
    controls[element.name] =
      element.type === "checkbox"
        ? element.checked
        : element.type === "number"
          ? Number(element.value)
          : element.value;
  }
}

// --------------------------------------------------------------- events

$("#setup").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!$("#setup").reportValidity()) return;
  create(buildRequest(defaults, form, availability), selectedAvailable());
});
$("#setup").addEventListener("change", (event) => {
  if (readSetup(event.target)) scheduleCapabilities();
});
$("#run").onclick = () => {
  if (session.running) {
    session.pause();
    $("#status").textContent = "Paused at step " + session.snapshot.step;
    syncButtons();
  } else runLoop();
};
$("#step").onclick = async () => {
  try {
    onSnapshot(await session.step());
    if (session.snapshot.done) await analyze();
  } catch (error) {
    fail(error);
  }
};
$("#restart").onclick = () => create(request, requestChannels);
$("#live-select").addEventListener("change", () => {
  liveSelected = [...$("#live-select").querySelectorAll("input:checked")].map(
    (box) => box.value,
  );
  renderActive();
});
$("#export-evidence").onclick = async () => {
  try {
    download(
      "qft-evidence-seed-" + sessionSeed() + ".cbor",
      await session.evidence(),
      "application/cbor",
    );
  } catch (error) {
    fail(error);
  }
};
$("#export-checkpoint").onclick = async () => {
  try {
    download(
      "qft-session-seed-" + sessionSeed() + ".agc",
      await session.checkpoint(),
      "application/octet-stream",
    );
  } catch (error) {
    fail(error);
  }
};
$("#restore-checkpoint").onchange = async (event) => {
  try {
    const bytes = await fileBytes(event.target, "A checkpoint");
    if (!bytes) return;
    const snapshot = await session.restore(bytes);
    // The checkpoint carries its own request: the page adopts the one Rust
    // resolved, so a restored run shows and restarts its own configuration.
    // It is analysed over every channel it measured.
    request = await session.request();
    requestChannels = [];
    form = formFromRequest(defaults, request);
    baseAnalysis = request.spectroscopy.analysis;
    controls = controlsFromAnalysis(baseAnalysis, request.seed);
    controls.channels = [];
    liveSelected = [];
    report = null;
    presentation = null;
    clearError();
    renderSetup();
    onSnapshot(snapshot);
    $("#status").textContent = "Checkpoint restored at step " + snapshot.step;
    renderRun();
    // The restored session already carries a measurement: analyse it so the
    // other panels show its numbers without touching a control first.
    await analyze();
  } catch (error) {
    fail(error);
  } finally {
    event.target.value = "";
  }
};
$("#import-evidence").onchange = async (event) => {
  try {
    const bytes = await fileBytes(event.target, "Spectroscopy evidence");
    if (!bytes) return;
    ensureControls();
    controls.channels = [];
    const analysis = currentAnalysis();
    const next = await session.importEvidence(bytes, analysis);
    // Imported evidence has its own presentation binding: the charts read the
    // same Rust plots as a live session, so none of them is left empty.
    adoptReport(next, await session.presentation(analysis));
    $("#status").textContent = "Evidence re-analysed in Rust";
    tabs.select("correlators");
  } catch (error) {
    fail(error);
  } finally {
    event.target.value = "";
  }
};
$("#import-archive").onchange = async (event) => {
  try {
    const bytes = await fileBytes(event.target, "A run archive");
    if (!bytes) return;
    const measurement = buildRequest(defaults, form, availability).spectroscopy
      .measurement;
    ensureControls();
    controls.channels = selectedAvailable();
    const analysis = currentAnalysis();
    const next = await session.importArchive(bytes, { measurement, analysis });
    adoptReport(next, await session.presentation(analysis));
    $("#status").textContent = "Archive measured and analysed in Rust";
    tabs.select("correlators");
  } catch (error) {
    fail(error);
  } finally {
    event.target.value = "";
  }
};
for (const id of ["#analysis", "#fit-controls"]) {
  $(id).addEventListener("submit", (event) => event.preventDefault());
  $(id).addEventListener("change", (event) => {
    readControls(event.currentTarget);
    renderActive();
    analyze();
  });
}
$("#mapping").addEventListener("submit", (event) => event.preventDefault());
$("#mapping").addEventListener("change", () => {
  controls.assignments = Object.fromEntries(
    [...$("#mapping").querySelectorAll('[name="assign"]')].map((select) => [
      select.dataset.channel,
      select.value,
    ]),
  );
  controls.anchors = [
    ...$("#mapping").querySelectorAll('[name="anchor"]:checked'),
  ].map((box) => box.value);
  analyze();
});
$("#correlator-channel").onchange = renderActive;
$("#log-y").onchange = renderActive;
$("#fit-channel").onchange = renderActive;
$("#presentation").onclick = () => {
  showPresentation = !showPresentation;
  renderCorrelators();
};
$("#workbench").addEventListener("click", (event) => {
  const button = event.target.closest("[data-export]");
  if (!button) return;
  const [group, index] = button.dataset.export.split(":");
  const chart = charts[group]?.[Number(index)];
  if (chart)
    download(
      "qft-" + group + "-" + index + ".svg",
      chartSVG(chart),
      "image/svg+xml",
    );
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && session.running) {
    session.pause();
    syncButtons();
  }
});
window.addEventListener("pagehide", () => worker.terminate());
window.addEventListener("resize", () => {
  // A soft keyboard resizes the window while a value is still being typed.
  if (
    !document.activeElement?.matches?.(
      'input[type="number"], input[type="text"]',
    )
  )
    renderActive();
});

try {
  $("#engine").textContent = "Loading Rust engine…";
  defaults = await client.defaults();
  form = formFromDefaults(defaults);
  // The catalog rows carry the availability of the default request.
  availability = availabilityIndex(defaults.catalog);
  $("#engine").textContent = "Rust engine ready";
  renderSetup();
  await refreshCapabilities();
} catch (error) {
  $("#engine").textContent = "Engine unavailable";
  fail(error);
}
