import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chartSVG, escapeXML } from "../../web/euclidean-gas/lecture/plots.js";
import {
  MAX_ADVANCE,
  MESSAGE_TYPES,
  createClient,
  createDispatcher,
  createQueue,
  createWasmEngine,
  serve,
  transferables,
} from "../../web/euclidean-gas/qft/worker.js";
import {
  availabilityIndex,
  buildAnalysis,
  buildRequest,
  chartTable,
  cloudChart,
  collectNotes,
  comparisonTables,
  controlsFromAnalysis,
  correlatorChart,
  correlatorTable,
  coverageTable,
  createSession,
  effectiveRateChart,
  fitTable,
  flowTable,
  formFromDefaults,
  formFromRequest,
  liveCharts,
  measuredLive,
  presentationCharts,
  presentationNotes,
  progressOf,
  variantRequest,
  windowScanCharts,
} from "../../web/euclidean-gas/qft/model.js";
import {
  channelListHTML,
  chartCardsHTML,
  comparisonHTML,
  notesHTML,
  tableHTML,
  variantOptionsHTML,
} from "../../web/euclidean-gas/qft/views.js";
import {
  COLOR_REASON,
  ODD_REASON,
  RATE_QUANTITY,
  REJECTED,
  REPORT_NOTE,
  capabilities,
  createFakeEngine,
  createFakeWasm,
  defaults,
  presentation,
  report,
  snapshot,
} from "./qft-simulator-support.mjs";

const page = (name) =>
  new URL("../../web/euclidean-gas/" + name, import.meta.url);
const scalar = () => report().channels[0];

// A Worker-shaped loopback: the client posts to a scope served in-process.
function loopback(engine) {
  const scope = {
      posted: [],
      postMessage(reply, transfer) {
        scope.posted.push({ reply, transfer });
        queueMicrotask(() => worker.onmessage({ data: reply }));
      },
    },
    worker = {
      postMessage(data) {
        scope.onmessage({ data });
      },
    };
  serve(scope, engine);
  return { scope, client: createClient(worker) };
}

test("The worker protocol routes every message type to the injected engine", async () => {
  const engine = createFakeEngine();
  const { client } = loopback(engine);
  assert.deepEqual(await client.defaults(), defaults);
  assert.deepEqual(await client.capabilities(defaults.request), capabilities);
  assert.equal((await client.create(defaults.request)).step, 0);
  assert.equal((await client.advance(8)).step, 8);
  assert.equal((await client.snapshot()).step, 8);
  assert.equal(
    (await client.analyze(defaults.request.spectroscopy.analysis)).frames,
    240,
  );
  assert.equal((await client.presentation({}))[0].plots.length, 1);
  assert.ok((await client.evidence()) instanceof Uint8Array);
  const checkpoint = await client.checkpoint();
  assert.equal((await client.restore(checkpoint)).step, 8);
  assert.ok(
    await client.import_evidence({ bytes: new Uint8Array(3), analysis: {} }),
  );
  assert.ok(
    await client.import_archive({
      bytes: new Uint8Array(3),
      config: { analysis: {} },
    }),
  );
  assert.equal(await client.dispose(), null);
  assert.deepEqual(
    engine.calls.map((c) => c.type),
    [...MESSAGE_TYPES],
  );
  assert.equal(engine.calls.find((c) => c.type === "advance").payload, 8);
});

test("Unknown operations and out-of-range step budgets are rejected before the engine", async () => {
  const engine = createFakeEngine();
  const dispatch = createDispatcher(engine);
  await assert.rejects(
    dispatch("fit_masses"),
    /Unknown spectroscopy operation/,
  );
  for (const steps of [0, 1.5, MAX_ADVANCE + 1, "8", undefined])
    await assert.rejects(dispatch("advance", { steps }), /between 1 and 64/);
  assert.equal(engine.calls.length, 0);
});

test("Requests are answered one at a time in arrival order, and an engine error is returned as text", async () => {
  const engine = createFakeEngine({ delays: { create: 25 } });
  const handle = createQueue(createDispatcher(engine));
  const order = [];
  const replies = await Promise.all(
    [
      { id: 1, type: "create", payload: defaults.request },
      { id: 2, type: "advance", payload: { steps: 4 } },
      { id: 3, type: "evidence" },
      { id: 4, type: "dispose" },
      { id: 5, type: "advance", payload: { steps: 4 } },
      { id: 6, type: "defaults" },
    ].map((message) =>
      handle(message).then((reply) => {
        order.push(reply.id);
        return reply;
      }),
    ),
  );
  assert.deepEqual(order, [1, 2, 3, 4, 5, 6]);
  assert.equal(replies[1].result.step, 4);
  assert.deepEqual(replies[4], { id: 5, error: "no live session" });
  assert.deepEqual(replies[5].result, defaults);
});

test("Binary results are transferred, JSON results are not", () => {
  const bytes = new Uint8Array([1, 2, 3]);
  assert.deepEqual(transferables({ id: 1, result: bytes }), [bytes.buffer]);
  assert.deepEqual(transferables({ id: 1, result: { step: 1 } }), []);
  assert.deepEqual(
    transferables({ id: 1, result: new Uint8Array(new ArrayBuffer(8), 2, 3) }),
    [],
  );
  assert.deepEqual(transferables({ id: 1, error: "x" }), []);
});

test("The wasm adapter passes JSON strings to Rust, frees replaced sessions and re-analyses imports without a session", async () => {
  const wasm = createFakeWasm();
  const engine = createWasmEngine(async () => wasm.module);
  await assert.rejects(engine.advance(4), /Create or restore/);
  await assert.rejects(engine.analyze({}), /before requesting an analysis/);
  await engine.create(defaults.request);
  assert.deepEqual(wasm.log[0], ["create", JSON.stringify(defaults.request)]);
  await engine.create(defaults.request);
  assert.equal(wasm.freed(), 1);
  await engine.analyze({ fit: "both" });
  assert.deepEqual(wasm.log.at(-1), ["analyze", '{"fit":"both"}']);

  const evidence = new Uint8Array([9, 9]);
  await engine.import_evidence({ bytes: evidence, analysis: { fit: "both" } });
  assert.equal(wasm.freed(), 2);
  await engine.analyze({ fit: "window_scan" });
  assert.deepEqual(wasm.log.at(-1), [
    "spectroscopy_analyze",
    evidence,
    '{"fit":"window_scan"}',
  ]);
  await assert.rejects(engine.checkpoint(), /Create or restore/);

  const archive = new Uint8Array([7]);
  await engine.import_archive({
    bytes: archive.buffer,
    config: { measurement: { max_lag: 8 }, analysis: { fit: "both" } },
  });
  await engine.analyze({ fit: "window_scan" });
  assert.deepEqual(JSON.parse(wasm.log.at(-1)[1]), {
    measurement: { max_lag: 8 },
    analysis: { fit: "window_scan" },
  });
  await assert.rejects(
    engine.import_evidence({ bytes: "not bytes", analysis: {} }),
    /binary CBOR/,
  );
  await engine.restore(new Uint8Array([1]));
  await engine.dispose();
  assert.equal(wasm.freed(), 3);
});

test("The session runs a bounded step budget per tick, pauses and resumes", async () => {
  const engine = createFakeEngine({ steps: 64 });
  const session = createSession(engine, {
    budget: 16,
    breathe: async () => {},
  });
  await session.create(defaults.request);
  const seen = [];
  await session.run((s) => {
    seen.push(s.step);
    if (s.step === 32) session.pause();
  });
  assert.deepEqual(seen, [16, 32]);
  assert.equal(session.running, false);
  await session.run((s) => seen.push(s.step));
  assert.deepEqual(seen, [16, 32, 48, 64]);
  assert.equal(session.snapshot.done, true);
  assert.ok(
    engine.calls
      .filter((c) => c.type === "advance")
      .every((c) => c.payload === 16),
  );
  await session.step();
  assert.equal(engine.count("advance"), 4, "a finished run is not advanced");
});

test("Re-analysis calls analyze only: it never advances or re-creates the session", async () => {
  const engine = createFakeEngine({ steps: 32 });
  const session = createSession(engine, { breathe: async () => {} });
  await session.create(defaults.request);
  await session.run();
  const advances = engine.count("advance");
  const base = defaults.request.spectroscopy.analysis;
  const controls = controlsFromAnalysis(base, 7);
  for (const estimator of ["frame_mean", "source_frozen", "euclidean_time"]) {
    const result = await session.analyze(
      buildAnalysis(base, { ...controls, estimator }),
    );
    assert.equal(result.analysis.estimator, estimator);
  }
  assert.equal(engine.count("advance"), advances);
  assert.equal(engine.count("create"), 1);
  assert.equal(engine.count("analyze"), 3);
});

test("Imported evidence is analysed and re-analysed without a live session", async () => {
  const engine = createFakeEngine();
  const session = createSession(engine);
  const base = defaults.request.spectroscopy.analysis;
  const imported = await session.importEvidence(new Uint8Array([1]), base);
  assert.equal(imported.notes[0], REPORT_NOTE);
  assert.equal(session.live, false);
  assert.equal(session.imported, "evidence");
  await session.analyze(base);
  assert.equal(engine.count("advance") + engine.count("create"), 0);
});

test("A replaced session stops the loop of the previous one", async () => {
  const engine = createFakeEngine({ steps: 640, delays: { advance: 2 } });
  const session = createSession(engine, {
    budget: 16,
    breathe: async () => {},
  });
  await session.create(defaults.request);
  const first = session.run();
  await new Promise((resolve) => setTimeout(resolve, 8));
  await session.create(defaults.request);
  await first;
  assert.equal(session.snapshot.step, 0);
  assert.equal(session.running, false);
});

test("The analysis adapter changes only the user-facing fields of the Rust AnalysisConfig", () => {
  const base = defaults.request.spectroscopy.analysis;
  const controls = controlsFromAnalysis(base, 7);
  assert.deepEqual(buildAnalysis(base, controls), base);
  assert.deepEqual(
    buildAnalysis(base, { ...controls, channels: ["a/distance"] }).channels,
    ["a/distance"],
    "the analysis is restricted to the ticked catalog rows",
  );
  const changed = buildAnalysis(
    base,
    {
      ...controls,
      resampling: "bootstrap",
      block: "fixed",
      blockFrames: 12,
      samples: 300,
      resampleSeed: 11,
      stability: true,
      gevp: true,
      assignments: { "meson/scalar/standard": "pion", "baryon/real": "" },
      anchors: ["pion"],
    },
    ["a", "b"],
  );
  assert.deepEqual(changed.resampling, {
    kind: "bootstrap",
    block: { kind: "fixed", frames: 12 },
    samples: 300,
    seed: 11,
  });
  assert.deepEqual(changed.stability, {}, "Rust chooses the stability grid");
  assert.deepEqual(changed.gevp, [{ id: "selected", channels: ["a", "b"] }]);
  assert.deepEqual(changed.assignments, { "meson/scalar/standard": "pion" });
  assert.deepEqual(changed.window_scan, base.window_scan);
  assert.deepEqual(changed.multi_exponential, base.multi_exponential);
  assert.deepEqual(changed.reference, base.reference);
  assert.equal(changed.svd_cut, base.svd_cut);
});

test("The request keeps Rust defaults, sends each catalog specification once and omits unavailable channels", () => {
  const form = formFromDefaults(defaults);
  assert.deepEqual(form.channels, [
    "meson/scalar/standard/distance",
    "meson/scalar/standard/cloning",
    "meson/pseudoscalar/standard/distance",
  ]);
  form.channels.push("baryon/real/triplet");
  form.walkers = 128;
  const availability = availabilityIndex(capabilities);
  const request = buildRequest(defaults, form, availability);
  assert.equal(request.run.walkers, 128);
  assert.equal(request.run.initial_lower, -1);
  assert.equal(request.chunk, 16);
  assert.deepEqual(
    request.spectroscopy.measurement.channels,
    [defaults.catalog[0].spec, defaults.catalog[2].spec],
    "two rows of one specification are one measured channel specification",
  );
  assert.equal(request.spectroscopy.measurement.max_lag, 8);
  assert.equal(defaults.request.run.walkers, 64, "defaults are not mutated");
  const probe = buildRequest(defaults, form, new Map(), { probe: true });
  assert.equal(probe.spectroscopy.measurement.channels.length, 4);
  JSON.stringify(request);
});

test("Choosing a variant restarts the form from the request Rust resolved for it", () => {
  assert.equal(variantRequest(defaults, "euclidean_gas").run.dimensions, 2);
  assert.equal(
    variantRequest(defaults, "latent_fractal_gas"),
    defaults.request,
    "a variant without a request falls back to the default request",
  );
  const form = formFromRequest(
    defaults,
    variantRequest(defaults, "euclidean_gas"),
  );
  assert.deepEqual(
    [form.variant, form.walkers, form.dimensions, form.steps],
    ["euclidean_gas", 32, 2, 128],
  );
  const request = buildRequest(defaults, form, availabilityIndex(capabilities));
  assert.equal(request.run.initial_lower, -2, "the variant run is kept whole");
  assert.equal(request.variant, "euclidean_gas");
});

test("Unavailable channels are disabled and show the Rust reason verbatim", () => {
  const availability = availabilityIndex(capabilities);
  assert.deepEqual(availability.get("baryon/real/triplet"), {
    available: false,
    reason: COLOR_REASON,
  });
  assert.deepEqual(availability.get("meson/scalar/standard/distance"), {
    available: true,
    reason: null,
  });
  assert.equal(
    availabilityIndex(defaults.catalog).get("u1/phase/q1/distance").available,
    true,
    "the catalog rows carry the availability of the default request",
  );

  const html = channelListHTML(
    defaults.catalog,
    ["meson/scalar/standard/distance", "baryon/real/triplet"],
    availability,
  );
  assert.ok(html.includes("Unavailable: " + escapeXML(COLOR_REASON)));
  assert.ok(html.includes("Unavailable: " + escapeXML(ODD_REASON)));
  const baryon = html.slice(html.indexOf('value="baryon/real/triplet"'));
  assert.match(
    baryon.slice(0, 80),
    /^value="baryon\/real\/triplet" disabled aria-describedby=/,
  );
  assert.match(
    html,
    /value="meson\/scalar\/standard\/distance" checked><code>meson\/scalar\/standard\/distance<\/code>/,
  );
  assert.ok(
    html.includes(escapeXML(defaults.catalog[0].descriptor.definition)),
  );
  assert.ok(html.includes("Book: def-sm-meson-operators"));
  assert.ok(html.includes("Elements: cloning_pair"));
  assert.ok(html.includes("Exchange: odd"));
  assert.ok(html.includes("Spatial parity: odd"));
  assert.ok(html.includes("No correlator"));
  assert.ok(html.includes("Not a book operator"));
  assert.ok(html.includes("d = 3"));
  assert.equal(html.match(/<fieldset class="family">/g).length, 3);
});

test("Book-only variants are listed but cannot be selected", () => {
  const html = variantOptionsHTML(defaults.variants, "viscous_euclidean_gas");
  assert.match(html, /value="viscous_euclidean_gas" selected>/);
  assert.match(html, /value="latent_fractal_gas" disabled>.*\(book only\)/);
});

test("Undefined correlator points become gaps, never zeros", () => {
  const chart = correlatorChart(scalar());
  const line = chart.series[0].points;
  assert.deepEqual(line[3], [3, null]);
  assert.deepEqual(line[4], [4, 0.14]);
  for (const series of chart.series)
    assert.equal(series.points[3][1], null, series.name);
  assert.ok(chart.series.every((s) => s.points.every((p) => p[1] !== 0)));

  const svg = chartSVG({ ...chart, series: [chart.series[0]] });
  const path = svg.match(/<path d="(M[^"]+)" stroke="#51d2b7"/)[1];
  assert.equal(path.match(/M/g).length, 2, "the line restarts after the gap");
  assert.equal(path.match(/L/g).length, 2);
  const markers = chartSVG({ ...chart, series: [chart.series[1]] });
  assert.equal(markers.match(/<circle /g).length, 4);

  const rate = effectiveRateChart(scalar());
  assert.deepEqual(
    rate.series[0].points.map((p) => p[1]),
    [0.49, 0.5, null, null, null],
  );
  const table = correlatorTable(scalar());
  assert.deepEqual(table.rows[3], [3, 3, null, null, null, null]);
  assert.ok(tableHTML(table).includes("<td>—</td>"));
  const numeric = chartTable(chart);
  assert.deepEqual(numeric.columns, ["Series", "x", "y", "Error"]);
  assert.deepEqual(numeric.rows[1], ["C(τ)", 1, 0.61, 0.03]);
  assert.deepEqual(numeric.rows[3], ["C(τ)", 3, null, null]);
  assert.equal(
    numeric.rows.length,
    5,
    "band edges and markers are drawing geometry, not reported numbers",
  );
});

test("Error bands repeat the Rust value and error and the lag axis uses the reported time step", () => {
  const channel = scalar();
  channel.correlator = {
    ...channel.correlator,
    time_step: 0.5,
    time_unit: "step_dt",
  };
  const chart = correlatorChart(channel, { logY: true });
  assert.equal(chart.yScale, "log");
  assert.equal(chart.xLabel, "Lag τ (stride · dt)");
  assert.deepEqual(
    chart.series[0].points.map((p) => p[0]),
    [0, 0.5, 1, 1.5, 2],
  );
  assert.deepEqual(chart.series[2].points[1], [0.5, 0.61 + 0.03]);
  assert.deepEqual(chart.series[3].points[1], [0.5, 0.61 - 0.03]);
  assert.ok(chart.series[2].dashed && chart.series[3].dashed);
  assert.equal(correlatorChart(report().channels[2]), null);
});

test("Live charts, the walker cloud, progress and coverage counters come from the Rust snapshot", () => {
  const s = snapshot(32);
  assert.deepEqual(
    measuredLive(s).map((c) => c.id),
    ["meson/scalar/standard/distance"],
    "an unavailable channel has empty live arrays",
  );
  const charts = liveCharts(s, ["meson/scalar/standard/distance"]);
  assert.deepEqual(
    charts.map((c) => c.title),
    ["Live C(τ)", "Live effective decay rate"],
  );
  assert.deepEqual(charts[0].series[0].points[3], [3, null]);
  assert.deepEqual(charts[0].series[0].points[4], [4, 0.14]);
  assert.deepEqual(
    charts[1].series[0].points.map((p) => p[1]),
    [0.49, 0.5, null, null, null],
  );
  assert.deepEqual(liveCharts(s, ["u1/phase/q1/distance"]), []);
  assert.deepEqual(
    cloudChart(s).series[0].points,
    [
      [0.1, -0.2],
      [-0.5, 0.2],
    ],
    "flat positions are read row by row and ineligible walkers are left out",
  );
  assert.equal(
    cloudChart({ walkers: { dimension: 3, positions: [], eligible: [] } }),
    null,
  );
  const progress = progressOf(s);
  assert.deepEqual(
    [progress.step, progress.required, progress.done, progress.chunk],
    [32, 256, false, 16],
  );
  assert.deepEqual(progress.table.rows, [[0, 7, 32, 16, 1, "—"]]);
  assert.deepEqual(progressOf(null).table.rows, []);
  assert.equal(presentationCharts(presentation())[0].title, "C(τ)");
  assert.deepEqual(presentationNotes(presentation()), [
    { owner: "Spectroscopy", text: REPORT_NOTE },
  ]);
  assert.deepEqual(presentationCharts(null), []);
  assert.equal(flowTable(report()), null);
  assert.deepEqual(
    flowTable({ flow: { frames: 12, steps: [0, 4], roughness: [0.4, null] } })
      .rows,
    [
      [0, 0.4],
      [4, null],
    ],
  );
  const coverage = coverageTable(s.channels);
  assert.deepEqual(coverage.rows[0], [
    "meson/scalar/standard/distance",
    40,
    0,
    2400,
    12,
    3,
    0,
    0,
    1,
    0,
  ]);
});

test("A rejected model shows the Rust reason and an empty rate, never a zero", () => {
  const table = fitTable(report());
  const column = (name) => table.columns.indexOf(name);
  const [fitted, rejected, missing] = table.rows;
  assert.equal(fitted[column("Quantity")], RATE_QUANTITY);
  assert.equal(fitted[column("Rate")], 0.495);
  assert.equal(fitted[column("χ²")], 2.4);
  assert.equal(fitted[column("dof")], 3);
  assert.equal(fitted[column("Window (frames)")], "1–4");
  assert.equal(fitted[column("Effective block")], 8);
  assert.equal(fitted[column("τ_int")], 1.7);
  assert.equal(rejected[column("Rate")], null);
  assert.equal(rejected[column("Error")], null);
  assert.equal(rejected[column("Model rejected")], REJECTED);
  assert.equal(missing[1], "Unavailable: " + ODD_REASON);
  assert.ok(!table.columns.some((c) => /mass/i.test(c)));
  const html = tableHTML(table);
  assert.ok(html.includes(escapeXML(REJECTED)));
  assert.equal(windowScanCharts(scalar()).length, 2);
  assert.deepEqual(windowScanCharts(report().channels[1]), []);
});

test("Rust notes are rendered verbatim on every scope and the comparison keeps its hypothesis label", () => {
  for (const scope of ["report", "correlators", "fits", "physics", "all"]) {
    const notes = collectNotes(report(), scope);
    assert.equal(notes[0].text, REPORT_NOTE, scope);
    assert.ok(notesHTML(notes).includes(escapeXML(REPORT_NOTE)), scope);
  }
  assert.ok(
    collectNotes(report(), "fits").some((n) =>
      n.text.startsWith("Window average"),
    ),
  );
  assert.equal(notesHTML([]), "");
  const tables = comparisonTables(report());
  assert.equal(tables.label, "hypothesis mapping");
  assert.equal(tables.reference.rows[0].at(-1), "—");
  assert.equal(tables.anchors[0].scale, "—");
  assert.equal(tables.ratios.rows[0][2], "—");
  const html = comparisonHTML(tables);
  assert.ok(
    html.includes('<p class="hypothesis-label">hypothesis mapping</p>'),
  );
  assert.ok(
    collectNotes(report(), "physics").some(
      (n) => n.owner === "hypothesis mapping",
    ),
  );
  assert.equal(comparisonTables({ comparison: null }), null);
});

test("Chart cards carry an SVG export control and a numeric fallback table", () => {
  const html = chartCardsHTML([correlatorChart(scalar())], {
    width: 320,
    group: "correlators",
  });
  assert.match(html, /data-export="correlators:0"/);
  assert.match(html, /<svg [^>]*role="img"/);
  assert.match(html, /<details class="numeric"><summary>Numeric values/);
  assert.match(html, /<th scope="col">Series<\/th>/);
});

test("The host sources contain no sampling, fitting or statistics", async () => {
  const forbidden =
    /Math\.(random|exp|log|log10|log2|sqrt|pow|cosh|acosh|hypot)\b|crypto\.getRandomValues|\*\*/;
  for (const name of ["main.js", "model.js", "views.js", "worker.js"]) {
    const source = await readFile(page("qft/" + name), "utf8");
    assert.doesNotMatch(source, forbidden, name);
    assert.doesNotMatch(source, /particle mass|hadron mass/i, name);
  }
});

test("The page exposes five keyboard-operable tabs and the section navigation", async () => {
  const html = await readFile(page("qft.html"), "utf8");
  assert.equal(html.match(/role="tab"/g).length, 5);
  assert.equal(html.match(/role="tabpanel"/g).length, 5);
  assert.equal(html.match(/aria-selected="true"/g).length, 1);
  for (const name of ["setup", "run", "correlators", "fits", "physics"]) {
    assert.ok(html.includes('aria-controls="panel-' + name + '"'), name);
    assert.ok(html.includes('aria-labelledby="tab-' + name + '"'), name);
  }
  assert.match(html, /<title>Algorithmic Gas · QFT Simulator<\/title>/);
  assert.match(html, /href="\.\/"/);
  assert.match(html, /href="\.\/lecture\.html"/);
  for (const [file, href] of [
    ["index.html", "./qft.html"],
    ["lecture.html", "./qft.html"],
    ["../index.html", "euclidean-gas/qft.html"],
    ["../lab/index.html", "../euclidean-gas/qft.html"],
  ])
    assert.ok(
      (await readFile(page(file), "utf8")).includes('href="' + href + '"'),
      file + " links to the QFT Simulator",
    );
  const lab = await readFile(page("index.html"), "utf8");
  assert.match(lab, /<title>Algorithmic Gas Lab · Fragile<\/title>/);
  assert.doesNotMatch(lab, /Euclidean Gas Lab/);
});
