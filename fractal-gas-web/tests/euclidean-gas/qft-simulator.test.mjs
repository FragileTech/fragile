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
  GEVP_BASIS,
  availabilityIndex,
  buildAnalysis,
  buildRequest,
  calibrationTable,
  channelCharts,
  chartTable,
  cloudChart,
  collectNotes,
  comparisonTables,
  controlsFromAnalysis,
  correlatorTable,
  coverageTable,
  createSession,
  fitTable,
  flowTable,
  formFromDefaults,
  formFromRequest,
  gevpCharts,
  liveCharts,
  liveNotes,
  measuredLive,
  missingRecords,
  presentationCharts,
  presentationMetrics,
  presentationNotes,
  progressOf,
  provenanceTable,
  samplesTable,
  variantRequest,
  windowCharts,
} from "../../web/euclidean-gas/qft/model.js";
import {
  capabilitySummaryHTML,
  channelListHTML,
  chartCardsHTML,
  comparisonHTML,
  fitControlsHTML,
  notesHTML,
  tableHTML,
  variantOptionsHTML,
} from "../../web/euclidean-gas/qft/views.js";
import {
  CATALOG_REASON,
  ESTIMATOR_NOTE,
  FITTED,
  FIT_NOTE,
  NO_SIGNAL,
  RATE_QUANTITY,
  REPORT_NOTE,
  UNAVAILABLE,
  UNAVAILABLE_REASON,
  UNFITTED,
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
const channel = (id) => report().channels.find((c) => c.id === id);
const plotOf = (title, plot) =>
  presentation()
    .find((r) => r.title === title)
    .plots.find((p) => p.title === plot);

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
  assert.deepEqual(await client.request(), defaults.request);
  assert.equal(
    (await client.analyze(defaults.request.spectroscopy.analysis)).frames,
    report().frames,
  );
  assert.equal(
    (await client.presentation({}))[0].title,
    presentation()[0].title,
  );
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
  await assert.rejects(engine.presentation({}), /before requesting the plots/);
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
  await engine.presentation({ fit: "window_scan" });
  assert.deepEqual(wasm.log.at(-1), [
    "spectroscopy_presentation",
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
  await engine.presentation({ fit: "window_scan" });
  assert.equal(wasm.log.at(-1)[0], "spectroscopy_archive_presentation");
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

test("Re-analysis calls analyze and presentation only: it never advances or re-creates the session", async () => {
  const engine = createFakeEngine({ steps: 32 });
  const session = createSession(engine, { breathe: async () => {} });
  await session.create(defaults.request);
  await session.run();
  const advances = engine.count("advance");
  const base = defaults.request.spectroscopy.analysis;
  const controls = controlsFromAnalysis(base, 7);
  for (const estimator of ["frame_mean", "source_frozen", "euclidean_time"]) {
    const analysis = buildAnalysis(base, { ...controls, estimator });
    const result = await session.analyze(analysis);
    assert.equal(result.analysis.estimator, estimator);
    const plots = await session.presentation(analysis);
    assert.equal(plots[0].details.request.estimator, estimator);
  }
  assert.equal(engine.count("advance"), advances);
  assert.equal(engine.count("create"), 1);
  assert.equal(engine.count("analyze"), 3);
  assert.equal(engine.count("presentation"), 3);
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
  await session.presentation(base);
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
      assignments: { "meson/scalar/standard": "pion", "baryon/complex": "" },
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
  assert.deepEqual(changed.standard_model, base.standard_model);
  assert.equal(changed.frame_subtraction, base.frame_subtraction);
  assert.equal(changed.propagator_subtraction, base.propagator_subtraction);
  assert.equal(changed.svd_cut, base.svd_cut);
});

test("A GEVP basis is requested only inside the bounds Rust validates", () => {
  const base = defaults.request.spectroscopy.analysis;
  const controls = { ...controlsFromAnalysis(base, 7), gevp: true };
  const ids = (n) => Array.from({ length: n }, (_, i) => "channel/" + i);
  assert.deepEqual(
    buildAnalysis(base, controls, ids(GEVP_BASIS.max)).gevp,
    [{ id: "selected", channels: ids(GEVP_BASIS.max) }],
    "the largest basis Rust accepts is still sent",
  );
  for (const count of [GEVP_BASIS.min - 1, GEVP_BASIS.max + 1])
    assert.deepEqual(
      buildAnalysis(base, controls, ids(count)).gevp,
      [],
      "a basis of " + count + " channels would be rejected by Rust",
    );
  assert.match(
    fitControlsHTML(controls, GEVP_BASIS.max + 1),
    /name="gevp"[^>]* disabled/,
    "the control is disabled when no basis can be solved",
  );
  assert.ok(
    !fitControlsHTML(controls, GEVP_BASIS.max + 1).includes(
      'name="gevp" checked',
    ),
    "a disabled control is not shown as ticked",
  );
  assert.match(
    fitControlsHTML(controls, GEVP_BASIS.max),
    /name="gevp" checked/,
    "inside the bounds the control stays operable",
  );
});

test("The request keeps Rust defaults, sends each catalog specification once and omits unavailable channels", () => {
  const form = formFromDefaults(defaults);
  assert.deepEqual(form.channels, [
    "meson/scalar/standard/distance",
    "meson/scalar/standard/cloning",
    "meson/pseudoscalar/standard/distance",
    "baryon/complex/triplet",
  ]);
  form.channels.push("meson/pseudoscalar/abs2/distance");
  form.walkers = 128;
  const availability = availabilityIndex(capabilities);
  const request = buildRequest(defaults, form, availability);
  assert.equal(request.run.walkers, 128);
  assert.deepEqual(request.run.boundary, defaults.request.run.boundary);
  assert.equal(request.chunk, defaults.request.chunk);
  assert.deepEqual(
    request.spectroscopy.measurement.channels,
    [
      defaults.catalog[0].spec,
      defaults.catalog[2].spec,
      defaults.catalog[5].spec,
    ],
    "two rows of one specification are one measured channel specification",
  );
  assert.equal(
    request.spectroscopy.measurement.max_lag,
    defaults.request.spectroscopy.measurement.max_lag,
  );
  assert.equal(
    defaults.request.run.walkers,
    200,
    "the Rust defaults are not mutated",
  );
  const probe = buildRequest(defaults, form, new Map(), { probe: true });
  assert.equal(probe.spectroscopy.measurement.channels.length, 5);
  JSON.stringify(request);
});

test("Choosing a variant restarts the form from the request Rust resolved for it", () => {
  assert.equal(variantRequest(defaults, "euclidean").run.dimensions, 2);
  assert.equal(
    variantRequest(defaults, "latent"),
    defaults.request,
    "a variant without a request falls back to the default request",
  );
  const form = formFromRequest(defaults, variantRequest(defaults, "euclidean"));
  assert.deepEqual(
    [form.variant, form.dimensions],
    ["euclidean", 2],
    "the resolved variant request drives the form",
  );
  const request = buildRequest(defaults, form, availabilityIndex(capabilities));
  assert.deepEqual(
    request.run.boundary,
    variantRequest(defaults, "euclidean").run.boundary,
    "the variant run is kept whole",
  );
  assert.equal(request.variant, "euclidean");
});

test("Unavailable channels are disabled and show the Rust reason verbatim", () => {
  const availability = availabilityIndex(capabilities);
  assert.deepEqual(availability.get("meson/pseudoscalar/abs2/distance"), {
    available: false,
    reason: CATALOG_REASON,
  });
  assert.deepEqual(availability.get("meson/scalar/standard/distance"), {
    available: true,
    reason: null,
  });
  assert.equal(
    availabilityIndex(defaults.catalog).get("tensor/envelope/distance")
      .available,
    true,
    "the catalog rows carry the availability of the default request",
  );

  const html = channelListHTML(
    defaults.catalog,
    ["meson/scalar/standard/distance", "meson/pseudoscalar/abs2/distance"],
    availability,
  );
  assert.ok(html.includes("Unavailable: " + escapeXML(CATALOG_REASON)));
  const blocked = html.slice(
    html.indexOf('value="meson/pseudoscalar/abs2/distance"'),
  );
  assert.match(
    blocked.slice(0, 90),
    /^value="meson\/pseudoscalar\/abs2\/distance" disabled aria-describedby=/,
  );
  assert.match(
    html,
    /value="meson\/scalar\/standard\/distance" checked><code>meson\/scalar\/standard\/distance<\/code>/,
  );
  const meson = defaults.catalog[0];
  assert.ok(html.includes(escapeXML(meson.signature.descriptor.definition)));
  assert.ok(
    html.includes("Book: " + meson.signature.descriptor.book_label),
    "the book label of the signature is shown",
  );
  assert.ok(html.includes("Elements: cloning_pair"));
  assert.ok(html.includes("Exchange: even"));
  assert.ok(html.includes("Spatial parity: even"));
  assert.ok(html.includes("Assigned: f0_500"));
  assert.ok(html.includes("No correlator"), "tensor/envelope is not a curve");
  assert.ok(html.includes("Not a book operator"));
  assert.ok(html.includes("d = 3"));
  assert.equal(html.match(/<fieldset class="family">/g).length, 3);
});

test("The capability summary repeats the records Rust says are missing", () => {
  const html = capabilitySummaryHTML(
    capabilities,
    missingRecords(capabilities),
  );
  assert.ok(html.includes("Integrator step dt"));
  assert.ok(html.includes("periodic_box"));
  assert.ok(
    html.includes(
      escapeXML(capabilities.capabilities.missing.periodic_box),
      "the reason is Rust's",
    ),
  );
  assert.deepEqual(missingRecords({}), []);
});

test("Book-only variants are listed but cannot be selected", () => {
  const html = variantOptionsHTML(defaults.variants, "euclidean");
  assert.match(html, /value="euclidean" selected>/);
  assert.match(html, /value="latent" disabled>.*\(book only\)/);
});

test("Correlator and effective-rate curves are the Rust presentation series, bands included", () => {
  const charts = channelCharts(presentation(), FITTED, { logY: true });
  assert.deepEqual(
    charts.map((c) => c.title),
    [FITTED + " · Correlator", FITTED + " · Effective rate"],
  );
  const correlator = charts[0];
  assert.equal(correlator.yScale, "log", "only the correlator is logarithmic");
  assert.equal(charts[1].yScale, undefined);
  assert.equal(correlator.xLabel, "lag (frames)");
  assert.equal(correlator.yLabel, "C");

  // Every point is the one Rust sent, band edges included: the page adds
  // nothing to a value and subtracts nothing from it.
  const plot = plotOf(FITTED, "Correlator");
  assert.deepEqual(
    correlator.series.map((s) => s.name),
    plot.series.map((s) => s.name),
  );
  for (const [i, series] of correlator.series.entries())
    assert.deepEqual(series.points, plot.series[i].points, series.name);
  assert.deepEqual(
    correlator.series.map((s) => Boolean(s.dashed)),
    [false, true, true],
    "the two band edges Rust emits are drawn dashed",
  );
  assert.equal(
    new Set(correlator.series.map((s) => s.color)).size,
    1,
    "a band keeps the colour of its curve",
  );
  const numeric = chartTable(correlator);
  assert.deepEqual(numeric.columns, ["Series", "x", "y"]);
  assert.equal(
    numeric.rows.length,
    plot.series[0].points.length,
    "band edges are drawing geometry, not reported numbers",
  );
  assert.deepEqual(numeric.rows[0], ["C", ...plot.series[0].points[0]]);
  assert.deepEqual(channelCharts(presentation(), UNAVAILABLE), []);
  assert.deepEqual(channelCharts(null, FITTED), []);
});

test("A curve broken by an undefined point stays broken, and is never closed with a zero", () => {
  const broken = presentation().map((result) =>
    result.title === FITTED
      ? {
          ...result,
          plots: [
            {
              title: "Correlator",
              x_label: "lag (frames)",
              y_label: "C",
              series: [
                {
                  name: "C",
                  kind: "line",
                  points: [
                    [0, 1],
                    [1, 0.5],
                  ],
                },
                { name: "C", kind: "line", points: [[3, 0.2]] },
              ],
            },
          ],
        }
      : result,
  );
  const series = channelCharts(broken, FITTED)[0].series;
  assert.equal(series.length, 1, "the runs Rust split are drawn as one curve");
  assert.deepEqual(series[0].points, [
    [0, 1],
    [1, 0.5],
    [null, null],
    [3, 0.2],
  ]);
  const path = chartSVG({ title: "t", series }).match(/<path d="(M[^"]+)"/)[1];
  assert.equal(path.match(/M/g).length, 2, "the line restarts after the gap");
  assert.deepEqual(
    chartTable({ series }).rows.map((r) => r[1]),
    [0, 1, 3],
    "the gap is not a row",
  );
});

test("Fit windows and GEVP levels are plotted from the same presentation results", () => {
  const windows = windowCharts(presentation(), FITTED);
  assert.deepEqual(
    windows.map((c) => c.title),
    [
      FITTED + " · window scan: rate per window",
      FITTED + " · window scan: window weights",
      FITTED + " · stability scan: rate per window",
      FITTED + " · stability scan: window weights",
    ],
  );
  const weights = windows[1];
  assert.deepEqual(
    weights.series[0].points,
    plotOf(FITTED, "window scan: window weights").series[0].points,
  );
  assert.deepEqual(windowCharts(presentation(), UNFITTED), []);
  assert.deepEqual(gevpCharts(presentation()), [], "no basis was solved");
  assert.deepEqual(
    gevpCharts([
      {
        title: "GEVP mesons",
        plots: [
          {
            title: "Eigenvalues",
            x_label: "lag",
            y_label: "lambda",
            series: [{ name: "state 0", kind: "line", points: [[1, 0.5]] }],
          },
        ],
      },
    ]).map((c) => c.title),
    ["GEVP mesons · Eigenvalues"],
  );
});

test("The presentation panel shows every plot, metric and note Rust produced", () => {
  const charts = presentationCharts(presentation());
  assert.ok(charts.length >= 6);
  assert.ok(
    charts.every((c) => c.series.length),
    "a plot with no defined point is not drawn",
  );
  const metrics = presentationMetrics(presentation());
  assert.deepEqual(metrics.columns, ["Result", "Quantity", "Value", "Unit"]);
  const rate = metrics.rows.find(
    (row) => row[0] === FITTED && row[1] === "rate",
  );
  assert.equal(rate[2], channel(FITTED).mass.value);
  assert.equal(
    metrics.rows.find((row) => row[0] === UNFITTED && row[1] === "rate")[2],
    null,
    "a missing rate is an empty cell, never a zero",
  );
  const notes = presentationNotes(presentation());
  assert.ok(notes.some((n) => n.text === REPORT_NOTE));
  assert.ok(
    notes.some((n) => n.owner === FITTED && n.text.startsWith("Definition:")),
  );
  assert.deepEqual(presentationCharts(null), []);
});

test("The correlator and provenance tables copy the report without rescaling a lag", () => {
  const fitted = channel(FITTED);
  const table = correlatorTable(fitted);
  assert.deepEqual(table.columns, [
    "Lag",
    "Time unit",
    "Time step",
    "C(τ)",
    "Error",
    "Effective rate",
    "Rate error",
  ]);
  assert.deepEqual(table.rows[0], [
    fitted.correlator.lags[0],
    "frames",
    fitted.correlator.time_step,
    fitted.correlator.value[0],
    fitted.correlator.error[0],
    ...fitted.effective_mass[0],
  ]);
  assert.deepEqual(
    table.rows.at(-1).slice(5),
    [null, null],
    "the last lag has no effective rate and shows no number",
  );
  assert.ok(tableHTML(table).includes("<td>—</td>"));
  assert.deepEqual(correlatorTable(channel(UNAVAILABLE)), {
    columns: [],
    rows: [],
  });

  const provenance = provenanceTable(report());
  assert.deepEqual(provenance.rows[0], [
    "Calculation origin",
    "executed_algorithm_archive",
  ]);
  assert.deepEqual(provenance.rows[1], ["Precision", "f64"]);
  assert.equal(provenanceTable(null), null);

  const samples = samplesTable(report().channels);
  const row = samples.rows.find((r) => r[0] === FITTED);
  assert.equal(row[1], "source-frozen propagator");
  assert.equal(row[2], "sum of valid element weights");
  assert.equal(
    row[samples.columns.indexOf("Sampling unit")],
    fitted.correlator.samples_meta.sampling_unit,
  );
});

test("Live charts, the walker cloud, progress and coverage counters come from the Rust snapshot", () => {
  const s = snapshot(32);
  const live = measuredLive(s).map((c) => c.id);
  assert.ok(live.includes(FITTED));
  assert.ok(
    !live.includes(UNAVAILABLE),
    "an unavailable channel has empty live arrays",
  );
  const charts = liveCharts(s, [FITTED]);
  assert.deepEqual(
    charts.map((c) => c.title),
    ["Live C(τ)", "Live effective decay rate"],
  );
  const points = charts[0].series[0].points;
  const source = s.channels.find((c) => c.id === FITTED).correlator;
  assert.deepEqual(
    points,
    source.map((value, lag) => [lag, value]),
  );
  assert.deepEqual(liveCharts(s, [UNAVAILABLE]), []);
  const cloud = cloudChart(s);
  assert.equal(
    cloud.series[0].points.length,
    s.walkers.eligible.filter(Boolean).length,
    "flat positions are read row by row and ineligible walkers are left out",
  );
  assert.deepEqual(cloud.series[0].points[0], [
    s.walkers.positions[0],
    s.walkers.positions[1],
  ]);
  assert.equal(
    cloudChart({ walkers: { dimension: 3, positions: [], eligible: [] } }),
    null,
  );
  const progress = progressOf(s);
  assert.deepEqual(
    [progress.step, progress.required, progress.done, progress.chunk],
    [32, s.steps, false, s.chunk],
  );
  assert.deepEqual(progress.table.rows, [
    [0, s.replicas[0].seed, 32, 32, 1, "—"],
    [1, s.replicas[1].seed, 32, 32, 1, "—"],
  ]);
  assert.deepEqual(progressOf(null).table.rows, []);
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
  const counts = s.channels[0].coverage;
  assert.deepEqual(coverage.rows[0], [
    s.channels[0].id,
    "frame average",
    counts.frames,
    counts.empty_frames,
    counts.valid,
    counts.masked_historical,
    counts.masked_ineligible,
    counts.masked_color,
    counts.masked_identity,
    counts.masked_self,
    counts.masked_scale,
  ]);
  // A measured channel the live estimator cannot report keeps its coverage
  // counts, loses its estimator and carries the reason Rust wrote.
  const frozen = s.channels.findIndex((c) => c.note);
  assert.ok(frozen > 0);
  assert.equal(coverage.rows[frozen][1], "source-frozen propagator");
  assert.deepEqual(liveNotes(s), [
    ...s.notes,
    { owner: s.channels[frozen].id, text: ESTIMATOR_NOTE },
  ]);
  assert.deepEqual(liveNotes(null), []);
});

test("A channel without a rate shows the Rust reason and an empty rate, never a zero", () => {
  const table = fitTable(report());
  const column = (name) => table.columns.indexOf(name);
  const fitted = table.rows.find(
    (row) => row[0] === FITTED && row[1] === "Window scan",
  );
  const silent = table.rows.find((row) => row[0] === UNFITTED);
  const missing = table.rows.find((row) => row[0] === UNAVAILABLE);
  assert.equal(fitted[column("Quantity")], RATE_QUANTITY);
  assert.equal(fitted[column("Rate")], channel(FITTED).mass.value);
  assert.equal(
    fitted[column("χ²")],
    channel(FITTED).fits[0].diagnostics.chi2,
    "the diagnostics are copied, not recomputed",
  );
  assert.equal(
    fitted[column("Effective block")],
    channel(FITTED).correlator.samples_meta.effective_block,
  );
  assert.equal(silent[column("Rate")], null);
  assert.equal(silent[column("Error")], null);
  assert.equal(silent[column("No signal")], NO_SIGNAL);
  assert.equal(missing[1], "Unavailable: " + UNAVAILABLE_REASON);
  assert.ok(!table.columns.some((c) => /mass/i.test(c)));
  assert.ok(tableHTML(table).includes(escapeXML(NO_SIGNAL)));
  assert.ok(
    table.rows.some((row) => row[1] === "Stability scan"),
    "every fit method Rust ran gets a row",
  );
});

test("Rust notes are rendered verbatim on every scope and the comparison keeps its hypothesis label", () => {
  for (const scope of ["report", "correlators", "fits", "physics", "all"]) {
    const notes = collectNotes(report(), scope);
    assert.equal(notes[0].text, REPORT_NOTE, scope);
    assert.ok(notesHTML(notes).includes(escapeXML(REPORT_NOTE)), scope);
  }
  assert.ok(
    collectNotes(report(), "correlators").some(
      (n) => n.owner === FITTED && n.text === ESTIMATOR_NOTE,
    ),
  );
  assert.ok(collectNotes(report(), "fits").some((n) => n.text === FIT_NOTE));
  assert.equal(notesHTML([]), "");
  const tables = comparisonTables(report());
  assert.equal(tables.label, "hypothesis mapping");
  const pion = tables.reference.rows.find((r) => r[0] === "pion");
  assert.equal(pion[1], FITTED);
  assert.equal(pion[6], "source-frozen propagator");
  assert.equal(
    tables.reference.rows.find((r) => r[0] === "f0_500").at(-3),
    "—",
    "a reference without a measured rate stays empty",
  );
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
  const calibration = calibrationTable(report());
  assert.ok(
    calibration.rows.some((r) => r[0].startsWith("Pair weight N₁")),
    "every calibrated scale Rust reports has a row",
  );
});

test("Chart cards carry an SVG export control and a numeric fallback table", () => {
  const html = chartCardsHTML(channelCharts(presentation(), FITTED), {
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
    // No error band is built here: `value + error` belongs to Rust.
    assert.doesNotMatch(source, /sign \* error|value\s*[+-]\s*error/i, name);
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
