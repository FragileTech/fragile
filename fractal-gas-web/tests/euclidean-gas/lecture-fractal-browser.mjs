import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";

const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/partv-review/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
async function healthy() {
  assert.equal(
    await page.locator("#error").isVisible(),
    false,
    await page.locator("#error").textContent(),
  );
}
async function ready() {
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent.startsWith("Ready") &&
      !document.querySelector("#step").disabled,
    null,
    { timeout: 180000 },
  );
  await healthy();
}
async function open(id) {
  await page.goto(`${base}/euclidean-gas/lecture.html?demo=${id}`);
  await ready();
  assert.equal(await page.locator('[data-demo^="V-"]').count(), 20);
  assert.equal(await page.locator("#control-source").count(), 0);
}
async function step() {
  await page.locator("#step").click();
  await page.waitForFunction(
    () => !document.querySelector("#step").disabled,
    null,
    { timeout: 180000 },
  );
  await healthy();
}
async function result() {
  return JSON.parse(
    await page.locator(".calculation-details pre").textContent(),
  );
}
async function archive(name) {
  const [file] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#archive").click(),
  ]);
  const path = new URL(name, output).pathname;
  await file.saveAs(path);
  const data = JSON.parse(await readFile(path, "utf8"));
  assert.equal(data.archives.length, data.configs.length);
  assert.ok(data.archives[0].steps.length > 0);
  assert.ok(data.archives[0].anchors.length > 0);
  return { path, data };
}
function validScene(scene) {
  assert.ok(scene.nodes.length > 0);
  const ids = new Set(scene.nodes.map((node) => node.id));
  assert.equal(ids.size, scene.nodes.length);
  for (const node of scene.nodes)
    assert.ok(node.position.every(Number.isFinite));
  for (const edge of scene.edges) {
    assert.ok(
      ids.has(edge.source),
      "Visible edge source must be a recorded event",
    );
    assert.ok(
      ids.has(edge.target),
      "Visible edge target must be a recorded event",
    );
  }
  for (const face of scene.faces) {
    assert.ok(face.vertices.length >= 3);
    assert.ok(face.vertices.flat().every(Number.isFinite));
  }
}
async function viewOnly(name, controls) {
  const before = await archive(`${name}-before.json`);
  const frame = await page.locator("#frame").textContent();
  const measurement = await result();
  await controls();
  assert.equal(await page.locator("#frame").textContent(), frame);
  assert.deepEqual(await result(), measurement);
  const after = await archive(`${name}-after.json`);
  assert.deepEqual(
    after.data,
    before.data,
    "Scene manipulation must not mutate the executed trajectory",
  );
  assert.doesNotMatch(
    await page.locator(".scene svg").innerHTML(),
    /NaN|Infinity/,
  );
  return before;
}
try {
  await open("V-02");
  for (let i = 0; i < 11; i++) await step();
  assert.match(await page.locator("#status").textContent(), /^Complete/);
  const graph = await result();
  validScene(graph.details.scene);
  assert.ok(graph.details.scene.edges.length > 0);
  assert.equal(graph.details.scene.unpositioned_events, 0);
  const summary = graph.details.graph_summary;
  const display = graph.details.scene.display;
  assert.equal(summary.recorded_steps, 96);
  assert.equal(summary.measurement_scope, "complete recorded graph");
  assert.equal(summary.unresolved_sources, 0);
  assert.equal(
    graph.details.graph,
    undefined,
    "The displayed result contains a graph summary; complete provenance remains in the exported archive",
  );
  assert.equal(display.window_start_step, 89);
  assert.equal(display.window_end_step, 96);
  assert.equal(display.total_nodes, summary.nodes);
  assert.equal(display.total_edges, summary.edges);
  assert.equal(display.displayed_nodes, graph.details.scene.nodes.length);
  assert.equal(display.displayed_edges, graph.details.scene.edges.length);
  assert.equal(display.displayed_edges, display.selected_edges);
  assert.equal(display.displayed_nodes, display.selected_nodes);
  assert.ok(display.omitted_nodes > 0 && display.omitted_edges > 0);
  assert.equal(
    summary.edges,
    Object.values(summary.edges_by_kind).reduce((sum, n) => sum + n, 0),
  );
  for (const edge of graph.details.scene.edges) {
    const [epoch, targetStep] = edge.target.split(":").map(Number);
    assert.equal(epoch, display.epoch);
    assert.ok(targetStep >= 89 && targetStep <= 96);
  }
  assert.ok(
    graph.details.scene.edges.some(
      (edge) => Number(edge.source.split(":")[1]) < 89,
    ),
    "Historical source endpoints must remain visible for incoming edges",
  );
  assert.ok(
    graph.plots.some((plot) =>
      plot.series.some((series) => series.points.some(([step]) => step < 89)),
    ),
    "Scientific measurements must include the full recorded history",
  );
  assert.equal(
    graph.metrics.find(
      (m) => m.label === "Boundary of triangle boundary residual",
    ).value,
    0,
  );
  assert.equal(await page.locator(".scene").isVisible(), true);
  const graphEvidence = await viewOnly("graph", async () => {
    await page.locator('[name="event"]').selectOption({ index: 3 });
    await page.locator('[name="trace-relation"]').selectOption("ancestry");
    await page.locator(".scene-viewport").press("ArrowRight");
    await page.locator('[name="time-cut"]').fill("0.5");
    await page.locator('[name="time-cut"]').dispatchEvent("input");
  });
  assert.equal(graphEvidence.data.request.id, "V-02");
  await page.locator(".scene").screenshot({
    path: new URL("recorded-interaction-graph.png", output).pathname,
  });
  await page.locator("#archive-import").setInputFiles(graphEvidence.path);
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Archive validated",
    null,
    { timeout: 180000 },
  );
  await healthy();
  const imported = await result();
  assert.deepEqual(imported.plots, graph.plots);
  assert.deepEqual(imported.metrics, graph.metrics);
  assert.deepEqual(imported.details.scene, graph.details.scene);

  await open("V-09");
  const cells = await result();
  validScene(cells.details.scene);
  assert.ok(cells.details.scene.faces.length > 0);
  assert.ok(
    cells.details.scene.faces.every((f) => f.layer === "metric Voronoi cells"),
  );
  assert.ok(
    cells.details.mesh.cells.every((c) => c.area >= 0 && c.geometric_area >= 0),
  );
  assert.ok(Math.abs(cells.details.mesh.closure_error) < 1e-8);
  assert.ok(await page.locator(".scene polygon").count());
  await viewOnly("cells", async () => {
    await page.locator(".scene-viewport").press("ArrowUp");
    await page.locator('[name="vertical-scale"]').fill("30");
    await page.locator('[name="vertical-scale"]').dispatchEvent("input");
  });
  await page.locator(".scene").screenshot({
    path: new URL("recorded-metric-cells.png", output).pathname,
  });

  await open("V-11");
  await step();
  const slabs = await result();
  validScene(slabs.details.scene);
  assert.ok(slabs.details.scene.faces.length > 0);
  assert.ok(slabs.details.slab_slots.tracked_slots.length >= 2);
  assert.ok(
    slabs.details.scene.edges.some((e) => e.layer === "clone replacement"),
  );
  assert.ok(
    slabs.details.scene.edges.some((e) => e.layer === "kinetic motion"),
  );
  assert.equal(
    slabs.details.scene.nodes.length,
    3 * slabs.details.slab_slots.tracked_slots.length,
  );
  assert.ok(await page.locator(".scene polygon").count());
  await viewOnly("slabs", async () => {
    await page.locator('[name="event"]').selectOption({ index: 3 });
    await page.locator(".scene-viewport").press("ArrowRight");
    await page.locator('[name="time-cut"]').fill("0.5");
    await page.locator('[name="time-cut"]').dispatchEvent("input");
  });
  await page.locator(".scene").screenshot({
    path: new URL("recorded-material-slab.png", output).pathname,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 2,
    ),
  );
  await page.screenshot({
    path: new URL("mobile-slab.png", output).pathname,
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  await writeFile(
    new URL("validation.json", output),
    JSON.stringify(
      {
        passed: true,
        testedIds: ["V-02", "V-09", "V-11"],
        eventGraph: true,
        graphMeasurementSteps: 96,
        graphDisplayWindow: [89, 96],
        historicalSourceEndpoints: true,
        evidenceReplay: true,
        immutableViewControls: true,
        recordedMetricCells: true,
        recordedMaterialSlab: true,
        mobile: true,
        errors,
      },
      null,
      2,
    ),
  );
  console.log(
    "Part V focused browser checks passed: actual event graph, metric cells, material slab, immutable scene controls, evidence replay and mobile layout.",
  );
} finally {
  await browser.close();
}
