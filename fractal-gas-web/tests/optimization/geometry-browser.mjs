import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1560, height: 1150 } }),
  errors = [];
page.on("pageerror", (error) => errors.push(error.message));
await mkdir("/tmp/optimization-geometry", { recursive: true });
try {
  await page.goto(
    process.env.OPTIMIZATION_TEST_URL || "http://127.0.0.1:8081/optimization/",
  );
  await page.waitForFunction(
    () =>
      document.querySelector("#step") &&
      !document.querySelector("#step").disabled,
  );
  await page.locator("#geometry-controls").evaluate((e) => (e.open = true));
  await page.locator("#geometry-collect").check();
  await page.waitForFunction(
    () => !document.querySelector("#geometry-collect").disabled,
  );
  await page.locator("#record-history").check();
  for (const [dimensions, algorithm, perturbation] of [
    [2, "gas", "local_covariance"],
    [3, "euclidean", "cloning_guided"],
    [8, "wave", "adaptive_fractal"],
    [3, "cmaes_active", null],
    [2, "cmaes_bipop", null],
  ]) {
    await page.locator("#algorithm").selectOption(algorithm);
    await page.locator("#benchmark").selectOption("quadratic");
    await page.locator('[name="dimensions"]').fill(String(dimensions));
    if (perturbation)
      await page.locator("#perturbation").selectOption(perturbation);
    await page.locator("#apply").click();
    await page.waitForFunction(() => !document.querySelector("#step").disabled);
    for (let i = 0; i < 4; i++) {
      const old = await page.locator("#iteration").textContent();
      await page.locator("#step").click();
      await page.waitForFunction(
        (previous) =>
          document.querySelector("#iteration").textContent !== previous,
        old,
      );
      await page.waitForFunction(
        () => !document.querySelector("#step").disabled,
      );
    }
    await page.locator("#geometry-jumps").check();
    await page.locator("#geometry-field").check();
    await page.locator("#geometry-drift").check();
    const message = await page.locator("#geometry-status").textContent();
    assert.match(message, /models/);
    assert.doesNotMatch(message, /Invalid|invalid|NaN/);
    await page.locator("#view").selectOption("spatial");
    await page.locator("#geometry-scale").selectOption("normalized");
    if (dimensions > 3) await page.locator("#axis-x").selectOption("7");
    await page.locator("#world").screenshot({
      path: `/tmp/optimization-geometry/${dimensions}d-${algorithm}.png`,
    });
    await page.locator("#geometry-visible").uncheck();
    assert.match(
      await page.locator("#geometry-status").textContent(),
      /hidden/,
    );
    await page.locator("#geometry-visible").check();
    await page.locator("#view").selectOption("landscape");
    await page.waitForFunction(() =>
      document
        .querySelector("#geometry-status")
        .textContent.includes("coordinate plane"),
    );
  }
  await page.locator("#timeline").fill("1");
  await page.locator("#timeline").dispatchEvent("input");
  assert.match(
    await page.locator("#geometry-status").textContent(),
    /BIPOP CMA-ES/,
  );
  const downloaded = page.waitForEvent("download");
  await page.locator("#save").click();
  const download = await downloaded;
  await download.saveAs("/tmp/optimization-geometry/replay.fgopt");
  await page
    .locator("#file")
    .setInputFiles("/tmp/optimization-geometry/replay.fgopt");
  await page.waitForFunction(
    () => document.querySelector("#geometry-collect").disabled,
  );
  assert.match(
    await page.locator("#geometry-status").textContent(),
    /BIPOP CMA-ES/,
  );
  // Repeated frame replacement must release old Three.js buffers/materials.
  const memory = await page.evaluate(async () => {
    const { SwarmRenderer } = await import("./renderer.js");
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:300px;height:300px";
    document.body.append(canvas);
    const renderer = new SwarmRenderer(canvas, () => {});
    renderer.setConfig({ dimensions: 3, low: -10, high: 10 });
    const frame = new Float64Array(26);
    frame.set([1, 1, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0]);
    frame[20] = 1;
    const settings = {
      axes: [0, 1, 2],
      view: "spatial",
      pointSize: 2,
      opacity: 1,
      edges: "none",
      geometryVisible: true,
      geometryMethods: ["cloning_guided"],
      geometryScale: "normalized",
      geometryOpacity: 0.5,
      geometryScope: "all",
      geometryVectorScale: 1,
      geometryField: true,
      geometryDrift: true,
      geometryJumps: true,
      geometryClones: true,
    };
    const model = {
      anchor: [0, 0, 0],
      shape: [2, 1, 0, 1, 2, 0, 0, 0, 1],
      representation: "dense",
      columns: 3,
      scale: 0.5,
      field: [1, 0, 0],
      drift: [0.1, 0, 0],
    };
    const metadata = {
      geometry: {
        version: 1,
        dimensions: 3,
        methods: [
          {
            id: "cloning_guided",
            active: true,
            status: "ready",
            models: [model],
          },
        ],
        events: [
          { origin: [0, 0, 0], destination: [1, 1, 0], kind: "proposal" },
        ],
      },
    };
    let baseline;
    for (let i = 0; i < 40; i++) {
      renderer.update(frame, settings, [], 0, metadata);
      if (i === 5) baseline = renderer.renderer.info.memory.geometries;
    }
    const after = renderer.renderer.info.memory.geometries;
    renderer.setSurface(new Float64Array([0, 0, 0, 0]), 2, {
      ...settings,
      slice: [0, 0, 0],
    });
    const retained = renderer.geometrySummary.includes("Clone-guided");
    renderer.dispose();
    canvas.remove();
    return { baseline, after, retained };
  });
  assert.equal(memory.after, memory.baseline);
  assert.equal(memory.retained, true);
  assert.deepEqual(errors, []);
  console.log(
    "Covariance/jump browser checks passed in 2D, 3D and projected 8D.",
  );
} finally {
  await browser.close();
}
