import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/sandbox-review/", import.meta.url);
await mkdir(output, { recursive: true });
// SwiftShader gives headless Chromium a software WebGL implementation.
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
page.setDefaultTimeout(30000);
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});

const ready = () =>
  page.waitForFunction(
    () =>
      window.euclideanGasLab?.ready &&
      document.querySelector("#run-status")?.textContent === "Paused" &&
      !document.querySelector("#step")?.disabled,
  );
const noError = async (label) =>
  assert.equal(await page.locator("#error").isVisible(), false, label);
const frames = () =>
  page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => resolve())),
  );
async function step(count = 1) {
  for (let i = 0; i < count; i++) {
    const before = await page.locator("#metric-step").textContent();
    await page.locator("#step").click();
    await page.waitForFunction(
      (before) =>
        document.querySelector("#metric-step").textContent !== before &&
        !document.querySelector("#step").disabled,
      before,
    );
  }
  await frames();
}
async function surface(action) {
  const before = await page.evaluate(() => window.euclideanGasLab.view.applied);
  await action();
  await page.waitForFunction(
    (before) => window.euclideanGasLab.view.applied > before,
    before,
  );
  await frames();
}
async function apply(values) {
  for (const [id, value] of Object.entries(values)) {
    const control = page.locator(`#${id}`);
    if ((await control.evaluate((el) => el.tagName)) === "SELECT")
      await control.selectOption(String(value));
    else await control.fill(String(value));
  }
  await surface(() => page.locator("#apply").click());
  await ready();
  await noError(`apply ${JSON.stringify(values)}`);
}
const stats = () =>
  page.evaluate(() => window.euclideanGasLab.stage.pixelStats());

try {
  await page.goto(base + "/euclidean-gas/index.html");
  await ready();
  await noError("load");
  assert.equal(await page.locator("#view").inputValue(), "2d");
  assert.equal(await page.locator("#stage canvas").count(), 1);

  for (const view of ["2d", "spatial", "landscape"]) {
    if (view !== "2d")
      await surface(() => page.locator("#view").selectOption(view));
    await step(3);
    await noError(view);
    const pixels = await stats();
    assert.ok(
      pixels.distinctColors > 8,
      `${view} renders a non-blank stage: ${JSON.stringify(pixels)}`,
    );
    assert.equal(
      await page.locator("#stage canvas").count(),
      1,
      `${view} keeps a single stage context`,
    );
    assert.equal(
      await page.locator("#reset-camera").isVisible(),
      view !== "2d",
      view,
    );
    assert.equal(
      await page.locator("#z-axis-label").isVisible(),
      view === "spatial",
      view,
    );
    await page.screenshot({
      path: new URL(`view-${view}.png`, output).pathname,
    });
  }

  // Walker selection: by picking in the 3D view, then by slot number.
  await page.locator("#walker").fill("5");
  await frames();
  assert.equal(await page.locator("#walker-tag").textContent(), "#5");
  const point = await page.evaluate(() =>
    window.euclideanGasLab.stage.renderer.screenPoint(17),
  );
  await page.mouse.click(point[0], point[1]);
  await frames();
  const picked = await page.evaluate(() => window.euclideanGasLab.selected);
  assert.notEqual(picked, 5, "clicking a walker selects it");
  assert.equal(await page.locator("#walker-tag").textContent(), `#${picked}`);
  assert.ok(
    (await page.locator("#walker-details dd").first().textContent()) !== "—",
  );

  // Links and trails are display-only overlays.
  await page.locator("#display-options summary").click();
  await page.locator("#links").selectOption("distance");
  await page.locator("#trails").check();
  await step(2);
  assert.match(await page.locator("#links-note").textContent(), /\d+ drawn/);
  await page.locator("#links").selectOption("cloning");
  await frames();
  await noError("overlays");
  for (const id of ["point-size", "surface-opacity", "height-scale"]) {
    await page.locator(`#${id}`).fill(id === "surface-opacity" ? "0.5" : "6");
    await frames();
  }
  await page.locator("#reset-camera").click();
  await page.locator("#fit-view").click();
  assert.ok((await stats()).distinctColors > 8, "restyled stage still renders");

  // Four dimensions: hidden coordinates are sliced.
  await apply({ dimensions: 4 });
  assert.equal(await page.locator("#view").inputValue(), "landscape");
  assert.equal(await page.locator("#slice-controls input").count(), 2);
  await surface(() =>
    page.locator("#slice-2").evaluate((input) => {
      input.value = 1.5;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }),
  );
  assert.equal(
    await page.evaluate(() => window.euclideanGasLab.view.slice[2]),
    1.5,
  );
  await surface(() => page.locator("#slice-from-selected").click());
  await surface(() => page.locator("#view").selectOption("spatial"));
  assert.equal(await page.locator("#z-axis").inputValue(), "2");
  assert.equal(await page.locator("#slice-controls input").count(), 2);
  await surface(() => page.locator("#x-axis").selectOption("2"));
  assert.notEqual(await page.locator("#z-axis").inputValue(), "2");
  await step(1);
  await noError("4D");

  // Two dimensions: the depth axis becomes a plane.
  await apply({ dimensions: 2 });
  assert.equal(
    await page.locator("#z-axis option:checked").textContent(),
    "Plane",
  );
  assert.equal(await page.locator("#slice-controls input").count(), 0);
  assert.ok((await stats()).distinctColors > 8, "2D state in the spatial view");

  // The benchmark list comes from the engine catalog, grouped as in the catalog.
  assert.equal(await page.locator("#benchmark option").count(), 38);
  assert.ok((await page.locator("#benchmark optgroup").count()) >= 6);

  // A 2D-only function locks the dimension and starts over its whole domain.
  await page.locator("#benchmark").selectOption("eggholder");
  assert.equal(await page.locator("#dimensions").inputValue(), "2");
  assert.equal(
    await page.locator("#dimensions").evaluate((input) => input.readOnly),
    true,
  );
  assert.equal(await page.locator("#initial-box").inputValue(), "domain");
  await surface(() => page.locator("#apply").click());
  await ready();
  await noError("eggholder");
  assert.equal(
    await page.locator("#z-axis option:checked").textContent(),
    "Plane",
  );
  await step(1);
  assert.ok((await stats()).distinctColors > 8, "eggholder");

  // COCO BBOB: the dimension snaps to the suite, the objective runs on the host,
  // and the gap is measured against the instance optimum.
  await page.locator("#benchmark").selectOption("bbob_21");
  await page.locator("#dimensions").fill("4");
  await page.locator("#dimensions").dispatchEvent("change");
  assert.ok(
    ["3", "5"].includes(await page.locator("#dimensions").inputValue()),
  );
  await page.locator("#param-coco_instance").fill("3");
  await surface(() => page.locator("#apply").click());
  await ready();
  await noError("bbob_21");
  await step(2);
  assert.match(
    await page.locator("#execution-details").textContent(),
    /host f64 · bbob_f021_i03_d0[35]/,
  );
  assert.equal(await page.locator("#metric-gap-tile").isVisible(), true);
  assert.ok(Number(await page.locator("#metric-gap").textContent()) >= 0);
  await surface(() => page.locator("#view").selectOption("landscape"));
  assert.ok((await stats()).distinctColors > 8, "BBOB landscape");
  await page.screenshot({ path: new URL("bbob-21.png", output).pathname });
  await surface(() => page.locator("#view").selectOption("spatial"));

  // Lennard-Jones: 3 × atoms dimensions and the atom viewer.
  await page.locator("#benchmark").selectOption("lennard_jones");
  await page.locator("#param-n_atoms").fill("4");
  await page.locator("#param-n_atoms").dispatchEvent("input");
  assert.equal(await page.locator("#dimensions").inputValue(), "12");
  await surface(() => page.locator("#apply").click());
  await ready();
  await noError("lennard_jones");
  assert.equal(await page.locator("#molecule").isVisible(), true);
  await step(1);
  await page.screenshot({
    path: new URL("lennard-jones.png", output).pathname,
  });

  // Run and pause keep the 3D stage live.
  await page.locator("#run").click();
  await page.waitForFunction(
    () => Number(document.querySelector("#metric-step").textContent) >= 4,
  );
  await page.locator("#run").click();
  await ready();
  await noError("run");

  // Deep link and the unchanged 2D default.
  await page.goto(base + "/euclidean-gas/index.html?view=landscape");
  await ready();
  await page.waitForFunction(() => window.euclideanGasLab.view.applied > 0);
  assert.equal(await page.locator("#view").inputValue(), "landscape");
  assert.ok((await stats()).distinctColors > 8, "deep-linked landscape");

  // Phone width: no horizontal page scroll.
  await page.setViewportSize({ width: 390, height: 844 });
  await frames();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  assert.ok(overflow <= 0, `horizontal overflow at 390px: ${overflow}px`);
  assert.ok((await stats()).distinctColors > 8, "narrow stage still renders");
  await page.screenshot({
    path: new URL("narrow.png", output).pathname,
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log("euclidean gas sandbox browser: ok");
} finally {
  await browser.close();
}
