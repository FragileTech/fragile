import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/partv-review/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
async function ready() {
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent.startsWith("Ready") &&
      !document.querySelector("#step").disabled,
    { timeout: 60000 },
  );
  assert.equal(await page.locator("#error").isVisible(), false);
}
async function open(id) {
  await page.goto(`${base}/euclidean-gas/lecture.html?demo=${id}`);
  await ready();
}
async function step() {
  await page.locator("#step").click();
  await page.waitForFunction(() => !document.querySelector("#step").disabled);
  assert.equal(
    await page.locator("#error").isVisible(),
    false,
    await page.locator("#error").textContent(),
  );
}
try {
  await open("V-01");
  await step();
  await step();
  await page.locator('[data-demo="V-02"]').click();
  await ready();
  assert.equal(await page.locator("#frame").textContent(), "2 · latest");
  assert.ok((await page.locator(".scene polygon").count()) > 0);
  const frame = await page.locator("#frame").textContent();
  await page.locator('[name="event"]').selectOption({ index: 10 });
  await page.locator('[name="trace-relation"]').selectOption("ancestry");
  await page.locator(".scene-viewport").press("ArrowRight");
  assert.equal(await page.locator("#frame").textContent(), frame);
  await page
    .locator(".scene")
    .screenshot({ path: new URL("interaction-complex.png", output).pathname });
  const saved = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#archive").click(),
  ]);
  const archivePath = new URL("archive.json", output).pathname;
  await saved[0].saveAs(archivePath);
  await page.locator("#archive-import").setInputFiles(archivePath);
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "Archive validated",
  );
  assert.equal(
    await page.locator("#title").textContent(),
    "Imported Fractal Set archive",
  );
  await open("V-06");
  await step();
  await page.screenshot({
    path: new URL("adaptive-noise.png", output).pathname,
    fullPage: true,
  });
  await open("V-09");
  await page.locator("#control-geometry").selectOption("variable");
  await ready();
  assert.match(await page.locator("#charts").textContent(), /Variable-metric/);
  await page.locator(".advanced summary").click();
  await page.locator("#control-resolution").selectOption("24");
  await ready();
  await page.screenshot({
    path: new URL("variable-metric.png", output).pathname,
    fullPage: true,
  });
  await open("V-11");
  await step();
  await step();
  assert.ok((await page.locator(".scene polygon").count()) > 0);
  await page.locator('[name="event"]').selectOption({ index: 3 });
  await page.locator(".scene-viewport").press("ArrowRight");
  await page
    .locator(".scene")
    .screenshot({ path: new URL("spacetime-cell.png", output).pathname });
  await open("V-17");
  await step();
  await step();
  const snapshotFrame = await page.locator("#frame").textContent();
  await page.locator('[name="vertical-scale"]').fill("30");
  await page.locator('[name="vertical-scale"]').dispatchEvent("input");
  assert.equal(await page.locator("#frame").textContent(), snapshotFrame);
  await page
    .locator(".scene")
    .screenshot({ path: new URL("causal-orders.png", output).pathname });
  await open("V-20");
  await page.screenshot({
    path: new URL("curvature.png", output).pathname,
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await open("V-11");
  await step();
  await page.screenshot({
    path: new URL("mobile-debug.png", output).pathname,
    fullPage: true,
  });
  const overflow = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
    elements: [...document.querySelectorAll("body *")]
      .filter((e) => e.getBoundingClientRect().right > innerWidth + 1)
      .slice(0, 15)
      .map((e) => ({
        tag: e.tagName,
        cls: e.className,
        name: e.getAttribute("name"),
        width: e.getBoundingClientRect().width,
        right: e.getBoundingClientRect().right,
      })),
  }));
  console.log(JSON.stringify(overflow));
  assert.equal(overflow.scroll <= overflow.width, true);
  await page.screenshot({
    path: new URL("mobile.png", output).pathname,
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  await writeFile(
    new URL("validation.json", output),
    JSON.stringify(
      {
        sharedArchive: true,
        archiveImport: true,
        viewOnlyControls: true,
        adaptiveNoise: true,
        variableMetric: true,
        spacetime: true,
        physicalOrders: true,
        mobile: true,
        errors,
      },
      null,
      2,
    ),
  );
  console.log(
    "Part V shared archive, import/export, scene controls, adaptive/noise, variable geometry, spacetime and mobile passed",
  );
} finally {
  await browser.close();
}
