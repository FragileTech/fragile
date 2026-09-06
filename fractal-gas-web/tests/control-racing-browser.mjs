import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
const output = process.env.CONTROL_SCREENSHOTS || "/tmp/fractal-control-racing";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1050 } }),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
page.setDefaultTimeout(30000);
const ready = () =>
  page.waitForFunction(() => !document.getElementById("run").disabled);
async function graphicsReady() {
  await page.waitForFunction(() => {
    const gl = document.getElementById("world").getContext("webgl2");
    if (gl.isContextLost()) {
      window.graphicsStableSince = undefined;
      return false;
    }
    window.graphicsStableSince ??= performance.now();
    return (
      performance.now() - window.graphicsStableSince > 1000 &&
      !document
        .getElementById("performance-readout")
        .textContent.includes("· 0 draws")
    );
  });
}
try {
  await page.goto(base);
  await ready();
  assert.match(await page.title(), /Fragile Tech/);
  assert.match(await page.locator(".brand").textContent(), /fragile.tech/);
  assert.ok(
    await page
      .locator(".brand-logo")
      .evaluate((e) => e.complete && e.naturalWidth > 0),
  );
  assert.equal(await page.locator("#scenario option").count(), 6);
  await page.evaluate(() => {
    for (const [id, value] of Object.entries({
      walkers: 24,
      horizon: 8,
      frames: 4,
    }))
      document.getElementById(id).value = value;
  });
  await page.locator("#scenario").selectOption("racing");
  await ready();
  await page.waitForFunction(() =>
    document
      .getElementById("score-note")
      .textContent.includes("Laps completed"),
  );
  assert.match(
    await page.locator("#scene-title").textContent(),
    /VIOLET CIRCUIT/,
  );
  assert.match(
    await page.locator("#score-note").textContent(),
    /Laps completed.*1\/16/,
  );
  await page.locator("#view").click();
  await graphicsReady();
  await page.screenshot({
    path: `${output}/circuit-overview.png`,
    fullPage: true,
  });
  for (const algorithm of ["fmc", "icem", "mppi"]) {
    await page.locator("#algorithm").selectOption(algorithm);
    await ready();
    await page.locator("#step").click();
    await page.waitForFunction(
      () => document.getElementById("tick").textContent === "TICK 000004",
    );
  }
  await page.locator("#manual").check();
  const box = await page.locator("#world").boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.keyboard.down("w");
  await page.waitForFunction(
    () => Number(document.getElementById("tick").textContent.slice(5)) >= 100,
  );
  await page.keyboard.up("w");
  await page.locator("#manual").uncheck();
  await page.locator("#view").click();
  await page.locator("#focus").click();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -900);
  await page.screenshot({ path: `${output}/kart-detail.png`, fullPage: true });
  await page.locator("#experiments").click();
  assert.equal(await page.locator("#benchmark-goal").inputValue(), "gates");
  assert.equal(await page.locator("#benchmark-target").inputValue(), "16");
  await page.locator("#close-experiments").click();
  await page.locator("#motion-timeline").fill("0");
  await page.locator("#motion-timeline").dispatchEvent("input");
  await page.waitForFunction(
    () => document.getElementById("run-state").textContent === "WORLD REPLAY",
  );
  assert.equal(await page.locator("#score").textContent(), "0");
  await page.setViewportSize({ width: 768, height: 1024 });
  assert(await page.locator("#manual").isVisible());
  await page.screenshot({ path: `${output}/tablet.png`, fullPage: true });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  console.log(
    "Browser passed: docs branding, racing preset, all planners, keyboard driving, circuit/kart rendering, lap readout, replay and tablet layout.",
  );
} catch (e) {
  console.error(
    "Lab status:",
    await page.locator("#status").textContent(),
    errors,
  );
  await page.screenshot({ path: `${output}/error.png`, fullPage: true });
  throw e;
} finally {
  await browser.close();
}
