import { captureScreenshot } from "./helpers/screenshots.mjs";
import { prepareWorkspace, applyDraft } from "./helpers/workspace-ui.mjs";
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
const output = process.env.CONTROL_SCREENSHOTS || "/tmp/fractal-control-racing";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1050 } }),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
page.setDefaultTimeout(120000);
const ready = () =>
  page.waitForFunction(
    () => !document.getElementById("run").disabled,
    undefined,
    { polling: 100 },
  );
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
async function capture(name) {
  await graphicsReady();
  await captureScreenshot(page, { path: `${output}/${name}.png` });
}
try {
  await prepareWorkspace(page);
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
  assert.equal(await page.locator("#track option").count(), 6);
  assert.equal(await page.locator("#track-control").isVisible(), false);
  assert.equal(await page.locator("#circuit-preview").isVisible(), false);
  await page.evaluate(() => {
    for (const [id, value] of Object.entries({
      threads: 1,
      walkers: 24,
      horizon: 8,
      frames: 4,
    }))
      document.getElementById(id).value = value;
  });
  await page.locator("#scenario").selectOption("racing");
  await applyDraft(page);
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
  await capture("circuit-overview");
  for (const algorithm of ["fmc", "icem", "mppi"]) {
    await page.locator("#tab-controller").click();
    await page.locator("#algorithm").selectOption(algorithm);
    await applyDraft(page);
    await ready();
    await page.locator("#step").click();
    await page.waitForFunction(
      () => document.getElementById("tick").textContent === "TICK 000004",
    );
  }
  await page.locator("#mode-drive").click();
  await page.locator("#run").click();
  const box = await page.locator("#world").boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.keyboard.down("w");
  await page.waitForFunction(
    () => Number(document.getElementById("tick").textContent.slice(5)) >= 100,
  );
  await page.keyboard.up("w");
  await page.locator("#mode-inspect").click();
  await page.locator("#view").click();
  await page.locator("#focus").click();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -900);
  await capture("kart-detail");
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
  assert(await page.locator("#mode-drive").isVisible());
  await capture("tablet");
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  // Every library entry loads its own preview, checkpoint count and clean state.
  for (const viewport of [
    { width: 1536, height: 1050 },
    { width: 768, height: 1024 },
  ]) {
    await page.setViewportSize(viewport);
    if (!(await page.locator("#tab-setup").isVisible()))
      await page.locator("#toggle-settings").click();
    let lastOutline;
    for (const id of [
      "racing-roots",
      "racing-fearless",
      "racing-sepang",
      "racing-original",
      "racing-obstacle-field",
      "racing",
    ]) {
      if (!(await page.locator("#tab-setup").isVisible()))
        await page.locator("#toggle-settings").click();
      await page.locator("#tab-setup").click();
      await page.locator("#track").selectOption(id);
      await applyDraft(page);
      await ready();
      assert.equal(await page.locator("#scenario").inputValue(), "racing");
      assert.equal(await page.locator("#track-control").isVisible(), true);
      const scene = await (
        await page.request.get(new URL(`scenarios/${id}.json`, base).href)
      ).json();
      const preview = page.locator("#circuit-preview");
      assert.equal(await preview.isVisible(), true);
      assert.equal(await page.locator("#description").isVisible(), true);
      assert.equal(
        await preview.locator("strong").textContent(),
        scene.circuit.name,
      );
      assert.equal(
        await preview.locator(".circuit-difficulty").textContent(),
        scene.circuit.difficulty,
      );
      assert.match(
        await page.locator("#score-note").textContent(),
        new RegExp(`1/${scene.gates.length}`),
      );
      assert.equal(await page.locator("#score").textContent(), "0");
      const outline = await preview
        .locator("svg > path")
        .first()
        .getAttribute("d");
      assert.notEqual(outline, lastOutline);
      lastOutline = outline;
      if (scene.circuit.sources.length) {
        assert.equal(
          await preview.locator("a").getAttribute("href"),
          scene.circuit.sources[0].url,
        );
      } else assert.equal(await preview.locator("a").count(), 0);
      // Driving is independent of viewport size; exercise it once per circuit.
      if (viewport.width === 1536) {
        await page.locator("#mode-drive").click();
        await page.locator("#run").click();
        await page.locator("#world").click();
        await page.keyboard.down("w");
        await page.waitForFunction(
          () =>
            Number(document.getElementById("tick").textContent.slice(5)) >= 12,
        );
        await page.keyboard.up("w");
        await page.locator("#mode-inspect").click();
      }
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        ),
        false,
      );
      // Keep every layout assertion, with representative tablet screenshots.
      if (viewport.width === 1536 || id === "racing-roots" || id === "racing")
        await capture(`${id}-${viewport.width}`);
      console.log(
        `Checked ${id}: selection, preview and ${viewport.width}px layout${viewport.width === 1536 ? ", manual control" : ""}.`,
      );
    }
  }
  await page.locator("#scenario").selectOption("harvest");
  await applyDraft(page);
  await ready();
  assert.equal(await page.locator("#circuit-preview").isVisible(), false);
  assert.equal(await page.locator("#track-control").isVisible(), false);
  assert.deepEqual(errors, []);
  console.log(
    "Browser passed: docs branding, racing preset, all planners, keyboard driving, circuit/kart rendering, lap readout, replay and tablet layout.",
  );
} catch (e) {
  console.error(e);
  console.error(
    "Lab status:",
    await page
      .locator("#status")
      .textContent({ timeout: 1000 })
      .catch(() => "Page unavailable"),
    errors,
  );
  await page
    .screenshot({ path: `${output}/error.png`, timeout: 10000 })
    .catch(() => {});
  throw e;
} finally {
  await browser.close();
}
