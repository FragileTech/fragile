import { captureScreenshot } from "./helpers/screenshots.mjs";
import {
  prepareWorkspace,
  applyDraft,
  openFiles,
  closeFiles,
} from "./helpers/workspace-ui.mjs";
// Run against tools/serve-control.py after building both WebAssembly targets.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
const output =
  process.env.CONTROL_SCREENSHOTS || "/tmp/fractal-control-browser";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1000 } }),
  errors = [];
page.setDefaultTimeout(30000);
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
async function ready() {
  await page.waitForFunction(
    () =>
      !document.querySelector("main").inert &&
      document.getElementById("backend").textContent.includes("WEBASSEMBLY"),
  );
}
async function idle() {
  await page.waitForFunction(
    () => document.getElementById("status").textContent === "",
  );
}
async function graphicsReady() {
  await page.waitForFunction(() => {
    const gl = document.getElementById("world").getContext("webgl2");
    if (gl.isContextLost()) {
      window.graphicsStableSince = undefined;
      return false;
    }
    window.graphicsStableSince ??= performance.now();
    return performance.now() - window.graphicsStableSince > 1000;
  });
}
try {
  await prepareWorkspace(page);
  await page.goto(base);
  await ready();
  // New workspaces start paused; apply the six-frame replay fixture explicitly.
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000000",
  );
  await page.locator("#tab-controller").click();
  await page.locator("#algorithm").selectOption("fmc");
  await page.evaluate(() => {
    document.getElementById("threads").value = "2";
  });
  await page.locator("#walkers").fill("12");
  await page.locator("#horizon").fill("5");
  await page.locator("#frames").fill("6");
  await page.locator("#frames").press("Tab");
  await applyDraft(page);
  await ready();
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000006",
  );
  await idle();
  assert.match(
    await page.locator("#backend").innerText(),
    /THREADS? \/ WEBASSEMBLY/,
  );
  assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
  await graphicsReady();
  await captureScreenshot(page, {
    animations: "disabled",
    path: `${output}/harvest.png`,
    fullPage: false,
  });
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000012",
  );
  const downloadState = page.waitForEvent("download");
  await openFiles(page);
  await page.locator("#save-state").click();
  const stateFile = `${output}/state.fgcs`;
  await (await downloadState).saveAs(stateFile);
  await closeFiles(page);
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000018",
  );
  const chooser = page.waitForEvent("filechooser");
  await openFiles(page);
  await page.locator("#load-state").click();
  await (await chooser).setFiles(stateFile);
  await closeFiles(page);
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000012",
  );
  // Restoring a snapshot starts a new recording; record a decision to branch.
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000018",
  );
  const downloadRun = page.waitForEvent("download");
  await openFiles(page);
  await page.locator("#export-run").click();
  const archive = await downloadRun;
  const runFile = `${output}/${archive.suggestedFilename()}`;
  await archive.saveAs(runFile);
  await closeFiles(page);
  await page.locator("#timeline-decisions").click();
  await page.locator("#replay").click();
  await page.waitForFunction(
    () =>
      !document.querySelector("main").inert &&
      document.getElementById("run-state").textContent === "PAUSED",
  );
  await page.locator("#view").click();
  await page.locator("#clean").click();
  await page.locator("#mode-edit").click();
  await page.locator("#edit-json").click();
  const scene = JSON.parse(await page.locator("#scene-json").inputValue());
  const originalSceneName = scene.name;
  scene.name = "Edited test range";
  await page.locator("#scene-json").fill(JSON.stringify(scene));
  await page.locator("#apply-json").click();
  assert.equal(await page.locator("#pending-settings").isVisible(), true);
  await page.locator("#undo").click();
  assert.equal(await page.locator("#pending-settings").isVisible(), false);
  await page.locator("#edit-json").click();
  assert.equal(
    JSON.parse(await page.locator("#scene-json").inputValue()).name,
    originalSceneName,
  );
  await page.locator("#scene-json").fill(JSON.stringify(scene));
  await page.locator("#apply-json").click();
  await applyDraft(page);
  await ready();
  assert.equal(
    await page.locator("#scene-title").innerText(),
    "EDITED TEST RANGE",
  );
  await page.locator("#tab-setup").click();
  for (const scenario of ["ants", "tandem", "mining", "rocket"]) {
    await page.locator("#scenario").selectOption(scenario);
    await applyDraft(page);
    await ready();
    await page.locator("#step").click();
    await idle();
    assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
    console.log(`Checked ${scenario}: staged scene and controller step.`);
    await graphicsReady();
    await captureScreenshot(page, {
      animations: "disabled",
      path: `${output}/${scenario}.png`,
      fullPage: false,
    });
  }
  const runChooser = page.waitForEvent("filechooser");
  await openFiles(page);
  await page.locator("#import-run").click();
  await (await runChooser).setFiles(runFile);
  await closeFiles(page);
  await page.waitForFunction(
    () => document.getElementById("playback-status").textContent === "Replay",
  );
  await page.locator("#timeline-motion").click();
  await page.locator("#motion-resume").click();
  await ready();
  await page.locator("#tab-controller").click();
  await page.locator("#clock").selectOption("realtime");
  await applyDraft(page);
  await ready();
  await page.locator("#run").click();
  await page.waitForFunction(
    () => parseInt(document.getElementById("tick").textContent.slice(5)) >= 30,
  );
  await page.locator("#run").click();
  const paused = await page.locator("#tick").innerText();
  await page.waitForTimeout(250);
  assert.equal(await page.locator("#tick").innerText(), paused);
  await page.setViewportSize({ width: 768, height: 1024 });
  await graphicsReady();
  await captureScreenshot(page, {
    animations: "disabled",
    path: `${output}/tablet.png`,
    fullPage: false,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  await page.setViewportSize({ width: 1536, height: 1000 });
  await page.locator("#tab-controller").click();
  await page.locator("details:has(#algorithm-settings) > summary").click();
  assert.equal(await page.locator("#threads").getAttribute("max"), "64");
  // Exercise the threaded backend with a pool suitable for shared CI runners.
  // The contract suite separately checks all accepted counts through 64.
  await page.locator("#clock").selectOption("reproducible");
  await page.locator("#threads").fill("2");
  await page.locator("#threads").press("Tab");
  await applyDraft(page);
  await page.waitForFunction(() =>
    document.getElementById("backend").textContent.startsWith("2 THREADS"),
  );
  await page.locator("#step").click();
  await idle();
  assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
  for (const algorithm of ["icem", "mppi"]) {
    await page.locator("#algorithm").selectOption(algorithm);
    await applyDraft(page);
    await ready();
    assert.match(await page.locator("#backend").innerText(), /^2 THREADS/);
    await page.locator("#step").click();
    await idle();
    assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
  }
  await page.close();
  const fallback = await browser.newContext({ serviceWorkers: "block" });
  // Only the document headers determine isolation; assets use normal loading.
  await fallback.route(base, async (route) => {
    const response = await route.fetch(),
      headers = { ...response.headers() };
    delete headers["cross-origin-opener-policy"];
    delete headers["cross-origin-embedder-policy"];
    await route.fulfill({ response, headers });
  });
  const serialPage = await fallback.newPage();
  await prepareWorkspace(serialPage);
  await serialPage.goto(base, { waitUntil: "domcontentloaded" });
  await serialPage.waitForFunction(
    () => !document.getElementById("run").disabled,
    null,
    { timeout: 60000 },
  );
  assert.equal(await serialPage.locator("#tick").innerText(), "TICK 000000");
  assert.equal(
    await serialPage.locator("#backend").innerText(),
    "1 THREAD / WEBASSEMBLY",
  );
  await fallback.close();
  assert.deepEqual(errors, []);
  console.log(
    "Browser passed: five scenes, FMC, snapshot restore, archive, replay, editor, real-time clock, pause and responsive layout.",
  );
} finally {
  await browser.close();
}
