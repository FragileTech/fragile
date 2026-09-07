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
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1000 } }),
  errors = [];
page.setDefaultTimeout(30000);
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
async function ready() {
  await page.waitForFunction(() => !document.getElementById("run").disabled);
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
  await page.goto(base);
  await ready();
  // Keep the replay fixture at six frames after checking the new default.
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000012",
  );
  await page.locator("#frames").fill("6");
  await page.locator("#frames").press("Tab");
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
  await page.screenshot({ path: `${output}/harvest.png`, fullPage: true });
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000012",
  );
  const downloadState = page.waitForEvent("download");
  await page.locator("#save-state").click();
  const stateFile = `${output}/state.fgcs`;
  await (await downloadState).saveAs(stateFile);
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000018",
  );
  const chooser = page.waitForEvent("filechooser");
  await page.locator("#load-state").click();
  await (await chooser).setFiles(stateFile);
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000012",
  );
  const downloadRun = page.waitForEvent("download");
  await page.locator("#export-run").click();
  const runFile = `${output}/run.fgclab`;
  await (await downloadRun).saveAs(runFile);
  await page.locator("#replay").click();
  await page.waitForFunction(
    () => document.getElementById("run-state").textContent === "REPLAY",
  );
  await page.locator("#view").click();
  await page.locator("#clean").click();
  await page.locator("#edit").click();
  await page.locator("#edit-json").click();
  const scene = JSON.parse(await page.locator("#scene-json").inputValue());
  scene.name = "Edited test range";
  await page.locator("#scene-json").fill(JSON.stringify(scene));
  await page.locator("#apply-json").click();
  await ready();
  assert.equal(
    await page.locator("#scene-title").innerText(),
    "EDITED TEST RANGE",
  );
  await page.locator("#undo").click();
  await ready();
  assert.equal(
    await page.locator("#scene-title").innerText(),
    "ASTEROID HARVESTING",
  );
  await page.locator("#close-editor").click();
  for (const scenario of ["ants", "tandem", "mining", "rocket"]) {
    await page.locator("#scenario").selectOption(scenario);
    await ready();
    await page.locator("#step").click();
    await idle();
    assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
    await graphicsReady();
    await page.screenshot({
      path: `${output}/${scenario}.png`,
      fullPage: true,
    });
  }
  const runChooser = page.waitForEvent("filechooser");
  await page.locator("#import-run").click();
  await (await runChooser).setFiles(runFile);
  await ready();
  await page.waitForFunction(() =>
    document.getElementById("record-count").textContent.includes("Decision"),
  );
  await page.locator("#clock").selectOption("realtime");
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
  await page.screenshot({ path: `${output}/tablet.png`, fullPage: true });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  await page.setViewportSize({ width: 1536, height: 1000 });
  await page.locator(".controls summary").click();
  assert.equal(await page.locator("#threads").getAttribute("max"), "64");
  await page.locator("#threads").fill("64");
  await page.locator("#threads").press("Tab");
  await page.waitForFunction(() =>
    document.getElementById("backend").textContent.startsWith("64 THREADS"),
  );
  await page.locator("#step").click();
  await idle();
  assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
  for (const algorithm of ["icem", "mppi"]) {
    await page.locator("#algorithm").selectOption(algorithm);
    await ready();
    assert.match(await page.locator("#backend").innerText(), /^64 THREADS/);
    await page.locator("#step").click();
    await idle();
    assert.equal(await page.locator("#tick").innerText(), "TICK 000006");
  }
  const fallback = await browser.newContext({ serviceWorkers: "block" });
  await fallback.route("**/*", async (route) => {
    const response = await route.fetch(),
      headers = { ...response.headers() };
    delete headers["cross-origin-opener-policy"];
    delete headers["cross-origin-embedder-policy"];
    await route.fulfill({ response, headers });
  });
  const serialPage = await fallback.newPage();
  await serialPage.goto(base);
  await serialPage.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000006",
  );
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
