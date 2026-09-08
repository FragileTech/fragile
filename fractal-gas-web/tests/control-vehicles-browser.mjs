import { openFiles, closeFiles } from "./helpers/workspace-ui.mjs";
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { resolveBodies } from "../web/lab/agent-types.js";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    serviceWorkers: "block",
    viewport: { width: 900, height: 650 },
    reducedMotion: "reduce",
  });
  page.setDefaultTimeout(60000);
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.addInitScript(() => {
    localStorage.setItem("lab.workspace.onboarded", "true");
    Object.defineProperty(navigator, "serviceWorker", { value: undefined });
    window.vehicleInits = [];
    const Base = window.Worker;
    window.Worker = class extends Base {
      postMessage(data, ...args) {
        if (data.type === "init") vehicleInits.push(structuredClone(data));
        return super.postMessage(data, ...args);
      }
    };
  });
  // Like the camera tests, load real scenes/assets but render on demand. A
  // continuous software-rendering loop makes repeated large-track loads costly.
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: (await response.text()).replace(
        "installStyleControls();",
        "cancelAnimationFrame(renderer.frame); window.vehicleRenderer = renderer; installStyleControls();",
      ),
    });
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/");
  const ready = () =>
    page.waitForFunction(() => !document.getElementById("run").disabled);
  const latest = () => page.evaluate(() => vehicleInits.at(-1).scene);
  const change = async (action) => {
    const count = await page.evaluate(() => vehicleInits.length);
    await action();
    await page.waitForFunction(() => !document.body.dataset.loadingPreset);
    await closeFiles(page);
    if (await page.locator("#pending-settings").isVisible()) await page.locator("#apply-configuration").click();
    else return;
    await page.waitForFunction((n) => vehicleInits.length > n, count);
    await page.waitForFunction(() => !document.querySelector("main").inert);
    await ready();
  };
  const select = (id, value) =>
    change(() => page.locator(`#${id}`).selectOption(value));
  const setCount = (count) =>
    change(async () => {
      await page.locator("#ants-vehicle-count").fill(String(count));
      await page.locator("#ants-vehicle-count").press("Tab");
    });
  await ready();
  console.log("Vehicle browser check: initial scene ready");
  await page.locator("#tab-controller").click();
  for (const [id, value] of [
    ["walkers", "8"],
    ["horizon", "2"],
    ["frames", "6"],
  ]) {
    await page.locator(`#${id}`).fill(value);
    await page.locator(`#${id}`).press("Tab");
    await ready();
  }
  await page.locator("#tab-setup").click();
  const fleetWorld = ({ bodies, description, ...rest }) => rest;
  for (const environment of [
    "harvest",
    "ants",
    "tandem",
    "mining",
    "rocket",
    "racing",
  ]) {
    await select("scenario", environment);
    assert.equal(await page.locator("#ants-controls").isVisible(), true);
    assert.deepEqual(
      await page
        .locator("#ants-vehicle-type option:not([disabled])")
        .evaluateAll((options) => options.map((o) => o.value)),
      ["rocket", "drone", "kart", "harvester"],
    );
    const before = await latest();
    const oldBodies = resolveBodies(before);
    for (const type of ["drone", "rocket", "kart", "harvester"]) {
      await select("ants-vehicle-type", type);
      const scene = await latest();
      assert.equal(await page.locator("#scenario").inputValue(), environment);
      assert.equal(await page.locator("#tick").textContent(), "TICK 000000");
      assert.equal(
        await page.locator("#run").textContent(),
        "Run",
      );
      assert.deepEqual(fleetWorld(scene), fleetWorld(before));
      const bodies = resolveBodies(scene);
      for (const [i, body] of bodies.entries()) {
        if (oldBodies[i].controlled) {
          assert.equal(body.visual.model, type);
          assert.deepEqual(body.position, oldBodies[i].position);
        } else assert.deepEqual(body, oldBodies[i]);
      }
      const count = bodies.filter((b) => b.controlled).length;
      assert.match(
        await page.locator("#footer-stats").textContent(),
        new RegExp(`${count * (type === "rocket" ? 2 : 3)} ACTION DIMENSIONS`),
      );
      await page.evaluate(() => {
        vehicleRenderer.renderer.render(
          vehicleRenderer.world,
          vehicleRenderer.camera,
        );
      });
    }
    console.log(
      `${environment}: all four types preserve setup and reset paused`,
    );
  }
  // Racing has one remembered fleet choice across every track.
  const tracks = await page
    .locator("#track option")
    .evaluateAll((options) => options.map((o) => o.value));
  for (const track of tracks) {
    await select("track", track);
    assert.equal(
      await page.locator("#ants-vehicle-type").inputValue(),
      "harvester",
    );
    console.log(`${track}: Racing selection retained`);
  }
  await select("scenario", "ants");
  assert.equal(
    await page.locator("#ants-vehicle-type").inputValue(),
    "harvester",
  );
  await setCount(3);
  await select("ants-vehicle-type", "rocket");
  await page.locator("#world-physics").evaluate(e => e.open = true);
  await change(() => page.locator("#flight-mode").uncheck());
  await select("ants-vehicle-type", "drone");
  assert.equal((await latest()).environment.flight, false);
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000006",
  );
  await select("ants-vehicle-type", "kart");
  assert.equal(await page.locator("#tick").textContent(), "TICK 000000");
  await select("scenario", "mining");
  assert.equal(
    await page.locator("#ants-vehicle-type").inputValue(),
    "harvester",
  );
  await select("scenario", "ants");
  assert.equal(await page.locator("#ants-vehicle-type").inputValue(), "kart");
  assert.equal(await page.locator("#ants-vehicle-count").inputValue(), "3");

  // Import a mixed scene with edited geometry/settings; loading must not rewrite it.
  const imported = await latest();
  imported.bodies[0] = { agent_type: "drone", position: [10, 10], angle: 0.4 };
  imported.environment = { flight: false };
  imported.rewards = { ...(imported.rewards || {}), movement: 2 };
  await openFiles(page);
  await change(async () => {
    const choosing = page.waitForEvent("filechooser");
    await page.locator("#import-scene").click();
    await (
      await choosing
    ).setFiles({
      name: "mixed.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(imported)),
    });
  });
  assert.deepEqual(await latest(), imported);
  assert.equal(await page.locator("#ants-vehicle-type").inputValue(), "");
  assert.equal(
    // isDisabled() retargets options inside a label to the enabled select.
    await page
      .locator('#ants-vehicle-type option[value=""]')
      .evaluate((option) => option.disabled),
    true,
  );
  await closeFiles(page);
  await select("ants-vehicle-type", "harvester");
  const converted = await latest();
  assert.deepEqual(converted.bodies[0].position, [10, 10]);
  assert.deepEqual(converted.rewards, imported.rewards);
  assert.equal(converted.environment.flight, false);
  await openFiles(page);
  const downloading = page.waitForEvent("download");
  await page.locator("#export-scene").click();
  const chunks = [];
  for await (const chunk of await (await downloading).createReadStream())
    chunks.push(chunk);
  assert.deepEqual(JSON.parse(Buffer.concat(chunks).toString()), converted);
  assert.deepEqual(errors, []);
  console.log(
    "Vehicle selection, session memory, flight overrides, imports and exports passed",
  );
} finally {
  await browser.close();
}
