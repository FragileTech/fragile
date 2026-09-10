import assert from "node:assert/strict";
import { chromium } from "playwright";
import {
  prepareWorkspace,
  applyDraft,
  openFiles,
  closeFiles,
} from "./helpers/workspace-ui.mjs";

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    serviceWorkers: "block",
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await prepareWorkspace(page);
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text())
          .replace(
            "installStyleControls();",
            "cancelAnimationFrame(renderer.frame); installStyleControls();",
          )
          .replace(
            "loadPresets();",
            '$("threads").value = "1"; loadPresets();',
          ) +
        `
      window.wallTest = { loadScene, get scene() { return currentScene },
        get ready() { return ready }, get frame() { return lastLiveFrame },
        get worker() { return worker }, get pending() { return rewardChangePending } };`,
    });
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8097/lab/");
  await page.waitForFunction(() => window.wallTest?.ready);
  const setup = async () => {
    await page.locator("#tab-setup").click();
    await page.locator("#world-physics").evaluate((e) => (e.open = true));
  };
  for (const name of [
    "harvest",
    "mining",
    "ants",
    "tandem",
    "rocket",
    "racing",
  ]) {
    console.log(`Wall settings: ${name}`);
    await page.locator("#scenario").selectOption(name);
    await applyDraft(page);
    await page.waitForFunction(() => wallTest.ready);
    await setup();
    assert.equal(await page.locator("#lethal-walls").isChecked(), false);
    await page.locator("#tab-rewards").click();
    await page.locator("#reward-settings").evaluate((e) => (e.open = true));
    assert.equal(
      await page.locator("#lab-reward-wall_collision").isVisible(),
      true,
    );
    assert.equal(
      await page.locator("#lab-reward-wall_collision").inputValue(),
      "100",
    );
    await setup();
    await page.locator("#lethal-walls").check();
    assert.equal(
      await page.evaluate(() => wallTest.scene.physics.lethal_walls),
      false,
    );
    assert.equal(
      await page.locator("#apply-configuration").textContent(),
      "Apply and restart",
    );
    await applyDraft(page);
    assert.equal(
      await page.evaluate(() => wallTest.scene.physics.lethal_walls),
      true,
    );
    assert.equal(await page.evaluate(() => wallTest.frame.tick), 0);
    await setup();
    await page.locator("#lethal-walls").uncheck();
    await applyDraft(page);
    assert.equal(
      await page.evaluate(() => wallTest.scene.physics.lethal_walls),
      false,
    );
  }

  // Applied wall settings belong to their preset, not the next environment.
  await page.locator("#scenario").selectOption("mining");
  await applyDraft(page);
  await page.locator("#tab-rewards").click();
  await page.locator("#reward-settings").evaluate((e) => (e.open = true));
  await page.locator("#lab-reward-wall_collision").fill("5");
  await page
    .locator("#reward-terms")
    .getByRole("button", { name: "Apply to current run", exact: true })
    .click();
  await page.waitForFunction(
    () => !wallTest.pending && wallTest.scene.rewards.wall_collision === 5,
  );
  await setup();
  await page.locator("#lethal-walls").check();
  await applyDraft(page);
  await page.locator("#scenario").selectOption("harvest");
  await applyDraft(page);
  assert.equal(
    await page.evaluate(() => wallTest.scene.rewards.wall_collision),
    100,
  );
  assert.equal(
    await page.evaluate(() => wallTest.scene.physics.lethal_walls),
    false,
  );
  await page.locator("#scenario").selectOption("mining");
  await applyDraft(page);
  assert.equal(
    await page.evaluate(() => wallTest.scene.rewards.wall_collision),
    5,
  );
  assert.equal(
    await page.evaluate(() => wallTest.scene.physics.lethal_walls),
    true,
  );

  await page.evaluate(async () => {
    await wallTest.loadScene({
      version: 1,
      name: "Wall test",
      task: "harvest",
      size: [20, 20],
      physics: { dt: 0.1, substeps: 4, lethal_walls: false },
      bodies: [{ controlled: true, position: [0.5, 10] }],
      rewards: {
        progress: 0,
        distance_squared: 0,
        catch: 0,
        wall_collision: 2,
      },
    });
  });
  await page.waitForFunction(() => wallTest.ready && wallTest.frame.tick === 0);
  const manual = async (frames) =>
    page.evaluate((frames) => {
      wallTest.worker.postMessage({
        type: "manual",
        action: new Float32Array(wallTest.frame.action.length),
        frames,
      });
    }, frames);
  await manual(3);
  await page.waitForFunction(() => wallTest.frame.tick === 3);
  const before = await page.evaluate(() =>
    Array.from(new Uint8Array(wallTest.frame.state.buffer)),
  );
  await page.locator("#tab-rewards").click();
  await page.locator("#reward-settings").evaluate((e) => (e.open = true));
  await page.locator("#lab-reward-wall_collision").fill("7");
  await page
    .locator("#reward-terms")
    .getByRole("button", { name: "Apply to current run", exact: true })
    .click();
  await page.waitForFunction(
    () => !wallTest.pending && wallTest.scene.rewards.wall_collision === 7,
  );
  assert.deepEqual(
    await page.evaluate(() =>
      Array.from(new Uint8Array(wallTest.frame.state.buffer)),
    ),
    before,
  );
  await setup();
  await page.locator("#lethal-walls").check();
  await applyDraft(page);
  assert.equal(await page.evaluate(() => wallTest.frame.tick), 0);
  await manual(10);
  await page.waitForFunction(() => wallTest.frame.tick === 1);
  assert.equal(
    await page.evaluate(() => new Uint32Array(wallTest.frame.state.buffer)[7]),
    1,
  );

  // Export through the real UI, then reload the portable scene configuration.
  await openFiles(page);
  const downloading = page.waitForEvent("download");
  await page.locator("#export-scene").click();
  const chunks = [];
  for await (const chunk of await (await downloading).createReadStream())
    chunks.push(chunk);
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  assert.equal(exported.physics.lethal_walls, true);
  assert.equal(exported.rewards.wall_collision, 7);
  await closeFiles(page);
  await page.evaluate(
    (scene) =>
      wallTest.loadScene({
        ...scene,
        name: "Import target",
        physics: { ...scene.physics, lethal_walls: false },
        rewards: { ...scene.rewards, wall_collision: 0 },
      }),
    exported,
  );
  await page.waitForFunction(() => wallTest.ready && wallTest.frame.tick === 0);
  await openFiles(page);
  const choosing = page.waitForEvent("filechooser");
  await page.locator("#import-scene").click();
  await (
    await choosing
  ).setFiles({
    name: "wall-test.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(exported)),
  });
  await closeFiles(page);
  await page.waitForFunction(
    () => !document.getElementById("pending-settings").hidden,
  );
  await applyDraft(page);
  await page.waitForFunction(() => wallTest.ready && wallTest.frame.tick === 0);
  await setup();
  assert.equal(await page.locator("#lethal-walls").isChecked(), true);
  assert.equal(
    await page.locator("#lab-reward-wall_collision").inputValue(),
    "7",
  );
  assert.deepEqual(errors, []);
  console.log(
    "All preset toggles, live penalty, wall death, and JSON round trip passed",
  );
} finally {
  await browser.close();
}
