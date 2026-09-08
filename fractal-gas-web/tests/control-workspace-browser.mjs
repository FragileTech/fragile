import assert from "node:assert/strict";
import { chromium } from "playwright";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    serviceWorkers: "block",
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text()) +
        `\nwindow.workspaceTest={workspace,transitions,loadScene,runSession,replay,renderer,editor,settings,getWorker:()=>worker,getScene:()=>currentScene,getReady:()=>ready};`,
    });
  });
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8089/lab/");
  await page.waitForFunction(() => window.workspaceTest?.getReady(), null, {
    timeout: 60000,
  });
  await page.locator("#chooser-explore").click();
  const tick = () => page.locator("#tick").innerText();
  assert.match(await tick(), /000000/);
  for (const [width, height] of [
    [1280, 720],
    [1440, 900],
    [390, 844],
  ]) {
    await page.setViewportSize({ width, height });
    const b = await page.evaluate(() => ({
      w: innerWidth,
      h: innerHeight,
      sw: document.documentElement.scrollWidth,
      sh: document.documentElement.scrollHeight,
      run: document.querySelector("#run").getBoundingClientRect().toJSON(),
      world: document
        .querySelector("#viewport")
        .getBoundingClientRect()
        .toJSON(),
      timeline: document
        .querySelector(".timeline-dock")
        .getBoundingClientRect()
        .toJSON(),
    }));
    assert.ok(b.sw <= width + 1, JSON.stringify(b));
    assert.ok(b.sh <= height + 1, JSON.stringify(b));
    assert.ok(b.run.bottom <= height);
    assert.ok(b.world.height >= 100, JSON.stringify(b));
    assert.ok(b.timeline.bottom <= height + 1, JSON.stringify(b));
  }
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.locator("#tab-controller").click();
  await page.locator("#walkers").fill("8");
  await page.locator("#walkers").press("Tab");
  await page.locator("#horizon").fill("2");
  await page.locator("#horizon").press("Tab");
  assert.equal(await page.evaluate(() => workspaceTest.settings().horizon), 32);
  assert.match(await tick(), /000000/);
  const oldId = await page.evaluate(() => workspaceTest.replay.recording.id);
  await page.locator("#apply-configuration").click();
  await page.waitForFunction(
    (old) =>
      workspaceTest.getReady() &&
      !workspaceTest.transitions.busy &&
      workspaceTest.replay.recording.id !== old,
    oldId,
    { timeout: 60000 },
  );
  assert.equal(await page.evaluate(() => workspaceTest.settings().horizon), 2);
  assert.equal(await page.evaluate(() => workspaceTest.workspace.dirty), false);
  const saved = await page.evaluate(async () =>
    (await (await import("./storage/database.js")).listRuns()).map((r) => r.id),
  );
  assert.ok(saved.includes(oldId));
  console.log(
    "Layout, onboarding, staged settings, durable replacement passed",
  );
  await page.evaluate(() =>
    workspaceTest.loadScene({
      name: "Continuous driving fixture",
      task: "navigation",
      size: [20, 20],
      bodies: [
        {
          controlled: true,
          position: [10, 10],
          velocity: [0.1, 0],
          drag: 0,
          actuator: { kind: "kart" },
        },
      ],
    }),
  );
  await page.locator("#mode-drive").click();
  await page.locator("#run").click();
  await page.waitForFunction(
    () => document.querySelector("#tick").textContent !== "TICK 000000",
  );
  const t1 = await tick();
  await page.waitForFunction(
    (previous) => document.querySelector("#tick").textContent !== previous,
    t1,
  );
  await page.locator("#run").click();
  await page.waitForTimeout(70);
  const paused = await tick();
  await page.waitForTimeout(100);
  assert.equal(await tick(), paused);
  await page.locator("#run").click();
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  await page.waitForTimeout(60);
  const blur = await tick();
  await page.waitForTimeout(80);
  assert.equal(await tick(), blur);
  await page.locator("#mode-inspect").click();
  console.log("Continuous driving and focus-loss pause passed");
  await page.locator("#tab-rewards").click();
  await page.locator("#reward-settings").evaluate((e) => (e.open = true));
  const beforeReward = await tick(),
    runId = await page.evaluate(() => workspaceTest.replay.recording.id);
  await page.locator("#lab-reward-delivery").fill("150");
  await page.locator("#apply-configuration").click();
  await page.waitForFunction(
    () => workspaceTest.getScene().rewards?.delivery === 150,
  );
  assert.equal(await tick(), beforeReward);
  assert.equal(
    await page.evaluate(() => workspaceTest.replay.recording.id),
    runId,
  );
  await page.locator("#motion-timeline").fill("1");
  await page.locator("#motion-resume").click();
  await page.waitForFunction(
    (old) =>
      !workspaceTest.transitions.busy &&
      workspaceTest.replay.recording.id !== old,
    runId,
    { timeout: 60000 },
  );
  assert.equal(
    await page.evaluate(() => workspaceTest.replay.recording.parent.run),
    runId,
  );
  assert.equal(
    await page.evaluate(() => workspaceTest.workspace.readOnly),
    false,
  );
  console.log("Live rewards and preserved replay branching passed");
  await page.locator("#tab-controller").click();
  const retained = await page.evaluate(() => workspaceTest.replay.recording.id);
  await page.evaluate(() => {
    window.originalSave = workspaceTest.runSession.save;
    workspaceTest.runSession.save = async () => {
      throw Error("Simulated quota exhaustion");
    };
  });
  await page.locator("#horizon").fill("3");
  await page.locator("#horizon").press("Tab");
  await page.locator("#apply-configuration").click();
  await page.locator("#save-cancel").click();
  await page.waitForFunction(() => !workspaceTest.transitions.busy);
  assert.equal(
    await page.evaluate(() => workspaceTest.replay.recording.id),
    retained,
  );
  assert.equal(await page.evaluate(() => workspaceTest.workspace.dirty), true);
  await page.evaluate(
    () => (workspaceTest.runSession.save = window.originalSave),
  );
  await page.locator("#discard-configuration").click();
  console.log("Failed-save recovery passed");
  await page.locator("#horizon").fill("4");
  await page.locator("#horizon").press("Tab");
  const failure = await page.evaluate(async () => {
    const { loadScene, getScene, replay, workspace } = workspaceTest;
    const id = replay.recording.id,
      draft = JSON.stringify(workspace.draft);
    const result = await loadScene({ ...getScene(), size: [-1, -1] });
    return {
      result,
      retained: replay.recording.id === id,
      draftRetained: JSON.stringify(workspace.draft) === draft,
    };
  });
  assert.deepEqual(failure, {
    result: false,
    retained: true,
    draftRetained: true,
  });
  await page.locator("#discard-configuration").click();
  console.log("Failed replacement preserves the run and draft");
  await page.locator("#experiments").click();
  await page.locator("#variant-b").selectOption("random");
  await page.locator("#benchmark-frames").fill("12");
  await page.locator("#compare-branches").click();
  await page.waitForFunction(
    () => document.querySelector("#comparison-section").hidden === false,
    null,
    { timeout: 30000 },
  );
  await page.locator("#comparison-play").click();
  await page.locator("#close-experiments").click();
  console.log("Structured comparisons passed");
  assert.deepEqual(errors, []);
  await context.close();
} finally {
  await browser.close();
}
