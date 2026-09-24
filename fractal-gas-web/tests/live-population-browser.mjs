import assert from "node:assert/strict";
import { chromium } from "playwright";
const base =
  process.env.LIVE_POPULATION_TEST_URL || "http://127.0.0.1:8097/web/";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
const errors = [];
const deadline = setTimeout(() => {
  console.error("Browser acceptance timeout", errors);
  process.exit(2);
}, 300000);
try {
  const context = await browser.newContext({
    viewport: { width: 960, height: 640 },
    serviceWorkers: "block",
  });
  const opt = await context.newPage();
  opt.on("pageerror", (e) => errors.push(`optimization: ${e.message}`));
  await opt.route("**/optimization/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text()) +
        "\nwindow.liveTest={createSession,step,get:()=>({config,frames:recording?.frames.map(f=>Array.from(f)),metadata:recording?.metadata,busy,running})};",
    });
  });
  await opt.goto(new URL("optimization/", base).href);
  await opt.waitForFunction(() => window.optimizationReady, null, {
    timeout: 60000,
  });
  await opt.evaluate(async () => {
    document.getElementById("record-history").checked = true;
    await liveTest.createSession({
      algorithm: "wave",
      benchmark: "quadratic",
      dimensions: 2,
      walkers: 8,
      max_walkers: 32,
      elites: 1,
      periodic: true,
      seed: 7,
    });
    await liveTest.step();
  });
  const optBefore = await opt.evaluate(() => liveTest.get().frames.at(-1));
  await opt.locator("#active-walkers").fill("4");
  await opt.locator("#live-removal-policy").selectOption("cumulative_reward");
  await opt.locator("#apply-population").click();
  await opt.waitForFunction(() => liveTest.get().frames.at(-1)[1] === 4);
  const optAfter = await opt.evaluate(() => liveTest.get().frames.at(-1));
  assert.equal(optAfter[4], optBefore[4]);
  assert.equal(optAfter[5], optBefore[5]);
  await opt.locator("#active-walkers").fill("20");
  await opt.locator("#apply-population").click();
  await opt.waitForFunction(() => liveTest.get().frames.at(-1)[1] === 20);
  await opt.evaluate(() => liveTest.step());
  assert.equal(
    await opt.evaluate(() => liveTest.get().frames.at(-1)[4]),
    optBefore[4] + 1,
  );
  await opt.locator("#run").click();
  await opt.waitForFunction(() => liveTest.get().frames.at(-1)[4] >= 4);
  await opt.locator("#active-walkers").fill("12");
  await opt.locator("#apply-population").click();
  await opt.waitForFunction(() => liveTest.get().frames.at(-1)[1] === 12);
  await opt.locator("#pause").click();
  await opt.waitForFunction(() => !liveTest.get().busy);
  console.log(
    "Optimization UI: paused and running resizing preserves progress",
  );
  await opt.close();

  const control = await context.newPage();
  control.on("pageerror", (e) => {
    errors.push(`control: ${e.message}`);
    console.error(e.message);
  });
  control.on("console", (msg) => {
    if (msg.type() === "error") console.error("control console:", msg.text());
  });
  console.log("Loading control UI");
  await control.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text()) +
        "\nwindow.liveTest={workspace,loadScene,settings,renderer,get:()=>({ready,currentState:currentState&&Array.from(currentState),recordingId:replay.recording?.id,wave:!!lastDiagnostics?.wave}),worker:()=>worker};",
    });
  });
  await control.goto(new URL("lab/", base).href);
  await control.waitForFunction(() => window.liveTest?.get().ready, null, {
    timeout: 60000,
  });
  if (await control.locator("#chooser-explore").isVisible())
    await control.locator("#chooser-explore").click();
  console.log("Control initial ready");
  await control.evaluate(async () => {
    await liveTest.loadScene(
      {
        name: "Population test",
        size: [100, 100],
        bodies: [{ controlled: true, position: [50, 50] }],
      },
      false,
      {
        clock: "realtime",
        settings: {
          ...liveTest.settings(),
          algorithm: "fmc",
          walkers: 8,
          max_walkers: 32,
          horizon: 3,
          frames: 1,
          elites: 1,
          threads: 2,
          recording: 2,
        },
      },
    );
  });
  await control.waitForFunction(
    () => liveTest.get().ready && liveTest.settings().walkers === 8,
    null,
    { timeout: 60000 },
  );
  console.log("Control fixture ready");
  const controlBefore = await control.evaluate(() => liveTest.get());
  await control.locator("#walkers").evaluate((el) => {
    el.value = "4";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await control.waitForFunction(() =>
    document
      .getElementById("population-status")
      .textContent.startsWith("4 active"),
  );
  const controlAfter = await control.evaluate(() => liveTest.get());
  assert.deepEqual(controlAfter.currentState, controlBefore.currentState);
  assert.equal(controlAfter.recordingId, controlBefore.recordingId);
  assert.equal(await control.evaluate(() => liveTest.workspace.dirty), false);
  // Inspect-mode standalone Wave and its packed cloud must also resize live.
  await control.locator("#wave").evaluate((el) => el.click());
  await control.waitForFunction(() => liveTest.get().wave);
  await control.locator("#walkers").evaluate((el) => {
    el.value = "16";
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await control.waitForFunction(() =>
    document
      .getElementById("population-status")
      .textContent.startsWith("16 active"),
  );
  // Software WebGL can dominate headless event processing during continuous runs.
  await control.evaluate(()=>cancelAnimationFrame(liveTest.renderer.frame));
  console.log("Control paused checks done; starting execution");
  const runningBefore = await control.evaluate(()=>liveTest.get().currentState);
  await control.evaluate(()=>document.getElementById("run").click());
  await control.waitForFunction(before=>JSON.stringify(liveTest.get().currentState)!==JSON.stringify(before),runningBefore);
  await control.evaluate(()=>{const el=document.getElementById("walkers");el.value="12";el.dispatchEvent(new Event("change",{bubbles:true}));});
  await control.waitForFunction(()=>document.getElementById("population-status").textContent.startsWith("12 active"));
  await control.evaluate(()=>document.getElementById("run").click());
  assert.equal(await control.evaluate(()=>liveTest.get().recordingId),controlBefore.recordingId);
  console.log("Control UI: paused, standalone Wave, and running resizing preserve the run");
  await control.close();

  console.log("Loading arcade UI");
  const arcade = await context.newPage();
  arcade.on("pageerror", (e) => errors.push(`arcade: ${e.message}`));
  await arcade.route("**/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text()) +
        "\nwindow.liveTest={initRun,get:()=>({initialized,running})};",
    });
  });
  await arcade.goto(new URL("arcade.html", base).href);
  console.log("Arcade page loaded");
  await arcade.waitForFunction(
    () => !document.getElementById("btn-start").disabled,
    null,
    { timeout: 60000 },
  );
  console.log("Arcade initial ready");
  await arcade.evaluate(() => {
    document.getElementById("param-n").value = "8";
    document.getElementById("wave-max-walkers").value = "32";
    document.getElementById("param-dt-min").value = "1";
    document.getElementById("param-dt-max").value = "1";
    liveTest.initRun();
  });
  await arcade.waitForFunction(
    () =>
      document
        .getElementById("population-status")
        .textContent.startsWith("8 active / 32"),
    null,
    { timeout: 60000 },
  );
  await arcade.locator("#btn-start").evaluate(el=>el.click());
  await arcade.waitForFunction(
    () => Number(document.getElementById("stat-iteration").textContent) >= 3,
  );
  await arcade.locator("#btn-pause").evaluate(el=>el.click());
  await arcade.waitForTimeout(100);
  const arcadeBefore = await arcade.locator("#stat-iteration").textContent();
  await arcade.locator("#param-n").fill("4");
  await arcade.locator("#param-n").press("Tab");
  await arcade.waitForFunction(() =>
    document
      .getElementById("population-status")
      .textContent.startsWith("4 active"),
  );
  assert.equal(
    await arcade.locator("#stat-iteration").textContent(),
    arcadeBefore,
  );
  await arcade.locator("#param-n").fill("20");
  await arcade.locator("#param-n").press("Tab");
  await arcade.waitForFunction(() =>
    document
      .getElementById("population-status")
      .textContent.startsWith("20 active"),
  );
  await arcade.locator("#btn-start").evaluate(el=>el.click());
  await arcade.waitForFunction(
    (n) => Number(document.getElementById("stat-iteration").textContent) > n,
    Number(arcadeBefore),
  );
  await arcade.locator("#btn-pause").evaluate(el=>el.click());
  await arcade.locator("#btn-start").evaluate(el=>el.click());
  await arcade.locator("#param-n").evaluate(el=>{el.value="12";el.dispatchEvent(new Event("change",{bubbles:true}));});
  await arcade.waitForFunction(()=>document.getElementById("population-status").textContent.startsWith("12 active"));
  await arcade.locator("#btn-pause").evaluate(el=>el.click());
  console.log("Arcade UI: paused and running resizing preserve progress");
  assert.deepEqual(errors, []);
} finally {
  clearTimeout(deadline);
  await browser.close();
}
