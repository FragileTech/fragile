import { prepareWorkspace, applyDraft } from "./helpers/workspace-ui.mjs";
import { chromium } from "playwright";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await prepareWorkspace(page);
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await page.locator("#tab-controller").click();
  await page.evaluate(() => {
    document.getElementById("walkers").value = "12";
    document.getElementById("horizon").value = "5";
    document.getElementById("frames").value = "3";
    document.getElementById("elites").value = "0";
    document.getElementById("threads").value = "1";
  });
  await page.locator("#algorithm").selectOption("wave-jump");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);

  await page.locator("details:has(#algorithm-settings) > summary").click();
  const toggle = page.locator("#planner-consensus_prefix");
  assert.equal(await toggle.isChecked(), true);
  await toggle.uncheck();
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000015",
  );
  assert.match(await page.locator("#latency").textContent(), /PATH REWARD/);
  assert.equal(await page.locator("#trajectory-progress").isVisible(), true);
  await page.locator("#step").click();
  await page.waitForFunction(
    () => document.getElementById("tick").textContent === "TICK 000030",
  );
  assert.equal(await toggle.isChecked(), false);
  await toggle.check();
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  assert.equal(await toggle.isChecked(), true);
  assert.equal(await page.locator("#planner-max_horizon").inputValue(), "0");
  await page.locator("#planner-max_horizon").fill("1");
  await page.locator("#planner-max_horizon").press("Tab");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await page.locator("#step").click();
  await page.waitForFunction(() =>
    document
      .getElementById("status")
      .textContent.includes("Maximum search horizon"),
  );
  await page.locator("#planner-max_horizon").fill("0");
  await page.locator("#planner-max_horizon").press("Tab");
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  await toggle.uncheck();
  await applyDraft(page);
  await page.waitForFunction(() => !document.getElementById("run").disabled);
  // Worker timing checks need an idle event loop, without the live WebGL renderer.
  const workerPage = new URL("worker-tests.html", page.url()).href;
  await page.route(workerPage, (route) =>
    route.fulfill({
      contentType: "text/html",
      body: "<!doctype html><title>Worker tests</title>",
    }),
  );
  await page.goto(workerPage);
  const results = await page.evaluate(async (consensus) => {
    const scene = {
      size: [1000, 1000],
      bodies: [{ controlled: true, position: [500, 500], velocity: [2, 1] }],
    };
    const settings = {
      algorithm: "wave-jump",
      walkers: consensus ? 1 : 12,
      consensus_prefix: consensus,
      max_horizon: 24,
      horizon: 12,
      frames: 11,
      elites: 0,
      recording: 0,
    };
    const make = async (
      mode = "reproducible",
      worldScene = scene,
      plannerSettings = settings,
    ) => {
      const worker = new Worker(
        new URL("./simulation-worker.js", location.href),
        { type: "module" },
      );
      const messages = [],
        waiters = [];
      let failure;
      worker.onmessage = ({ data }) => {
        messages.push(data);
        if (data.type === "error") failure = new Error(data.message);
        for (const wake of [...waiters]) wake();
      };
      worker.onerror = (e) => {
        failure = new Error(e.message);
        for (const wake of [...waiters]) wake();
      };
      const wait = (predicate, start = 0) =>
        new Promise((resolve, reject) => {
          const timer = setTimeout(
            () => finish(new Error("Worker timeout")),
            20000,
          );
          const finish = (error, value) => {
            clearTimeout(timer);
            const i = waiters.indexOf(wake);
            if (i >= 0) waiters.splice(i, 1);
            error ? reject(error) : resolve(value);
          };
          const wake = () => {
            if (failure) return finish(failure);
            const value = messages.slice(start).find(predicate);
            if (value) finish(null, value);
          };
          waiters.push(wake);
          wake();
        });
      worker.postMessage({
        type: "init",
        scene: worldScene,
        settings: plannerSettings,
        mode,
        seed: 7,
        revision: 1,
        threads: 1,
      });
      await wait((d) => d.type === "ready");
      return { worker, messages, wait, send: (d) => worker.postMessage(d) };
    };
    const snapshot = async (host) => {
      const start = host.messages.length;
      host.send({ type: "snapshot" });
      return Array.from(
        (await host.wait((d) => d.type === "snapshot", start)).bytes,
      );
    };
    const finishStep = async (host) => {
      const start = host.messages.length;
      host.send({ type: "step" });
      return host.wait(
        (d) => d.type === "frame" && d.tick > 0 && !d.trajectoryProgress,
        start,
      );
    };
    const doomed = {
      size: [1000, 1000],
      physics: { lethal_walls: true },
      bodies: [
        { controlled: true, position: [100, 500], velocity: [300, 0], drag: 0 },
      ],
    };
    const fallback = await make("realtime", doomed, {
      ...settings,
      frames: 60,
    });
    let fallbackResult;
    try {
      fallback.send({ type: "run", value: true });
      await fallback.wait(
        (d) =>
          d.type === "frame" &&
          d.tick > 0 &&
          d.trajectoryProgress?.remaining < 60,
      );
      fallback.send({ type: "run", value: false });
      let start = fallback.messages.length;
      fallback.send({ type: "checkpoint" });
      const saved = await fallback.wait((d) => d.type === "checkpoint", start);
      const restored = await make("realtime", doomed, {
        ...settings,
        frames: 60,
      });
      try {
        start = restored.messages.length;
        restored.send({ ...saved, type: "restore-checkpoint" });
        await restored.wait((d) => d.type === "checkpoint-restored", start);
        const completed = await finishStep(fallback);
        await finishStep(restored);
        const originalWorld = await snapshot(fallback),
          restoredWorld = await snapshot(restored);
        const next = await finishStep(restored);
        fallbackResult = {
          completedTick: completed.tick,
          nextTick: next.tick,
          decisions: next.decisions,
          originalWorld,
          restoredWorld,
          edges: saved.execution.trajectory.length,
          deadRatio: fallback.messages.find((d) => d.type === "diagnostics")
            .metrics[9],
        };
      } finally {
        restored.worker.terminate();
      }
    } finally {
      fallback.worker.terminate();
    }
    const a = await make("realtime");
    try {
      a.send({ type: "run", value: true });
      await a.wait(
        (d) =>
          d.type === "frame" &&
          d.trajectoryProgress?.remaining < 11 &&
          d.tick > 0,
      );
      a.send({ type: "run", value: false });
      let start = a.messages.length;
      a.send({ type: "checkpoint" });
      const checkpoint = await a.wait((d) => d.type === "checkpoint", start);
      const paused = await snapshot(a);
      await new Promise((resolve) => setTimeout(resolve, 80));
      const stillPaused = await snapshot(a);
      const b = await make("realtime");
      try {
        start = b.messages.length;
        b.send({ ...checkpoint, type: "restore-checkpoint" });
        await b.wait((d) => d.type === "checkpoint-restored", start);
        const frameA = await finishStep(a),
          frameB = await finishStep(b);
        const finalA = await snapshot(a),
          finalB = await snapshot(b);
        // Step again must run exactly one fresh trajectory.
        const second = await finishStep(b);
        // An external world action invalidates a restored trajectory.
        start = b.messages.length;
        b.send({ ...checkpoint, type: "restore-checkpoint" });
        await b.wait((d) => d.type === "checkpoint-restored", start);
        start = b.messages.length;
        b.send({ type: "manual", action: new Float32Array(2), frames: 1 });
        const manual = await b.wait(
          (d) => d.type === "frame" && !d.trajectoryProgress,
          start,
        );
        const afterManual = await finishStep(b);
        // Saving an in-flight search also resumes deterministically.
        const c = await make();
        try {
          c.send({ type: "run", value: true });
          c.send({ type: "checkpoint" });
          const searchSaved = await c.wait((d) => d.type === "checkpoint");
          const frozen = await snapshot(c);
          const d = await make();
          try {
            start = d.messages.length;
            d.send({ ...searchSaved, type: "restore-checkpoint" });
            await d.wait((m) => m.type === "checkpoint-restored", start);
            await finishStep(c);
            await finishStep(d);
            return {
              paused,
              stillPaused,
              fallbackResult,
              checkpointHasExecution: !!checkpoint.execution,
              finalA,
              finalB,
              tick: frameA.tick,
              restoredTick: frameB.tick,
              decisions: frameA.decisions,
              secondTick: second.tick,
              secondDecisions: second.decisions,
              manualTick: manual.tick,
              afterManualTick: afterManual.tick,
              afterManualDecisions: afterManual.decisions,
              searchFinalA: await snapshot(c),
              searchFinalB: await snapshot(d),
              frozenTick: new DataView(
                Uint8Array.from(frozen).buffer,
              ).getUint32(32, true),
              initialSearchTicks: a.messages
                .filter((m) => m.type === "frame" && m.decisions === 0)
                .map((m) => m.tick),
            };
          } finally {
            d.worker.terminate();
          }
        } finally {
          c.worker.terminate();
        }
      } finally {
        b.worker.terminate();
      }
    } finally {
      a.worker.terminate();
    }
  }, process.env.CONSENSUS === "1");
  assert.equal(results.fallbackResult.deadRatio, 1);
  assert.equal(results.fallbackResult.edges, 1);
  assert.equal(results.fallbackResult.completedTick, 60);
  assert.equal(results.fallbackResult.nextTick, 120);
  assert.equal(results.fallbackResult.decisions, 2);
  assert.deepEqual(
    results.fallbackResult.originalWorld,
    results.fallbackResult.restoredWorld,
  );
  assert.equal(results.checkpointHasExecution, true);
  assert.deepEqual(results.paused, results.stillPaused);
  assert.deepEqual(results.finalA, results.finalB);
  assert.equal(results.tick, 132);
  assert.equal(results.restoredTick, 132);
  assert.equal(results.decisions, 1);
  assert.equal(results.secondTick, 264);
  assert.equal(results.secondDecisions, 2);
  assert.equal(results.afterManualTick, results.manualTick + 132);
  assert.equal(results.afterManualDecisions, 2);
  assert.deepEqual(results.searchFinalA, results.searchFinalB);
  assert.equal(results.frozenTick, 0);
  assert.ok(results.initialSearchTicks.every((tick) => tick === 0));
  console.log(
    "Wave Jump worker: trajectory execution, real-time search freeze, Pause, Step, and both checkpoint phases passed.",
  );
} finally {
  await browser.close();
}
