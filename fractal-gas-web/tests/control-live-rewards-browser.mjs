import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({
    serviceWorkers: "block",
    viewport: { width: 1536, height: 2400 },
  });
  page.setDefaultTimeout(20000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/lab/main.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: `${(await response.text()).replace("installStyleControls();", "cancelAnimationFrame(renderer.frame); installStyleControls();")}
      window.rewardTest = { renderer, editor, replay, loadScene, stop, applyRewardUpdate,
        get worker() { return worker }, get ready() { return ready },
        get scene() { return currentScene }, get frame() { return lastLiveFrame },
        get pending() { return rewardChangePending } };`,
    });
  });
  await page.addInitScript(() => localStorage.setItem("lab.workspace.onboarded", "true"));
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8099/lab/");
  await page.waitForFunction(() => window.rewardTest?.ready, null, {
    timeout: 60000,
  });
  await page.evaluate(async () => {
    for (const [id, value] of Object.entries({
      walkers: 16,
      horizon: 3,
      frames: 1,
      threads: 1,
    }))
      document.getElementById(id).value = value;
    document.getElementById("persistent-recording").checked = true;
    document.getElementById("clock").value = "realtime";
    await rewardTest.loadScene({
      version: 1,
      name: "Live rewards regression",
      size: [100, 100],
      physics: { dt: 0.1, substeps: 2 },
      bodies: [
        { position: [50, 50], velocity: [2, -3], drag: 0, controlled: true },
      ],
    });
  });
  await page.waitForFunction(
    () => rewardTest.ready && rewardTest.frame?.tick === 0,
  );
  console.log("Test world ready");
  await page.evaluate(() => {
    window.originalWorker = rewardTest.worker;
    window.originalRecording = rewardTest.replay.recording;
    window.before = Array.from(new Uint8Array(rewardTest.frame.state.buffer));
    window.camera = [
      rewardTest.renderer.zoom,
      ...rewardTest.renderer.viewCenter,
    ];
    rewardTest.worker.postMessage({
      type: "manual",
      action: new Float32Array(rewardTest.frame.action.length),
      frames: 3,
    });
  });
  await page.waitForFunction(() => rewardTest.frame.tick === 3);
  await page.evaluate(() => {
    window.before = Array.from(new Uint8Array(rewardTest.frame.state.buffer));
    window.lengthBefore = rewardTest.replay.recording.length;
  });
  await page.locator("#tab-rewards").click();
  await page.locator("#reward-settings > summary").click();
  const apply = async (weight) => {
    await page.locator("#lab-reward-distance_squared").fill(String(weight));
    await page
      .locator("#reward-terms")
      .getByRole("button", { name: "Apply to current run", exact: true })
      .click();
    await page.waitForFunction(
      (w) =>
        !rewardTest.pending && rewardTest.scene.rewards?.distance_squared === w,
      weight,
      { timeout: 60000 },
    );
  };
  await apply(2);
  console.log("Paused reward update applied");
  assert.deepEqual(
    await page.evaluate(() => ({
      sameWorker: originalWorker === rewardTest.worker,
      sameRecording: originalRecording === rewardTest.replay.recording,
      state: Array.from(new Uint8Array(rewardTest.frame.state.buffer)),
      camera: [rewardTest.renderer.zoom, ...rewardTest.renderer.viewCenter],
      running: rewardTest.frame.running,
      changes: rewardTest.replay.recording.rewardChanges.length,
      appended: rewardTest.replay.recording.length - lengthBefore,
    })),
    {
      sameWorker: true,
      sameRecording: true,
      state: await page.evaluate(() => before),
      camera: await page.evaluate(() => camera),
      running: false,
      changes: 1,
      appended: 1,
    },
  );
  await page.evaluate(() =>
    rewardTest.worker.postMessage({ type: "run", value: true }),
  );
  await page.waitForFunction(() => rewardTest.frame.tick > 3);
  await apply(3);
  console.log("Running reward update applied");
  const runningTick = await page.evaluate(() => rewardTest.frame.tick);
  await page.waitForFunction(
    (tick) => rewardTest.frame.tick > tick,
    runningTick,
  );
  await page.evaluate(() => rewardTest.stop());
  await page.waitForFunction(() => !rewardTest.frame.running);
  const failure = await page.evaluate(async () => {
    const before = Array.from(new Uint8Array(rewardTest.frame.state.buffer));
    await new Promise((resolve) => {
      const listener = ({ data }) => {
        if (data.type === "rewards-error") {
          rewardTest.worker.removeEventListener("message", listener);
          resolve();
        }
      };
      rewardTest.worker.addEventListener("message", listener);
      rewardTest.applyRewardUpdate(
        { ...rewardTest.scene, rewards: { distance_squared: -1 } },
        {},
      );
    });
    return {
      before,
      after: Array.from(new Uint8Array(rewardTest.frame.state.buffer)),
      weight: rewardTest.scene.rewards.distance_squared,
    };
  });
  assert.deepEqual(failure.before, failure.after);
  assert.equal(failure.weight, 3);
  const archive = await page.evaluate(async () => {
    const { importStoredFile } = await import(
      "/lab/storage/recording-store.js"
    );
    const { deleteRun } = await import("/lab/storage/database.js");
    const r = rewardTest.replay.recording;
    const imported = await importStoredFile(await r.exportFile());
    const last = imported.rewardConfiguration();
    const result = {
      length: imported.length,
      expected: r.length,
      changes: imported.rewardChanges.length,
      weight: last.scene.rewards.distance_squared,
      originalWeight:
        imported.rewardConfiguration(0).scene.rewards?.distance_squared,
    };
    const cfg = imported.rewardConfiguration(0);
    rewardTest.applyRewardUpdate(
      cfg.scene,
      cfg.settings,
      await imported.getRows(0),
      cfg.root,
    );
    await deleteRun(imported.id);
    return result;
  });
  assert.equal(archive.length, archive.expected);
  assert.equal(archive.changes, 2);
  assert.equal(archive.weight, 3);
  assert.equal(archive.originalWeight, undefined);
  await page.waitForFunction(
    () => !rewardTest.pending && rewardTest.frame.tick === 0,
  );
  assert.equal(
    await page.evaluate(() => rewardTest.replay.recording.rewardChanges.length),
    3,
  );
  // Exercise the worker protocol independently of the UI for both clocks and
  // queued Wave Jump trajectories, including updates during an active search.
  for (const algorithm of ["fmc", "wave-jump"]) {
    for (const mode of ["reproducible", "realtime"]) {
      const result = await page.evaluate(
        async ({ algorithm, mode }) => {
          const worker = new Worker("/lab/simulation-worker.js", {
            type: "module",
          });
          const scene = {
            version: 1,
            size: [100, 100],
            physics: { dt: 0.01, substeps: 2 },
            bodies: [
              { position: [50, 50], velocity: [1, 0], controlled: true },
            ],
          };
          return await new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
              worker.terminate();
              reject(new Error(`${algorithm}/${mode} timed out`));
            }, 30000);
            let requested = false,
              boundary,
              latest,
              after = false,
              decisions = 0;
            worker.onerror = (event) => {
              clearTimeout(timer);
              worker.terminate();
              reject(new Error(event.message));
            };
            worker.onmessage = ({ data }) => {
              if (["error", "rewards-error"].includes(data.type)) {
                clearTimeout(timer);
                worker.terminate();
                reject(new Error(data.message));
                return;
              }
              if (data.type === "ready")
                worker.postMessage({ type: "run", value: true });
              if (
                data.type === "diagnostics" &&
                after &&
                data.revision !== 21
              ) {
                clearTimeout(timer);
                worker.terminate();
                reject(new Error("Stale plan accepted"));
                return;
              }
              if (data.type === "rewards-updated") {
                boundary = data.tick;
                decisions = data.decisions;
                const expected = new Uint8Array(latest.state.buffer).subarray(
                  0,
                  data.root.length - 32,
                );
                if (
                  !data.root.subarray(32).every((v, i) => v === expected[i])
                ) {
                  clearTimeout(timer);
                  worker.terminate();
                  reject(new Error("Update changed physical state"));
                  return;
                }
                after = true;
              }
              if (data.type === "frame") {
                latest = data;
                if (data.tick > 0 && !requested) {
                  requested = true;
                  worker.postMessage({
                    type: "update-rewards",
                    scene: { ...scene, rewards: { distance_squared: 4 } },
                    coefficients: { reward_coef: 2, distance_coef: 0.5 },
                  });
                }
                if (
                  after &&
                  data.tick > boundary &&
                  data.decisions > decisions
                ) {
                  clearTimeout(timer);
                  worker.terminate();
                  resolve({ progressed: true, running: data.running });
                }
              }
            };
            worker.postMessage({
              type: "init",
              scene,
              settings: {
                algorithm,
                walkers: 16,
                horizon: 3,
                frames: 2,
                distance_coef: 1,
                reward_coef: 1,
                noise: 0.2,
                elites: 1,
                inertial: true,
                recording: 1,
              },
              revision: 20,
              seed: 7,
              mode,
              threads: 1,
            });
          });
        },
        { algorithm, mode },
      );
      assert.deepEqual(result, { progressed: true, running: true });
      console.log(`${algorithm}/${mode}: no state reset or stale plan`);
    }
  }
  assert.deepEqual(errors, []);
  console.log(
    "Live reward UI: paused/running updates, preserved world/camera/history, rollback, archive and historical continuation passed",
  );
} finally {
  await browser.close();
}
