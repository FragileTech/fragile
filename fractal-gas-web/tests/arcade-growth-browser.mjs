// Resource UI, recovery after a refused budget, and Graph capacity growth.
import assert from "node:assert/strict";
import { chromium } from "playwright";
const startingN = Number(process.env.ARCADE_GRAPH_WALKERS || 48);
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await page.goto(
    process.env.ARCADE_TEST_URL || "http://127.0.0.1:8091/web/arcade.html",
  );
  await page.waitForFunction(
    () => !document.getElementById("btn-start").disabled,
  );
  await page.evaluate(() => {
    const n = document.getElementById("param-n");
    n.value = "2";
    n.dispatchEvent(new Event("change"));
    const workers = document.getElementById("param-workers");
    workers.value = "20";
    workers.dispatchEvent(new Event("change"));
    const memory = document.getElementById("param-memory");
    memory.value = "1";
    memory.dispatchEvent(new Event("change"));
  });
  await page.waitForFunction(
    () =>
      !document.getElementById("btn-start").disabled &&
      document
        .getElementById("resource-hint")
        .textContent.startsWith("2 workers"),
  );
  await page.reload();
  await page.waitForFunction(
    () => !document.getElementById("btn-start").disabled,
  );
  assert.equal(await page.locator("#param-workers").inputValue(), "20");
  assert.equal(await page.locator("#param-memory").inputValue(), "1");
  const result = await page.evaluate(async (startingN) => {
    const w = new Worker("worker.js", { type: "module" });
    const rom = await (await fetch("sonic.rom")).arrayBuffer();
    const params = {
      n: 1000,
      distCoef: 1,
      rewardCoef: 1,
      useCumulativeReward: true,
      dtMin: 1,
      dtMax: 2,
      nElite: 2,
      seed: 7,
      obsMode: 1,
      console: 2,
      game: 1,
      world: 0,
      stage: 0,
      algorithm: 1,
      maxWalkers: 1500,
    };
    return new Promise((resolve, reject) => {
      let refused = false,
        populationRefused = false,
        steps = 0,
        maximum = startingN,
        cap,
        peak = 0;
      const timeout = setTimeout(() => fail("Graph growth timed out"), 900000);
      function fail(message) {
        clearTimeout(timeout);
        w.terminate();
        reject(new Error(message));
      }
      w.onerror = (e) => fail(e.message);
      w.onmessage = ({ data: m }) => {
        if (m.type === "error") {
          if (!refused && /too small/.test(m.message)) {
            refused = true;
            w.postMessage({
              type: "init",
              rom,
              params,
              resources: { workers: 20, memoryGiB: 2 },
            });
          } else if (
            refused &&
            !populationRefused &&
            /estimated safe maximum/.test(m.message)
          ) {
            populationRefused = true;
            w.postMessage({
              type: "init",
              rom,
              params: { ...params, n: startingN },
              resources: { workers: 20, memoryGiB: 8 },
            });
          } else fail(`${m.message}; after ${steps} updates, cap ${cap}, peak ${peak}, resources ${JSON.stringify(m.resources)}`);
        }
        if (m.type === "ready") {
          cap = m.maxWalkers;
          w.postMessage({ type: "start" });
        }
        if (m.type === "step") {
          peak = Math.max(peak, m.resources.allocatedBytes);
          maximum = Math.max(maximum, m.stats.walkerCount);
          if (++steps === 100) w.postMessage({ type: "dispose" });
        }
        if (m.type === "disposed") {
          clearTimeout(timeout);
          w.terminate();
          resolve({ refused, populationRefused, steps, maximum, cap, peak });
        }
      };
      w.postMessage({
        type: "init",
        rom,
        params,
        resources: { workers: 20, memoryGiB: 1 },
      });
    });
  }, startingN);
  assert.equal(result.refused, true);
  assert.equal(result.populationRefused, true);
  assert.ok(result.maximum > startingN);
  assert.ok(result.maximum <= result.cap && result.cap < 1500);
  console.log("Graph RGB growth and resource recovery", result);
} finally {
  await browser.close();
}
