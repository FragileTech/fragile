// Build Arcade and run serve.py first. Full matrix needs a desktop with >8 GiB free.
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { chromium, firefox } from "playwright";
const engine = process.env.ARCADE_BROWSER === "firefox" ? firefox : chromium;
const browser = await engine.launch({
  headless: true,
  args: engine === chromium ? ["--no-sandbox"] : [],
});
const base = process.env.ARCADE_TEST_URL || "http://127.0.0.1:8091/web/";
const results = [];
const selectedModes = (process.env.ARCADE_MODES || "3,0,2,1")
  .split(",")
  .map(Number);
const selectedAlgorithms = (process.env.ARCADE_ALGORITHMS || "0,1,2,3")
  .split(",")
  .map(Number);
try {
  for (const obsMode of selectedModes)
    for (const algorithm of selectedAlgorithms) {
      const page = await browser.newPage();
      page.on("console", (message) => {
        if (message.text().startsWith("Arcade acceptance:"))
          console.error(message.text());
      });
      await page.goto(new URL("./", base).href);
      const result = await page.evaluate(
        async ({ obsMode, algorithm, timeoutMs }) => {
          const worker = new Worker("worker.js", { type: "module" });
          const params = {
            n: 1000,
            nThreads: 20,
            distCoef: 1,
            rewardCoef: 1,
            useCumulativeReward: true,
            dtMin: 1,
            dtMax: 2,
            nElite: 2,
            seed: 7,
            obsMode,
            console: 2,
            game: 1,
            world: 0,
            stage: 0,
            algorithm,
            maxWalkers: 1000,
            horizon: 4,
            maxHorizon: 8,
            consensusPrefix: true,
          };
          const rom = await (await fetch("sonic.rom")).arrayBuffer();
          return new Promise((resolve, reject) => {
            const begun = performance.now();
            let steps = 0,
              cycles = 0,
              priorPhase = "",
              peak = 0,
              mainPeak = 0,
              ready;
            let activeStart = 0;
            const timer = setTimeout(() => fail("Large run timed out"), timeoutMs);
            function fail(message) {
              clearTimeout(timer);
              worker.terminate();
              reject(new Error(message));
            }
            worker.onerror = (e) => fail(e.message);
            worker.onmessage = ({ data: m }) => {
              if (m.resources) {
                peak = Math.max(peak, m.resources.allocatedBytes);
                mainPeak = Math.max(mainPeak, m.resources.mainBytes);
                if (m.resources.allocatedBytes > m.resources.budgetBytes)
                  return fail("Combined memory budget exceeded");
              }
              if (m.type === "error") return fail(m.message);
              if (m.type === "ready") {
                ready = m.resources;
                activeStart = performance.now();
                worker.postMessage({ type: "start" });
              }
              if (m.type === "step") {
                steps++;
                if (steps % 10 === 0)
                  console.log(`Arcade acceptance: mode ${obsMode}, algorithm ${algorithm}, ${steps} updates, ${cycles} cycles, ${(peak / 1024 ** 3).toFixed(3)} GiB`);
                if (priorPhase === "playing" && m.stats.phase !== "playing")
                  cycles++;
                priorPhase = m.stats.phase;
                if (
                  (algorithm < 2 && steps >= 100) ||
                  (algorithm >= 2 && cycles >= 3)
                ) {
                  worker.postMessage({ type: "pause" });
                  const ms = performance.now() - activeStart;
                  worker.onmessage = ({ data: final }) => {
                    if (final.type === "disposed") {
                      clearTimeout(timer);
                      worker.terminate();
                      resolve({
                        obsMode,
                        algorithm,
                        steps,
                        cycles,
                        msPerUpdate: ms / steps,
                        peakBytes: peak,
                        mainPeakBytes: mainPeak,
                        ready,
                        elapsedMs: performance.now() - begun,
                      });
                    }
                  };
                  worker.postMessage({ type: "dispose" });
                }
              }
              if (m.type === "allDead" || m.type === "gameDone")
                fail("Population/game ended before acceptance count");
            };
            worker.postMessage(
              {
                type: "init",
                rom,
                params,
                resources: { workers: 20, memoryGiB: 8 },
              },
              [rom],
            );
          });
        },
        { obsMode, algorithm, timeoutMs: Number(process.env.ARCADE_TIMEOUT_MS || 900000) },
      );
      assert.equal(result.ready.workers, 20);
      assert.ok(result.mainPeakBytes > 512 * 1024 ** 2);
      assert.ok(result.mainPeakBytes <= 4 * 1024 ** 3);
      results.push(result);
      console.log(JSON.stringify(result));
      await page.close();
      await writeFile(
        process.env.ARCADE_REPORT || "/tmp/arcade-resources-results.json",
        JSON.stringify(results, null, 2),
      );
    }
} finally {
  await browser.close();
}
