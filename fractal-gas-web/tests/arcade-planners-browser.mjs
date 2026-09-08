// Run after building arcade WASM and serving fractal-gas-web with serve.py.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const base = process.env.ARCADE_TEST_URL || "http://127.0.0.1:8091/web/";
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(() => !document.getElementById("btn-start").disabled);
  for (const [id, value] of [["param-n", "4"], ["param-dt-min", "2"],
      ["param-dt-max", "2"], ["param-horizon", "3"]]) {
    await page.locator(`#${id}`).evaluate((el, v) => { el.value = v; }, value);
  }
  assert.deepEqual(await page.locator("#algo-select button").allTextContents(),
    ["Wave", "Graph", "FMC", "Jump Wave"]);
  for (const algorithm of [2, 3, 1, 0]) {
    await page.locator(`[data-algo="${algorithm}"]`).click();
    await page.waitForFunction(() => !document.getElementById("btn-start").disabled);
    assert.equal(await page.locator("#planner-settings").isVisible(), algorithm >= 2);
    assert.equal(await page.locator("#param-max-walkers-row").isVisible(), algorithm === 1);
    assert.equal(await page.locator("#screen-title").textContent(), algorithm >= 2 ? "Played game" : "Best walker");
    if (algorithm === 3) {
      assert.equal(await page.locator("#param-consensus").isChecked(), true);
      await page.locator("#param-consensus").uncheck();
      assert.equal(await page.locator("#param-max-horizon-row").isVisible(), false);
      await page.locator("#param-consensus").check();
    }
    await page.locator("#btn-start").click();
    await page.waitForFunction(planner => planner
      ? Number(document.getElementById("stat-played-frames").textContent.replaceAll(",", "")) >= 2
      : Number(document.getElementById("stat-iteration").textContent) >= 2, algorithm >= 2);
    await page.locator("#btn-pause").click();
    await page.locator("#btn-reset").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("Reset"));
  }
  await page.locator('[data-algo="3"]').click();
  await page.waitForFunction(() => !document.getElementById("btn-start").disabled);
  await page.locator("#param-max-horizon").fill("1");
  await page.locator("#param-max-horizon").press("Tab");
  await page.waitForFunction(() => document.getElementById("status").textContent.includes("Maximum search horizon"));
  assert.equal(await page.locator("#btn-reset").isEnabled(), true);
  await page.locator("#param-max-horizon").fill("0");
  await page.locator("#param-max-horizon").press("Tab");
  await page.locator("#btn-start").click();
  await page.waitForFunction(() => Number(document.getElementById("stat-played-frames").textContent) >= 2);
  await page.locator("#btn-pause").click();
  await page.screenshot({ path: "/tmp/arcade-solvers.png", fullPage: true });
  assert.deepEqual(errors, []);
  await page.close();

  for (const game of [
    { console: 0, game: 0, rom: "test-rom.nes", name: "NES Mario", world: 1, stage: 1 },
    { console: 1, game: 0, rom: "roms/atari/breakout.bin", name: "Atari Breakout", world: 1, stage: 1 },
    { console: 1, game: 1, rom: "roms/atari/montezuma_revenge.bin", name: "Atari Montezuma", world: 1, stage: 1 },
    { console: 2, game: 1, rom: "sonic.rom", name: "Genesis Sonic", world: 0, stage: 0 },
  ]) {
    for (const [algorithm, consensusPrefix] of [[2, true], [3, true], [3, false]]) {
      const page = await browser.newPage();
      const fixture = new URL("planner-worker-test.html", base).href;
      await page.route(fixture, route => route.fulfill({
        contentType: "text/html", body: "<!doctype html><title>Arcade planner test</title>",
        headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" },
      }));
      await page.goto(fixture);
      const result = await page.evaluate(async ({ game, algorithm, consensusPrefix }) => {
        const worker = new Worker(new URL("worker.js", location.href), { type: "module" });
        const messages = [];
        let failure;
        worker.onmessage = ({ data }) => {
          messages.push(data);
          if (data.type === "error") failure = data.message;
        };
        worker.onerror = e => { failure = e.message; };
        const wait = async predicate => {
          const deadline = performance.now() + 60000;
          while (!predicate()) {
            if (failure) throw new Error(failure);
            if (performance.now() > deadline) throw new Error("Worker timeout: " + JSON.stringify(messages.at(-1)?.stats));
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        };
        const steps = () => messages.filter(m => m.type === "step");
        const last = () => steps().at(-1)?.stats;
        const params = {
          n: 4, nThreads: 1, distCoef: 1, rewardCoef: 1, useCumulativeReward: true,
          dtMin: 2, dtMax: 2, nElite: 1, seed: 7, obsMode: 3,
          console: game.console, game: game.game, world: game.world, stage: game.stage,
          algorithm, horizon: 3, consensusPrefix, maxHorizon: 6,
        };
        try {
          const response = await fetch(game.rom);
          if (!response.ok) throw new Error("Missing test ROM: " + game.rom);
          const rom = await response.arrayBuffer();
          worker.postMessage({ type: "init", rom, params }, [rom]);
          await wait(() => messages.some(m => m.type === "ready"));
          worker.postMessage({ type: "start" });
          await wait(() => last()?.playedFrames >= 8 || last()?.gameDone);
          worker.postMessage({ type: "pause" });
          await new Promise(resolve => setTimeout(resolve, 150));
          const pausedCount = steps().length;
          const pausedFrames = last().playedFrames;
          await new Promise(resolve => setTimeout(resolve, 150));
          if (steps().length !== pausedCount) throw new Error("Pause kept advancing");
          const firstRun = steps().slice();
          worker.postMessage({ type: "start" });
          await wait(() => last().playedFrames > pausedFrames || last().gameDone);
          worker.postMessage({ type: "reset" });
          await wait(() => messages.some(m => m.type === "resetDone"));
          messages.length = 0;
          worker.postMessage({ type: "start" });
          await wait(() => last()?.playedFrames >= firstRun.find(m => m.stats.playedFrames > 0).stats.playedFrames);
          worker.postMessage({ type: "pause" });
          await new Promise(resolve => setTimeout(resolve, 100));
          const first = firstRun.find(m => m.stats.playedFrames > 0);
          const repeated = steps().find(m => m.stats.playedFrames > 0);
          const equal = (a, b) => a.byteLength === b.byteLength && new Uint8Array(a).every((v, i) => v === new Uint8Array(b)[i]);
          if (!equal(first.frame, repeated.frame)) throw new Error("Reset did not replay the same committed frame");
          let prior = null;
          for (const msg of firstRun) {
            const s = msg.stats;
            if (s.algorithm !== algorithm) throw new Error("Wrong solver selected");
            if (prior && s.searchAdvanced) {
              if (s.playedFrames !== prior.stats.playedFrames || !equal(msg.frame, prior.frame))
                throw new Error("Search changed committed playback");
            }
            if (prior && s.playedFrames < prior.stats.playedFrames) throw new Error("Played clock moved backwards");
            prior = msg;
          }
          // Cancel a deliberately long search while it is still planning.
          worker.postMessage({ type: "setParams", params: { ...params, horizon: 100, maxHorizon: 200 } });
          messages.length = 0;
          worker.postMessage({ type: "start" });
          await wait(() => steps().length > 0);
          worker.postMessage({ type: "reset" });
          await wait(() => messages.some(m => m.type === "resetDone"));
          return { steps: firstRun.length, playedFrames: pausedFrames, modes: [...new Set(firstRun.map(m => m.stats.executionMode).filter(Boolean))] };
        } finally { worker.terminate(); }
      }, { game, algorithm, consensusPrefix });
      console.log(game.name, algorithm === 2 ? "FMC" : consensusPrefix ? "Jump Wave shared" : "Jump Wave full", result);
      await page.close();
    }
  }
} finally {
  await browser.close();
}
