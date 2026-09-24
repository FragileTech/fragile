import assert from "node:assert/strict";
import { chromium, firefox } from "playwright";
const engine = process.env.ARCADE_BROWSER === "firefox" ? firefox : chromium;
const browser = await engine.launch({ headless: true, args: engine === chromium ? ["--no-sandbox"] : [] });
try {
  const page = await browser.newPage();
  await page.goto(process.env.ARCADE_TEST_URL || "http://127.0.0.1:8096/web/");
  for (const consoleId of (process.env.ARCADE_CONSOLES || "0,1,2").split(",").map(Number)) for (const algorithm of (process.env.ARCADE_ALGORITHMS || "0,1,2,3").split(",").map(Number)) {
    const result = await page.evaluate(async ({ consoleId, algorithm }) => {
      const { playbackController } = await import("./playback-controller.js");
      const base = new URL("./", location.href).href;
      // Test-only instrumentation, capturing the selected state's expected frame.
      let source = await (await fetch("worker.js")).text();
      source = source.replaceAll('"./', '"' + base).replaceAll('self.location.href', JSON.stringify(base));
      source += `\nconst realPost = post; post = (type, data, transfer = []) => {
        if (type === "trajectoryRecording" && !data.error) {
          data.expected = new Uint8Array(fg.renderWalkerFrame(data.walker)).buffer;
          transfer.push(data.expected);
        }
        realPost(type, data, transfer);
      };`;
      const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
      const rom = await (await fetch(["test-rom.nes", "roms/atari/breakout.bin", "sonic.rom"][consoleId])).arrayBuffer();
      const params = { n: 4, nThreads: 2, distCoef: 1, rewardCoef: 1, useCumulativeReward: true,
        dtMin: 1, dtMax: 2, nElite: 1, seed: 7, obsMode: 3, console: consoleId,
        game: consoleId === 1 ? 0 : 1, world: consoleId === 0 ? 1 : 0,
        stage: consoleId === 0 ? 1 : 0, algorithm, maxWalkers: 48,
        horizon: 2, maxHorizon: 4, consensusPrefix: true, freezePrefixAfter: 1 };
      const hash = buffer => { let h = 2166136261; for (const b of new Uint8Array(buffer)) h = Math.imul(h ^ b, 16777619); return h >>> 0; };
      async function run(play) {
        const worker = new Worker(url, { type: "module" });
        let player, timer, frames = 0, expected, length = 0, searchDone = false;
        const trace = [];
        let request = 1, refused = false;
        const checkLimit = play && consoleId === 0 && algorithm === 0;
        return await new Promise((resolve, reject) => {
          const cleanup = () => { clearTimeout(timer); player.dispose(); worker.terminate(); };
          const fail = error => { cleanup(); reject(new Error(error)); };
          const finish = () => { if (searchDone && (!play || frames >= 3)) { cleanup(); resolve({ trace, frames, refused }); } };
          timer = setTimeout(() => fail("Independent playback test timed out"), 180000);
          player = playbackController({ sendSearch: msg => worker.postMessage(msg), config: () => ({ rom, params }),
            memory: bytes => { if (bytes > 256 * 1024 ** 2) fail("Playback budget exceeded"); },
            receive: msg => {
              if (msg.error) {
                if (checkLimit && !refused && msg.error.includes("32 MiB")) {
                  refused = true; request++;
                  player.send({ type: "trajectorySelect", walker: -1, request });
                  return;
                }
                return fail(msg.error);
              }
              if (msg.type === "trajectorySelected") length = msg.length;
              if (msg.type === "trajectoryFrame" && msg.ready) {
                if (msg.index === length - 1 && hash(msg.frame) !== expected) return fail("Replay differs from captured state");
                frames++;
              }
              if (frames >= 3) { finish(); return; }
              player.send({ type: "trajectoryFrame", index: msg.ready && msg.index === length - 1 ? 0 : length - 1, request });
            } });
          worker.onerror = e => fail(e.message);
          worker.onmessage = ({ data: m }) => {
            if (m.type === "error") return fail(m.message);
            if (m.type === "ready") worker.postMessage({ type: "start" });
            if (m.type === "trajectoryRecording") { expected = hash(m.expected); player.captured(checkLimit && !refused ? { ...m, actions: new ArrayBuffer(32 * 1024 ** 2) } : m); }
            if (m.type === "step") {
              if (trace.length < 12) {
                trace.push([m.stats.iteration, m.stats.totalFrames, m.stats.meanReward,
                  m.stats.maxReward, m.stats.numCloned, m.stats.phase, m.stats.playedFrames]);
                if (play && trace.length === 2) player.send({ type: "trajectorySelect", walker: -1, request });
              }
              if (trace.length >= 12) { searchDone = true; worker.postMessage({ type: "pause" }); finish(); }
            }
          };
          worker.postMessage({ type: "init", rom: rom.slice(0), params, resources: { workers: 2, memoryGiB: 8 } });
        });
      }
      try { return { baseline: await run(false), playback: await run(true) }; }
      finally { URL.revokeObjectURL(url); }
    }, { consoleId, algorithm });
    assert.deepEqual(result.playback.trace, result.baseline.trace);
    assert.ok(result.playback.frames >= 3);
    if (consoleId === 0 && algorithm === 0) assert.ok(result.playback.refused);
    console.log(`Independent playback console ${consoleId}, algorithm ${algorithm}: frames match; search traces identical`);
  }
} finally { await browser.close(); }
