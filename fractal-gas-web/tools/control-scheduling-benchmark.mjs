// Paired comparisons against a preserved pre-change WASM engine directory.
// Alternate execution order, and assert identical states/results/selected paths.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { NativeEngine, loadNative } from "../web/lab/native.js";
import "../web/lab/controllers/index.js";
import { createController } from "../web/lab/controllers/registry.js";
const [baselineDir, output = "/tmp/control-scheduling.json"] =
  process.argv.slice(2);
if (!baselineDir)
  throw Error(
    "Usage: node tools/control-scheduling-benchmark.mjs BASELINE_ENGINE_DIR [OUTPUT.json]",
  );
const scene = JSON.parse(
  await readFile(
    new URL("../web/lab/scenarios/harvest.json", import.meta.url),
    "utf8",
  ),
);
const samples = [];
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const threadCounts = process.env.CONTROL_BENCH_THREADS
  ? process.env.CONTROL_BENCH_THREADS.split(",").map(Number)
  : [1, 8];
for (const threads of threadCounts) {
  const { default: create } = await import(
    pathToFileURL(
      join(
        resolve(baselineDir),
        threads > 1 ? "control-threaded.mjs" : "control.mjs",
      ),
    )
  );
  const modules = [
    await create({ controlThreads: threads }),
    await loadNative(threads > 1, threads),
  ];
  const definition = structuredClone(scene);
  definition.environment = { ...definition.environment, flight: true };
  const fixed = structuredClone(definition);
  fixed.physics.lethal_walls = false;
  const source = new NativeEngine(modules[0], fixed);
  source.reset(7);
  const free = source.states();
  source.step(source.neutralAction(), 180);
  const resting = source.states();
  const engines = modules.map((m) => new NativeEngine(m, fixed, 128, threads));
  for (const heavy of [0, 16, 128]) {
    const states = new Float32Array(128 * source.stride);
    for (let i = 0; i < 128; i++)
      states.set(i < heavy ? resting : free, i * source.stride);
    const action = engines[0].neutralAction(),
      times = [[], []];
    for (let repeat = 0; repeat < 6; repeat++) {
      for (const k of repeat % 2 ? [1, 0] : [0, 1]) {
        engines[k].restoreRows(states);
        const start = performance.now();
        engines[k].step(action, 12);
        const elapsed = performance.now() - start;
        if (repeat) times[k].push(elapsed);
      }
      assert.deepEqual(engines[0].snapshot(), engines[1].snapshot());
      assert.deepEqual(engines[0].results(), engines[1].results());
    }
    const row = {
      kind: "fixed",
      threads,
      worlds: 128,
      framesPerWorld: 12,
      heavyWorlds: heavy,
      staticMs: median(times[0]),
      dynamicMs: median(times[1]),
      times,
    };
    samples.push(row);
    console.log(JSON.stringify(row));
  }
  engines.forEach((e) => e.dispose());
  source.dispose();
  const settings = {
    algorithm: "wave-jump",
    walkers: 128,
    horizon: 64,
    frames: 12,
    noise: 0.2,
    elites: 0,
    inertial: true,
    recording: 1,
  };
  const strategies = modules.map((m) =>
    createController(m, definition, settings, threads),
  );
  const live = new NativeEngine(modules[0], definition);
  for (const [index, seed] of [7, 7, 11, 19].entries()) {
    live.reset(seed);
    const root = live.snapshot(),
      times = [],
      results = [];
    for (const k of index % 2 ? [1, 0] : [0, 1]) {
      const c = strategies[k].controller;
      console.log(
        `Search threads=${threads} seed=${seed} ${k ? "dynamic" : "static"}${index ? "" : " warm-up"}`,
      );
      const start = performance.now();
      c.begin(root, seed);
      let iterations = 0;
      while (!c.advance()) {
        assert.ok(
          ++iterations <= 2 * settings.horizon,
          "Search exceeded its maximum horizon",
        );
      }
      times[k] = performance.now() - start;
      results[k] = c.result();
    }
    assert.deepEqual(
      strategies[0].engine.states(true, 128),
      strategies[1].engine.states(true, 128),
    );
    assert.deepEqual(results[0].trajectory, results[1].trajectory);
    assert.equal(results[0].selectedLeaf, results[1].selectedLeaf);
    assert.equal(results[0].selectedReward, results[1].selectedReward);
    if (index) {
      const row = {
        kind: "search",
        threads,
        seed,
        staticMs: times[0],
        dynamicMs: times[1],
        depth: results[0].searchDepth,
      };
      samples.push(row);
      console.log(JSON.stringify(row));
    }
  }
  strategies.forEach((s) => s.dispose());
  live.dispose();
  await writeFile(output, JSON.stringify({ samples }, null, 2) + "\n");
}
console.log(`Identical physics and selected trajectories; saved ${output}`);
