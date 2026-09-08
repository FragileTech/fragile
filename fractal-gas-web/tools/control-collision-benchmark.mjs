// Compare release engines without changing search budgets. Preserve an old
// engine directory before rebuilding, then pass it with --engine-dir.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { NativeEngine } from "../web/lab/native.js";
import "../web/lab/controllers/index.js";
import { createController } from "../web/lab/controllers/registry.js";
const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const engineDir = resolve(
  option(
    "--engine-dir",
    fileURLToPath(new URL("../web/lab/engine", import.meta.url)),
  ),
);
const output = resolve(
  option("--output", "/tmp/control-collision-benchmark.json"),
);
const rootDir = resolve(option("--roots", "/tmp/control-collision-roots"));
const seeds = option("--seeds", "7,11,19").split(",").map(Number);
const threadsList = option("--threads", "1,8").split(",").map(Number);
const search = !args.includes("--fixed-only");
const scene = JSON.parse(
  await readFile(
    new URL("../web/lab/scenarios/harvest.json", import.meta.url),
    "utf8",
  ),
);
const rows = [];
await mkdir(rootDir, { recursive: true });
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
for (const threads of threadsList) {
  const { default: create } = await import(
    pathToFileURL(
      join(engineDir, threads > 1 ? "control-threaded.mjs" : "control.mjs"),
    )
  );
  const module = await create({ controlThreads: threads });
  for (const flight of [false, true]) {
    const definition = structuredClone(scene);
    definition.environment = { ...definition.environment, flight };
    // Fixed work disables episode termination, then reuses exactly the same
    // serialized starting states across old and new engines. No search involved.
    const fixedScene = structuredClone(definition);
    fixedScene.physics.lethal_walls = false;
    const fixed = new NativeEngine(module, fixedScene, 32, threads);
    for (const seed of seeds) {
      const rootPath = join(rootDir, `${flight}-${seed}.bin`);
      let root;
      try {
        root = new Uint8Array(await readFile(rootPath));
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
        fixed.reset(seed);
        fixed.step(fixed.neutralAction(), 180);
        root = fixed.snapshot();
        await writeFile(rootPath, root);
      }
      const action = fixed.neutralAction(),
        times = [];
      let frames = 0,
        collisions = 0;
      for (let repeat = 0; repeat < 4; repeat++) {
        fixed.restore(root);
        const start = performance.now();
        fixed.step(action, 120);
        const elapsed = performance.now() - start;
        if (repeat) times.push(elapsed);
        const results = fixed.results();
        frames = 0;
        collisions = 0;
        for (let i = 0; i < 32; i++) {
          frames += results[i * 4 + 1];
          collisions += results[i * 4 + 3];
        }
        if (frames !== 32 * 120)
          throw Error(`Fixed workload stopped early: ${frames}`);
      }
      const row = {
        kind: "fixed",
        threads,
        flight,
        seed,
        medianMs: median(times),
        times,
        frames,
        collisions,
      };
      rows.push(row);
      console.log(JSON.stringify(row));
    }
    fixed.dispose();
    if (search) {
      const live = new NativeEngine(module, definition);
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
      // Warm the actual full-size search once; individual seed results remain
      // visible rather than hiding divergent trajectories in a single timing.
      const strategy = createController(module, definition, settings, threads);
      for (const seed of [seeds[0], ...seeds]) {
        live.reset(seed);
        strategy.engine.resetProfile();
        const start = performance.now();
        strategy.controller.begin(live.snapshot(), seed);
        while (!strategy.controller.advance()) {}
        const searchMs = performance.now() - start;
        const result = strategy.controller.result();
        const row = {
          kind: "search",
          threads,
          flight,
          seed,
          searchMs,
          frames: strategy.engine.profile()[1],
          depth: result.searchDepth,
          mode: result.executionMode,
          committedFrames: result.trajectory.reduce((n, e) => n + e.frames, 0),
        };
        if (!strategy.warmed) strategy.warmed = true;
        else {
          rows.push(row);
          console.log(JSON.stringify(row));
        }
      }
      strategy.dispose();
      live.dispose();
    }
    await writeFile(
      output,
      JSON.stringify({ engineDir, seeds, rows }, null, 2) + "\n",
    );
  }
}
console.log(`Saved ${output}`);
