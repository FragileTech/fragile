// node tests/fractal-wasm-benchmark.mjs /absolute/path/to/web
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { statSync } from "node:fs";
const root = resolve(process.argv[2] || "fractal-gas-web/web");
const load = (name) => import(pathToFileURL(resolve(root, name)).href);
const { NativeEngine, loadNative } = await load("lab/native.js");
const { createController } = await load("lab/controllers/index.js");
const { NativeOptimization } = await load("optimization/native.js");
const { default: createOptimization } = await load(
  "optimization/engine/optimization.mjs",
);
const control = await loadNative(false),
  optimization = await createOptimization();
const scene = {
  size: [1000, 1000],
  environment: { flight: true },
  bodies: [{ controlled: true, position: [500, 500] }],
};
const median = (a) => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
console.log("workload,us_per_operation,process_peak_rss_kib,wasm_heap_bytes");
function bench(label, setup, module) {
  if (process.argv[3] && process.argv[3] !== label) return;
  const times = [];
  for (let repeat = 0; repeat < 9; repeat++) {
    const run = setup();
    for (let i = 0; i < 8; i++) run.step();
    const start = performance.now();
    for (let i = 0; i < 40; i++) run.step();
    times.push(((performance.now() - start) * 1000) / 40);
    run.dispose();
  }
  console.log(
    [
      label,
      median(times),
      process.resourceUsage().maxRSS,
      module.HEAPU8.buffer.byteLength,
    ].join(","),
  );
}
for (let mode = 0; mode < 3; mode++)
  bench(
    `lab-${mode}`,
    () => {
      const engine = new NativeEngine(control, scene);
      engine.reset(7);
      engine.begin({ walkers: 256, frames: 1, elites: 4, recording: mode }, 7);
      return { step: () => engine.waveStep(), dispose: () => engine.dispose() };
    },
    control,
  );
for (const algorithm of ["fmc", "wave-jump"])
  bench(
    `lab-${algorithm}-cycle`,
    () => {
      const settings = {
        algorithm,
        walkers: 128,
        horizon: 8,
        max_horizon: 16,
        frames: 1,
        elites: 4,
        recording: 1,
        consensus_prefix: true,
      };
      const world = new NativeEngine(control, scene);
      world.reset(7);
      const snapshot = world.snapshot();
      world.dispose();
      const strategy = createController(control, scene, settings);
      return {
        step: () => {
          strategy.controller.begin(snapshot, 7);
          while (!strategy.controller.advance()) {}
          strategy.controller.result();
        },
        dispose: () => strategy.dispose(),
      };
    },
    control,
  );
for (const algorithm of ["wave", "graph", "fmc", "wave_jump", "euclidean", "euclidean-no-force"])
  bench(
    `optimization-${algorithm}`,
    () => {
      const engine = new NativeOptimization(optimization);
      engine.create({
        algorithm: algorithm.replace("-no-force", ""),
        potential_force: algorithm !== "euclidean-no-force",
        benchmark: "sphere",
        walkers: 128,
        max_walkers: 512,
        dimensions: 8,
        horizon: 8,
        max_horizon: 16,
        periodic: true,
        companion: "uniform",
        clone_companion: "uniform",
      });
      return {
        step: () => engine.check(optimization._fgo_step(engine.handle)),
        dispose: () => optimization._fgo_destroy(engine.handle),
      };
    },
    optimization,
  );
console.error(
  JSON.stringify({
    control_wasm_bytes: statSync(resolve(root, "lab/engine/control.wasm")).size,
    optimization_wasm_bytes: statSync(
      resolve(root, "optimization/engine/optimization.wasm"),
    ).size,
  }),
);
