// Headless reproducible benchmark runner. Requires the WASM engine build.
import { readFile, writeFile } from "node:fs/promises";
import { loadNative } from "../web/lab/native.js";
import { runBenchmark } from "../web/lab/experiments.js";
const [sceneFile, specFile, output = "control-benchmark.json"] =
  process.argv.slice(2);
if (!sceneFile || !specFile)
  throw new Error(
    "Usage: node tools/control-benchmark.mjs SCENE.json SPEC.json [REPORT.json]",
  );
const scene = JSON.parse(await readFile(sceneFile, "utf8")),
  spec = JSON.parse(await readFile(specFile, "utf8"));
const report = await runBenchmark({
  module: await loadNative(false),
  scene: Array.isArray(scene) ? scene[0] : scene,
  scenes: Array.isArray(scene) ? scene : undefined,
  spec,
  progress: (_, i, n) => console.log(`Episode ${i}/${n}`),
});
await writeFile(output, JSON.stringify(report, null, 2) + "\n");
console.log(`Saved ${output}`);
