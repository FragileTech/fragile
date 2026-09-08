// Run compare-cma.py first. Same configurations; compare outcomes, not RNG draws.
import { readFile, writeFile } from "node:fs/promises";
import create from "../web/optimization/engine/optimization.mjs";
import { NativeOptimization } from "../web/optimization/native.js";
const output = new URL("../../outputs/optimization/cma/", import.meta.url);
const reference = JSON.parse(
  await readFile(new URL("comparison.json", output), "utf8"),
);
const native = new NativeOptimization(await create());
const rows = [];
for (const r of reference) {
  const start = performance.now();
  native.create(r.config);
  for (;;) {
    const status = native.status();
    if (status.finished || status.budget_exhausted) break;
    native.step();
  }
  const frame = native.snapshot();
  rows.push({
    ...r,
    best: frame[9],
    evaluations: frame[5],
    seconds: (performance.now() - start) / 1000,
    status: native.status(),
  });
}
native.dispose();
await writeFile(
  new URL("wasm-comparison.json", output),
  JSON.stringify(rows, null, 2) + "\n",
);
const median = (values) =>
  values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
const lines = [
  "# WASM/native comparison",
  "",
  "Same five seeds and configurations as comparison.md; medians over complete runs. Different standard-library normal sampling is permitted. This small sample is descriptive, not a claim of statistical equivalence. Times include initialization; CMA uses float64 coordinates, Wave/GAS float32. Objective quality does not measure reward-density fidelity.",
  "",
  "| Benchmark | Algorithm | Native best | WASM best | WASM evaluations | WASM seconds |",
  "|---|---|---:|---:|---:|---:|",
];
for (const benchmark of ["bbob_10", "rosenbrock", "rastrigin"])
  for (const algorithm of ["cmaes_active", "cmaes_bipop", "wave", "gas"]) {
    const group = rows.filter(
      (r) => r.benchmark === benchmark && r.algorithm === algorithm,
    );
    const original = reference.filter(
      (r) => r.benchmark === benchmark && r.algorithm === algorithm,
    );
    lines.push(
      `| ${benchmark} | ${algorithm} | ${median(original.map((r) => r.best)).toPrecision(9)} | ${median(group.map((r) => r.best)).toPrecision(9)} | ${median(group.map((r) => r.evaluations))} | ${median(group.map((r) => r.seconds)).toFixed(5)} |`,
    );
  }
await writeFile(new URL("wasm-comparison.md", output), lines.join("\n") + "\n");
console.log(lines.join("\n"));
