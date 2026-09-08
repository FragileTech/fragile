import test from "node:test";
import assert from "node:assert/strict";
import { Recording } from "../../web/optimization/recording.js";
import { exportFixedBudgetCSV } from "../../web/optimization/fixed-budget.js";

test("IOHanalyzer CSV uses actual evaluation checkpoints and keeps direction/settings", () => {
  const recording = new Recording({
    benchmark: "bbob_24",
    algorithm: "wave_jump",
    dimensions: 5,
    seed: 7,
    coco_instance: 3,
    max_evaluations: 100,
    objective: "maximize",
    perturbation: "gaussian",
    perturbation_std: 0.1,
  });
  for (const [iteration, evaluations, best] of [
    [0, 1, 12],
    [1, 17, 20],
    [2, 17, 20],
    [3, 33, 25],
  ]) {
    const frame = new Float64Array(12);
    frame[4] = iteration;
    frame[5] = evaluations;
    frame[9] = best;
    recording.append(frame);
  }
  const lines = exportFixedBudgetCSV(recording).trim().split("\n");
  assert.equal(lines.length, 4);
  assert.match(
    lines[0],
    /"evaluations","best","function","algorithm","dimension","run"/,
  );
  assert.match(lines[3], /^"33","25","bbob_24","wave_jump","5"/);
  assert.match(lines[3], /"maximize","100","fgopt-3","gaussian","0.1"$/);
  assert.ok(!lines.some((line) => line.startsWith('"100",')));
});

test("CSV rejects recordings without finite objectives", () => {
  const recording = new Recording({ dimensions: 3 });
  assert.throws(() => exportFixedBudgetCSV(recording), /No finite/);
});

test("GAS CSV identifies feature switches without claiming a fixed adaptive deviation", () => {
  const recording = new Recording({
    algorithm: "gas",
    benchmark: "sphere",
    dimensions: 2,
    gas_tabu: false,
    gas_local_search: true,
    gas_local_evaluations: 200,
    perturbation: "gas_adaptive",
    perturbation_std: 1,
    seed: 7,
  });
  const frame = new Float64Array(12);
  frame[5] = 20;
  frame[9] = 0;
  recording.append(frame);
  const csv = exportFixedBudgetCSV(recording);
  assert.match(csv, /"gas_tabu","gas_local_search","gas_local_evaluations"/);
  assert.match(csv, /"gas_adaptive","","false","true","200"/);
});
