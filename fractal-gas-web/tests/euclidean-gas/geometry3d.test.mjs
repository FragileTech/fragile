import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  bestIndex,
  buildSurface,
  companionPairs,
  contourLevels,
  contourSegments,
  coord,
  heightOf,
  metricRange,
  robustScale,
  shade,
  writeInstance,
} from "../../web/euclidean-gas/geometry3d.js";
import { TrailBuffer } from "../../web/euclidean-gas/trails.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});

test("Domain coordinates map onto the display cube", () => {
  assert.equal(coord(-5.12, -5.12, 5.12), -10);
  assert.equal(coord(5.12, -5.12, 5.12), 10);
  assert.equal(coord(0, -5.12, 5.12), 0);
  assert.ok(Number.isNaN(coord(NaN, -1, 1)));
});

test("Robust scale anchors at the optimum side and ignores nonfinite samples", () => {
  const values = Float64Array.from({ length: 101 }, (_, i) => i);
  const withHoles = Float64Array.from([...values, NaN, Infinity]);
  const low = robustScale(withHoles, "minimize");
  assert.deepEqual(low, { base: 0, orient: 1, scale: 30 });
  const high = robustScale(withHoles, "maximize");
  assert.deepEqual(high, { base: 100, orient: -1, scale: 30 });
  assert.equal(robustScale(Float64Array.from([NaN, NaN])), null);
  assert.equal(robustScale(Float64Array.from([3, 3, 3])).scale, 3 * 1e-8);
});

test("Heights are monotone, signed and zero at the anchor", () => {
  for (const direction of ["minimize", "maximize"]) {
    const s = robustScale(Float64Array.from([-4, 0, 1, 9, 100]), direction);
    assert.equal(heightOf(s.base, s, 3), 0);
    let previous = -Infinity;
    for (const v of [-4, 0, 1, 9, 100]) {
      const h = heightOf(v, s, 3);
      assert.ok(h > previous);
      assert.ok(direction === "minimize" ? h >= 0 : h <= 0);
      previous = h;
    }
    assert.equal(shade(s.base, s), 0);
    assert.equal(shade(direction === "minimize" ? 1e12 : -1e12, s), 1);
    const levels = contourLevels(s, 9);
    assert.equal(levels.length, 9);
    assert.ok(
      levels.every((level) =>
        direction === "minimize" ? level > s.base : level < s.base,
      ),
    );
  }
});

test("Surface triangles touching nonfinite samples are skipped", () => {
  const n = 4,
    values = Float64Array.from({ length: n * n }, (_, k) => k % n);
  const s = robustScale(values);
  const full = buildSurface(values, n, (v) => v, s);
  assert.equal(full.indices.length, (n - 1) * (n - 1) * 6);
  assert.deepEqual(Array.from(full.positions.slice(0, 3)), [-10, -10, 0]);
  assert.deepEqual(Array.from(full.positions.slice(-3)), [10, 10, 3]);
  values[5] = NaN;
  const holed = buildSurface(values, n, (v) => v, s);
  assert.equal(holed.indices.length, full.indices.length - 6 * 3);
  assert.ok(!Array.from(holed.indices).includes(5));
  assert.ok(holed.positions.every(Number.isFinite));
  assert.equal(holed.t[5], 0);
});

test("Marching triangles trace a linear ramp at the expected abscissa", () => {
  const n = 5,
    values = Float64Array.from({ length: n * n }, (_, k) => k % n);
  const { positions, indices } = buildSurface(values, n, () => 0, null);
  const segments = contourSegments(values, positions, indices, [1.5], 0.25);
  // One segment per triangle of the crossed column: 2 triangles × 4 rows.
  assert.equal(segments.length, 8 * 6);
  for (let k = 0; k < segments.length; k += 3) {
    assert.ok(Math.abs(segments[k] - coord(1.5, 0, 4)) < 1e-6);
    assert.equal(segments[k + 2], 0.25);
  }
  assert.equal(
    contourSegments(values, positions, indices, [9]).length,
    0,
    "levels outside the samples draw nothing",
  );
});

test("Best walker honours direction, eligibility and finiteness", () => {
  const raw = Float64Array.from([3, -1, NaN, -7, 12]);
  assert.equal(bestIndex(raw, [1, 1, 1, 0, 1], "minimize"), 1);
  assert.equal(bestIndex(raw, [1, 1, 1, 0, 1], "maximize"), 4);
  assert.equal(bestIndex(raw, [0, 0, 1, 0, 0], "minimize"), -1);
  assert.deepEqual(metricRange(raw, [1, 1, 1, 0, 1]), [-1, 12]);
  assert.deepEqual(metricRange(raw, [0, 0, 0, 0, 0]), [Infinity, -Infinity]);
});

test("Instance matrices are uniform-scale translations", () => {
  const m = new Float32Array(32).fill(9);
  writeInstance(m, 1, 1, 2, 3, 0.5);
  assert.deepEqual(
    Array.from(m.slice(16)),
    [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 1, 2, 3, 1],
  );
  assert.ok(m.slice(0, 16).every((v) => v === 9));
});

test("Companion links use only donors frozen at the step being reported", async () => {
  const config = JSON.parse(default_config());
  config.walkers = 24;
  config.gas.distance_donors.count = 3;
  const gas = await BrowserGas.create(JSON.stringify(config));
  try {
    const frame = await gas.step(1),
      report = frame.report,
      n = config.walkers;
    assert.ok(report.distance_sources.length);
    assert.equal(
      Math.max(...report.distance_sources.map((source) => source.frame)),
      report.step - 1,
      "current-frame donors are frozen before the step counter advances",
    );
    const distance = companionPairs(report, "distance", n);
    assert.equal(distance.skippedHistorical, 0);
    assert.ok(distance.count > 0 && distance.count <= n * 3);
    for (let p = 0; p < distance.count; p++) {
      assert.notEqual(distance.out[2 * p], distance.out[2 * p + 1]);
      assert.ok(distance.out[2 * p + 1] < n);
    }
    const cloning = companionPairs(report, "cloning", n, distance.out);
    assert.ok(cloning.count <= n);
    const only = companionPairs(report, "distance", n, null, 5);
    assert.ok(only.count <= 3);
    for (let p = 0; p < only.count; p++) assert.equal(only.out[2 * p], 5);
    assert.equal(companionPairs(null, "distance", n).count, 0);
  } finally {
    gas.free();
  }
});

test("Historical donors are counted, not drawn", async () => {
  const config = JSON.parse(default_config());
  config.walkers = 16;
  config.gas.distance_donors.history_window = 3;
  const gas = await BrowserGas.create(JSON.stringify(config));
  try {
    await gas.step(3);
    const report = (await gas.step(1)).report;
    const historical = report.distance_companions.indices.filter(
      (index, slot) =>
        report.distance_companions.valid[slot] &&
        report.distance_sources[index].frame !== report.step - 1,
    ).length;
    assert.ok(historical > 0, "the window must expose earlier frames");
    const pairs = companionPairs(report, "distance", config.walkers);
    assert.equal(pairs.skippedHistorical, historical);
  } finally {
    gas.free();
  }
});

function frameOf(step, positions, { choices, generations, d = 2 } = {}) {
  const rows = positions.length / d;
  return {
    step,
    population: {
      observations: {
        fields: {
          positions: {
            rows,
            item_shape: [d],
            values: Float64Array.from(positions),
          },
        },
      },
      rewards: { raw: Float64Array.from({ length: rows }, (_, i) => i) },
      generations: generations || Array(rows).fill(0),
    },
    report: choices ? { clone_plan: { choices } } : null,
  };
}
const kept = (n) => Array(n).fill({ accepted: false, revival: false });
const flat = (positions, offset, raw, target) => {
  target[0] = positions[offset];
  target[1] = positions[offset + 1];
  target[2] = raw;
  return true;
};

test("Trails wrap around and keep at most frames − 1 segments per walker", () => {
  const trails = new TrailBuffer({ frames: 4, maxWalkers: 2 });
  assert.equal(trails.capacity, 3 * 3 * 6);
  const out = new Float32Array(trails.capacity);
  for (let step = 0; step < 9; step++)
    trails.push(
      frameOf(step, [step, 0, step, 1, step, 2], { choices: kept(3) }),
      [1, 1, 1],
    );
  const floats = trails.segments(flat, out);
  assert.equal(floats, 3 * 2 * 6, "third walker is beyond maxWalkers");
  assert.deepEqual(Array.from(out.slice(0, 6)), [5, 0, 0, 6, 0, 0]);
  assert.deepEqual(
    Array.from(out.slice(floats - 6, floats)),
    [7, 1, 1, 8, 1, 1],
  );
  trails.push(frameOf(8, [9, 9, 9, 9, 9, 9]), [1, 1, 1]);
  assert.equal(
    trails.segments(flat, out),
    floats,
    "repeated steps are ignored",
  );
});

test("Trails break at clones, revivals, generation changes and ineligible walkers", () => {
  const out = new Float32Array(600);
  const run = (second, eligible = [1, 1]) => {
    const trails = new TrailBuffer({ frames: 8, maxWalkers: 4 });
    trails.push(frameOf(0, [0, 0, 1, 1], { choices: kept(2) }), [1, 1]);
    trails.push(second, eligible);
    return trails.segments(flat, out) / 6;
  };
  assert.equal(run(frameOf(1, [0, 1, 1, 2], { choices: kept(2) })), 2);
  assert.equal(
    run(
      frameOf(1, [0, 1, 1, 2], {
        choices: [
          { accepted: true, revival: false },
          { accepted: false, revival: false },
        ],
      }),
    ),
    1,
  );
  assert.equal(
    run(
      frameOf(1, [0, 1, 1, 2], {
        choices: [
          { accepted: false, revival: true },
          { accepted: false, revival: true },
        ],
      }),
    ),
    0,
  );
  assert.equal(
    run(frameOf(1, [0, 1, 1, 2], { generations: [0, 4] })),
    1,
    "without a report the generation counter marks the clone",
  );
  assert.equal(run(frameOf(1, [0, 1, 1, 2], { choices: kept(2) }), [1, 0]), 1);
  assert.equal(run(frameOf(1, [0, NaN, 1, 2], { choices: kept(2) })), 1);
});

test("Trails break at step gaps and periodic wraps", () => {
  const out = new Float32Array(600);
  const trails = new TrailBuffer({ frames: 8, maxWalkers: 4 });
  trails.push(frameOf(0, [0, 0], { choices: kept(1) }), [1]);
  trails.push(frameOf(1, [0, 1], { choices: kept(1) }), [1]);
  trails.push(frameOf(5, [0, 2], { choices: kept(1) }), [1]);
  assert.equal(trails.segments(flat, out) / 6, 1);
  const options = { periodic: true, low: -5, high: 5 };
  const wrapped = new TrailBuffer({ frames: 8, maxWalkers: 4 });
  wrapped.push(frameOf(0, [4.9, 0], { choices: kept(1) }), [1], options);
  wrapped.push(frameOf(1, [-4.9, 0], { choices: kept(1) }), [1], options);
  wrapped.push(frameOf(2, [-4.5, 0], { choices: kept(1) }), [1], options);
  assert.equal(wrapped.segments(flat, out) / 6, 1);
  assert.deepEqual(Array.from(out.slice(0, 2)), [Math.fround(-4.9), 0]);
});

test("A newly selected walker starts with an empty trail", () => {
  const out = new Float32Array(600);
  const trails = new TrailBuffer({ frames: 8, maxWalkers: 1 });
  const positions = (step) => [step, 0, step, 1, step, 2];
  for (let step = 0; step < 3; step++)
    trails.push(
      frameOf(step, positions(step), { choices: kept(3) }),
      [1, 1, 1],
    );
  assert.equal(trails.segments(flat, out) / 6, 2);
  trails.track(2);
  assert.equal(trails.segments(flat, out) / 6, 2);
  trails.push(frameOf(3, positions(3), { choices: kept(3) }), [1, 1, 1]);
  assert.equal(trails.segments(flat, out) / 6, 3, "one point is not a segment");
  trails.push(frameOf(4, positions(4), { choices: kept(3) }), [1, 1, 1]);
  assert.equal(trails.segments(flat, out) / 6, 5);
  trails.track(0);
  assert.ok(!trails.slots.has(2), "the previous selection is released");
  trails.push(frameOf(5, [0, 0, 0, 1], { choices: kept(2) }), [1, 1]);
  assert.equal(trails.segments(flat, out), 0, "a new population resets trails");
});
