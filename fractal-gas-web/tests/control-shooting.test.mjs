import test from "node:test";
import assert from "node:assert/strict";
import { instantiateController } from "../web/lab/controllers/index.js";
import { SeededRandom } from "../web/lab/controllers/registry.js";
import { ColoredNoise } from "../web/lab/controllers/colored-noise.js";
import { encodeObject, decodeObject } from "../web/lab/storage/codec.js";

// An unrelated deterministic control system: no bodies, geometry or physics
// fields exposed to the planner. Reward is a known separable quadratic.
class QuadraticEngine {
  constructor(n = 64, channels = [{ low: -1, high: 1 }], target = [0.7]) {
    Object.assign(this, {
      worlds: n,
      channels,
      dim: channels.length,
      stride: 1,
      controlled: 0,
      target,
    });
    this.rows = new Float32Array(n);
    this.out = new Float32Array(n * 4);
    this.frames = 0;
  }
  broadcast() {
    this.rows.fill(0);
  }
  step(actions, durations) {
    for (let w = 0; w < this.worlds; w++) {
      let reward = 0;
      for (let d = 0; d < this.dim; d++) {
        const u = actions[w * this.dim + d],
          c = this.channels[d];
        assert.ok(
          Number.isFinite(u) && u >= c.low - 1e-6 && u <= c.high + 1e-6,
        );
        const normalized = (2 * (u - c.low)) / (c.high - c.low) - 1;
        reward -= (normalized - this.target[d]) ** 2;
      }
      this.out[w * 4] = reward * durations[w];
      this.out[w * 4 + 1] = durations[w];
      this.rows[w] += durations[w];
      this.frames += durations[w];
    }
  }
  results() {
    return this.out.slice();
  }
  states() {
    return this.rows.slice();
  }
  metrics() {
    return new Float32Array(16);
  }
  checkpoint() {
    return encodeObject({
      rows: this.rows,
      out: this.out,
      frames: this.frames,
    });
  }
  restoreCheckpoint(bytes) {
    Object.assign(this, decodeObject(bytes));
  }
}
const defaults = { horizon: 8, frames: 1, search_iterations: 3 };
const make = (algorithm, settings = {}, engine = new QuadraticEngine()) =>
  instantiateController(algorithm, engine, { ...defaults, ...settings });
const finish = (c) => {
  while (!c.advance()) {}
  return c.result();
};
const root = new Uint8Array([1, 2, 3]);

for (const algorithm of ["icem", "mppi"]) {
  test(`${algorithm} improves a known objective with heterogeneous action bounds`, () => {
    const engine = new QuadraticEngine(
      128,
      [
        { low: 2, high: 8 },
        { low: -4, high: -2 },
        { low: 0, high: 1 },
      ],
      [0.7, -0.5, 0.2],
    );
    const c = make(
      algorithm,
      { search_iterations: 5, mppi_temperature: 0.05 },
      engine,
    );
    c.begin(root, 19);
    const before = c.result().action;
    const after = finish(c).action;
    const loss = (u) =>
      u.reduce((sum, value, d) => {
        const b = engine.channels[d];
        return (
          sum +
          ((2 * (value - b.low)) / (b.high - b.low) - 1 - engine.target[d]) ** 2
        );
      }, 0);
    assert.ok(
      loss(after) < loss(before) * 0.5,
      `${loss(after)} vs ${loss(before)}`,
    );
  });
  test(`${algorithm} checkpoints are immutable and resume partial searches and later decisions`, () => {
    const c = make(algorithm);
    c.begin(root, 43);
    c.advance();
    c.advance();
    const saved = decodeObject(encodeObject(c.checkpoint()));
    const frozen = encodeObject(saved);
    const expected = finish(c);
    c.begin(root, 44);
    const warm = c.state.mean.slice();
    const next = finish(c);
    for (let trial = 0; trial < 2; trial++) {
      c.restore(saved);
      assert.deepEqual(finish(c), expected);
      c.begin(root, 44);
      assert.deepEqual(c.state.mean, warm);
      assert.deepEqual(finish(c), next);
      assert.deepEqual(encodeObject(saved), frozen);
    }
    const fresh = make(algorithm);
    fresh.restore(saved);
    assert.deepEqual(finish(fresh), expected);
    assert.throws(
      () => make(algorithm, { frames: 2 }).restore(saved),
      /checkpoint/,
    );
    const bad = structuredClone(saved);
    bad.state.samples[0] = NaN;
    assert.throws(() => c.restore(bad), /checkpoint/);
  });
  test(`${algorithm} supports a single sample, single step and early results`, () => {
    const c = make(
      algorithm,
      { horizon: 1, search_iterations: 1 },
      new QuadraticEngine(1),
    );
    c.begin(root, 1);
    assert.ok(c.result().action.every(Number.isFinite));
    assert.equal(c.advance(), true);
    const result = c.result();
    assert.equal(c.advance(), true);
    assert.deepEqual(c.result(), result);
    assert.equal(result.work.simulatorFrames, 1);
    assert.equal(result.budgetUsed, 1);
  });
  test(`${algorithm} rejects invalid dimensions and hyperparameters`, () => {
    for (const settings of [
      { horizon: 0 },
      { frames: -1 },
      { search_iterations: 1.5 },
      { search_iterations: 0 },
    ])
      assert.throws(() => make(algorithm, settings), /Invalid/);
    const key = algorithm === "icem" ? "icem_sigma" : "mppi_temperature";
    assert.throws(() => make(algorithm, { [key]: 0 }), /Invalid/);
  });
}

test("iCEM decays real simulator work, shifts elites, and resets uncertainty", () => {
  const engine = new QuadraticEngine(64),
    c = make("icem", { icem_decay: 2, icem_keep_fraction: 0.5 }, engine);
  c.begin(root, 7);
  const result = finish(c);
  assert.equal(result.work.simulatorFrames, (64 + 32 + 16) * 8);
  assert.equal(engine.frames, result.work.simulatorFrames);
  const previous = c.state.mean.slice(),
    elites = c.state.elites.slice();
  c.begin(root, 8);
  assert.deepEqual(c.state.mean.subarray(0, 7), previous.subarray(1));
  assert.deepEqual(c.state.elites.subarray(0, 7), elites.subarray(1, 8));
  assert.ok(c.state.std.every((v) => v === 0.5));
  c.advance();
  const keep = 3;
  assert.deepEqual(
    c.state.samples.subarray((64 - keep) * 8, (64 - keep + 1) * 8),
    c.state.elites.subarray(0, 8),
  );
});

test("power-law Gaussian noise has unit ensemble variance and temporal correlation", () => {
  const rng = new SeededRandom(321),
    out = new Float64Array(32);
  const stats = (beta) => {
    const noise = new ColoredNoise(32, beta);
    let squares = 0,
      differences = 0;
    for (let j = 0; j < 2000; j++) {
      noise.fill(rng, out);
      for (let t = 0; t < 32; t++) {
        squares += out[t] ** 2;
        if (t) differences += (out[t] - out[t - 1]) ** 2;
      }
    }
    return { variance: squares / 64000, variation: differences / 62000 };
  };
  const white = stats(0),
    red = stats(2);
  assert.ok(Math.abs(white.variance - 1) < 0.08);
  assert.ok(Math.abs(red.variance - 1) < 0.08);
  assert.ok(red.variation < white.variation / 4);
});

test("MPPI applies the Gaussian importance correction and a stable weighted update", () => {
  const c = make(
    "mppi",
    { horizon: 1, mppi_sigma: 0.5, mppi_temperature: 2 },
    new QuadraticEngine(2),
  );
  c.begin(root, 1);
  c.state.mean[0] = 0.25;
  c.state.samples.set([0.75, -0.25]);
  c.state.rewards.set([1e30, 1e30]);
  c.update();
  // Equal rewards: weights proportional to exp(-u * epsilon / sigma^2).
  const a = Math.exp(-0.5),
    b = Math.exp(0.5);
  const expected = (a * 0.75 + b * -0.25) / (a + b);
  assert.ok(Math.abs(c.state.mean[0] - expected) < 1e-6);
  // Saturation must not erase the latent noise used by importance sampling.
  c.state.mean[0] = 0;
  c.state.samples.set([3, -1]);
  c.state.rewards.fill(0);
  c.update();
  assert.equal(c.state.mean[0], 1);
});
