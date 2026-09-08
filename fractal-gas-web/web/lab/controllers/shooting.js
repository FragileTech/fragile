import { SeededRandom, emptyTree } from "./registry.js";

export const shootingParameters = {
  search_iterations: {
    label: "Search rounds",
    help: "How many complete candidate-plan populations the controller evaluates before committing the first action. More rounds spend more simulation budget to refine the plan.",
    default: 3,
    min: 1,
    max: 128,
    step: 1,
  },
};
export function options(settings, definitions) {
  return Object.fromEntries(
    Object.entries(definitions).map(([key, d]) => {
      const value = settings[key] ?? d.default;
      if (
        !Number.isFinite(value) ||
        value < d.min ||
        value > d.max ||
        (d.step === 1 && !Number.isInteger(value))
      )
        throw new Error(`Invalid ${key}: expected ${d.min}–${d.max}`);
      return [key, value];
    }),
  );
}
export const clip = (x) => Math.max(-1, Math.min(1, x));
export function shift(sequence, dim) {
  // Repeat the last control when extending the receding horizon.
  sequence.copyWithin(0, dim);
}

// Shared rollout machinery. All coordinates are normalized to [-1, 1]; only
// this adapter knows channel bounds. World rows remain completely opaque.
export class ShootingController {
  constructor(engine, settings, algorithm, definitions) {
    this.engine = engine;
    this.algorithm = algorithm;
    this.config = options(settings, {
      ...shootingParameters,
      horizon: { min: 1, max: 4096, step: 1 },
      frames: { min: 1, max: 4096, step: 1 },
      ...definitions,
    });
    this.N = engine.worlds;
    this.D = engine.dim;
    this.H = this.config.horizon;
    this.L = this.H * this.D;
    if (
      !Number.isInteger(this.N) ||
      this.N < 1 ||
      this.N > 8192 ||
      !Number.isInteger(this.D) ||
      this.D < 1 ||
      engine.channels.length !== this.D ||
      engine.channels.some(
        (c) =>
          !Number.isFinite(c.low) ||
          !Number.isFinite(c.high) ||
          c.low > c.high,
      )
    )
      throw new Error("Invalid shooting engine descriptor");
    if (this.N * this.L > 32 * 1024 * 1024)
      throw new Error(
        "Controller samples exceed 128 MiB; reduce population or horizon",
      );
    this.signature = JSON.stringify([
      algorithm,
      this.config,
      this.N,
      engine.channels,
    ]);
    this.actions = new Float32Array(this.N * this.D);
    for (let i = 0; i < this.actions.length; i++) {
      const c = engine.channels[i % this.D];
      this.actions[i] = c.low + (c.high - c.low) / 2;
    }
    this.durations = new Int32Array(this.N);
    this.state = {
      initialized: false,
      needsSample: true,
      hasBest: false,
      iteration: 0,
      depth: 0,
      steps: 0,
      simulatorFrames: 0,
      trajectories: 0,
      activeCount: this.N,
      bestScore: -Number.MAX_VALUE,
      mean: new Float32Array(this.L),
      samples: new Float32Array(this.N * this.L),
      rewards: new Float64Array(this.N),
      bestAction: new Float32Array(this.D),
    };
  }
  begin(root, seed) {
    const s = this.state;
    this.root = root.slice();
    this.rng = new SeededRandom(seed);
    if (s.initialized) this.shiftPlan();
    s.initialized = true;
    s.iteration = s.depth = s.steps = s.simulatorFrames = s.trajectories = 0;
    s.needsSample = true;
    s.hasBest = false;
    s.bestScore = -Number.MAX_VALUE;
    s.activeCount = this.N;
    s.rewards.fill(0);
    this.resetSearch();
    this.engine.broadcast(this.root);
  }
  shiftPlan() {
    shift(this.state.mean, this.D);
  }
  resetSearch() {}
  advance() {
    const s = this.state,
      e = this.engine;
    if (!s.initialized)
      throw new Error("Call begin before advancing a controller");
    if (s.iteration === this.config.search_iterations) return true;
    if (s.needsSample) {
      this.sample();
      s.rewards.fill(0);
      s.depth = 0;
      s.needsSample = false;
      e.broadcast(this.root);
    }
    // Zero durations skip physics for unused capacity, including iCEM's
    // decayed population. The native batch still copies these opaque rows.
    this.durations.fill(0);
    this.durations.fill(this.config.frames, 0, s.activeCount);
    for (let w = 0; w < s.activeCount; w++)
      for (let d = 0; d < this.D; d++) {
        const c = e.channels[d],
          u = clip(s.samples[w * this.L + s.depth * this.D + d]);
        this.actions[w * this.D + d] = c.low + ((u + 1) * (c.high - c.low)) / 2;
      }
    e.step(this.actions, this.durations);
    const results = e.results();
    for (let w = 0; w < s.activeCount; w++) {
      if (
        !Number.isFinite(results[w * 4]) ||
        !Number.isFinite(results[w * 4 + 1])
      )
        throw new Error("Non-finite rollout result");
      s.rewards[w] += results[w * 4];
      s.simulatorFrames += results[w * 4 + 1];
    }
    s.steps++;
    if (++s.depth === this.H) {
      s.trajectories += s.activeCount;
      this.update();
      s.iteration++;
      s.needsSample = true;
    }
    return s.iteration === this.config.search_iterations;
  }
  result() {
    const s = this.state,
      e = this.engine,
      metrics = e.metrics();
    const selected = s.hasBest ? s.bestAction : s.mean;
    const action = Float32Array.from(
      e.channels,
      (c, d) => c.low + ((clip(selected[d]) + 1) * (c.high - c.low)) / 2,
    );
    metrics[8] = s.steps;
    metrics[9] = metrics[3] / this.N;
    metrics[11] =
      s.rewards.subarray(0, s.activeCount).reduce((a, b) => a + b, 0) /
      s.activeCount;
    metrics[12] = Math.max(...s.rewards.subarray(0, s.activeCount));
    return {
      action,
      metrics,
      cloud: e.states().slice(0, s.activeCount * e.stride),
      tree: emptyTree(e, this.root),
      budgetUsed: s.steps / (this.H * this.config.search_iterations),
      work: {
        simulatorFrames: s.simulatorFrames,
        trajectories: s.trajectories,
      },
    };
  }
  checkpoint() {
    if (!this.state.initialized) throw new Error("No search to checkpoint");
    return {
      version: 1,
      algorithm: this.algorithm,
      signature: this.signature,
      root: this.root.slice(),
      bytes: this.engine.checkpoint(),
      rng: this.rng.state,
      state: structuredClone(this.state),
    };
  }
  restore(saved) {
    const bad = () => {
      throw new Error("Invalid shooting controller checkpoint");
    };
    if (
      saved?.version !== 1 ||
      saved.algorithm !== this.algorithm ||
      saved.signature !== this.signature ||
      !(saved.root instanceof Uint8Array) ||
      !saved.root.length ||
      !(saved.bytes instanceof Uint8Array) ||
      !Number.isInteger(saved.rng) ||
      saved.rng <= 0 ||
      saved.rng > 0xffffffff
    )
      bad();
    const s = saved.state;
    if (!s || Object.keys(s).length !== Object.keys(this.state).length) bad();
    for (const [key, value] of Object.entries(this.state)) {
      const candidate = s[key];
      if (ArrayBuffer.isView(value)) {
        if (
          !(candidate instanceof value.constructor) ||
          candidate.length !== value.length ||
          !candidate.every(Number.isFinite)
        )
          bad();
      } else if (
        typeof candidate !== typeof value ||
        (typeof value === "number" && !Number.isFinite(candidate))
      )
        bad();
    }
    for (const key of [
      "iteration",
      "depth",
      "steps",
      "simulatorFrames",
      "trajectories",
      "activeCount",
    ])
      if (!Number.isSafeInteger(s[key]) || s[key] < 0) bad();
    if (
      !s.initialized ||
      s.iteration > this.config.search_iterations ||
      s.depth > this.H ||
      s.activeCount < 1 ||
      s.activeCount > this.N ||
      s.steps !== s.iteration * this.H + (s.needsSample ? 0 : s.depth) ||
      (!s.needsSample && s.depth >= this.H) ||
      (s.iteration === this.config.search_iterations && !s.needsSample) ||
      s.mean.some((v) => v < -1 || v > 1) ||
      s.bestAction.some((v) => v < -1 || v > 1)
    )
      bad();
    this.validateState(s, bad);
    this.engine.restoreCheckpoint(saved.bytes);
    this.root = saved.root.slice();
    this.rng = new SeededRandom(saved.rng);
    this.state = structuredClone(s);
  }
  validateState() {}
}
