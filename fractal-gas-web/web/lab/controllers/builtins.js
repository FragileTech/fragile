import { registerController, SeededRandom, emptyTree } from "./registry.js";
import { shootingParameters } from "./shooting.js";
registerController("fmc", {
  label: "Fractal Monte Carlo",
  create: ({ engine, settings }) => ({
    begin(root, seed) {
      this.root = root;
      engine.restore(root);
      engine.begin(settings, seed);
    },
    advance() {
      return engine.advance();
    },
    result() {
      return {
        action: engine.action(),
        budgetUsed: engine.metrics()[8] / settings.horizon,
        tree: engine.tree(),
        cloud: engine.states(true, settings.walkers),
        metrics: engine.metrics(),
      };
    },
    checkpoint() {
      return {
        version: 1,
        algorithm: "fmc",
        root: this.root,
        bytes: engine.checkpoint(),
      };
    },
    restore(saved) {
      engine.restoreCheckpoint(saved.bytes);
      this.root = saved.root;
    },
  }),
});
registerController("random", {
  label: "Seeded random baseline",
  create: ({ engine }) => ({
    begin(root, seed) {
      engine.restore(root);
      this.root = root;
      this.rng = new SeededRandom(seed);
      this.action = engine.neutralAction();
    },
    advance() {
      this.action = Float32Array.from(
        engine.channels,
        (c) => c.low + this.rng.uniform() * (c.high - c.low),
      );
      return true;
    },
    result() {
      const metrics = engine.metrics();
      metrics[8] = 1;
      return {
        action: this.action.slice(),
        budgetUsed: 1,
        tree: emptyTree(engine, this.root),
        cloud: engine.states(),
        metrics,
      };
    },
    checkpoint() {
      return {
        version: 1,
        algorithm: "random",
        bytes: engine.checkpoint(),
        rng: this.rng.state,
        root: this.root,
        action: this.action,
      };
    },
    restore(s) {
      engine.restoreCheckpoint(s.bytes);
      this.rng = new SeededRandom(s.rng);
      this.root = s.root;
      this.action = s.action;
    },
  }),
});
registerController("cem", {
  label: "Cross-entropy shooting",
  parameters: shootingParameters,
  worlds: (s) => s.walkers,
  create: ({ engine, settings }) =>
    new CrossEntropyController(engine, settings),
});
// Generic shooting optimizer: it only uses descriptors, state broadcasts,
// batched actions/rewards and snapshots. No body geometry or task assumptions.
class CrossEntropyController {
  constructor(engine, settings) {
    this.engine = engine;
    this.settings = settings;
  }
  begin(root, seed) {
    this.root = root;
    this.rng = new SeededRandom(seed);
    this.iteration = 0;
    this.depth = 0;
    this.steps = 0;
    this.bestAction = undefined;
    const e = this.engine,
      H = this.settings.horizon,
      D = e.dim;
    this.mean = new Float32Array(H * D);
    this.std = new Float32Array(H * D);
    for (let i = 0; i < H * D; i++) {
      const c = e.channels[i % D];
      this.mean[i] = (c.low + c.high) / 2;
      this.std[i] = (c.high - c.low) / 2;
    }
    this.sample();
  }
  sample() {
    const e = this.engine,
      N = e.worlds,
      H = this.settings.horizon,
      D = e.dim;
    if (N * H * D > 32 * 1024 * 1024)
      throw new Error(
        "Controller samples exceed 128 MiB; reduce population or horizon",
      );
    this.samples = new Float32Array(N * H * D);
    this.rewards = new Float32Array(N);
    this.depth = 0;
    e.broadcast(this.root);
    for (let w = 0; w < N; w++)
      for (let k = 0; k < H * D; k++) {
        const c = e.channels[k % D];
        this.samples[w * H * D + k] = Math.max(
          c.low,
          Math.min(c.high, this.mean[k] + this.std[k] * this.rng.normal()),
        );
      }
  }
  advance() {
    if (this.iteration >= (this.settings.search_iterations || 3)) return true;
    const e = this.engine,
      N = e.worlds,
      H = this.settings.horizon,
      D = e.dim,
      u = new Float32Array(N * D);
    for (let w = 0; w < N; w++)
      u.set(
        this.samples.subarray(
          (w * H + this.depth) * D,
          (w * H + this.depth + 1) * D,
        ),
        w * D,
      );
    e.step(u, this.settings.frames);
    this.steps++;
    const result = e.results();
    for (let w = 0; w < N; w++) this.rewards[w] += result[w * 4];
    if (++this.depth < H) return false;
    const ranked = Array.from({ length: N }, (_, i) => i).sort(
        (a, b) => this.rewards[b] - this.rewards[a] || a - b,
      ),
      K = Math.max(1, Math.floor(N * 0.15));
    for (let k = 0; k < H * D; k++) {
      let sum = 0;
      for (let j = 0; j < K; j++) sum += this.samples[ranked[j] * H * D + k];
      const mean = sum / K;
      let variance = 0;
      for (let j = 0; j < K; j++)
        variance += (this.samples[ranked[j] * H * D + k] - mean) ** 2;
      this.mean[k] = mean;
      this.std[k] = Math.max(0.03, Math.sqrt(variance / K));
    }
    this.bestAction = this.samples.slice(
      ranked[0] * H * D,
      ranked[0] * H * D + D,
    );
    if (++this.iteration >= (this.settings.search_iterations || 3)) return true;
    this.sample();
    return false;
  }
  result() {
    const e = this.engine,
      metrics = e.metrics();
    metrics[8] = this.steps;
    metrics[9] = metrics[3] / e.worlds;
    metrics[11] = this.rewards.reduce((s, v) => s + v, 0) / e.worlds;
    metrics[12] = Math.max(...this.rewards);
    return {
      action: (this.bestAction || this.mean.subarray(0, e.dim)).slice(),
      budgetUsed:
        this.steps /
        (this.settings.horizon * (this.settings.search_iterations || 3)),
      tree: emptyTree(e, this.root),
      cloud: e.states(),
      metrics,
    };
  }
  checkpoint() {
    return {
      version: 1,
      algorithm: "cem",
      bytes: this.engine.checkpoint(),
      root: this.root,
      rng: this.rng.state,
      iteration: this.iteration,
      depth: this.depth,
      steps: this.steps,
      mean: this.mean,
      std: this.std,
      samples: this.samples,
      rewards: this.rewards,
      bestAction: this.bestAction,
    };
  }
  restore(s) {
    const e = this.engine,
      H = this.settings.horizon,
      D = e.dim,
      N = e.worlds;
    for (const [key, size] of Object.entries({
      mean: H * D,
      std: H * D,
      samples: N * H * D,
      rewards: N,
    }))
      if (
        !(s[key] instanceof Float32Array) ||
        s[key].length !== size ||
        !s[key].every(Number.isFinite)
      )
        throw new Error("Invalid controller checkpoint " + key);
    if (
      !Number.isInteger(s.iteration) ||
      s.iteration < 0 ||
      s.iteration > (this.settings.search_iterations || 3) ||
      !Number.isInteger(s.depth) ||
      s.depth < 0 ||
      s.depth > H ||
      !Number.isInteger(s.steps) ||
      s.steps < 0 ||
      s.std.some((v) => v < 0)
    )
      throw new Error("Invalid controller checkpoint progress");
    if (
      s.bestAction &&
      (!(s.bestAction instanceof Float32Array) ||
        s.bestAction.length !== D ||
        !s.bestAction.every(Number.isFinite))
    )
      throw new Error("Invalid checkpoint action");
    e.restoreCheckpoint(s.bytes);
    this.rng = new SeededRandom(s.rng);
    for (const key of [
      "root",
      "iteration",
      "depth",
      "steps",
      "mean",
      "std",
      "samples",
      "rewards",
      "bestAction",
    ])
      this[key] = s[key];
  }
}
