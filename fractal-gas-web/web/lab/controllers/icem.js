import { registerController } from "./registry.js";
import {
  ShootingController,
  shootingParameters,
  clip,
  shift,
} from "./shooting.js";
import { ColoredNoise } from "./colored-noise.js";

export const icemParameters = {
  ...shootingParameters,
  icem_elite_fraction: {
    label: "Elite fraction",
    help: "Fraction of sampled plans retained as elites after each iCEM round. Higher values make updates steadier but less selective.",
    default: 0.1,
    min: 0.01,
    max: 0.5,
    step: 0.01,
  },
  icem_keep_fraction: {
    label: "Reuse elite fraction",
    help: "Fraction of the previous elite plans reused in the next round. Reuse reduces noise and computation while preserving promising sequences.",
    default: 0.3,
    min: 0,
    max: 1,
    step: 0.05,
  },
  icem_decay: {
    label: "Population decay factor",
    help: "How quickly iCEM reduces its active rollout population over search rounds. Larger values focus later rounds on fewer candidates.",
    default: 1.25,
    min: 1,
    max: 10,
    step: 0.05,
  },
  icem_beta: {
    label: "Noise spectral exponent",
    help: "Temporal correlation of iCEM exploration noise. Zero is white noise; larger values make neighboring actions vary more smoothly.",
    default: 2,
    min: 0,
    max: 4,
    step: 0.1,
  },
  icem_alpha: {
    label: "Distribution momentum",
    help: "How much of the previous mean and spread iCEM keeps when fitting the elites. Higher values damp sudden distribution changes.",
    default: 0.1,
    min: 0,
    max: 0.99,
    step: 0.05,
  },
  icem_sigma: {
    label: "Initial normalized noise",
    help: "Initial standard deviation of normalized action samples. Larger values explore more broadly at the start of each plan.",
    default: 0.5,
    min: 0.001,
    max: 2,
    step: 0.05,
  },
  icem_min_sigma: {
    label: "Minimum normalized noise",
    help: "Lower bound on iCEM's action spread. It prevents the search distribution from collapsing and losing exploration completely.",
    default: 0.01,
    min: 0.0001,
    max: 1,
    step: 0.001,
  },
};
// Pinneri et al., CoRL 2020: https://proceedings.mlr.press/v155/pinneri21a.html
export class ICEMController extends ShootingController {
  constructor(engine, settings) {
    super(engine, settings, "icem", icemParameters);
    if (this.config.icem_min_sigma > this.config.icem_sigma)
      throw new Error("icem_min_sigma must not exceed icem_sigma");
    this.K = Math.max(1, Math.floor(this.N * this.config.icem_elite_fraction));
    Object.assign(this.state, {
      std: new Float32Array(this.L),
      elites: new Float32Array(this.K * this.L),
      eliteCount: 0,
    });
    this.noise = new ColoredNoise(this.H, this.config.icem_beta);
  }
  shiftPlan() {
    super.shiftPlan();
    const s = this.state;
    for (let w = 0; w < s.eliteCount; w++) {
      const elite = s.elites.subarray(w * this.L, (w + 1) * this.L);
      shift(elite, this.D);
      for (let d = 0; d < this.D; d++)
        elite[this.L - this.D + d] = clip(
          s.mean[this.L - this.D + d] +
            this.config.icem_sigma * this.rng.normal(),
        );
    }
  }
  resetSearch() {
    this.state.std.fill(this.config.icem_sigma);
  }
  sample() {
    const s = this.state,
      c = this.config;
    // `walkers` is the total capacity including reused elites and the mean.
    s.activeCount = Math.min(
      this.N,
      Math.max(2 * this.K, Math.floor(this.N / c.icem_decay ** s.iteration)),
    );
    const meanSlot = s.iteration === c.search_iterations - 1 ? 1 : 0;
    const keep = Math.min(
      Math.floor(s.eliteCount * c.icem_keep_fraction),
      s.activeCount - meanSlot,
    );
    for (let w = 0; w < s.activeCount - keep - meanSlot; w++) {
      for (let d = 0; d < this.D; d++)
        this.noise.fill(this.rng, s.samples, w * this.L + d, this.D);
      for (let k = 0; k < this.L; k++) {
        const i = w * this.L + k;
        s.samples[i] = clip(s.mean[k] + s.std[k] * s.samples[i]);
      }
    }
    // Re-evaluate retained sequences from the current root, also for stochastic engines.
    s.samples.set(
      s.elites.subarray(0, keep * this.L),
      (s.activeCount - keep - meanSlot) * this.L,
    );
    if (meanSlot) s.samples.set(s.mean, (s.activeCount - 1) * this.L);
  }
  update() {
    const s = this.state,
      c = this.config;
    const ranked = Array.from({ length: s.activeCount }, (_, i) => i).sort(
      (a, b) => s.rewards[b] - s.rewards[a] || a - b,
    );
    if (s.rewards[ranked[0]] > s.bestScore) {
      s.bestScore = s.rewards[ranked[0]];
      s.bestAction.set(
        s.samples.subarray(ranked[0] * this.L, ranked[0] * this.L + this.D),
      );
      s.hasBest = true;
    }
    s.eliteCount = this.K;
    for (let w = 0; w < this.K; w++)
      s.elites.set(
        s.samples.subarray(ranked[w] * this.L, (ranked[w] + 1) * this.L),
        w * this.L,
      );
    for (let k = 0; k < this.L; k++) {
      let mean = 0,
        variance = 0;
      for (let w = 0; w < this.K; w++)
        mean += s.elites[w * this.L + k] / this.K;
      for (let w = 0; w < this.K; w++)
        variance += (s.elites[w * this.L + k] - mean) ** 2 / this.K;
      s.mean[k] = c.icem_alpha * s.mean[k] + (1 - c.icem_alpha) * mean;
      s.std[k] = Math.max(
        c.icem_min_sigma,
        c.icem_alpha * s.std[k] + (1 - c.icem_alpha) * Math.sqrt(variance),
      );
    }
  }
  validateState(s, bad) {
    if (
      ![0, this.K].includes(s.eliteCount) ||
      s.std.some((v) => v <= 0) ||
      s.elites.some((v) => v < -1 || v > 1) ||
      s.samples.some((v) => v < -1 || v > 1)
    )
      bad();
  }
}
registerController("icem", {
  label: "iCEM · improved cross-entropy",
  parameters: icemParameters,
  worlds: (s) => s.walkers,
  create: ({ engine, settings }) => new ICEMController(engine, settings),
});
