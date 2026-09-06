import { registerController } from "./registry.js";
import { ShootingController, shootingParameters, clip } from "./shooting.js";

export const mppiParameters = {
  ...shootingParameters,
  mppi_temperature: {
    label: "Temperature (reward units)",
    default: 1,
    min: 0.000001,
    max: 1000000,
    step: 0.1,
  },
  mppi_sigma: {
    label: "Normalized exploration noise",
    default: 0.5,
    min: 0.001,
    max: 2,
    step: 0.05,
  },
};
// Information-theoretic MPPI with a fixed diagonal Gaussian covariance.
// Williams et al.: https://arxiv.org/abs/1707.02342 (Algorithm 1, without SGF).
export class MPPIController extends ShootingController {
  constructor(engine, settings) {
    super(engine, settings, "mppi", mppiParameters);
    this.weights = new Float64Array(this.N);
  }
  sample() {
    const s = this.state;
    s.activeCount = this.N;
    for (let w = 0; w < this.N; w++)
      for (let k = 0; k < this.L; k++)
        // Retain the *unclipped* Gaussian latent sample for the likelihood
        // correction and update; the shared rollout adapter clips actuation.
        s.samples[w * this.L + k] =
          s.mean[k] + this.config.mppi_sigma * this.rng.normal();
  }
  update() {
    const s = this.state,
      c = this.config,
      weights = this.weights;
    const reference = Math.max(...s.rewards);
    let max = -Infinity;
    for (let w = 0; w < this.N; w++) {
      let correction = 0;
      for (let k = 0; k < this.L; k++)
        correction +=
          (s.mean[k] * (s.samples[w * this.L + k] - s.mean[k])) /
          c.mppi_sigma ** 2;
      // Cost = -reward + lambda * u^T Sigma^-1 epsilon. The omitted
      // u^T Sigma^-1 u term is constant across samples and cancels in softmax.
      weights[w] = (s.rewards[w] - reference) / c.mppi_temperature - correction;
      max = Math.max(max, weights[w]);
    }
    let total = 0;
    for (let w = 0; w < this.N; w++)
      total += weights[w] = Math.exp(weights[w] - max);
    for (let k = 0; k < this.L; k++) {
      let delta = 0;
      for (let w = 0; w < this.N; w++)
        delta += (weights[w] / total) * (s.samples[w * this.L + k] - s.mean[k]);
      s.mean[k] = clip(s.mean[k] + delta);
    }
    s.bestAction.set(s.mean.subarray(0, this.D));
    s.hasBest = true;
  }
}
registerController("mppi", {
  label: "MPPI · path integral control",
  parameters: mppiParameters,
  worlds: (s) => s.walkers,
  create: ({ engine, settings }) => new MPPIController(engine, settings),
});
