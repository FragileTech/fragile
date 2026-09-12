import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";

await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});

test("Actual harmonic WASM relaxation matches independent BAOAB moments across 64 replicas", async () => {
  const N = 64,
    replicas = 64,
    h = 0.04,
    damping = Math.exp(-h),
    innovationVariance = 1 - damping * damping,
    checkpoints = [1, 5, 25, 50, 100, 200, 300];
  // Derive the transition directly from the five physical stages. This test
  // deliberately does not use the lecture's Gaussian-reference helpers.
  function transition(x, v, innovation = 0) {
    v -= (h * x) / 2;
    x += (h * v) / 2;
    v = damping * v + innovation;
    x += (h * v) / 2;
    v -= (h * x) / 2;
    return [x, v];
  }
  const colX = transition(1, 0),
    colV = transition(0, 1),
    noise = transition(0, 0, 1),
    matrix = [
      [colX[0], colV[0]],
      [colX[1], colV[1]],
    ],
    transpose = (A) => A[0].map((_, j) => A.map((row) => row[j])),
    multiply = (A, B) =>
      A.map((row) =>
        B[0].map((_, j) =>
          row.reduce((sum, value, k) => sum + value * B[k][j], 0),
        ),
      );
  let covariance = [
      [1 / 3, 0],
      [0, 0],
    ],
    initialPropagation = [
      [1, 0],
      [0, 1],
    ];
  const predictions = [];
  for (let step = 1; step <= checkpoints.at(-1); step++) {
    covariance = multiply(multiply(matrix, covariance), transpose(matrix)).map(
      (row, i) =>
        row.map((value, j) => value + innovationVariance * noise[i] * noise[j]),
    );
    initialPropagation = multiply(matrix, initialPropagation);
    if (checkpoints.includes(step))
      predictions.push({
        mean: 2 * covariance[0][0],
        // Uniform[-1,1] has fourth cumulant -2/15. Keep its contribution
        // rather than approximating early-time position samples as Gaussian.
        se: Math.sqrt(
          (4 * covariance[0][0] ** 2 -
            (4 / 15) * initialPropagation[0][0] ** 4) /
            (N * replicas),
        ),
      });
  }
  const measured = checkpoints.map(() => 0);
  for (let replica = 0; replica < replicas; replica++) {
    const config = JSON.parse(default_config());
    Object.assign(config, {
      benchmark: "sphere",
      potential: "quadratic",
      reward_shift: [],
      walkers: N,
      dimensions: 2,
      initial_lower: -1,
      initial_upper: 1,
    });
    Object.assign(config.gas, {
      backend: "cpu",
      precision: "f64",
      seed: 10007 + 7919 * replica,
      boundary: { kind: "unbounded" },
      clone_transform: {
        position_field: null,
        jitter: null,
        jitter_amplitude: 0,
        velocity_field: null,
        restitution: null,
      },
      kinetic: {
        integrator: {
          kind: "baoab",
          positions: "positions",
          velocities: "velocities",
          dt: h,
          friction: 1,
        },
        noise: {
          innovation: "gaussian",
          geometry: {
            kind: "isotropic",
            scale: { kind: "constant", values: [Math.SQRT2] },
          },
        },
      },
    });
    config.gas.fitness.reward_exponent = 0;
    config.gas.fitness.diversity_exponent = 0;
    const run = await BrowserGas.create(JSON.stringify(config));
    try {
      let step = 0;
      for (const [index, checkpoint] of checkpoints.entries()) {
        let frame;
        while (step < checkpoint) {
          const count = Math.min(16, checkpoint - step);
          frame = await run.step(count);
          step += count;
        }
        const positions = frame.population.observations.fields.positions.values;
        measured[index] +=
          positions.reduce((sum, x) => sum + x * x, 0) / (N * replicas);
      }
    } finally {
      run.free();
    }
  }
  for (const [index, prediction] of predictions.entries()) {
    const z = Math.abs(measured[index] - prediction.mean) / prediction.se;
    // Deterministic seeds; four SE also leave room for portable RNG math.
    assert.ok(
      z < 4,
      `t=${h * checkpoints[index]}: measured ${measured[index]}, ` +
        `predicted ${prediction.mean}, deviation ${z} SE`,
    );
  }
  assert.ok(Math.abs(predictions.at(-1).mean - 2) < 0.0001);
});
