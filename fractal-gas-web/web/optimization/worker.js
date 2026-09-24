import create from "./engine/optimization.mjs";
import { NativeOptimization, frameInfo } from "./native.js";
const ready = create().then((module) => new NativeOptimization(module));
let config;
let queue = Promise.resolve();
self.onmessage = ({ data: message }) => {
  queue = queue.then(async () => {
    try {
      const engine = await ready;
      let result;
      const start = performance.now();
      if (message.type === "catalog") result = engine.catalog();
      else if (message.type === "create") {
        config = engine.create(message.config);
        result = { config, frame: engine.snapshot(), status: engine.status() };
      } else if (message.type === "setPopulation") {
        const before = frameInfo(engine.snapshot());
        const count = Math.max(
          before.n,
          message.walkers +
            (["fmc", "wave_jump"].includes(config.algorithm) ? 1 : 0),
        );
        if (
          (12 + count * before.stride) * 8 + 2048 >
          (message.remaining ?? Infinity)
        )
          throw new Error("Population update exceeds the recording limit");
        const status = engine.setPopulation(
          message.walkers,
          message.removal_policy,
        );
        Object.assign(config, {
          walkers: message.walkers,
          removal_policy: message.removal_policy,
        });
        result = { config, frame: engine.snapshot(), status };
      } else if (message.type === "step") {
        const before = frameInfo(engine.snapshot());
        const status = engine.status();
        if (status.finished)
          throw new Error(`Optimizer finished: ${status.stop_reason}`);
        if (status.budget_exhausted)
          throw new Error(
            "Evaluation budget reached before the next complete generation or step.",
          );
        const maxN = status.next_population;
        // Reserve bounded status JSON space before advancing the optimizer.
        const metadataBytes =
          config.algorithm === "graph"
            ? JSON.stringify(status).length * 2 +
              maxN * (config.dimensions * 24 + 128)
            : ["wave", "fmc", "wave_jump"].includes(config.algorithm) ||
                config.algorithm.startsWith("cmaes_")
              ? 2048
              : 0;
        if ((12 + maxN * before.stride) * 8 + metadataBytes > message.remaining)
          throw new Error(
            "Recording reached 64 MiB. Save this run and reset to continue.",
          );
        result = {
          frame: engine.step(),
          status: engine.status(),
          simulationMs: performance.now() - start,
        };
      } else if (message.type === "surface") {
        const { resolution, axes, slice } = message;
        if (
          ![64, 128, 192].includes(resolution) ||
          slice.length !== config.dimensions ||
          !slice.every(Number.isFinite)
        )
          throw new Error("Invalid objective slice");
        const count = resolution * resolution,
          values = new Float64Array(count);
        const chunk = Math.max(
          1,
          Math.min(256, Math.floor(1048576 / config.dimensions)),
        );
        for (let start = 0; start < count; start += chunk) {
          const n = Math.min(chunk, count - start),
            positions = new Float32Array(n * config.dimensions);
          for (let j = 0; j < n; j++) {
            positions.set(slice, j * config.dimensions);
            positions[j * config.dimensions + axes[0]] =
              config.low +
              ((config.high - config.low) * ((start + j) % resolution)) /
                (resolution - 1);
            if (config.dimensions > 1)
              positions[j * config.dimensions + axes[1]] =
                config.low +
                ((config.high - config.low) *
                  Math.floor((start + j) / resolution)) /
                  (resolution - 1);
          }
          values.set(engine.sample(positions, config.dimensions), start);
        }
        result = { values };
      } else if (message.type === "sample")
        result = {
          values: engine.sample(message.positions, config.dimensions),
        };
      else throw new Error("Unknown worker request");
      const transfers = [result.frame?.buffer, result.values?.buffer].filter(
        Boolean,
      );
      self.postMessage({ id: message.id, result }, transfers);
    } catch (error) {
      self.postMessage({ id: message.id, error: error.message });
    }
  });
};
