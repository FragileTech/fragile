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
        result = { config, frame: engine.snapshot() };
      } else if (message.type === "step") {
        const before = frameInfo(engine.snapshot());
        const maxN =
          config.algorithm === "graph"
            ? Math.min(config.max_walkers, before.n + config.walkers)
            : before.n;
        if ((12 + maxN * before.stride) * 8 > message.remaining)
          throw new Error(
            "Recording reached 64 MiB. Save this run and reset to continue.",
          );
        result = {
          frame: engine.step(),
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
