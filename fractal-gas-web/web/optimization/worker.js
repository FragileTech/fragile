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
      } else if (message.type === "geometry") {
        result = {
          status: engine.setGeometryDiagnostics(message.enabled),
          frame: engine.snapshot(),
        };
      } else if (["updateSettings", "setPopulation"].includes(message.type)) {
        const patch =
          message.type === "setPopulation"
            ? {
                walkers: message.walkers,
                removal_policy: message.removal_policy,
              }
            : message.patch;
        const next = engine.previewSettings(patch);
        const before = frameInfo(engine.snapshot());
        const count = Math.max(
          before.n,
          next.walkers +
            (["fmc", "wave_jump"].includes(config.algorithm) ? 1 : 0),
        );
        const bytes =
          new TextEncoder().encode(JSON.stringify(engine.status())).byteLength +
          6 * new TextEncoder().encode(JSON.stringify(next)).byteLength +
          4096;
        if (
          (12 + count * before.stride) * 8 + bytes >
          (message.remaining ?? Infinity)
        )
          throw new Error("Settings update exceeds the recording limit");
        const revision = engine.status().settings_revision;
        const status = engine.updateSettings(patch);
        config = engine.config();
        result = {
          config,
          frame: engine.snapshot(),
          status,
          changed: status.settings_revision !== revision,
        };
      } else if (message.type === "exportBasins") {
        result = { text: engine.exportBasins() };
      } else if (message.type === "importBasins") {
        const required =
          message.text.length * 6 +
          JSON.stringify(engine.status()).length * 6 +
          engine.snapshot().byteLength +
          8192;
        if (required > (message.remaining ?? Infinity))
          throw new Error("Archive update exceeds the recording limit");
        result = {
          status: engine.importBasins(message.text),
          frame: engine.snapshot(),
          config: engine.config(),
        };
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
        const settingsBytes =
          (config.controller_enabled
            ? 64 * (config.dimensions * 24 + 4096)
            : 0) +
          6 * new TextEncoder().encode(JSON.stringify(status)).byteLength +
          4096;
        if (
          (12 + maxN * before.stride) * 8 +
            metadataBytes +
            settingsBytes +
            (status.geometry_capacity_bytes || 0) >
          message.remaining
        )
          throw new Error(
            "Recording reached 64 MiB. Save this run and reset to continue.",
          );
        const frame = engine.step(),
          nextStatus = engine.status();
        result = {
          frame,
          status: nextStatus,
          config: engine.config(),
          simulationMs: performance.now() - start,
          roundChanged:
            nextStatus.controller?.round !== status.controller?.round,
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
