import { validateLabConfig } from "./config.js";

let gas = null;
let module = null;
let config = null;
let queue = Promise.resolve();
const modules = new Map();

async function load(profile) {
  if (!modules.has(profile)) {
    const promise = import(`./engine/${profile}/gas.js`)
      .then(async (m) => {
        await m.default({
          module_or_path: new URL(
            `./engine/${profile}/gas_bg.wasm`,
            import.meta.url,
          ),
        });
        return m;
      })
      .catch((error) => {
        modules.delete(profile);
        throw error;
      });
    modules.set(profile, promise);
  }
  return modules.get(profile);
}
async function prepare(nextConfig) {
  validateLabConfig(nextConfig);
  const gpu = nextConfig.gas.backend === "wgpu";
  if (gpu && nextConfig.gas.precision !== "f32")
    throw new Error(
      "Browser WebGPU requires f32. Select WASM CPU explicitly for f64.",
    );
  let device = "WASM CPU · one worker";
  if (gpu) {
    if (!self.navigator.gpu)
      throw new Error(
        "WebGPU is unavailable in this browser or context. Select WASM CPU; no fallback was applied.",
      );
    const adapter = await self.navigator.gpu.requestAdapter();
    if (!adapter)
      throw new Error("No WebGPU adapter was found. No fallback was applied.");
    if (
      nextConfig.gas.max_batch_elements * 4 >
      adapter.limits.maxStorageBufferBindingSize
    ) {
      throw new Error(
        "The configured batch memory budget exceeds this adapter's storage-buffer limit. Reduce max_batch_elements in the configuration.",
      );
    }
    device = `WebGPU · ${adapter.info?.description || adapter.info?.device || "adapter"} · host-orchestrated`;
  }
  return { module: await load(gpu ? "webgpu" : "cpu"), device };
}
function transferFrame(frame) {
  const Float = config.gas.precision === "f64" ? Float64Array : Float32Array;
  const transfers = [];
  const float = (values) => {
    const array = new Float(values);
    transfers.push(array.buffer);
    return array;
  };
  const integer = (values) => {
    const array = new Uint32Array(values);
    transfers.push(array.buffer);
    return array;
  };
  for (const field of Object.values(frame.population.observations.fields))
    field.values = float(field.values);
  frame.population.rewards.raw = float(frame.population.rewards.raw);
  if (frame.report) {
    frame.report.pre_clone_rewards.raw = float(
      frame.report.pre_clone_rewards.raw,
    );
    for (const key of [
      "oriented_reward",
      "separation",
      "reward_z",
      "diversity_z",
      "fitness",
    ])
      frame.report.pre_clone_fitness[key] = float(
        frame.report.pre_clone_fitness[key],
      );
    frame.report.distance_companions.indices = integer(
      frame.report.distance_companions.indices,
    );
    frame.report.cloning_companions.indices = integer(
      frame.report.cloning_companions.indices,
    );
  }
  return { frame, transfers };
}
async function handle({ id, type, payload }) {
  try {
    let result;
    let transfers = [];
    if (type === "defaults") {
      const m = await load("cpu");
      result = {
        config: JSON.parse(m.default_config()),
        capabilities: m.capabilities(),
      };
    } else if (type === "initialize") {
      const prepared = await prepare(payload);
      const next = await prepared.module.BrowserGas.create(
        JSON.stringify(payload),
      );
      gas?.free();
      gas = next;
      module = prepared.module;
      config = JSON.parse(gas.config_json());
      const packed = transferFrame(gas.snapshot());
      result = { config, device: prepared.device, frame: packed.frame };
      transfers = packed.transfers;
    } else if (type === "restore") {
      const decoder = await load("cpu");
      const savedConfig = JSON.parse(
        decoder.checkpoint_config(new Uint8Array(payload)),
      );
      const prepared = await prepare(savedConfig);
      const m = prepared.module;
      const next = await m.BrowserGas.restore(new Uint8Array(payload));
      const nextConfig = JSON.parse(next.config_json());
      gas?.free();
      gas = next;
      module = m;
      config = nextConfig;
      const packed = transferFrame(gas.snapshot());
      result = {
        config,
        frame: packed.frame,
        device: `${config.gas.backend === "cpu" ? "WASM CPU" : "WebGPU hybrid"} · restored`,
      };
      transfers = packed.transfers;
    } else {
      if (!gas) throw new Error("Initialize a run first.");
      if (type === "step" || type === "snapshot") {
        const frame =
          type === "step"
            ? await gas.step(payload?.count || 1)
            : gas.snapshot();
        const packed = transferFrame(frame);
        result = packed.frame;
        transfers = packed.transfers;
      } else if (type === "checkpoint") {
        result = gas.checkpoint();
        transfers = [result.buffer];
      } else if (type === "landscape")
        result = gas.landscape(
          payload.x,
          payload.y,
          payload.resolution,
          payload.center,
        );
      else throw new Error(`Unknown worker request: ${type}`);
    }
    self.postMessage({ id, ok: true, result }, transfers);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
}
self.onmessage = ({ data }) => {
  queue = queue.then(
    () => handle(data),
    () => handle(data),
  );
};
