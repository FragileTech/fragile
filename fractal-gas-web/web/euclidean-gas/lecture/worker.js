import { demos, metadata, parameters } from "./catalog.js";
import { importedArchiveModel } from "./fractal.js";

let modulePromise,
  model,
  tracked = new Set(),
  queue = Promise.resolve();
async function wasm() {
  if (!modulePromise)
    modulePromise = import("../engine/cpu/gas.js")
      .then(async (module) => {
        await module.default();
        return module;
      })
      .catch((error) => {
        modulePromise = null;
        throw error;
      });
  return modulePromise;
}
const engine = {
  async inspectArchive(archive) {
    return (await wasm()).partv_archive(JSON.stringify(archive));
  },
  async geometry(request) {
    return (await wasm()).partv_geometry(JSON.stringify(request));
  },
  async analysis(request) {
    return (await wasm()).partv_analysis(JSON.stringify(request));
  },
  async defaults() {
    return JSON.parse((await wasm()).default_config());
  },
  async create(config) {
    const gas = await (await wasm()).BrowserGas.create(JSON.stringify(config));
    tracked.add(gas);
    const free = gas.free.bind(gas);
    gas.free = () => {
      if (tracked.delete(gas)) free();
    };
    return gas;
  },
  async restore(bytes) {
    const gas = await (await wasm()).BrowserGas.restore(new Uint8Array(bytes));
    tracked.add(gas);
    const free = gas.free.bind(gas);
    gas.free = () => {
      if (tracked.delete(gas)) free();
    };
    return gas;
  },
};
function dispose() {
  try {
    model?.dispose?.();
  } finally {
    model = null;
    for (const gas of [...tracked]) gas.free();
  }
}
async function dispatch(type, payload) {
  if (type === "catalog") return metadata;
  if (type === "dispose") {
    dispose();
    return null;
  }
  if (type === "initialize") {
    const demo = demos.find((demo) => demo.id === payload.id);
    if (!demo) throw new Error("Unknown experiment");
    if (
      !Number.isInteger(payload.seed) ||
      payload.seed < 0 ||
      payload.seed > 4294967295
    )
      throw new Error("Seed must be an integer from 0 to 4294967295");
    const params = parameters(demo, payload.params);
    const start = performance.now();
    if (
      payload.reuse &&
      model?.reuse?.({ id: demo.id, params, seed: payload.seed })
    )
      return {
        snapshot: model.snapshot(),
        initializationMs: performance.now() - start,
        params,
        checkpoint: Boolean(model.checkpoint),
        archive: Boolean(model.archive),
        initialTick: model.snapshot().step,
      };
    dispose();
    try {
      model = await demo.create({ params, seed: payload.seed, engine });
    } catch (error) {
      dispose();
      throw error;
    }
    return {
      snapshot: model.snapshot(),
      initializationMs: performance.now() - start,
      params,
      checkpoint: Boolean(model.checkpoint),
      archive: Boolean(model.archive),
    };
  }
  if (type === "archive_import") {
    dispose();
    model = await importedArchiveModel(payload, engine);
    return model.snapshot();
  }
  if (!model) throw new Error("Initialize an experiment first");
  if (type === "step") {
    await model.step();
    return model.snapshot();
  }
  if (type === "archive") {
    if (!model.archive)
      throw new Error("This reference experiment has no trajectory archive.");
    return await model.archive();
  }
  if (type === "checkpoint") {
    if (!model.checkpoint)
      throw new Error(
        "This experiment uses seed and parameter replay. Save the experiment JSON.",
      );
    return await model.checkpoint();
  }
  throw new Error("Unknown operation");
}
self.onmessage = ({ data: { id, type, payload } }) => {
  queue = queue.then(async () => {
    try {
      self.postMessage({ id, result: await dispatch(type, payload) });
    } catch (error) {
      self.postMessage({ id, error: String(error?.message || error) });
    }
  });
};
