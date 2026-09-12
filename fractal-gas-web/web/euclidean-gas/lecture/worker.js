import { demos, metadata, parameters } from "./catalog.js";
import { renderSnapshot } from "./run-model.js";
let model,
  queue = Promise.resolve();
let compiled;
async function wasm() {
  if (!compiled)
    compiled = import("../engine/cpu/gas.js")
      .then(async (module) => {
        await module.default();
        return module;
      })
      .catch((error) => {
        compiled = null;
        throw error;
      });
  return compiled;
}
const engine = {
  async lectureCreate(request) {
    return (await wasm()).LectureExperiment.create(JSON.stringify(request));
  },
};
function dispose() {
  model?.dispose?.();
  model = null;
}
async function importEvidence(payload) {
  const evidence = payload?.archive || payload?.evidence || payload;
  if (
    !evidence?.request ||
    !Array.isArray(evidence.archives) ||
    !Array.isArray(evidence.configs)
  )
    throw new Error(
      "Import an experiment evidence bundle containing its request, configurations, and executed archives.",
    );
  const result = await (await wasm()).lecture_analyze(JSON.stringify(evidence));
  result.details.lecture_id = evidence.request.id;
  result.details.calculation_origin = evidence.continuation
    ? "independent_algorithm_continuations"
    : "executed_algorithm_archive";
  const raw = {
    id: evidence.request.id,
    step: evidence.archives[0].steps.length,
    run_steps: evidence.archives.map((a) => a.steps.length),
    budgets: evidence.archives.map((a) => a.steps.length),
    done: true,
    result,
  };
  model = {
    snapshot: () => ({ ...renderSnapshot(raw), imported: true }),
    archive: () => evidence,
    async step() {},
  };
  return model.snapshot();
}
async function dispatch(type, payload) {
  if (type === "catalog") return metadata;
  if (type === "dispose") {
    dispose();
    return null;
  }
  if (type === "initialize") {
    dispose();
    const demo = demos.find((d) => d.id === payload.id);
    if (!demo) throw new Error("Unknown experiment");
    if (
      !Number.isInteger(payload.seed) ||
      payload.seed < 0 ||
      payload.seed > 4294967295
    )
      throw new Error("Invalid seed");
    const params = parameters(demo, payload.params),
      start = performance.now();
    model = await demo.create({ params, seed: payload.seed, engine });
    return {
      snapshot: model.snapshot(),
      initializationMs: performance.now() - start,
      params,
      checkpoint: true,
      archive: true,
    };
  }
  if (
    ["archive_import", "qft_archive_import", "qft_results_import"].includes(
      type,
    )
  ) {
    dispose();
    return importEvidence(payload);
  }
  if (!model) throw new Error("Initialize an experiment first");
  if (type === "step") {
    await model.step();
    return model.snapshot();
  }
  if (type === "archive") return model.archive();
  if (type === "checkpoint") {
    if (!model.checkpoint)
      throw new Error("Imported evidence has no resumable session");
    return model.checkpoint();
  }
  throw new Error("Unknown experiment operation");
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
