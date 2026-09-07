import "./controllers/index.js";
import { createController, emptyTree } from "./controllers/registry.js";
import { loadNative, NativeEngine, controlThreads } from "./native.js";
import { validateTrajectory } from "./trajectory.js";
let engine,
  strategy,
  riskEngine,
  revision,
  sequence = 0,
  settings,
  currentRoot,
  resumeSaved = false;
const yieldTask = () => new Promise((resolve) => setTimeout(resolve, 0));
function report(error) {
  postMessage({ type: "error", message: String(error.message || error) });
}
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      revision = data.revision;
      settings = data.settings;
      const requestedThreads = controlThreads(data.threads ?? 8);
      let threaded = !!self.crossOriginIsolated,
        module;
      try {
        module = await loadNative(threaded, requestedThreads);
      } catch (error) {
        if (!threaded) throw error;
        threaded = false;
        module = await loadNative(false);
      }
      const threads = threaded ? requestedThreads : 1;
      strategy = createController(module, data.scene, settings, threads);
      engine = strategy.engine;
      // Share the module's prewarmed pthread budget with the Wave engine by
      // keeping this small, independent diagnostic batch on its caller thread.
      riskEngine = new NativeEngine(module, data.scene, 16, 1);
      postMessage({ type: "ready", threads });
      return;
    }
    if (data.type === "checkpoint") {
      sequence++;
      if (!strategy.controller.checkpoint)
        throw new Error("Controller does not support checkpoints");
      if (!currentRoot || !sameBytes(currentRoot, data.root)) {
        strategy.controller.begin(data.root, data.seed);
        currentRoot = data.root;
      }
      // Saving pauses this exact search. The next plan must continue it on
      // both the original and restored worker; beginning again would shift
      // a stateful controller's warm-start plan a second time.
      resumeSaved = true;
      postMessage({
        type: "checkpoint",
        id: data.id,
        checkpoint: strategy.controller.checkpoint(),
      });
      return;
    }
    if (data.type === "restore-checkpoint") {
      sequence++;
      if (
        data.checkpoint.algorithm !== strategy.id ||
        !strategy.controller.restore
      )
        throw new Error("Controller checkpoint type mismatch");
      strategy.controller.restore(data.checkpoint);
      currentRoot = data.checkpoint.root;
      resumeSaved = true;
      postMessage({ type: "checkpoint-restored" });
      return;
    }
    if (data.type !== "plan") return;
    const seq = ++sequence,
      started = performance.now();
    if (!resumeSaved || !sameBytes(currentRoot, data.root))
      strategy.controller.begin(data.root, data.seed);
    resumeSaved = false;
    currentRoot = data.root;
    let done = false;
    do {
      done = strategy.controller.advance();
      await yieldTask();
      if (seq !== sequence) return;
    } while (
      !done &&
      (!data.budget || performance.now() - started < data.budget)
    );
    if (seq !== sequence) return;
    const result = strategy.controller.result();
    const action = Float32Array.from(result.action);
    if (
      action.length !== engine.dim ||
      !action.every(
        (v, i) =>
          Number.isFinite(v) &&
          v >= engine.channels[i].low - 1e-6 &&
          v <= engine.channels[i].high + 1e-6,
      )
    )
      throw new Error("Controller returned an invalid action");
    const trajectory = result.trajectory
      ? validateTrajectory(result.trajectory, engine.channels)
      : undefined;
    const tree = result.tree || emptyTree(engine, data.root),
      metrics = result.metrics || engine.metrics(),
      cloud = result.cloud || engine.states();
    // Independent continuations of the actual selected action, never the Wave
    // population's death rate. This estimate has a stated horizon and sample count.
    riskEngine.broadcast(data.root);
    const riskActions = new Float32Array(16 * engine.dim);
    for (let i = 0; i < 16; i++) riskActions.set(action, i * engine.dim);
    riskEngine.step(riskActions, settings.frames);
    let rng = (data.seed ^ 0xa73826b5) >>> 0;
    for (let k = 0; k < riskActions.length; k++) {
      rng ^= rng << 13;
      rng ^= rng >>> 17;
      rng ^= rng << 5;
      riskActions[k] =
        engine.channels[k % engine.dim].low +
        ((rng >>> 0) / 4294967296) *
          (engine.channels[k % engine.dim].high -
            engine.channels[k % engine.dim].low);
    }
    riskEngine.step(riskActions, settings.frames * 2);
    const risk = riskEngine.metrics()[3] / 16;
    postMessage(
      {
        type: "plan",
        revision: data.revision,
        target: data.target,
        root: data.root,
        action,
        trajectory,
        selectedLeaf: result.selectedLeaf,
        selectedReward: result.selectedReward,
        executionMode: result.executionMode,
        searchDepth: result.searchDepth,
        tree,
        cloud,
        metrics,
        risk,
        riskSamples: 16,
        riskFrames: settings.frames * 3,
        elapsed: performance.now() - started,
        profile: engine.profile(),
        budgetUsed: result.budgetUsed,
        checkpoint: data.checkpoint
          ? strategy.controller.checkpoint?.()
          : undefined,
      },
      [
        action.buffer,
        ...(trajectory?.map((edge) => edge.action.buffer) || []),
        tree.meta.buffer,
        tree.values.buffer,
        tree.root.buffer,
        cloud.buffer,
        metrics.buffer,
      ],
    );
  } catch (error) {
    report(error);
  }
};

function sameBytes(a, b) {
  return a?.length === b?.length && a.every((v, i) => v === b[i]);
}
