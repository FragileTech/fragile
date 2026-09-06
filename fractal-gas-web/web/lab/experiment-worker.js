import { loadNative, NativeEngine } from "./native.js";
import { runBenchmark, runEpisode, validateExperiment } from "./experiments.js";
import { exportRecording } from "./archive.js";
let generation = 0;
self.onmessage = async ({ data }) => {
  if (data.type === "cancel") {
    generation++;
    return;
  }
  const id = ++generation;
  try {
    const module = await loadNative(false),
      cancelled = () => id !== generation;
    if (data.type === "benchmark") {
      const scenes = data.scenarios
        ? await Promise.all(
            data.scenarios.map(async (name) => {
              if (!/^[a-z][a-z0-9_-]*$/.test(name))
                throw new Error("Invalid preset name");
              const response = await fetch(`./scenarios/${name}.json`);
              if (!response.ok)
                throw new Error("Unable to load benchmark scene");
              return response.json();
            }),
          )
        : undefined;
      const report = await runBenchmark({
        scenes,
        module,
        scene: data.scene,
        spec: data.spec,
        cancelled,
        progress: (trial, index, total) =>
          postMessage({ type: "progress", trial, index, total }),
      });
      if (!cancelled()) postMessage({ type: "benchmark", report });
    } else if (data.type === "compare") {
      validateExperiment(data.spec);
      const branches = [];
      for (const settings of data.spec.variants) {
        const result = await runEpisode({
          module,
          scene: data.scene,
          root: data.root,
          settings,
          seed: data.spec.seeds[0],
          maxFrames: data.spec.maxFrames,
          goal: data.spec.goal,
          record: true,
          cancelled,
        });
        branches.push({
          stats: result.stats,
          archive: exportRecording(data.scene, settings, [], result.motion),
        });
        postMessage({
          type: "progress",
          index: branches.length,
          total: data.spec.variants.length,
        });
      }
      if (!cancelled()) postMessage({ type: "comparison", branches });
    } else if (data.type === "profile") {
      const worlds = Math.min(1024, Math.max(1, data.worlds || 256)),
        e = new NativeEngine(module, data.scene, worlds);
      try {
        e.resetProfile();
        const rows = e.copyStates(),
          indices = Int32Array.from(
            { length: worlds },
            (_, i) => worlds - 1 - i,
          ),
          actions = new Float32Array(worlds * e.dim);
        const start = performance.now();
        for (let i = 0; i < 32; i++) {
          e.copyStates(rows);
          e.restoreRows(rows);
          e.gather(indices);
        }
        const transferMs = performance.now() - start;
        e.step(actions, 16);
        postMessage({
          type: "profile",
          worlds,
          profile: e.profile(),
          transferMs,
          wasmBytes: module.HEAPU8.byteLength,
        });
      } finally {
        e.dispose();
      }
    }
  } catch (error) {
    if (id === generation)
      postMessage({ type: "error", message: error.message });
  }
};
