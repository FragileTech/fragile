import { registerController, emptyTree } from "./registry.js";
import { branchActions } from "../timing.js";

registerController("wave-jump", {
  label: "Wave Jump",
  create: ({ engine, settings }) => ({
    begin(root, seed) {
      this.root = root;
      engine.restore(root);
      engine.begin(
        { ...settings, recording: Math.max(1, settings.recording || 0) },
        seed,
      );
    },
    advance() {
      return engine.advance();
    },
    result() {
      const selectedLeaf = engine.bestLeaf(),
        tree = engine.tree();
      const trajectory = branchActions(tree, selectedLeaf).filter(
        (edge) => edge.frames > 0,
      );
      if (!trajectory.length)
        throw new Error(
          "Wave Jump found no executable trajectory. Reset or change the search settings.",
        );
      const row =
        tree.meta.findIndex(
          (value, i) => i % 5 === 0 && value === selectedLeaf,
        ) / 5;
      return {
        action: trajectory[0].action,
        trajectory,
        selectedLeaf,
        selectedReward: tree.values[row * (3 + tree.dim + tree.poseDim)],
        budgetUsed: engine.metrics()[8] / settings.horizon,
        tree: settings.recording === 0 ? emptyTree(engine, this.root) : tree,
        cloud: engine.states(true, settings.walkers),
        metrics: engine.metrics(),
      };
    },
    checkpoint() {
      return {
        version: 1,
        algorithm: "wave-jump",
        root: this.root,
        bytes: engine.checkpoint(),
      };
    },
    restore(saved) {
      engine.restoreCheckpoint(saved.bytes);
      this.root = saved.root;
    },
  }),
});
