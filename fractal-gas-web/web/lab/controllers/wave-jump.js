import { registerController, emptyTree } from "./registry.js";

registerController("wave-jump", {
  label: "Wave Jump",
  parameters: {
    consensus_prefix: {
      type: "boolean",
      label: "Stop at first bifurcation",
      default: true,
      help: "Execute the exact path shared by surviving final walkers. Extend the search if they disagree immediately; at the maximum horizon take one best-path action.",
    },
    max_horizon: {
      label: "Maximum search horizon",
      default: 0,
      min: 0,
      max: 4096,
      step: 1,
      help: "0 automatically uses twice the normal horizon (up to 4096). An explicit limit must be at least the normal horizon. Used only with Stop at first bifurcation.",
    },
  },
  create: ({ engine, settings }) => ({
    begin(root, seed) {
      this.root = root;
      engine.restore(root);
      engine.begin({ ...settings, algorithm: "wave-jump" }, seed);
    },
    advance() {
      return engine.advance();
    },
    result() {
      const result = engine.planResult(),
        tree = engine.tree();
      const row =
        tree.meta.findIndex(
          (value, i) => i % 5 === 0 && value === result.selectedLeaf,
        ) / 5;
      return {
        ...result,
        selectedReward: tree.values[row * (3 + tree.dim + tree.poseDim)],
        budgetUsed: result.searchDepth / settings.horizon,
        tree: settings.recording === 0 ? emptyTree(engine, this.root) : tree,
        cloud: engine.states(true, settings.walkers),
        metrics: engine.metrics(),
      };
    },
    checkpoint() {
      return {
        version: 2,
        algorithm: "wave-jump",
        root: this.root,
        bytes: engine.checkpoint(),
      };
    },
    restore(saved) {
      if (saved.version !== 2)
        throw new Error("Unsupported Wave Jump checkpoint version");
      engine.restoreCheckpoint(saved.bytes);
      this.root = saved.root;
    },
  }),
});
