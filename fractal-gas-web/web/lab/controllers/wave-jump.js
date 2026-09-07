import { registerController, emptyTree } from "./registry.js";
import { branchActions } from "../timing.js";

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
      this.consensus = settings.consensus_prefix ?? true;
      if (typeof this.consensus !== "boolean")
        throw new Error("Invalid consensus_prefix setting");
      this.normalHorizon = settings.horizon;
      const maximum = settings.max_horizon ?? 0;
      this.maxHorizon =
        maximum === 0 ? Math.min(4096, 2 * settings.horizon) : maximum;
      if (
        this.consensus &&
        (!Number.isInteger(this.maxHorizon) ||
          this.maxHorizon < settings.horizon ||
          this.maxHorizon > 4096)
      )
        throw new Error(
          "Maximum search horizon must be an integer between the normal horizon and 4096, or 0 for automatic.",
        );
      this.done = false;
      this.prefix = 0;
      engine.restore(root);
      engine.begin(
        {
          ...settings,
          horizon: this.consensus ? this.maxHorizon : settings.horizon,
          recording: Math.max(1, settings.recording || 0),
        },
        seed,
      );
    },
    advance() {
      if (this.done) return true;
      const done = engine.advance();
      if (!this.consensus) return (this.done = done);
      if (done || engine.metrics()[8] >= this.normalHorizon) {
        this.prefix = engine.commonAncestor();
        const executable =
          this.prefix &&
          branchActions(engine.tree(), this.prefix).some(
            (edge) => edge.frames > 0,
          );
        this.done = Boolean(done || executable);
      }
      return this.done;
    },
    result() {
      const selectedLeaf = engine.bestLeaf(),
        tree = engine.tree();
      let trajectory = branchActions(tree, selectedLeaf).filter(
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
      // A terminal winner means no final walker survived. Commit only the
      // first executable action, then let the host search from the new world.
      let executionMode = "full path";
      if (tree.meta[row * 5 + 4] & 1) {
        trajectory.splice(1);
        executionMode = "all-dead fallback";
      } else if (this.consensus) {
        const shared = branchActions(tree, this.prefix).filter(
          (edge) => edge.frames > 0,
        );
        if (shared.length) {
          trajectory = shared;
          executionMode = "shared prefix";
        } else {
          trajectory.splice(1);
          executionMode = "horizon fallback";
        }
      }
      return {
        action: trajectory[0].action,
        trajectory,
        selectedLeaf,
        executionMode,
        searchDepth: engine.metrics()[8],
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
        consensus: this.consensus,
        normalHorizon: this.normalHorizon,
        maxHorizon: this.maxHorizon,
        done: this.done,
        prefix: this.prefix,
      };
    },
    restore(saved) {
      engine.restoreCheckpoint(saved.bytes);
      this.root = saved.root;
      this.consensus = saved.consensus ?? false;
      this.normalHorizon = saved.normalHorizon ?? settings.horizon;
      this.maxHorizon = saved.maxHorizon ?? settings.horizon;
      this.done = saved.done ?? false;
      this.prefix = saved.prefix ?? 0;
    },
  }),
});
