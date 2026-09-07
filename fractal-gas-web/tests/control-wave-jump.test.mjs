import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { createController } from "../web/lab/controllers/index.js";
import { TrajectoryCursor, validateTrajectory } from "../web/lab/trajectory.js";
import { instantiateController } from "../web/lab/controllers/registry.js";
import { runEpisode } from "../web/lab/experiments.js";
const scene = {
  size: [1000, 1000],
  bodies: [{ controlled: true, position: [500, 500], velocity: [2, 1] }],
};
const settings = {
  algorithm: "wave-jump",
  walkers: 12,
  horizon: 5,
  frames: 3,
  elites: 2,
  recording: 0,
};
const module = await loadNative(false);

test("Wave Jump records ancestry with recording off and replays the winning leaf exactly", () => {
  const world = new NativeEngine(module, scene),
    strategy = createController(module, scene, settings);
  try {
    world.reset(7);
    strategy.controller.begin(world.snapshot(), 19);
    strategy.controller.advance();
    const saved = strategy.controller.checkpoint();
    while (!strategy.controller.advance()) {}
    const decision = strategy.controller.result();
    assert.equal(decision.tree.meta.length, 0);
    const recordedTree = strategy.engine.tree();
    const metric = decision.metrics[12];
    assert.equal(decision.selectedReward, metric);
    assert.ok(decision.trajectory.length > 1);
    strategy.controller.restore(saved);
    while (!strategy.controller.advance()) {}
    const resumed = strategy.controller.result();
    for (const key of [
      "action",
      "trajectory",
      "tree",
      "cloud",
      "selectedLeaf",
      "selectedReward",
    ])
      assert.deepEqual(resumed[key], decision[key]);
    for (const edge of decision.trajectory)
      world.step(edge.action, edge.frames);
    const width = 3 + recordedTree.poseDim + recordedTree.dim;
    const i =
      recordedTree.meta.findIndex(
        (v, i) => i % 5 === 0 && v === decision.selectedLeaf,
      ) / 5;
    assert.ok(i >= 0);
    // Replay native ancestry from its saved root and compare the entire world.
    strategy.engine.check(
      module._fgc_replay_node(strategy.engine.h, decision.selectedLeaf),
    );
    assert.deepEqual(world.snapshot(), strategy.engine.snapshot());
    assert.equal(decision.selectedReward, recordedTree.values[i * width]);
  } finally {
    strategy.dispose();
    world.dispose();
  }
});

test("trajectory cursor preserves partial actions and rejects invalid checkpoints", () => {
  const channels = [{ low: -1, high: 1 }];
  const edges = [
    { action: [0.25], frames: 2 },
    { action: [-0.5], frames: 1 },
  ];
  const a = new TrajectoryCursor(edges, channels);
  a.advance();
  const saved = a.checkpoint(),
    b = new TrajectoryCursor(saved.trajectory, channels, saved);
  const sequence = [];
  while (!b.done) {
    sequence.push(b.action[0]);
    b.advance();
  }
  assert.deepEqual(sequence, [0.25, -0.5]);
  assert.throws(
    () => new TrajectoryCursor(edges, channels, { index: 0, remaining: 3 }),
  );
  assert.throws(() =>
    validateTrajectory([{ action: [NaN], frames: 1 }], channels),
  );
  assert.throws(() =>
    validateTrajectory([{ action: [0], frames: 0 }], channels),
  );
});

test("experiments count one decision per full trajectory and stop at the frame limit", async () => {
  const { stats } = await runEpisode({
    module,
    scene,
    settings,
    seed: 7,
    maxFrames: 20,
  });
  assert.equal(stats.frames, 20);
  assert.equal(stats.decisions, 2);
});

test("Wave Jump experiments stop partway through a trajectory on success", async () => {
  const { stats } = await runEpisode({
    module,
    scene,
    settings,
    seed: 7,
    maxFrames: 50,
    goal: { metric: "survival", target: 5 },
  });
  assert.equal(stats.frames, 5);
  assert.equal(stats.decisions, 1);
  assert.equal(stats.success, true);
});

test("Wave Jump executes a shortened terminal edge and stops", async () => {
  const lethalScene = {
    physics: { lethal_walls: true },
    holes: [
      [
        [30, 10],
        [35, 10],
        [35, 30],
        [30, 30],
      ],
    ],
    bodies: [
      {
        position: [10, 20],
        velocity: [3000, 0],
        drag: 0,
        controlled: true,
        vertices: [
          [-0.5, -0.5],
          [0.5, -0.5],
          [0.5, 0.5],
          [-0.5, 0.5],
        ],
      },
    ],
  };
  const world = new NativeEngine(module, lethalScene),
    strategy = createController(module, lethalScene, settings);
  try {
    strategy.controller.begin(world.snapshot(), 7);
    while (!strategy.controller.advance()) {}
    const result = strategy.controller.result();
    assert.equal(result.trajectory.length, 1);
    assert.ok(result.trajectory[0].frames < settings.frames);
    const { stats } = await runEpisode({
      module,
      scene: lethalScene,
      settings,
      seed: 7,
      maxFrames: 50,
    });
    assert.equal(stats.dead, true);
    assert.equal(stats.decisions, 1);
    assert.equal(stats.frames, result.trajectory[0].frames);
  } finally {
    strategy.dispose();
    world.dispose();
  }
});

test("Wave Jump skips zero-duration ancestry and reports an empty path", () => {
  const tree = {
    dim: 1,
    poseDim: 0,
    root: new Uint8Array(),
    meta: new Uint32Array([
      1, 0, 0, 0, 0, 2, 1, 1, 2, 0, 3, 2, 2, 0, 1, 4, 3, 3, 1, 1,
    ]),
    values: new Float32Array([
      0, 0, 0, 0, 2, 2, 0, 0.5, 2, 0, 0, 0.75, 3, 1, 0, 1,
    ]),
  };
  const engine = {
    bestLeaf: () => 4,
    tree: () => tree,
    metrics: () => new Float32Array(16),
    states: () => new Float32Array(),
  };
  const controller = instantiateController("wave-jump", engine, {
    ...settings,
    recording: 2,
  });
  assert.deepEqual(
    controller.result().trajectory.map((e) => [e.action[0], e.frames]),
    [[0.5, 2]],
  );
  tree.meta[19] = 0; // An alive winner still executes the entire branch.
  assert.deepEqual(
    controller.result().trajectory.map((e) => [e.action[0], e.frames]),
    [
      [0.5, 2],
      [1, 1],
    ],
  );
  engine.bestLeaf = () => 1;
  assert.throws(() => controller.result(), /no executable trajectory/);
});

test("all-dead lookahead commits one safe action and experiments replan", async () => {
  const doomed = {
    size: [100, 100],
    physics: { lethal_walls: true },
    bodies: [
      { controlled: true, position: [50, 50], velocity: [60, 0], drag: 0 },
    ],
  };
  const config = { ...settings, horizon: 12, frames: 11, elites: 0 };
  const world = new NativeEngine(module, doomed),
    strategy = createController(module, doomed, config);
  try {
    strategy.controller.begin(world.snapshot(), 7);
    while (!strategy.controller.advance()) {}
    const result = strategy.controller.result();
    assert.equal(result.metrics[9], 1);
    assert.equal(result.selectedReward, result.metrics[12]);
    const tree = strategy.engine.tree();
    const row =
      tree.meta.findIndex((v, i) => i % 5 === 0 && v === result.selectedLeaf) /
      5;
    assert.ok(tree.meta[row * 5 + 2] > 1);
    assert.equal(result.trajectory.length, 1);
    assert.equal(result.trajectory[0].frames, 11);
    world.step(result.action, 11);
    assert.equal(world.metrics()[3], 0);
    const { stats } = await runEpisode({
      module,
      scene: doomed,
      settings: config,
      seed: 7,
      maxFrames: 22,
    });
    assert.equal(stats.frames, 22);
    assert.equal(stats.decisions, 2);
    assert.equal(stats.dead, false);
  } finally {
    strategy.dispose();
    world.dispose();
  }
});
