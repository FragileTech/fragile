import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { prepareRewardEngines } from "../web/lab/live-rewards.js";

const module = await loadNative();
const agent = (x, y = 20) => ({
  controlled: true,
  position: [x, y],
  drag: 0,
  radius: 0.01,
});
const fixture = (overrides = {}) => ({
  task: "tandem",
  size: [100, 100],
  environment: { flight: false },
  physics: { dt: 0.1, substeps: 4 },
  rewards: {
    formation: 0,
    distance_squared: 0,
    collision: 0,
    wall_collision: 0,
  },
  bodies: [agent(20), agent(26)],
  gates: [
    { position: [20, 20], radius: 1 },
    { position: [26, 20], radius: 1 },
  ],
  ...overrides,
});
function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-4, `${actual} != ${expected}`);
}
function step(engine, frames = 1) {
  engine.step(engine.neutralAction(), frames);
  return engine.results()[0];
}
function edit(engine, { positions = {}, velocities = {}, stages = {} }) {
  const row = engine.states(),
    words = new Uint32Array(row.buffer);
  for (const [b, [x, y]] of Object.entries(positions)) {
    row[8 + Number(b)] = x;
    row[8 + engine.bodies + Number(b)] = y;
  }
  for (const [b, [x, y]] of Object.entries(velocities)) {
    row[8 + 2 * engine.bodies + Number(b)] = x;
    row[8 + 3 * engine.bodies + Number(b)] = y;
  }
  for (const [c, stage] of Object.entries(stages))
    words[8 + 7 * engine.bodies + Number(c)] = stage;
  engine.restoreRows(row);
}
const stages = (engine) =>
  Array.from(
    new Uint32Array(engine.states().buffer).slice(
      8 + 7 * engine.bodies,
      8 + 7 * engine.bodies + engine.controlled,
    ),
  );

test("staggered arrivals wait, unlock next frame, and wrap in authored order", () => {
  const engine = new NativeEngine(
    module,
    fixture({ rewards: { ...fixture().rewards, progress: 0 } }),
  );
  try {
    close(step(engine), 15);
    assert.deepEqual(stages(engine), [1, 0]);
    edit(engine, { positions: { 0: [26, 20], 1: [28, 20] } });
    close(step(engine), 0);
    assert.deepEqual(stages(engine), [1, 0]);
    edit(engine, { positions: { 1: [20, 20] } });
    close(step(engine), 15);
    assert.deepEqual(stages(engine), [1, 1]);
    close(step(engine), 15);
    assert.deepEqual(stages(engine), [2, 1]);
    edit(engine, { positions: { 0: [20, 20], 1: [26, 20] } });
    close(step(engine), 15);
    assert.deepEqual(stages(engine), [2, 2]);
    close(step(engine), 15);
    assert.deepEqual(stages(engine), [3, 2]);
    assert.equal(new Uint32Array(engine.states().buffer)[6], 5);
  } finally {
    engine.dispose();
  }
});

test("proximity scale and complete-team bonus are independent of team size", () => {
  for (const count of [1, 2, 4]) {
    // All agents approach along rays at the same speed, without contacts.
    const bodies = Array.from({ length: count }, (_, i) => {
      const angle = (2 * Math.PI * i) / count;
      return {
        ...agent(50 + 10 * Math.cos(angle), 50 + 10 * Math.sin(angle)),
        velocity: [-2 * Math.cos(angle), -2 * Math.sin(angle)],
      };
    });
    const engine = new NativeEngine(
      module,
      fixture({ bodies, gates: [{ position: [50, 50], radius: 9.9 }] }),
    );
    try {
      close(step(engine), 30 + 9.9 / (9.9 + 9.8));
      assert.deepEqual(stages(engine), Array(count).fill(1));
    } finally {
      engine.dispose();
    }
  }
});

test("cleared agents contribute to positive proximity; mean precedes the transform", () => {
  const engine = new NativeEngine(
    module,
    fixture({
      bodies: [agent(28), { position: [80, 80] }, agent(20)],
      gates: [
        { position: [30, 20], radius: 1 },
        { position: [50, 20], radius: 1 },
      ],
    }),
  );
  try {
    edit(engine, { stages: { 0: 5 } });
    close(step(engine), 1 / 7); // Mean distance (2 + 10) / 2 = 6.
    assert.deepEqual(stages(engine), [5, 0]);
    edit(engine, { positions: { 0: [29, 20] } });
    close(step(engine), 1 / 6.5); // Bringing the cleared agent closer improves reward.
    edit(engine, { velocities: { 0: [-10, 0] } });
    close(step(engine), 1 / 7); // Moving away still earns a positive amount.
    edit(engine, { velocities: { 0: [0, 0] } });
    close(step(engine, 3), 3 / 7); // Stationary agents keep earning proximity.
  } finally {
    engine.dispose();
  }
});

test("overlapping gates advance once per frame with zero-weight synchronization", () => {
  for (const gate of [0, 30]) {
    const engine = new NativeEngine(
      module,
      fixture({
        rewards: { formation: 0, distance_squared: 0, progress: 0, gate },
        bodies: [agent(20), agent(20.5)],
        gates: [
          { position: [20, 20], radius: 2 },
          { position: [20, 20], radius: 2 },
        ],
      }),
    );
    try {
      edit(engine, { stages: { 0: 1 } });
      close(step(engine), gate / 2);
      assert.deepEqual(stages(engine), [1, 1]);
      close(step(engine), gate);
      assert.deepEqual(stages(engine), [2, 2]);
    } finally {
      engine.dispose();
    }
  }
});

test("proximity is bounded, scales with weight, and zero disables only its payment", () => {
  for (const weight of [0, 1, 7]) {
    const engine = new NativeEngine(
      module,
      fixture({
        bodies: [agent(20)],
        rewards: { ...fixture().rewards, progress: weight, gate: 0 },
        gates: [{ position: [20, 20], radius: 2 }],
      }),
    );
    try {
      close(step(engine), weight);
      assert.deepEqual(stages(engine), [1]);
      edit(engine, { positions: { 0: [30, 20] } });
      close(step(engine), weight / 6);
    } finally {
      engine.dispose();
    }
  }
});

test("final crossing scores old shared target and switches proximity next frame", () => {
  const engine = new NativeEngine(module, fixture());
  try {
    edit(engine, { positions: { 1: [20.5, 20] }, stages: { 0: 1 } });
    close(step(engine), 15 + 1 / 1.25);
    assert.deepEqual(stages(engine), [1, 1]);
    close(step(engine), 1 / 6.75);
  } finally {
    engine.dispose();
  }
});

test("snapshot, batched stepping and live reweighting retain checkpoint barriers", () => {
  const scene = fixture();
  const engine = new NativeEngine(module, scene);
  let replacement;
  try {
    step(engine);
    const saved = engine.snapshot();
    edit(engine, { positions: { 1: [20.5, 20] } });
    step(engine);
    engine.restore(saved);
    assert.deepEqual(stages(engine), [1, 0]);
    edit(engine, { velocities: { 1: [-10, 0] } });
    const start = engine.snapshot();
    const batchReward = step(engine, 6),
      end = engine.states();
    engine.restore(start);
    let serialReward = 0;
    for (let i = 0; i < 6; ++i) serialReward += step(engine);
    close(batchReward, serialReward);
    assert.deepEqual(engine.states(), end);
    engine.restore(saved);
    const before = engine.states();
    replacement = prepareRewardEngines(
      engine,
      scene,
      {
        ...scene,
        rewards: { ...scene.rewards, progress: 2, gate: 60 },
      },
      {},
    );
    assert.deepEqual(replacement.engine.states(), before);
    edit(replacement.engine, { positions: { 1: [20.5, 20] } });
    close(step(replacement.engine), 30 + 2 / 1.25);
    assert.deepEqual(stages(replacement.engine), [1, 1]);
  } finally {
    engine.dispose();
    replacement?.engine.dispose();
    replacement?.predict.dispose();
  }
});

test("non-tandem remains independent and scenes without checkpoints are valid", () => {
  const engine = new NativeEngine(module, fixture({ task: "navigation" }));
  try {
    close(step(engine), 30);
    edit(engine, { positions: { 0: [26, 20], 1: [28, 20] } });
    close(step(engine), 30);
    assert.deepEqual(stages(engine), [2, 0]);
  } finally {
    engine.dispose();
  }
  for (const bodies of [[agent(20)], [{ position: [20, 20] }]]) {
    const empty = new NativeEngine(module, fixture({ bodies, gates: [] }));
    try {
      close(step(empty), 0);
    } finally {
      empty.dispose();
    }
  }
});
