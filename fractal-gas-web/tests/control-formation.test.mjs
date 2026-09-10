import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import {
  rewardDefaults,
  rewardValues,
  withRewards,
} from "../web/lab/reward-settings.js";
import { prepareRewardEngines } from "../web/lab/live-rewards.js";

const module = await loadNative();
const fixture = (overrides = {}) => ({
  task: "tandem",
  size: [100, 100],
  physics: { dt: 0.1, substeps: 4 },
  environment: { flight: false },
  formation_distance: 5,
  bodies: [
    { controlled: true, position: [20, 20], drag: 0 },
    { controlled: true, position: [26, 20], drag: 0 },
  ],
  ...overrides,
});
function run(scene, frames = 1) {
  const engine = new NativeEngine(module, scene);
  try {
    engine.step(engine.neutralAction(), frames);
    return {
      reward: engine.results()[0],
      metrics: engine.metrics(),
      states: engine.states(),
    };
  } finally {
    engine.dispose();
  }
}
function close(actual, expected, tolerance = 1e-5) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`,
  );
}

test("formation defaults match omitted settings, presets and explicit overrides", async () => {
  const defaults = rewardDefaults({ task: "tandem" });
  assert.deepEqual(defaults, {
    catch: 0,
    distance_squared: 1,
    hooked_rock_distance: 0,
    progress: 1,
    wall_collision: 100,
    collision: 2,
    pickup: 0,
    delivery: 0,
    gate: 30,
    formation: 50,
    full_reward: 0,
  });
  const preset = JSON.parse(
    await readFile(
      new URL("../web/lab/scenarios/tandem.json", import.meta.url),
      "utf8",
    ),
  );
  assert.deepEqual(rewardValues(preset), defaults);
  assert.equal(preset.formation_distance, 4);
  const scene = fixture({ gates: [{ position: [20, 20], radius: 1 }] });
  close(run(scene).reward, (50 * 5) / 6 + 15 + 0.25);
  assert.equal(
    run(scene).metrics[7],
    1,
    "individual checkpoint crossings still count",
  );
  close(
    run({ ...scene, rewards: { gate: 17, formation: 0.15 } }).reward,
    8.875,
  );
  assert.equal(
    rewardValues({ task: "tandem", rewards: { pickup: 17 } }).full_reward,
    0,
  );
  assert.equal(
    rewardValues({ task: "tandem", cargo: { full_reward: 19 } }).full_reward,
    19,
  );
  assert.equal(rewardDefaults({ task: "navigation" }).progress, 1);
  assert.equal(rewardDefaults({ task: "navigation" }).gate, 30);
  assert.equal(rewardDefaults({ task: "harvest" }).formation, 0);
  assert.equal(
    withRewards(scene, { ...rewardValues(scene), formation: 100 }).rewards
      .formation,
    100,
  );
  assert.throws(
    () => withRewards(scene, { ...rewardValues(scene), formation: 100.01 }),
    /between 0 and 100/,
  );
});

test("formation scores current separation, once per frame, regardless of progress", () => {
  for (const substeps of [1, 4, 16])
    for (const progress of [0, 2])
      for (const [distance, expected] of [
        [5, 1],
        [6, 5 / 6],
        [4, 5 / 6],
        [50, 0.1],
      ]) {
        const scene = fixture({
          physics: { dt: 0.1, substeps },
          rewards: { formation: 1, progress },
          bodies: [
            { controlled: true, position: [20, 20] },
            { controlled: true, position: [20 + distance, 20] },
          ],
        });
        close(run(scene, 5).reward, 5 * expected);
      }
  assert.equal(run(fixture({ rewards: { formation: 0 } })).reward, 0);
  assert.equal(run(fixture({ task: "navigation" })).reward, 0);
  assert.equal(run(fixture({ bodies: [{ position: [20, 20] }] })).reward, 0);
  assert.equal(
    run(fixture({ bodies: [{ controlled: true, position: [20, 20] }] })).reward,
    0,
  );
});

test("different pair targets multiply once, with scalar fallback and body indices", () => {
  const scene = fixture({
    rewards: { formation: 1 },
    formation_pairs: [
      { a: 2, b: 0, distance: 3 },
      { a: 0, b: 3, distance: 4 },
    ],
    bodies: [
      { controlled: true, position: [20, 20] },
      { cargo: true, position: [80, 80] },
      { controlled: true, position: [23, 20] },
      { controlled: true, position: [20, 24] },
    ],
  });
  close(run(scene).reward, 1);
  const doubled = structuredClone(scene);
  doubled.bodies[2].position = [26, 20];
  doubled.bodies[3].position = [20, 28];
  close(run(doubled).reward, 0.125);
  const rotated = structuredClone(doubled);
  for (const body of rotated.bodies)
    body.position = [100 - body.position[1], body.position[0]];
  close(run(rotated).reward, 0.125);
  const updated = withRewards(scene, { ...rewardValues(scene), formation: 2 });
  assert.deepEqual(updated.formation_pairs, scene.formation_pairs);
  close(run(JSON.parse(JSON.stringify(updated))).reward, 2);
});

test("formation pair configuration rejects malformed references and targets", () => {
  const valid = { a: 0, b: 1, distance: 5 };
  for (const formation_pairs of [
    null,
    {},
    "pairs",
    [null],
    [1],
    [{}],
    [{ a: 0, b: 1 }],
    [{ a: 0, distance: 5 }],
    [{ b: 1, distance: 5 }],
    [valid, valid],
    [valid, { a: 1, b: 0, distance: 6 }],
    ...[-1, 0.5, 2, "0", true, null].map((a) => [{ ...valid, a }]),
    [{ ...valid, b: 0 }],
    ...[0, -1, 0.099, 1001, "5", true, null, NaN, Infinity].map((distance) => [
      { ...valid, distance },
    ]),
  ]) {
    assert.throws(
      () => new NativeEngine(module, fixture({ formation_pairs })),
      undefined,
      JSON.stringify(formation_pairs),
    );
  }
  assert.throws(
    () =>
      new NativeEngine(
        module,
        fixture({
          bodies: [
            { controlled: true, position: [20, 20] },
            { position: [25, 20] },
          ],
          formation_pairs: [valid],
        }),
      ),
  );
  for (const distance of [0.1, 1000]) {
    const engine = new NativeEngine(
      module,
      fixture({ formation_pairs: [{ ...valid, distance }] }),
    );
    engine.dispose();
  }
});

test("formation adds to travel and collision rewards without changing physics", () => {
  const scene = fixture({ rewards: { formation: 1 } });
  for (const body of scene.bodies) body.velocity = [2, 3];
  close(run(scene, 5).reward, 5 * (5 / 6 + 0.13), 1e-4);
  const wall = fixture({
    rewards: { formation: 1 },
    bodies: [
      { controlled: true, position: [0.5, 10] },
      { controlled: true, position: [5.5, 10] },
    ],
  });
  close(run(wall).reward, -99);
  const contact = fixture({
    rewards: { formation: 1 },
    bodies: [
      { controlled: true, position: [20, 20] },
      { controlled: true, position: [20.8, 20] },
    ],
  });
  const withFormation = run(contact);
  const withoutFormation = run({ ...contact, rewards: { formation: 0 } });
  assert.deepEqual(withFormation.states, withoutFormation.states);
  assert.ok(withoutFormation.reward < 0);
  const rows = withFormation.states;
  const distance = Math.hypot(rows[8] - rows[9], rows[10] - rows[11]);
  close(
    withFormation.reward - withoutFormation.reward,
    5 / (5 + Math.abs(5 - distance)),
  );
});

test("live formation weights preserve state, pair targets and future frame accumulation", () => {
  const scene = fixture({ formation_pairs: [{ a: 0, b: 1, distance: 5 }] });
  const engine = new NativeEngine(module, scene);
  let replacement;
  try {
    engine.step(engine.neutralAction(), 3);
    const before = engine.states();
    replacement = prepareRewardEngines(
      engine,
      scene,
      {
        ...scene,
        rewards: { formation: 2 },
      },
      {},
    );
    assert.deepEqual(replacement.engine.states(), before);
    assert.deepEqual(replacement.scene.formation_pairs, scene.formation_pairs);
    const snapshot = replacement.engine.snapshot();
    replacement.engine.step(replacement.engine.neutralAction(), 3);
    const batch = replacement.engine.results()[0];
    close(batch, 5);
    const future = replacement.engine.states();
    replacement.engine.restore(snapshot);
    let separate = 0;
    for (let i = 0; i < 3; ++i) {
      replacement.engine.step(replacement.engine.neutralAction(), 1);
      separate += replacement.engine.results()[0];
    }
    close(separate, batch);
    assert.deepEqual(replacement.engine.states(), future);
  } finally {
    engine.dispose();
    replacement?.engine.dispose();
    replacement?.predict.dispose();
  }
});
