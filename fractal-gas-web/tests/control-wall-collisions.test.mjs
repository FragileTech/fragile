import test from "node:test";
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { rewardValues, withRewards } from "../web/lab/reward-settings.js";
import { prepareRewardEngines } from "../web/lab/live-rewards.js";

const module = await loadNative(false);
const fixture = (overrides = {}) => ({
  task: "navigation",
  size: [20, 20],
  physics: { dt: 0.1, substeps: 4 },
  environment: { flight: false },
  bodies: [{ controlled: true, position: [0.5, 10], radius: 0.5, drag: 0 }],
  rewards: { progress: 0, distance_squared: 0, collision: 50, catch: 0 },
  ...overrides,
});
function run(scene, frames = 1) {
  const engine = new NativeEngine(module, scene);
  try {
    engine.step(engine.neutralAction(), frames);
    return {
      reward: engine.results()[0],
      words: new Uint32Array(engine.states().buffer),
    };
  } finally {
    engine.dispose();
  }
}

test("wall defaults and edits are available in every task, including imported harvest scenes", () => {
  for (const task of ["navigation", "harvest", "forage", "tandem"]) {
    const scene = fixture({ task });
    assert.equal(rewardValues(scene).wall_collision, 100);
    assert.equal(run(scene).reward, -100);
    assert.equal(
      run({ ...scene, rewards: { ...scene.rewards, collision: 0 } }).reward,
      -100,
    );
    for (const wall_collision of [0, 2, 3.5, 10000]) {
      const edited = withRewards(scene, {
        ...rewardValues(scene),
        wall_collision,
      });
      assert.equal(run(edited).reward, -wall_collision || 0);
      assert.equal(
        rewardValues(JSON.parse(JSON.stringify(edited))).wall_collision,
        wall_collision,
      );
    }
    for (const wall_collision of [-1, 10001, "2", true]) {
      assert.throws(
        () =>
          new NativeEngine(module, {
            ...scene,
            rewards: { ...scene.rewards, wall_collision },
          }),
      );
      assert.throws(() =>
        withRewards(scene, { ...rewardValues(scene), wall_collision }),
      );
    }
    for (const wall_collision of [NaN, Infinity])
      assert.throws(() =>
        withRewards(scene, { ...rewardValues(scene), wall_collision }),
      );
  }
});

test("sustained and corner contacts cost once per vehicle per frame, independent of substeps", () => {
  for (const substeps of [1, 4, 8, 32]) {
    for (const position of [
      [0.5, 10],
      [0.5, 0.5],
    ]) {
      const scene = fixture({
        physics: { dt: 0.1, substeps },
        bodies: [
          { controlled: true, position, radius: 0.5 },
          { controlled: true, position: [19.5, 15], radius: 0.5 },
        ],
      });
      assert.equal(run(scene, 5).reward, -1000);
      const engine = new NativeEngine(module, scene);
      try {
        const root = engine.snapshot();
        engine.step(engine.neutralAction(), 5);
        const batch = engine.snapshot();
        engine.restore(root);
        let total = 0;
        for (let i = 0; i < 5; i++) {
          engine.step(engine.neutralAction(), 1);
          total += engine.results()[0];
        }
        assert.equal(total, -1000);
        assert.deepEqual(engine.snapshot(), batch);
      } finally {
        engine.dispose();
      }
    }
  }
});

test("obstacle walls and fast impacts charge the wall penalty independently of death", () => {
  const scenes = [
    fixture(),
    fixture({
      holes: [
        [
          [8, 8],
          [12, 8],
          [12, 12],
          [8, 12],
        ],
      ],
      bodies: [{ controlled: true, position: [7.5, 10], radius: 0.5 }],
    }),
    fixture({
      bodies: [
        {
          controlled: true,
          position: [3, 10],
          velocity: [-80, 0],
          radius: 0.5,
          restitution: 0,
        },
      ],
    }),
  ];
  for (const base of scenes)
    for (const lethal_walls of [false, true])
      for (const wall_collision of [0, 2]) {
        const scene = {
          ...base,
          physics: { ...base.physics, lethal_walls },
          rewards: { ...base.rewards, wall_collision },
        };
        const first = run(scene);
        assert.equal(first.reward, -wall_collision || 0);
        assert.equal(first.words[7], Number(lethal_walls));
        if (lethal_walls) {
          const longer = run(scene, 10);
          assert.equal(
            longer.words[0],
            1,
            "death must stop before the next physics frame",
          );
          assert.equal(longer.reward, -wall_collision || 0);
          assert.deepEqual(longer.words, first.words);
        }
      }
});

test("a single vehicle wall collision ends the whole multi-vehicle world", () => {
  const result = run(
    fixture({
      physics: { lethal_walls: true },
      bodies: [
        { controlled: true, position: [10, 10] },
        { controlled: true, position: [0.5, 5] },
      ],
    }),
    10,
  );
  assert.equal(result.words[7], 1);
  assert.equal(result.words[0], 1);
  assert.equal(result.reward, -100);
});

test("cargo and physical hooks neither incur wall penalties nor kill the world", () => {
  for (const fast of [false, true]) {
    const scene = fixture({
      task: "harvest",
      physics: { dt: 0.1, substeps: 4, lethal_walls: true },
      bodies: [
        { controlled: true, position: [10, 10] },
        {
          cargo: true,
          position: [fast ? 2 : 0.5, 5],
          velocity: [fast ? -40 : 0, 0],
          radius: 0.5,
        },
      ],
      tethers: [{ a: 0, b: -1, stiffness: 0, damping: 0 }],
    });
    const engine = new NativeEngine(module, scene);
    try {
      const rows = engine.states(),
        hook = 2;
      rows[8 + hook] = fast ? 2 : 0.2;
      rows[8 + engine.bodies + hook] = 15;
      rows[8 + 2 * engine.bodies + hook] = fast ? -40 : 0;
      engine.restoreRows(rows);
      engine.step(engine.neutralAction(), 3);
      assert.equal(engine.results()[0], 0);
      assert.equal(new Uint32Array(engine.states().buffer)[7], 0);
    } finally {
      engine.dispose();
    }
  }
});

test("body contacts retain a separate penalty and do not trigger wall death", () => {
  const result = run(
    fixture({
      physics: { dt: 0.1, substeps: 1, lethal_walls: true },
      bodies: [
        { controlled: true, position: [9.5, 10], radius: 0.5 },
        { position: [10.4, 10], radius: 0.5 },
      ],
    }),
  );
  assert.equal(result.reward, -50);
  assert.equal(result.words[7], 0);
});

test("live wall-penalty updates preserve state and change only future rewards", () => {
  const scene = fixture(),
    engine = new NativeEngine(module, scene);
  let next;
  try {
    engine.step(engine.neutralAction(), 2);
    const before = engine.states();
    next = prepareRewardEngines(
      engine,
      scene,
      { ...scene, rewards: { ...scene.rewards, wall_collision: 7 } },
      {},
    );
    assert.deepEqual(next.engine.states(), before);
    next.engine.step(next.engine.neutralAction(), 1);
    assert.equal(next.engine.results()[0], -7);
    assert.deepEqual(engine.states(), before);
  } finally {
    engine.dispose();
    next?.engine.dispose();
    next?.predict.dispose();
  }
});

test("every shipped Control Lab preset explicitly enables wall penalties without death", async () => {
  const directory = new URL("../web/lab/scenarios/", import.meta.url);
  for (const file of await readdir(directory)) {
    if (!file.endsWith(".json")) continue;
    const scene = JSON.parse(await readFile(new URL(file, directory)));
    assert.equal(scene.rewards.wall_collision, 100, file);
    assert.equal(scene.physics.lethal_walls, false, file);
  }
});
