import test from "node:test";
import assert from "node:assert/strict";
import { WorkspaceState } from "../web/lab/workspace-state.js";
import { ConfigurationTransition } from "../web/lab/configuration-transition.js";
import { DriveClock } from "../web/lab/drive-clock.js";
import { configurationDiff } from "../web/lab/variant-settings.js";
test("draft edits cannot alter active configuration; discard restores nested state", () => {
  const s = new WorkspaceState();
  s.commit({ bodies: [{ mass: 1 }] }, { horizon: 32 });
  s.draft.scene.bodies[0].mass = 2;
  s.draft.settings.horizon = 64;
  assert.equal(s.active.scene.bodies[0].mass, 1);
  assert.equal(s.active.settings.horizon, 32);
  assert.equal(s.changes.length, 2);
  s.discard();
  assert.equal(s.dirty, false);
});
test("save and prepare failures retain the original session", async () => {
  const t = new ConfigurationTransition(),
    events = [];
  await assert.rejects(
    t.run({
      quiesce: () => events.push("paused"),
      save: () => {
        throw Error("quota");
      },
      prepare: () => events.push("prepared"),
      commit: () => events.push("committed"),
    }),
    /quota/,
  );
  assert.deepEqual(events, ["paused"]);
  await assert.rejects(
    t.run({
      quiesce: () => {},
      save: () => {},
      prepare: () => {
        throw Error("invalid");
      },
      commit: () => events.push("committed"),
    }),
    /invalid/,
  );
  assert.equal(t.busy, false);
  assert.ok(!events.includes("committed"));
});
test("replacement commits only after saving and initialization", async () => {
  const events = [],
    t = new ConfigurationTransition();
  await t.run({
    quiesce: async () => events.push("paused"),
    save: async () => events.push("saved"),
    prepare: async () => {
      events.push("prepared");
      return 7;
    },
    commit: async (c) => events.push(c),
  });
  assert.deepEqual(events, ["paused", "saved", "prepared", 7]);
});
test("continuous drive clock bounds catchup to five fixed steps", () => {
  const c = new DriveClock(1 / 60);
  let steps = 0;
  c.advance(0, () => steps++);
  c.advance(40, () => steps++);
  assert.equal(steps, 3);
  const result = c.advance(10000, () => steps++);
  assert.equal(result.count, 5);
  assert.equal(result.slow, true);
  assert.equal(c.advance(10000, () => steps++).count, 0);
  c.reset();
  c.advance(12000, () => false);
  assert.equal(c.next, null);
});
test("comparison difference shows effective change", () =>
  assert.deepEqual(configurationDiff({ horizon: 2 }, { horizon: 4 }), [
    "horizon: 2 → 4",
  ]));
