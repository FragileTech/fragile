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
  assert.deepEqual(events, ["prepared", "paused"]);
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
  assert.deepEqual(events, ["prepared", "paused", "saved", 7]);
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

test("invalid replacement never requests a save; save failure disposes its candidate", async () => {
  const t = new ConfigurationTransition();
  let saved = 0,
    disposed = 0;
  await assert.rejects(
    t.run({
      prepare: () => {
        throw Error("invalid scene");
      },
      quiesce: () => {},
      save: () => saved++,
      commit: () => {},
    }),
    /invalid scene/,
  );
  assert.equal(saved, 0);
  await assert.rejects(
    t.run({
      prepare: () => ({ worker: { terminate: () => disposed++ } }),
      quiesce: () => {},
      save: () => {
        throw Error("quota");
      },
      commit: () => {},
    }),
    /quota/,
  );
  assert.equal(disposed, 1);
});

test("failed retries keep recovery available until saving succeeds", async () => {
  const { RunSession } = await import("../web/lab/run-session.js");
  let attempts = 0,
    prompts = 0;
  const session = new RunSession({
    getRecording: () => ({
      retry: async () => {
        throw Error("quota");
      },
    }),
  });
  session.save = async () => {
    if (++attempts < 3) throw Error("quota");
  };
  session.failure = async () => {
    prompts++;
    return "retry";
  };
  await session.preserve();
  assert.equal(attempts, 3);
  assert.equal(prompts, 2);
});

test("Drive only advertises keyboard inputs that actuate the selected vehicle", async () => {
  const { supportsKey } = await import("../web/lab/manual-control.js");
  const channels = [
    { body: 0, name: "thrust", low: 0, high: 1 },
    { body: 1, name: "throttle", low: -1, high: 1 },
  ];
  assert.equal(supportsKey(channels, 0, "w"), true);
  assert.equal(supportsKey(channels, 0, "s"), false);
  assert.equal(supportsKey(channels, 1, "s"), true);
});
