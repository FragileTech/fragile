import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine, controlThreads } from "../web/lab/native.js";
import { equalBytes, acceptPlan, branchActions } from "../web/lab/timing.js";
import {
  Recording,
  exportRecording,
  importRecording,
} from "../web/lab/archive.js";

const scene = JSON.parse(
  await readFile(
    new URL("../web/lab/scenarios/rocket.json", import.meta.url),
    "utf8",
  ),
);
const presetIds = JSON.parse(
  await readFile(
    new URL("../web/lab/scenario-catalog.json", import.meta.url),
    "utf8",
  ),
).map((p) => p.id);
const module = await loadNative(false);
test("thread selection accepts 1–64 and rejects invalid counts", () => {
  for (const count of [1, 4, 8, 32, 64])
    assert.equal(controlThreads(count), count);
  for (const count of [0, -1, 65, 3.5, NaN, Infinity, "64"])
    assert.throws(() => controlThreads(count), /integer from 1 to 64/);
});
test("WASM state restores exactly; branch replay and recording round trip", () => {
  const engine = new NativeEngine(module, scene);
  try {
    engine.begin(
      { walkers: 24, horizon: 6, frames: 6, recording: 1, elites: 2 },
      19,
    );
    while (!engine.advance()) {}
    const tree = engine.tree(),
      states = engine.states(true, 24),
      saved = engine.snapshot();
    const leaf = tree.meta[tree.meta.length - 5];
    engine.restore(tree.root);
    for (const edge of branchActions(tree, leaf))
      engine.step(edge.action, edge.frames);
    const replayed = engine.states();
    assert(
      equalBytes(
        new Uint8Array(replayed.buffer, 0, engine.words * 4),
        new Uint8Array(states.buffer, 23 * engine.stride * 4, engine.words * 4),
      ),
    );
    engine.restore(saved);
    engine.step(engine.action(), 6);
    const future = engine.snapshot();
    engine.restore(saved);
    engine.step(engine.action(), 6);
    assert(equalBytes(future, engine.snapshot()));
    const record = new Recording();
    record.append({ tree, decision: 1, action: engine.action(), selectedReward: 12.5, executionMode: "controller", settings: { horizon: 6 }, rewards: { progress: 2 } });
    const restored = importRecording(
      exportRecording(scene, {}, record.entries),
    );
    assert.deepEqual(restored.recording.entries[0].tree, tree);
    assert.equal(restored.recording.entries[0].selectedReward, 12.5);
    assert.deepEqual(restored.recording.entries[0].rewards, { progress: 2 });
    assert.deepEqual(restored.recording.entries[0].settings, { horizon: 6 });
    const corrupt = saved.slice();
    corrupt[corrupt.length - 1] ^= 1;
    assert.throws(() => engine.restore(corrupt), /checksum/);
    assert(equalBytes(future, engine.snapshot()));
  } finally {
    engine.dispose();
  }
});
test("deadline acceptance checks exact root, target tick and revision", () => {
  const root = new Uint8Array([1, 2, 3]),
    result = { root, target: 6, revision: 4 };
  assert(acceptPlan(result, 6, 4, root.slice()));
  assert(!acceptPlan(result, 7, 4, root));
  assert(!acceptPlan(result, 5, 4, root));
  assert(!acceptPlan(result, 6, 5, root));
  assert(!acceptPlan(result, 6, 4, new Uint8Array([1, 2, 4])));
});
test("all catalog scenario presets advance joint actions in WASM", async () => {
  for (const key of presetIds) {
    const config = JSON.parse(
      await readFile(
        new URL(`../web/lab/scenarios/${key}.json`, import.meta.url),
        "utf8",
      ),
    );
    const engine = new NativeEngine(module, config, 4);
    try {
      const root = engine.snapshot();
      engine.step(new Float32Array(4 * engine.dim), 3);
      assert.equal(engine.metrics()[4], 3);
      engine.restore(root);
      assert.equal(engine.metrics()[4], 0);
    } finally {
      engine.dispose();
    }
  }
});
