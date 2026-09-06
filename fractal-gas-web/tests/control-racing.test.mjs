import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import {
  createEnvironment,
  registerEnvironment,
} from "../web/lab/visuals/environments/index.js";
import {
  sceneReadout,
  registerSceneMetric,
} from "../web/lab/scene-presentation.js";
import { MotionRecording, WorldCapture, bytesOf } from "../web/lab/motion.js";
const scene = JSON.parse(
  await readFile(
    new URL("../web/lab/scenarios/racing.json", import.meta.url),
    "utf8",
  ),
);
const module = await loadNative();

function drive(engine) {
  const s = engine.states(),
    bits = new Uint32Array(s.buffer);
  const gate = scene.gates[bits[engine.info[6]] % scene.gates.length].position;
  const heading = Math.atan2(gate[1] - s[9], gate[0] - s[8]) - s[12];
  const error = Math.atan2(Math.sin(heading), Math.cos(heading));
  return new Float32Array([0.4, Math.max(-1, Math.min(1, error * 1.2)), 0]);
}

test("kart completes ordered laps without contacts and replays exact lap progress", () => {
  const e = new NativeEngine(module, scene);
  try {
    assert.deepEqual(
      e.channels.map((c) => c.name),
      ["throttle", "steering", "brake"],
    );
    assert.equal(e.words * 4, 64);
    const recording = new MotionRecording(e.info, e.snapshot(), 1 / 60);
    const capture = new WorldCapture(e, ({ packet, label }) =>
      recording.append(packet, label),
    );
    capture.capture(e.neutralAction(), 0, "Start grid");
    let contacts = 0;
    for (let i = 0; i < 1400 && e.metrics()[7] < 32; i++) {
      capture.step(drive(e), 3, i + 1);
      contacts += e.metrics()[2];
    }
    assert.equal(e.metrics()[7], 32);
    assert.equal(contacts, 0);
    assert.equal(sceneReadout(scene, e.metrics()).score, 2);
    const final = e.snapshot();
    for (const index of [0, 400, recording.length - 1]) {
      e.restoreRows(recording.rows(index));
      assert.deepEqual(bytesOf(e.states()), bytesOf(recording.rows(index)));
      const bits = new Uint32Array(recording.rows(index).buffer);
      assert.equal(e.metrics()[7], bits[6]);
    }
    assert.deepEqual(e.snapshot(), final);
    const checkpoint = e.snapshot(),
      action = drive(e);
    e.step(action, 6);
    const future = e.snapshot();
    e.restore(checkpoint);
    e.step(action, 6);
    assert.deepEqual(e.snapshot(), future);
  } finally {
    e.dispose();
  }
});

test("finish line alone and out-of-order checkpoints do not award laps", () => {
  const e = new NativeEngine(module, scene);
  try {
    e.step(e.neutralAction(), 100);
    assert.equal(e.metrics()[7], 0);
    const rows = e.states();
    rows[8] = scene.gates[5].position[0];
    rows[9] = scene.gates[5].position[1];
    e.restoreRows(rows);
    e.step(e.neutralAction(), 1);
    assert.equal(e.metrics()[7], 0);
  } finally {
    e.dispose();
  }
});

test("circuit renderer shares physical boundaries and highlights restored checkpoint", () => {
  const env = createEnvironment(scene),
    e = new NativeEngine(module, scene);
  try {
    assert.equal(env.replacesGates, true);
    assert.ok(env.group.getObjectByName("Asphalt racing surface"));
    env.update(e.states(), e.info);
    assert.equal(
      env.group.getObjectByName("Checkpoint 1").material.opacity,
      0.95,
    );
    const state = e.states();
    new Uint32Array(state.buffer)[e.info[6]] = 7;
    env.update(state, e.info);
    assert.equal(
      env.group.getObjectByName("Checkpoint 8").material.opacity,
      0.95,
    );
    assert.equal(
      env.group.getObjectByName("Checkpoint 1").material.opacity,
      0.12,
    );
    assert.ok(env.group.children.filter((c) => c.isInstancedMesh).length >= 6);
  } finally {
    e.dispose();
    env.group.traverse((o) => {
      o.geometry?.dispose();
      o.material?.dispose();
    });
  }
});

test("environment and readout extensions require no task-specific application branches", () => {
  registerEnvironment("test_environment", (scene) => ({
    group: { name: scene.name },
  }));
  assert.equal(
    createEnvironment({
      name: "Example",
      environment: { kind: "test_environment" },
    }).group.name,
    "Example",
  );
  registerSceneMetric("test_count", (m) => m[2]);
  const custom = {
    presentation: {
      score: { metric: "test_count", divisor: 3, label: "Cycles" },
      progress: { metric: "test_count", cycle: 3, label: "Next" },
    },
  };
  assert.deepEqual(sceneReadout(custom, [0, 0, 8]), {
    score: 2,
    label: "Cycles · Next 3/3",
  });
});

test("lab branding is copied exactly from the documentation", async () => {
  for (const name of ["logo.png", "favicon.png"])
    assert.deepEqual(
      await readFile(new URL(`../web/lab/branding/${name}`, import.meta.url)),
      await readFile(new URL(`../../docs/${name}`, import.meta.url)),
    );
});
