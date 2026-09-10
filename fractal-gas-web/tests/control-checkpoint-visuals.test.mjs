import test from "node:test";
import assert from "node:assert/strict";
import { CheckpointPresentation } from "../web/lab/visuals/checkpoints.js";
import { palette } from "../web/lab/visuals/primitives.js";
import { WorldPlayback } from "../web/lab/playback.js";

function fixture(count = 3, task = "tandem", gates = 3) {
  let now = 1000;
  let writes = 0;
  const label = {
    style: {},
    hidden: true,
    remove() {
      this.removed = true;
    },
    get textContent() {
      return this.text;
    },
    set textContent(value) {
      this.text = value;
      ++writes;
    },
  };
  const canvas = {
    ownerDocument: { createElement: () => label },
    parentElement: { append() {} },
  };
  // Offset deliberately includes interspersed passive bodies.
  const info = [1, count + 2, count, 0, 0, 0, 8 + 7 * (count + 2)];
  const p = new CheckpointPresentation(
    {
      task,
      gates: Array.from({ length: gates }, (_, i) => ({
        position: [10 + 10 * i, 20],
        radius: 3,
      })),
    },
    info,
    canvas,
    { now: () => now },
  );
  const update = (counters, tick, discontinuity = false) => {
    const state = new Float32Array(info[6] + count);
    const bits = new Uint32Array(state.buffer);
    bits[0] = tick;
    bits.set(counters, info[6]);
    p.update(state, { discontinuity });
  };
  return {
    p,
    update,
    label,
    time(value) {
      now = value;
    },
    writes: () => writes,
  };
}

test("asynchronous final playback frame keeps its forward-update flag after pause", async () => {
  const oldRequest = globalThis.requestAnimationFrame;
  const oldCancel = globalThis.cancelAnimationFrame;
  globalThis.requestAnimationFrame = () => 1;
  globalThis.cancelAnimationFrame = () => {};
  const shown = [];
  try {
    const playback = new WorldPlayback({
      show: (frame, index, context) => shown.push({ index, ...context }),
    });
    playback.attach({
      length: 2,
      dt: 0.1,
      getFrame: async (index) => ({ index }),
    });
    await playback.seek(0);
    assert.deepEqual(shown.at(-1), { index: 0, discontinuity: true });
    playback.play();
    playback.animate(0);
    playback.animate(100);
    assert.equal(playback.playing, false);
    await Promise.resolve();
    assert.deepEqual(shown.at(-1), { index: 1, discontinuity: false });
  } finally {
    globalThis.requestAnimationFrame = oldRequest;
    globalThis.cancelAnimationFrame = oldCancel;
  }
});

test("shared highlight and label count controlled agents only", () => {
  for (const count of [1, 2, 4]) {
    const f = fixture(count);
    try {
      f.update(Array(count).fill(0), 0);
      assert.equal(f.p.active, 0);
      assert.equal(f.label.textContent, `Checkpoint 1 · 0/${count} crossed`);
      assert.equal(f.p.markers[0].ring.material.color.getHex(), palette.gold);
      f.update(Array(count).fill(1), 1);
      assert.equal(f.p.active, 1);
      assert.equal(f.label.textContent, `Checkpoint 2 · 0/${count} crossed`);
      assert.equal(f.p.markers[0].ring.material.color.getHex(), palette.green);
    } finally {
      f.p.dispose();
    }
  }
});

test("partial arrivals flash, fade for 600ms, and restart on subsequent arrivals", () => {
  const f = fixture();
  try {
    f.update([0, 0, 0], 0);
    f.update([1, 0, 0], 1);
    assert.equal(f.label.textContent, "Checkpoint 1 · 1/3 crossed");
    assert.equal(f.p.markers[0].ring.material.color.getHex(), palette.green);
    f.time(1300);
    f.p.paint();
    assert.notEqual(f.p.markers[0].ring.material.color.getHex(), palette.green);
    f.update([1, 1, 0], 2);
    assert.equal(f.p.flashes[0], 1300);
    f.time(1400);
    f.update([1, 1, 1], 3);
    assert.equal(f.p.active, 1);
    assert.equal(f.p.markers[1].ring.material.color.getHex(), palette.gold);
    assert.equal(f.p.markers[0].ring.material.color.getHex(), palette.green);
    f.time(2000);
    f.p.paint();
    assert.equal(f.p.markers[0].ring.material.color.getHex(), 0x756a59);
  } finally {
    f.p.dispose();
  }
});

test("seek, backward ticks/counters and initial restore never manufacture crossings", () => {
  const f = fixture();
  try {
    f.update([5, 2, 2], 100);
    assert.equal(f.label.textContent, "Checkpoint 3 · 1/3 crossed");
    assert.ok(f.p.flashes.every((v) => v === -Infinity));
    f.update([6, 6, 6], 200, true);
    assert.equal(f.p.active, 0);
    assert.ok(f.p.flashes.every((v) => v === -Infinity));
    f.update([8, 8, 8], 150);
    assert.ok(f.p.flashes.every((v) => v === -Infinity));
    f.update([1, 1, 1], 151);
    assert.ok(f.p.flashes.every((v) => v === -Infinity));
    f.update([5, 5, 5], 160);
    assert.equal(f.p.flashes[1], 1000); // Only the latest crossing, checkpoint 5.
    assert.equal(f.p.flashes[0], -Infinity);
    assert.equal(f.p.flashes[2], -Infinity);
  } finally {
    f.p.dispose();
  }
});

test("disabled animation retains steady feedback and updates reuse resources", () => {
  const f = fixture();
  const rings = f.p.markers.map((m) => m.ring.material);
  const geometry = f.p.ringGeometry;
  try {
    f.update([0, 0, 0], 0);
    const writes = f.writes();
    for (let i = 1; i <= 100; ++i) f.update([0, 0, 0], i);
    assert.equal(f.writes(), writes);
    f.p.setEnabled(false);
    f.update([1, 0, 0], 101);
    assert.equal(f.label.textContent, "Checkpoint 1 · 1/3 crossed");
    assert.equal(f.p.markers[0].ring.material.color.getHex(), palette.gold);
    f.p.setEnabled(true);
    assert.ok(f.p.flashes.every((v) => v === -Infinity));
    assert.equal(f.p.ringGeometry, geometry);
    assert.deepEqual(
      f.p.markers.map((m) => m.ring.material),
      rings,
    );
  } finally {
    f.p.dispose();
  }
  assert.equal(f.label.removed, true);
  assert.equal(f.p.group.children.length, 0);
});

test("only tandem scenes with controlled agents and gates create feedback", () => {
  for (const [count, task, gates] of [
    [2, "navigation", 3],
    [0, "tandem", 3],
    [2, "tandem", 0],
  ]) {
    const f = fixture(count, task, gates);
    assert.equal(f.p.label, undefined);
    assert.equal(f.p.markers.length, 0);
    f.p.dispose();
  }
});
