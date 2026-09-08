import test from "node:test";
import assert from "node:assert/strict";
import {
  Recording,
  importRecording,
  RECORDING_LIMIT,
  validateFrame,
} from "../../web/optimization/recording.js";
import { frameInfo, row } from "../../web/optimization/native.js";
function frame(iteration = 0) {
  return new Float64Array([
    1,
    2,
    2,
    1,
    iteration,
    2,
    2,
    0,
    1,
    1,
    2,
    0,
    1,
    2,
    0,
    0,
    1,
    2,
    1,
    1,
    1,
    0,
    0,
    1,
    3,
    4,
    0,
    0,
    3,
    4,
    1,
    0,
    0,
    1,
    0,
    1,
  ]);
}
test("portable replay preserves all typed frame values", () => {
  const r = new Recording({
    dimensions: 2,
    algorithm: "euclidean",
    benchmark: "sphere",
  });
  r.append(frame());
  r.append(frame(1));
  const imported = importRecording(r.export());
  assert.deepEqual(imported.frames, r.frames);
  assert.equal(frameInfo(imported.frames[0]).alive, 2);
  assert.deepEqual(Array.from(row(imported.frames[0], 1).x), [3, 4]);
});
test("reject corrupt checksums, schemas, and incompatible engine versions", () => {
  const r = new Recording({ dimensions: 2 });
  r.append(frame());
  const json = JSON.parse(r.export());
  json.frames[0].checksum = 0;
  assert.throws(() => importRecording(JSON.stringify(json)), /checksum/);
  json.engine = "other";
  assert.throws(() => importRecording(JSON.stringify(json)), /Unsupported/);
  const invalid = frame();
  invalid[12 + 4 + 3] = 99;
  assert.throws(() => validateFrame(invalid, { dimensions: 2 }), /companion/);
});
test("recording limit preserves the previous complete frame", () => {
  const r = new Recording({ dimensions: 2 });
  r.append(frame());
  r.bytes = RECORDING_LIMIT - 8;
  assert.throws(() => r.append(frame(1)), /64 MiB/);
  assert.equal(r.frames.length, 1);
});
test("nonfinite invalid walkers are retained, finite active walkers required", () => {
  const f = frame();
  f[6] = 1;
  f[12 + 4 + 2] = 0;
  f[12] = NaN;
  f[12 + 4] = Infinity;
  validateFrame(f, { dimensions: 2 });
  f[12 + 4 + 2] = 1;
  assert.throws(() => validateFrame(f, { dimensions: 2 }), /Nonfinite/);
});
test("duplicate and nonmonotone iterations are rejected", () =>
  assert.throws(
    () => validateFrame(frame(1), { dimensions: 2 }, 1),
    /counters/,
  ));
