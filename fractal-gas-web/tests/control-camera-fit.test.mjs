import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";
import {
  ARENA_MARGIN,
  arenaBounds,
  arenaHalfSpan,
} from "../web/lab/camera-fit.js";

const scenarioDirectory = new URL("../web/lab/scenarios/", import.meta.url);

test("stock scenario bounds fit with the selected margin", async () => {
  const files = (await readdir(scenarioDirectory)).filter((file) =>
    file.endsWith(".json"),
  );
  for (const file of files) {
    const scene = JSON.parse(
      await readFile(new URL(file, scenarioDirectory), "utf8"),
    );
    const bounds = arenaBounds(scene);
    const points = scene.boundary;
    assert(points.length >= 3, `${file}: collision boundary is required`);
    assert.equal(bounds.minX, Math.min(...points.map(([x]) => x)));
    assert.equal(bounds.maxX, Math.max(...points.map(([x]) => x)));
    assert.equal(bounds.minY, Math.min(...points.map(([, y]) => y)));
    assert.equal(bounds.maxY, Math.max(...points.map(([, y]) => y)));
    assert.deepEqual(bounds.center, [
      (bounds.minX + bounds.maxX) / 2,
      (bounds.minY + bounds.maxY) / 2,
    ]);
    for (const aspect of [16 / 9, 4 / 3, 9 / 16]) {
      const span = arenaHalfSpan(bounds, aspect);
      for (const [x, y] of points) {
        assert(Math.abs(x - bounds.center[0]) <= span * aspect);
        assert(Math.abs(y - bounds.center[1]) <= span);
      }
      assert.equal(
        span,
        (ARENA_MARGIN / 2) *
          Math.max(
            bounds.maxY - bounds.minY,
            (bounds.maxX - bounds.minX) / aspect,
          ),
      );
    }
  }
});

test("camera bounds fall back to the full scene size", () => {
  const bounds = arenaBounds({
    size: [20, 10],
    boundary: [
      [0, 0],
      [1, 1],
    ],
  });
  assert.deepEqual(bounds, {
    minX: 0,
    maxX: 20,
    minY: 0,
    maxY: 10,
    center: [10, 5],
  });
  assert.equal(arenaHalfSpan(bounds, 2), 5.25);
  assert.equal(arenaHalfSpan(bounds, 0), 10.5);
});

test("invalid scene sizes use the renderer defaults", () => {
  assert.deepEqual(arenaBounds({ size: [0, NaN] }), {
    minX: 0,
    maxX: 64,
    minY: 0,
    maxY: 44,
    center: [32, 22],
  });
});

test("degenerate boundaries fall back to the scene size", () => {
  assert.deepEqual(
    arenaBounds({
      size: [20, 10],
      boundary: [
        [3, 3],
        [3, 3],
        [3, 3],
      ],
    }),
    {
      minX: 0,
      maxX: 20,
      minY: 0,
      maxY: 10,
      center: [10, 5],
    },
  );
});
