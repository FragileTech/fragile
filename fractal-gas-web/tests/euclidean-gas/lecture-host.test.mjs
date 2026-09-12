import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  demos,
  metadata,
  parameters,
} from "../../web/euclidean-gas/lecture/catalog.js";
import { chartSVG } from "../../web/euclidean-gas/lecture/plots.js";
import {
  rng,
  histogram,
  covariance2,
  pushBounded,
} from "../../web/euclidean-gas/lecture/math.js";

test("All 42 descriptors have distinct IDs, controls, and chapter placements", async () => {
  const placements = JSON.parse(
    await readFile(
      new URL(
        "../../web/euclidean-gas/lecture/placements.json",
        import.meta.url,
      ),
    ),
  );
  assert.equal(new Set(demos.map((demo) => demo.id)).size, 42);
  assert.equal(new Set(placements.map((item) => item.chapter)).size, 21);
  assert.deepEqual(
    placements.map((item) => item.id),
    demos.map((item) => item.id),
  );
  assert.doesNotThrow(() => structuredClone(metadata));
  for (const demo of demos) {
    assert.ok(
      demo.title && demo.question && demo.prediction && demo.explanation,
      demo.id,
    );
    assert.equal(
      new Set(demo.controls.map((control) => control.key)).size,
      demo.controls.length,
    );
    assert.deepEqual(
      parameters(demo),
      Object.fromEntries(
        demo.controls.map((control) => [control.key, control.value]),
      ),
    );
    for (const control of demo.controls) {
      const input = {
        [control.key]:
          control.type === "select" ? "invalid-selection" : Infinity,
      };
      assert.throws(() => parameters(demo, input), demo.id + "/" + control.key);
    }
  }
});
test("Plot rendering escapes labels and excludes nonfinite/log-invalid values", () => {
  const svg = chartSVG({
    title: '<script>alert("x")</script>',
    xScale: "log",
    yScale: "log",
    series: [
      {
        name: "Actual <measure>",
        points: [
          [0, 1],
          [1, 2],
          [10, 20],
          [NaN, 5],
          [Infinity, 6],
        ],
      },
    ],
  });
  assert.ok(!svg.includes("<script>"));
  assert.ok(svg.includes("&lt;script&gt;"));
  assert.ok(!svg.includes("NaN"));
  assert.ok(!svg.includes("Infinity"));
  assert.match(svg, /<svg/);
  const matrix = chartSVG({
    title: "Covariance",
    matrix: [
      [1, 0.5],
      [0.5, 2],
    ],
    rowLabels: ["x", "v"],
  });
  assert.match(matrix, /Row 1, column 2: 0.5/);
});
test("Bounded histories, deterministic Gaussian samples, density mass and covariance", () => {
  const a = rng(7),
    b = rng(7);
  assert.deepEqual(
    Array.from({ length: 100 }, () => a.normal()),
    Array.from({ length: 100 }, () => b.normal()),
  );
  const history = [];
  for (let i = 0; i < 1000; i++) pushBounded(history, i, 80);
  assert.equal(history.length, 80);
  assert.equal(history[0], 920);
  const bins = histogram([0.1, 0.2, 0.8, 2], 0, 1, 10);
  assert.ok(
    Math.abs(bins.reduce((sum, [, density]) => sum + density / 10, 0) - 0.75) <
      1e-12,
  );
  assert.deepEqual(
    covariance2([
      [1, 2],
      [-1, -2],
    ]),
    [
      [1, 2],
      [2, 4],
    ],
  );
});
