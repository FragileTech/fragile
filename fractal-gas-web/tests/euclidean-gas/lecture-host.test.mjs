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

test("All 128 descriptors have distinct IDs, controls, and chapter placements", async () => {
  const placements = JSON.parse(
    await readFile(
      new URL(
        "../../web/euclidean-gas/lecture/placements.json",
        import.meta.url,
      ),
    ),
  );
  assert.equal(new Set(demos.map((demo) => demo.id)).size, 128);
  assert.equal(new Set(placements.map((item) => item.chapter)).size, 34);
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
test("Published chapter figures use the current scientific controls and reviewed captions", async () => {
  const [captions, manifest] = await Promise.all([
    readFile(
      new URL("../../web/euclidean-gas/lecture/captions.json", import.meta.url),
      "utf8",
    ).then(JSON.parse),
    readFile(
      new URL(
        "../../../docs/_static_theory/gas-demos/manifest.json",
        import.meta.url,
      ),
      "utf8",
    ).then(JSON.parse),
  ]);
  for (const demo of demos) {
    const published = manifest.find((entry) => entry.id === demo.id);
    assert.ok(published, demo.id);
    assert.deepEqual(
      published.controls,
      demo.controls,
      `${demo.id}: regenerate lecture assets`,
    );
    assert.deepEqual(published.params, parameters(demo), demo.id);
    assert.equal(published.prediction, captions[demo.id].prediction, demo.id);
    assert.equal(
      published.question,
      captions[demo.id].question || demo.question,
      demo.id,
    );
    assert.equal(published.title, demo.title, demo.id);
    assert.equal(published.kind, demo.kind, demo.id);
  }
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
