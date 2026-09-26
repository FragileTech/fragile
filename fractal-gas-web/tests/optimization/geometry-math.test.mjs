import test from "node:test";
import assert from "node:assert/strict";
import {
  projectedCovariance,
  covarianceFactor,
  covarianceSegments,
  periodicSegments,
  cmaBoundary,
  validateGeometry,
} from "../../web/optimization/geometry.js";
const close = (a, b) =>
  assert.ok(Math.abs(a - b) < 1e-9 * Math.max(1, Math.abs(b)), `${a} != ${b}`);
const model = {
  anchor: [1, 2, 3],
  representation: "dense",
  columns: 3,
  shape: [5, 4, 0, 4, 5, 0, 0, 0, 16],
  scale: 2,
};
test("projected rotated covariance preserves scale, axes and rank", () => {
  const c = projectedCovariance(model, [0, 1, 2]);
  const l = covarianceFactor(c);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      close(
        l[i].reduce((sum, v, k) => sum + v * l[j][k], 0),
        c[i][j],
      );
  assert.deepEqual(projectedCovariance(model, [2, 0]), [
    [64, 0],
    [0, 20],
  ]);
  const repeated = covarianceFactor(projectedCovariance(model, [0, 0, 1]));
  assert.ok(repeated.flat().every(Number.isFinite));
  assert.ok(
    covarianceSegments(model, [0, 1, 2], false, -10, 10)
      .flat(2)
      .every(Number.isFinite),
  );
});
test("low-rank projection agrees with its dense covariance without dimension truncation", () => {
  const low = {
    anchor: [0, 0, 0],
    representation: "diagonal_low_rank",
    columns: 2,
    shape: [1, 2, 3, 4, 5, 6],
    scale: 1,
  };
  assert.deepEqual(projectedCovariance(low, [0, 2]), [
    [5, 12],
    [12, 41],
  ]);
  const dense = {
    ...low,
    representation: "dense",
    columns: 3,
    shape: [5, 8, 12, 8, 19, 24, 12, 24, 41],
  };
  assert.deepEqual(
    projectedCovariance(low, [1, 2], true),
    projectedCovariance(dense, [1, 2], true),
  );
  const c = projectedCovariance(low, [0, 1, 2], true);
  close(c[0][0] + c[1][1] + c[2][2], 3);
});
test("degenerate models remain finite and invalid covariance is rejected", () => {
  const zero = { ...model, shape: Array(9).fill(0) };
  assert.deepEqual(covarianceFactor(projectedCovariance(zero, [0, 1])), [
    [0, 0],
    [0, 0],
  ]);
  assert.throws(
    () =>
      covarianceFactor([
        [1, 2],
        [2, 1],
      ]),
    /positive semidefinite/,
  );
  assert.throws(
    () =>
      covarianceFactor([
        [NaN, 0],
        [0, 1],
      ]),
    /Nonfinite/,
  );
});
test("periodic jumps split at boundaries including simultaneous corner crossings", () => {
  assert.deepEqual(periodicSegments([9, 0], [-9, 0], -10, 10, true), [
    [
      [9, 0],
      [10, 0],
    ],
    [
      [-10, 0],
      [-9, 0],
    ],
  ]);
  assert.deepEqual(periodicSegments([9, 9], [-9, -9], -10, 10, true), [
    [
      [9, 9],
      [10, 10],
    ],
    [
      [-10, -10],
      [-9, -9],
    ],
  ]);
  assert.deepEqual(periodicSegments([9, 0], [-9, 0], -10, 10, false), [
    [
      [9, 0],
      [-9, 0],
    ],
  ]);
});
test("CMA transformation matches bounded identity and quadratic boundary section", () => {
  close(cmaBoundary(0, -5, 5), 0);
  close(cmaBoundary(-5.3, -5, 5), -5);
  close(cmaBoundary(-5, -5, 5), -4.925);
  for (const x of [-100, 100, 1e5])
    assert.ok(cmaBoundary(x, -5, 5) >= -5 && cmaBoundary(x, -5, 5) <= 5);
});
test("recorded diagnostics validate shapes and vectors before rendering", () => {
  const g = {
    version: 1,
    dimensions: 3,
    methods: [{ id: "cloning_guided", models: [model] }],
    events: [],
  };
  validateGeometry(g, 3);
  assert.throws(() => validateGeometry(g, 2));
  assert.throws(() =>
    validateGeometry(
      {
        ...g,
        events: [
          { origin: [0, 0, 0], destination: [1, NaN, 0], kind: "proposal" },
        ],
      },
      3,
    ),
  );
  assert.throws(() =>
    validateGeometry({ ...g, methods: [{ id: "unknown", models: [] }] }, 3),
  );
});
