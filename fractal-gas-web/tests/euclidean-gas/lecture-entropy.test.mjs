import test from "node:test";
import assert from "node:assert/strict";
import { demos } from "../../web/euclidean-gas/lecture/entropy.js";
import {
  deterministicDemo,
  engine,
  params,
  metric,
} from "./lecture-rust-support.mjs";
for (const demo of demos)
  test(`${demo.id} uses reproducible compiled gas measurements`, () =>
    deterministicDemo(demo));
for (const id of ["IV-05", "IV-07"])
  test(`${id} differentiates the recorded normalization`, async () => {
    const demo = demos.find((d) => d.id === id);
    const model = await demo.create({
      params: params(demo),
      seed: 516,
      engine,
    });
    try {
      assert.ok(
        Math.abs(metric(model.snapshot(), "Derivative residual")) <
          1e-5 *
            (1 +
              Math.abs(metric(model.snapshot(), "Analytic first derivative"))),
      );
    } finally {
      model.dispose();
    }
  });
test("Twelfth-order fitness jets run in the Rust library", async () => {
  const demo = demos.find((d) => d.id === "IV-07");
  const model = await demo.create({
    params: { ...params(demo), order: 12 },
    seed: 7,
    engine,
  });
  try {
    assert.ok(
      model
        .snapshot()
        .result.details.jet_multi_indices.some((a) => a[0] === 12),
    );
  } finally {
    model.dispose();
  }
});
test("Harmonic expectation is computed alongside the executed trajectory", async () => {
  const demo = demos.find((d) => d.id === "IV-10");
  const model = await demo.create({ params: params(demo), seed: 7, engine });
  try {
    const result = model.snapshot().result;
    assert.equal(
      result.details.harmonic_prediction.length,
      result.details.archive_steps,
    );
    assert.ok(
      result.plots.some((p) =>
        p.series.some(
          (s) =>
            s.name ===
            "Conditional expectation from recorded initial population",
        ),
      ),
    );
  } finally {
    model.dispose();
  }
});

test("IV-07 separates fixed assignments, the changing law and numerical resolution", async () => {
  const demo = demos.find((d) => d.id === "IV-07");
  for (const controls of [{ landscape: "quadratic" }, { landscape: "multiwell" }, { beta: 0 }]) {
    const model = await demo.create({
      params: { ...params(demo), ...controls }, seed: 7, engine,
    });
    try {
      const result = model.snapshot().result;
      const average = result.details.companion_average;
      assert.equal(average.available, true);
      assert.equal(average.count, 81);
      assert.ok(Math.abs(metric(model.snapshot(), "Assignment probability sum") - 1) < 1e-12);
      assert.ok(Math.abs(metric(model.snapshot(), "Mean derivative residual")) <
        2e-5 * (1 + Math.abs(average.production_finite_difference)));
      if (controls.beta === 0) {
        assert.ok(metric(model.snapshot(), "Companion fitness standard deviation") < 1e-12);
        assert.ok(Math.abs(average.law_derivative) < 1e-12);
      }
      const window = result.details.window_validation;
      assert.ok(window.accepted_radius <= window.requested_radius);
      assert.ok(metric(model.snapshot(), "Maximum scaled error, staggered checks") <= window.scaled_tolerance);
      assert.equal(window.analytic_remainder_bound, null);
      assert.equal(result.details.coefficient_checks.length, 6);
      assert.ok(result.details.coefficient_checks.every((c) =>
        Number.isFinite(c.independent_coefficient) && typeof c.resolved === "boolean"));
      assert.ok(result.plots.some((p) => p.title.includes("Companion fluctuations")));
      assert.ok(result.plots.some((p) => p.title.includes("Reward landscape")));
    } finally { model.dispose(); }
  }
});
