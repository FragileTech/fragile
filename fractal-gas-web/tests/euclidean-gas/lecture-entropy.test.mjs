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
