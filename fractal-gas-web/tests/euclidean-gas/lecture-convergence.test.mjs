import test from "node:test";
import assert from "node:assert/strict";
import { demos } from "../../web/euclidean-gas/lecture/convergence.js";
import {
  deterministicDemo,
  engine,
  params,
  metric,
} from "./lecture-rust-support.mjs";
for (const demo of demos)
  test(`${demo.id} uses reproducible compiled gas measurements`, () =>
    deterministicDemo(demo));
for (const id of ["II-03", "II-04"])
  test(`${id} measured transport obeys the coupling upper bound`, async () => {
    const demo = demos.find((d) => d.id === id);
    const model = await demo.create({
      params: params(demo),
      seed: 516,
      engine,
    });
    try {
      const snapshot = model.snapshot();
      assert.ok(
        metric(snapshot, "Between-run marginal W2 squared") <=
          metric(snapshot, "Between-run independent coupling cost") + 1e-12,
      );
      const evidence = await model.archive();
      assert.equal(evidence.archives.length, 2);
      assert.ok(evidence.archives.every((a) => a.steps.length > 0));
    } finally {
      model.dispose();
    }
  });
test("Collective fluctuations use independent recorded run seeds", async () => {
  const demo = demos.find((d) => d.id === "III-07");
  const model = await demo.create({ params: params(demo), seed: 7, engine });
  try {
    const seeds = model.snapshot().result.details.run_seeds;
    assert.equal(new Set(seeds).size, seeds.length);
    assert.ok(seeds.length > 1);
  } finally {
    model.dispose();
  }
});
