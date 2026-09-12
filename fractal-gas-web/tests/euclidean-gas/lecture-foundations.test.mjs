import test from "node:test";
import assert from "node:assert/strict";
import { demos } from "../../web/euclidean-gas/lecture/foundations.js";
import {
  deterministicDemo,
  engine,
  params,
  metric,
  finiteMeasured,
} from "./lecture-rust-support.mjs";
for (const demo of demos)
  test(`${demo.id} uses reproducible compiled gas measurements`, () =>
    deterministicDemo(demo));
test("Cloning probabilities are recorded before the executed draws", async () => {
  const demo = demos.find((d) => d.id === "I-07");
  const model = await demo.create({ params: params(demo), seed: 516, engine });
  try {
    const evidence = await model.archive();
    for (const step of evidence.archives[0].steps)
      for (const choice of step.report.clone_plan.choices)
        assert.ok(choice.probability >= 0 && choice.probability <= 1);
    assert.ok(
      Number.isFinite(
        metric(model.snapshot(), "Acceptance martingale residual sum"),
      ),
    );
  } finally {
    model.dispose();
  }
});
test("An all-ineligible recorded population has an explicit extinction outcome", async () => {
  const demo = demos.find((d) => d.id === "I-04");
  const model = await demo.create({
    params: { ...params(demo), survivors: 0 },
    seed: 7,
    engine,
  });
  try {
    const snapshot = model.snapshot();
    finiteMeasured(snapshot);
    assert.equal(snapshot.done, true);
    assert.equal(metric(snapshot, "Eligible walkers"), 0);
    assert.equal(snapshot.result.details.terminal_outcome, "extinction");
  } finally {
    model.dispose();
  }
});
