import test from "node:test";
import assert from "node:assert/strict";
import { demos } from "../../web/euclidean-gas/lecture/fractal.js";
import { LectureExperiment, lecture_analyze } from "./lecture-rust-support.mjs";
for (const demo of demos)
  test(
    demo.id + " measures the executed Fractal Set and replays its archive",
    async () => {
      const run = await LectureExperiment.create(
        JSON.stringify({ id: demo.id, seed: 516, steps: 16, parameters: {} }),
      );
      try {
        let s;
        do {
          s = await run.advance(8);
        } while (!s.done);
        assert.ok(
          s.result.plots.some((p) => p.series.some((s) => s.points.length)),
        );
        const evidence = run.evidence();
        assert.ok(evidence.archives[0].steps.length === 16);
        const r = await lecture_analyze(JSON.stringify(evidence));
        assert.deepEqual(r.plots, s.result.plots);
        if (demo.id === "V-02")
          assert.equal(
            s.result.metrics.find(
              (m) => m.label === "Boundary of triangle boundary residual",
            ).value,
            0,
          );
        const restored = await LectureExperiment.restore(run.checkpoint());
        try {
          assert.deepEqual(restored.evidence().archives, evidence.archives);
        } finally {
          restored.free();
        }
      } finally {
        run.free();
      }
    },
  );
