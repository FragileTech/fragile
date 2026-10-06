import test from "node:test";
import assert from "node:assert/strict";
import { demos } from "../../web/euclidean-gas/lecture/partvi.js";
import { LectureExperiment, lecture_analyze } from "./lecture-rust-support.mjs";
for (const demo of demos)
  test(
    demo.id + " computes real gas measurements through the compiled session",
    async () => {
      const run = await LectureExperiment.create(
        JSON.stringify({ id: demo.id, seed: 7, steps: 32, parameters: {} }),
      );
      try {
        let s;
        do {
          s = await run.advance(8);
        } while (!s.done);
        assert.ok(s.result);
        assert.ok(
          s.result.plots.some((p) => p.series.some((s) => s.points.length)),
        );
        assert.ok(
          [
            "executed_algorithm_archive",
            "independent_algorithm_continuations",
          ].includes(s.result.details.calculation_origin),
        );
        const evidence = run.evidence();
        assert.ok(evidence.archives.every((a) => a.steps.length > 0));
        const r = await lecture_analyze(JSON.stringify(evidence));
        assert.deepEqual(r.plots, s.result.plots);
        if (demo.id === "VI-36") {
          assert.equal(s.result.details.readout, "twistor");
          for (const f of s.result.details.fits || [])
            assert.equal(f.mass, null);
        }
      } finally {
        run.free();
      }
    },
  );
test("A result bundle without executed evidence cannot be imported", async () => {
  await assert.rejects(async () =>
    lecture_analyze(
      JSON.stringify({ results: [{ experiment: 1, plots: [] }] }),
    ),
  );
});
