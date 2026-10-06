import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  LectureExperiment,
  lecture_analyze,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
export { LectureExperiment, lecture_analyze };
export const engine = {
  lectureCreate: (request) => LectureExperiment.create(JSON.stringify(request)),
};
export function params(demo) {
  return Object.fromEntries(demo.controls.map((c) => [c.key, c.value]));
}
export function finiteMeasured(snapshot) {
  assert.ok(snapshot.result, "Rust analysis result");
  assert.ok(
    snapshot.charts.some((c) => c.series.some((s) => s.points.length > 0)),
    "measured points",
  );
  for (const chart of snapshot.charts)
    for (const series of chart.series)
      for (const point of series.points)
        assert.ok(
          point.length === 2 && point.every(Number.isFinite),
          `${chart.title}: ${point}`,
        );
  assert.ok(snapshot.result.details.calculation_origin.includes("archive"));
}
export function metric(snapshot, label) {
  const item = snapshot.result.metrics.find((m) => m.label === label);
  assert.ok(item, `Missing metric ${label}`);
  return item.value;
}
export async function deterministicDemo(demo) {
  const input = { id: demo.id, params: params(demo), seed: 7, engine };
  let a, b;
  try {
    a = await demo.create(input);
    finiteMeasured(a.snapshot());
    await a.step();
    const expected = a.snapshot();
    const evidence = await a.archive();
    assert.ok(evidence.archives.every((a) => a.steps.length > 0));
    const recomputed = await lecture_analyze(JSON.stringify(evidence));
    assert.deepEqual(
      recomputed.plots,
      expected.result.plots,
      "Rust archive recomputation reproduces plotted measurements",
    );
    b = await demo.create(input);
    await b.step();
    assert.deepEqual(b.snapshot(), expected, "same seed/configuration replays");
    assert.deepEqual(
      a.snapshot(),
      expected,
      "rendering does not advance dynamics",
    );
  } finally {
    a?.dispose();
    b?.dispose();
  }
}
