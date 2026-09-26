import test from "node:test";
import assert from "node:assert/strict";
import create from "../../web/optimization/engine/optimization.mjs";
import {
  NativeOptimization,
  NativePopulationCoordinator,
  populationJson,
} from "../../web/optimization/native.js";
import {
  encodePopulationRecording,
  decodePopulationRecording,
} from "../../web/optimization/populations.js";
const preset = (n) => ({
  seed: 17,
  max_evaluations: 10000,
  defaults: {
    algorithm: "wave",
    benchmark: "quadratic",
    dimensions: 2,
    walkers: 16,
    elites: 5,
    boundary: "periodic",
  },
  members: Array.from({ length: n }, (_, i) => ({
    id: `swarm-${i}`,
    settings: { perturbation_std: 0.01 * (i + 1) },
  })),
});
test("WASM coordinator exchanges complete foreign rows and respects shared budgets", async () => {
  const coordinator = new NativePopulationCoordinator(await create());
  const config = coordinator.create(preset(4));
  const engines = await Promise.all(
    config.members.map(async (member) => {
      const m = await create(),
        e = new NativeOptimization(m);
      e.create(member.settings);
      e.exchange = (request) =>
        populationJson(m, "_fgo_exchange", e.handle, request);
      e.exchange({
        op: "configure",
        id: member.id,
        count: member.exchange_count,
      });
      return e;
    }),
  );
  try {
    coordinator.request({
      op: "initialize",
      reports: engines.map((e) => e.exchange({ op: "capture" })),
    });
    const { archive } = coordinator.request({ op: "prepare" });
    engines.forEach((e) => e.exchange({ op: "sync", archive }));
    coordinator.request({
      op: "admit",
      reports: engines.map((e) => e.exchange({ op: "capture" })),
    });
    engines.forEach((e) => e.step());
    const before = engines.map((e) => e.snapshot());
    const reports = engines.map((e) => e.exchange({ op: "capture" }));
    // Arrival order is deliberately unrelated to member order.
    const { plans } = coordinator.request({
      op: "finish",
      reports: reports.toReversed(),
    });
    assert.equal(plans.length, 4);
    for (let i = 0; i < 4; i++) {
      assert.equal(plans[i].imports.length, 5);
      assert.ok(plans[i].imports.every((x) => x.walker.source !== plans[i].id));
      engines[i].exchange({ op: "stage", imports: plans[i].imports });
      assert.deepEqual(engines[i].snapshot(), before[i]);
    }
    engines.forEach((e) => e.exchange({ op: "commit" }));
    const status = coordinator.request({ op: "commit" });
    assert.equal(status.round, 1);
    assert.equal(status.exchanges, 1);
    assert.equal(status.members.length, 4);
    assert.ok(status.global_elites.length <= 20);
    const encoded = encodePopulationRecording(config, [
      { status, frames: engines.map((e) => e.snapshot()) },
    ]);
    const decoded = decodePopulationRecording(encoded);
    assert.equal(decoded.frames.length, 1);
    assert.deepEqual(decoded.frames[0].frames[0], engines[0].snapshot());
  } finally {
    engines.forEach((e) => e.dispose());
    coordinator.dispose();
  }
});
test("single member can retain every walker as an elite", async () => {
  const coordinator = new NativePopulationCoordinator(await create());
  const input = preset(1);
  input.defaults.walkers = 5;
  const c = coordinator.create(input);
  const e = new NativeOptimization(await create());
  e.create(c.members[0].settings);
  try {
    populationJson(e.m, "_fgo_exchange", e.handle, {
      op: "configure",
      id: "swarm-0",
      count: 5,
    });
    const report = populationJson(e.m, "_fgo_exchange", e.handle, {
      op: "capture",
    });
    assert.equal(report.frame.destinations.length, 0);
    coordinator.request({ op: "initialize", reports: [report] });
  } finally {
    e.dispose();
    coordinator.dispose();
  }
});
test("recording validates population and walker shapes", () => {
  assert.throws(() => decodePopulationRecording("{}"), /Invalid population/);
  assert.throws(
    () =>
      decodePopulationRecording(
        JSON.stringify({
          format: "fractal-populations",
          version: 1,
          config: { members: [{}] },
          frames: [{ frames: [[1, 2, 2]] }],
        }),
      ),
    /dimensions/,
  );
});
