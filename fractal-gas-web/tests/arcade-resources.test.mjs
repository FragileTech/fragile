import test from "node:test";
import assert from "node:assert/strict";
import { resourcePlan, GiB, MiB, PAGE } from "../web/arcade-resources.js";

test("Auto uses reported CPUs, caps at twenty, and never creates idle walker slots", () => {
  assert.equal(
    resourcePlan({ n: 1000, console: 2 }, { workers: "auto" }, 64).workers,
    20,
  );
  assert.equal(
    resourcePlan({ n: 2, console: 2 }, { workers: 20 }, 64).workers,
    2,
  );
  assert.equal(
    resourcePlan({ n: 48, console: 0 }, { workers: "auto" }, 0).workers,
    3,
  );
});
test("partition enforces the combined budget with a four GiB main ceiling", () => {
  for (const memoryGiB of [1, 2, 4, 8])
    for (let workers = 1; workers <= 20; workers++) {
      if ((512 + 256) * MiB + workers * 64 * MiB > memoryGiB * GiB) {
        assert.throws(
          () => resourcePlan({ n: 1000, console: 2 }, { workers, memoryGiB }),
          /too small/,
        );
        continue;
      }
      const p = resourcePlan({ n: 1000, console: 2 }, { workers, memoryGiB });
      assert.ok(p.mainLimitBytes >= 512 * MiB && p.mainLimitBytes <= 4 * GiB);
      assert.ok(p.shimLimitBytes >= 64 * MiB && p.shimLimitBytes <= 2 * GiB);
      assert.equal(p.shimLimitBytes % PAGE, 0);
      assert.ok(p.mainLimitBytes + workers * p.shimLimitBytes + p.playbackLimitBytes <= p.budgetBytes);
    }
});
test("legacy callers and non-Sonic consoles get usable defaults", () => {
  const p = resourcePlan({ n: 48, nThreads: 4, console: 0 });
  assert.equal(p.workers, 4);
  assert.equal(p.budgetBytes, 8 * GiB);
  assert.equal(p.mainLimitBytes, 4 * GiB);
  assert.equal(p.farmWorkers, 0);
});
test("invalid limits and fractional workers are rejected", () => {
  for (const workers of [0, 21, 1.5, "4", NaN])
    assert.throws(
      () => resourcePlan({ n: 1000, console: 2 }, { workers }),
      /Workers/,
    );
  assert.throws(
    () => resourcePlan({ n: 1000, console: 2 }, { memoryGiB: 32 }),
    /memory limit/,
  );
  assert.throws(() => resourcePlan({ n: 1025, console: 2 }), /Walkers/);
});
