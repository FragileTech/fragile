import test from "node:test";
import assert from "node:assert/strict";
import { configuration, bestNode } from "../../web/llm/config.js";
import { Recording, importRecording } from "../../web/llm/recording.js";
import { TokenEnvironment } from "../../web/llm/environment.js";
import { NativeLlm } from "../../web/llm/native.js";
import {
  OpenRouter,
  parseCompletion,
  CONTINUATION_PROBE_TEXT,
} from "../../web/llm/openrouter.js";
import { initialRun, stoppingReason } from "../../web/llm/run-control.js";
import { completion, fakeOpenRouter } from "./fixtures.mjs";

async function setup(
  overrides = {},
  sample = ({ count }) => completion("x".repeat(count)),
) {
  const config = configuration({
    walkers: 4,
    chunk_tokens: 2,
    sequence_tokens: 20,
    max_walkers: 32,
    iterations: 100,
    concurrency: 2,
    ...overrides,
  });
  const record = new Recording(config),
    calls = [];
  record.metadata({ dimensions: 2 });
  const api = {
    dimensions: 2,
    async generate(_config, prefix, count, request) {
      const call = { ...request, prefix, count, number: calls.length + 1 };
      calls.push(call);
      const result = parseCompletion(await sample(call), count);
      return { ...result, logical_request_id: `sample-${call.number}` };
    },
    async embed(_model, texts) {
      return texts.map((text) => [text.length + 1, 1]);
    },
  };
  const env = new TokenEnvironment(config, api, record);
  const native = await NativeLlm.create({ ...config, dimensions: 2 }, (r) =>
    env.transition(r),
  );
  return {
    config,
    record,
    calls,
    env,
    native,
    async step() {
      env.commit(await native.advance());
      return record.data.snapshots.at(-1);
    },
    async run() {
      while (!record.data.run.stop_reason) await this.step();
    },
  };
}

for (const algorithm of ["wave", "graph"]) {
  test(`${algorithm}: archive EOS, recycle its slot, and stop at the cumulative target`, async () => {
    const r = await setup({ algorithm }, ({ number, count }) =>
      completion(
        "x".repeat(count),
        number === 2 ? -0.01 : -1,
        number === 2 || number > 4 ? "stop" : "length",
      ),
    );
    try {
      const first = await r.step();
      const terminal = structuredClone(
        r.record.data.nodes.find((n) => n.status === 1),
      );
      assert.ok(first.walkers.some((w) => w.node === terminal.id));
      assert.equal(r.record.data.run.eos_node_ids.length, 1);
      await r.run();
      const data = importRecording(r.record.export());
      assert.equal(data.run.stop_reason, "eos_target");
      assert.ok(data.run.eos_node_ids.length >= r.config.walkers);
      assert.deepEqual(data.nodes[terminal.id], terminal);
      assert.ok(!r.calls.some((c) => c.source === terminal.id));
      const recycled = data.snapshots
        .flatMap((s) => s.decisions)
        .find((d) => d.evaluated === terminal.id && d.cloned);
      assert.ok(
        recycled,
        "An EOS slot must clone even when its answer has the best score",
      );
      assert.equal(recycled.best_protected, false);
      assert.equal(data.nodes[recycled.donor].status, 0);
      assert.equal(bestNode(data.nodes, "total").id, terminal.id);
      for (const n of data.nodes.slice(1))
        assert.equal(data.nodes[n.parent].status, 0);
      const count = r.calls.length;
      await r.env.transition([{ source: 0, action: 999, duration: 2 }]);
      assert.equal(
        r.calls.length,
        count,
        "The run ending must block fresh actions too",
      );
      assert.throws(() => r.env.commit(data.snapshots.at(-1)), /ended/);
    } finally {
      r.native.close();
    }
  });

  test(`${algorithm}: exact shared token budget, including discarded and inherited work`, async () => {
    const r = await setup(
      { algorithm, sequence_tokens: 5, chunk_tokens: 2, concurrency: 3 },
      ({ count, number }) =>
        completion("x".repeat(number === 1 ? 1 : count), -0.1 * number),
    );
    try {
      await r.run();
      const data = importRecording(r.record.export());
      assert.equal(
        data.run.generated_tokens,
        r.calls.reduce(
          (sum, c, i) => sum + data.attempts[i].token_data.length,
          0,
        ),
      );
      assert.ok(data.run.generated_tokens <= 20);
      assert.ok(
        ["token_budget", "no_active_branches"].includes(data.run.stop_reason),
      );
      if (data.run.stop_reason === "token_budget")
        assert.equal(data.run.generated_tokens, 20);
      assert.ok(r.calls.some((c) => c.count === 1));
      assert.equal(
        data.run.generated_tokens,
        data.nodes.reduce((sum, n) => sum + n.token_data.length, 0),
      );
      assert.ok(data.nodes.some((n) => n.parent > 0));
      if (algorithm === "graph")
        assert.ok(
          data.snapshots.some(
            (s) =>
              s.walkers.filter((w) => w.node !== null).length >
              r.config.walkers,
          ),
          "The fixed run budget must survive Graph population growth",
        );
    } finally {
      r.native.close();
    }
  });

  test(`${algorithm}: empty EOS completes successfully without an empty Best answer`, async () => {
    const r = await setup({ algorithm }, () => completion("", 0, "stop"));
    try {
      await r.run();
      assert.equal(r.record.data.run.stop_reason, "eos_target");
      assert.equal(r.record.data.run.generated_tokens, 0);
      assert.equal(r.record.data.run.eos_node_ids.length, r.config.walkers);
      assert.equal(bestNode(r.record.data.nodes, "total"), null);
      importRecording(r.record.export());
    } finally {
      r.native.close();
    }
  });
}

test("concurrent reservations shorten the final request and preserve partial skipped rows", async () => {
  const r = await setup({
    walkers: 2,
    sequence_tokens: 3,
    chunk_tokens: 2,
    concurrency: 4,
  });
  try {
    const rows = await r.env.transition(
      Array.from({ length: 5 }, (_, i) => ({
        source: 0,
        action: i,
        duration: i === 0 ? 1 : 2,
      })),
    );
    assert.deepEqual(
      r.calls.map((c) => c.count),
      [1, 2, 2, 1],
    );
    assert.equal(r.record.data.run.generated_tokens, 6);
    assert.equal(rows[4].skipped, true);
    assert.equal(rows[4].id, 0);
    assert.equal(rows[4].status, 0);
    assert.ok(r.env.pending.every((n) => n.status === 0));
    await r.env.transition([{ source: 0, action: 99, duration: 2 }]);
    assert.equal(r.calls.length, 4);
  } finally {
    r.native.close();
  }
});

test("short replies release reserved tokens for subsequent queued requests", async () => {
  const r = await setup(
    { walkers: 2, sequence_tokens: 3, chunk_tokens: 3, concurrency: 1 },
    () => completion("x"),
  );
  try {
    await r.env.transition(
      Array.from({ length: 8 }, (_, action) => ({
        source: 0,
        action,
        duration: 3,
      })),
    );
    assert.deepEqual(
      r.calls.map((c) => c.count),
      [3, 3, 3, 3, 2, 1],
    );
    assert.equal(r.record.data.run.generated_tokens, 6);
  } finally {
    r.native.close();
  }
});

test("EOS target stops queued dispatch and retains extra already-running completions", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const r = await setup({ walkers: 2, concurrency: 2 }, async ({ number }) => {
    if (number === 1) await gate;
    if (number === 3) release();
    return completion("x", -0.5, "stop");
  });
  try {
    const rows = await r.env.transition(
      Array.from({ length: 10 }, (_, action) => ({
        source: 0,
        action,
        duration: 2,
      })),
    );
    assert.equal(r.calls.length, 3);
    assert.equal(r.env.pending.length, 3);
    assert.equal(rows.filter((r) => r.skipped).length, 7);
    assert.equal(new Set(r.env.pending.map((n) => n.id)).size, 3);
  } finally {
    r.native.close();
  }
});

test("cached EOS replay never adds tokens or completions", async () => {
  const r = await setup({}, () => completion("x", -0.5, "stop"));
  try {
    const request = { source: 0, action: 1, duration: 2 };
    const first = await r.env.transition([request, request]);
    assert.deepEqual(first[0], first[1]);
    await r.env.transition([request]);
    assert.equal(r.calls.length, 1);
    assert.equal(r.env.control.completions, 1);
    assert.equal(r.record.data.run.generated_tokens, 1);
    assert.equal(r.env.pending.length, 1);
  } finally {
    r.native.close();
  }
});

for (const algorithm of ["wave", "graph"])
  test(`${algorithm}: exhaustion preserves capped and EOS traces without starting fresh branches`, async () => {
    const r = await setup(
      { algorithm, sequence_tokens: 2 },
      ({ number, count }) =>
        number === 1
          ? completion("x", -0.5, "stop")
          : completion("x".repeat(count)),
    );
    try {
      await r.run();
      assert.equal(r.calls.length, 4);
      assert.equal(r.record.data.run.stop_reason, "no_active_branches");
      assert.equal(r.record.data.run.generated_tokens, 7);
      assert.equal(r.record.data.run.eos_node_ids.length, 1);
      assert.equal(r.record.data.nodes.filter((n) => n.status === 2).length, 3);
    } finally {
      r.native.close();
    }
  });

test("the cumulative EOS target ends a committed population with active walkers remaining", async () => {
  const r = await setup({ concurrency: 1 }, ({ number }) =>
    completion("xx", -0.5, [1, 2, 5, 6].includes(number) ? "stop" : "length"),
  );
  try {
    await r.run();
    const data = importRecording(r.record.export());
    assert.equal(data.run.stop_reason, "eos_target");
    assert.equal(data.run.eos_node_ids.length, 4);
    assert.equal(r.calls.length, 6);
    assert.ok(
      data.snapshots
        .at(-1)
        .walkers.some((w) => w.alive && data.nodes[w.node]?.status === 0),
    );
  } finally {
    r.native.close();
  }
});

test("stopping precedence is EOS, tokens, exhaustion, then iterations", () => {
  const c = configuration({
    walkers: 2,
    chunk_tokens: 2,
    sequence_tokens: 2,
    iterations: 1,
  });
  const run = { ...initialRun(c), eos_node_ids: [1, 2], generated_tokens: 4 };
  assert.equal(stoppingReason(c, run, { walkers: [] }, [], 1), "eos_target");
  run.eos_node_ids = [];
  assert.equal(stoppingReason(c, run, { walkers: [] }, [], 1), "token_budget");
  run.generated_tokens = 1;
  assert.equal(
    stoppingReason(c, run, { walkers: [] }, [], 1),
    "no_active_branches",
  );
  assert.equal(
    stoppingReason(
      c,
      run,
      { walkers: [{ alive: true, node: 0 }] },
      [{ status: 0 }],
      1,
    ),
    "iteration_limit",
  );
});

test("v3 progress is validated while v1/v2 retain their original stopping semantics", async () => {
  const r = await setup({ iterations: 1, sequence_tokens: 2 });
  try {
    await r.run();
    const saved = JSON.parse(r.record.export());
    assert.equal(importRecording(JSON.stringify(saved)).version, 3);
    for (const version of [1, 2]) {
      const legacy = structuredClone(saved);
      legacy.version = version;
      legacy.config.walkers = 2; // Legacy records are not subject to the new run allowance.
      assert.ok(
        saved.run.generated_tokens >
          legacy.config.walkers * legacy.config.sequence_tokens,
      );
      delete legacy.run;
      legacy.snapshots.forEach((s) => {
        delete s.run;
        if (version === 1) delete s.decisions;
      });
      assert.equal(importRecording(JSON.stringify(legacy)).version, version);
    }
    for (const modify of [
      (d) => d.run.eos_node_ids.push(1),
      (d) => d.run.generated_tokens++,
      (d) => (d.run.stop_reason = "eos_target"),
      (d) => d.snapshots[0].run.generated_tokens++,
      (d) => (d.nodes[1].requested_tokens = 0),
    ]) {
      const corrupt = structuredClone(saved);
      modify(corrupt);
      assert.throws(() => importRecording(JSON.stringify(corrupt)));
    }
  } finally {
    r.native.close();
  }
});

test("capability probes never continue an EOS response", async () => {
  const fake = fakeOpenRouter();
  let generation = 0;
  const api = new OpenRouter("test", {
    fetchImpl: async (url, options) => {
      if (!url.endsWith("chat/completions")) return fake.fetch(url, options);
      generation++;
      return {
        ok: true,
        status: 200,
        headers: { get: () => null },
        json: async () => completion("done", -0.5, "stop"),
      };
    },
  });
  await assert.rejects(
    api.prepare(configuration()),
    /ended before continuation/,
  );
  assert.equal(generation, 1);
});

for (const restarted of [
  CONTINUATION_PROBE_TEXT,
  "Here's a short explanation:",
]) {
  test(`preflight rejects a provider restarting with ${JSON.stringify(restarted)}`, async () => {
    const fake = fakeOpenRouter();
    let generation = 0;
    const api = new OpenRouter("test", {
      fetchImpl: async (url, options) => {
        if (!url.endsWith("chat/completions")) return fake.fetch(url, options);
        generation++;
        return {
          ok: true,
          status: 200,
          headers: { get: () => null },
          json: async () =>
            completion(
              generation === 1
                ? CONTINUATION_PROBE_TEXT.slice(0, 8)
                : restarted.slice(0, 8),
            ),
        };
      },
    });
    await assert.rejects(
      api.prepare(configuration()),
      /restarted or changed the answer/,
    );
    assert.equal(generation, 2);
  });
}
