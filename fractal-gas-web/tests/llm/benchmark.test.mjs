import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, appendFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BenchmarkRunner } from "../../web/llm/benchmark.js";
import {
  manifest,
  collectRuns,
  exportBenchmark,
  parseBenchmark,
  processBenchmark,
  fractalRecording,
  usageTotals,
  Journal,
} from "../../web/llm/benchmark-data.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import {
  DiskBenchmarkStore,
  lockDirectory,
} from "../../tools/llm-benchmark-storage.mjs";
import { main } from "../../tools/llm-benchmark.mjs";
import { importRecording } from "../../web/llm/recording.js";
import { fakeOpenRouter, completion } from "./fixtures.mjs";
const settings = (overrides = {}) => ({
  comparison: "both",
  repetitions: 2,
  config: {
    walkers: 3,
    chunk_tokens: 2,
    sequence_tokens: 5,
    iterations: 4,
    max_walkers: 12,
    concurrency: 2,
    ...overrides,
  },
});
const key = "secret-benchmark-key";
async function runBenchmark(input = settings(), options = {}) {
  const fake = fakeOpenRouter(),
    store = new MemoryBenchmarkStore(manifest(input));
  const runner = new BenchmarkRunner(store, key, {
    fetchImpl: fake.fetch,
    ...options,
  });
  await runner.run();
  return { fake, store, runner, ...collectRuns(store.manifest, store.events) };
}
for (const algorithm of ["wave", "graph"])
  for (const comparison of ["population", "tokens", "both"]) {
    test(`${algorithm}/${comparison}: engine, baselines, accounting and archive`, async () => {
      const { store, fake, runs } = await runBenchmark({
        ...settings({ algorithm }),
        comparison,
      });
      assert.equal(runs.size, 2 * (comparison === "both" ? 4 : 3));
      for (let trial = 0; trial < 2; trial++) {
        const methods = [...runs.values()].filter((r) => r.trial === trial),
          fractal = methods.find((r) => r.method === "fractal");
        assert.ok(fractal.generated_tokens > 0);
        assert.equal(fractal.recording_version, 3);
        assert.ok(
          fractal.run.generated_tokens <=
            fractal.config.walkers * fractal.config.sequence_tokens,
        );
        assert.equal(
          fractal.completion_success,
          fractal.stop_reason === "eos_target",
        );
        assert.equal(fractal.config.seed, 7 + trial);
        importRecording(JSON.stringify(fractalRecording(fractal)));
        for (const r of methods) {
          assert.equal(r.status, "completed");
          assert.equal(
            r.generated_tokens,
            r.nodes.slice(1).reduce((n, row) => n + row.token_data.length, 0),
          );
          if (r.method === "independent_population")
            assert.equal(r.trajectories.length, 3);
          if (r.method === "independent_tokens")
            assert.equal(r.generated_tokens, fractal.generated_tokens);
          if (r.method === "temperature_zero") {
            assert.equal(r.trajectories.length, 1);
            assert.equal(r.config.temperature, 0);
          }
          if (r.method !== "fractal")
            for (const node of r.nodes.slice(1)) {
              const parent = r.nodes[node.parent];
              assert.ok(
                parent.id === 0 || parent.trajectory_id === node.trajectory_id,
              );
            }
        }
      }
      assert.ok(fake.peak <= 2);
      const exported = exportBenchmark(store.manifest, store.events);
      assert.ok(!exported.includes(key));
      const archive = parseBenchmark(exported);
      assert.deepEqual(
        processBenchmark(archive.manifest, archive.events),
        processBenchmark(store.manifest, store.events),
      );
      const bodies = store.events.filter(
        (e) =>
          e.type === "request_start" &&
          e.run_id?.includes("temperature_zero") &&
          e.payload.path === "chat/completions",
      );
      assert.ok(
        bodies.length > 0 &&
          bodies.every((e) => e.payload.request.temperature === 0),
      );
      const corrupt = structuredClone(archive);
      corrupt.events.find(
        (e) => e.type === "generation",
      ).payload.nodes[0].logp = 10;
      assert.throws(() => exportBenchmark(corrupt.manifest, corrupt.events));
    });
  }
test("identical text still uses fresh independent requests; short stops retain accounting", async () => {
  const fake = fakeOpenRouter();
  let generation = 0;
  const fetchImpl = async (url, options) => {
    if (!url.endsWith("chat/completions")) return fake.fetch(url, options);
    const body = JSON.parse(options.body);
    if (
      body.messages[0].content.startsWith("Copy the following text exactly") ||
      body.messages[0].content.startsWith("Continue this list")
    )
      return fake.fetch(url, options);
    generation++;
    return {
      ok: true,
      status: 200,
      json: async () => completion("x", -0.5, "stop"),
      headers: { get: () => null },
    };
  };
  const { store, runs } = await runBenchmark(
    { ...settings(), repetitions: 1 },
    { fetchImpl },
  );
  const independent = [...runs.values()].find(
    (r) => r.method === "independent_population",
  );
  const fractal = [...runs.values()].find((r) => r.method === "fractal");
  assert.equal(fractal.stop_reason, "eos_target");
  assert.equal(fractal.completion_success, true);
  assert.equal(fractal.run.eos_node_ids.length, 3);
  assert.equal(independent.accepted.length, 3);
  assert.equal(
    new Set(independent.nodes.slice(1).map((n) => n.trajectory_id)).size,
    3,
  );
  assert.equal(
    generation,
    [...runs.values()].reduce((n, r) => n + r.accepted.length, 0),
  );
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("three zero-progress rounds stop with preserved incomplete data", async () => {
  const fake = fakeOpenRouter();
  let phase = "";
  const store = new MemoryBenchmarkStore(
    manifest({ ...settings(), repetitions: 1, comparison: "tokens" }),
  );
  const fetchImpl = async (url, options) => {
    if (
      url.endsWith("chat/completions") &&
      phase.includes("independent_tokens")
    )
      return {
        ok: true,
        status: 200,
        json: async () => completion("", 0, "stop"),
        headers: { get: () => null },
      };
    return fake.fetch(url, options);
  };
  await assert.rejects(
    new BenchmarkRunner(store, key, {
      fetchImpl,
      onStatus: (s) => {
        phase = s;
      },
    }).run(),
    /no_token_progress/,
  );
  const r = [...collectRuns(store.manifest, store.events).runs.values()].find(
    (r) => r.method === "independent_tokens",
  );
  assert.equal(r.status, "incomplete");
  assert.equal(r.snapshots.length, 3);
  assert.equal(r.generated_tokens, 0);
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("explicit retry preserves completed methods and creates a separate attempt", async () => {
  const fake = fakeOpenRouter();
  let phase = "",
    fail = true;
  const store = new MemoryBenchmarkStore(
    manifest({ ...settings(), repetitions: 1 }),
  );
  const fetchImpl = async (url, options) => {
    if (
      fail &&
      phase.includes("independent_population") &&
      url.endsWith("/embeddings")
    )
      throw Error("offline");
    return fake.fetch(url, options);
  };
  const opts = {
    fetchImpl,
    onStatus: (s) => {
      phase = s;
    },
  };
  await assert.rejects(new BenchmarkRunner(store, key, opts).run());
  const before = [...collectRuns(store.manifest, store.events).runs.values()];
  assert.equal(before[0].status, "completed");
  assert.equal(before[1].status, "failed");
  assert.ok(before[1].accepted.length > 0);
  assert.equal(before[1].nodes.length, 1);
  const count = fake.requests.length;
  await assert.rejects(
    new BenchmarkRunner(store, key, opts).run(),
    /Explicitly retry/,
  );
  assert.equal(fake.requests.length, count);
  fail = false;
  await new BenchmarkRunner(store, key, opts).run({ retryIncomplete: true });
  const runs = [...collectRuns(store.manifest, store.events).runs.values()];
  assert.equal(runs.filter((r) => r.method === "fractal").length, 1);
  assert.equal(
    runs.filter((r) => r.method === "independent_population").length,
    2,
  );
  assert.equal(
    runs.find((r) => r.method === "independent_population" && r.attempt === 2)
      .status,
    "completed",
  );
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("storage failures latch writes; missing measurements stay unavailable", async () => {
  const store = new MemoryBenchmarkStore(manifest(settings()));
  let calls = 0;
  store.append = async () => {
    calls++;
    throw Error("quota");
  };
  const journal = new Journal(store);
  await assert.rejects(journal.append("error", { message: "first" }), /quota/);
  await assert.rejects(journal.append("error", { message: "second" }), /quota/);
  assert.equal(calls, 1);
  assert.equal(
    usageTotals([{ path: "chat/completions", usage: null }]).cost.total,
    null,
  );
});
test("disk tail recovery, writer exclusivity and CLI processing round trip", async () => {
  const dir = await mkdtemp(join(tmpdir(), "fgllm-bench-"));
  try {
    const { store } = await runBenchmark({ ...settings(), repetitions: 1 });
    await lockDirectory(dir, async () => {
      await assert.rejects(
        lockDirectory(dir, async () => {}),
        /active writer/,
      );
      const disk = await DiskBenchmarkStore.create(dir, store.manifest);
      for (const event of store.events) await disk.append(event);
      await appendFile(join(dir, "events.fgllmbench"), '{"seq":');
      const reopened = await DiskBenchmarkStore.open(dir, { recover: true });
      assert.deepEqual(await reopened.readEvents(), store.events);
    });
    const input = join(dir, "export.fgllmbench"),
      output = join(dir, "tables");
    await main(["export", "--output", dir, "--file", input], { env: {} });
    await main(["process", "--input", input, "--output", output], { env: {} });
    for (const [name, rows] of Object.entries(
      processBenchmark(store.manifest, store.events),
    )) {
      const contents = await readFile(join(output, `${name}.jsonl`), "utf8");
      assert.deepEqual(
        contents.trim() ? contents.trim().split("\n").map(JSON.parse) : [],
        rows,
      );
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("CLI generation and completed resume run without a browser", async () => {
  const dir = await mkdtemp(join(tmpdir(), "fgllm-bench-cli-"));
  try {
    const path = join(dir, "config.json"),
      output = join(dir, "run");
    await writeFile(path, JSON.stringify({ ...settings(), repetitions: 1 }));
    const fake = fakeOpenRouter(),
      options = {
        env: { OPENROUTER_API_KEY: key, INIT_CWD: dir },
        runnerOptions: { fetchImpl: fake.fetch, onStatus: () => {} },
      };
    await main(["run", "--config", "config.json", "--output", "run"], options);
    const count = fake.requests.length;
    await main(["resume", "--output", output], options);
    assert.equal(fake.requests.length, count);
    const disk = await DiskBenchmarkStore.open(output);
    parseBenchmark(exportBenchmark(disk.manifest, await disk.readEvents()));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("concurrent budget reservation shortens the final chunk and leaves partial trajectories", async () => {
  const fake = fakeOpenRouter();
  let phase = "",
    first = true;
  const fetchImpl = async (url, options) => {
    if (
      url.endsWith("chat/completions") &&
      phase.includes("fractal") &&
      first
    ) {
      first = false;
      return {
        ok: true,
        status: 200,
        json: async () => completion("a", -0.5),
        headers: { get: () => null },
      };
    }
    return fake.fetch(url, options);
  };
  const { store, runs } = await runBenchmark(
    {
      ...settings({ iterations: 1, sequence_tokens: 8 }),
      comparison: "tokens",
      repetitions: 1,
    },
    {
      fetchImpl,
      onStatus: (s) => {
        phase = s;
      },
    },
  );
  const r = [...runs.values()].find((r) => r.method === "independent_tokens");
  assert.equal(r.token_budget, 5);
  assert.deepEqual(r.accepted.map((a) => a.count).sort(), [1, 2, 2]);
  assert.ok(r.nodes.slice(1).every((n) => n.status === 0));
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("cancellation preserves accepted in-flight responses, committed boundaries and fresh retry", async () => {
  const fake = fakeOpenRouter(),
    store = new MemoryBenchmarkStore(
      manifest({ ...settings(), repetitions: 1 }),
    );
  let runner,
    stopped = false;
  const append = store.append.bind(store);
  store.append = async (event) => {
    await append(event);
    if (event.type === "generation" && !stopped) {
      stopped = true;
      runner.stop();
    }
  };
  runner = new BenchmarkRunner(store, key, { fetchImpl: fake.fetch });
  await assert.rejects(runner.run(), /Stopped/);
  const partial = [
    ...collectRuns(store.manifest, store.events).runs.values(),
  ][0];
  assert.equal(partial.status, "stopped");
  assert.equal(partial.run.stop_reason, null);
  assert.equal(partial.run.generated_tokens, partial.generated_tokens);
  assert.equal(partial.run.completion_target, partial.config.walkers);
  assert.equal(partial.snapshots.length, 1);
  parseBenchmark(exportBenchmark(store.manifest, store.events));
  await new BenchmarkRunner(store, key, { fetchImpl: fake.fetch }).run({
    retryIncomplete: true,
  });
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("storage failure prevents queued generation and keeps the durable journal prefix exportable", async () => {
  const fake = fakeOpenRouter(),
    store = new MemoryBenchmarkStore(
      manifest({ ...settings({ concurrency: 1 }), repetitions: 1 }),
    );
  const append = store.append.bind(store);
  store.append = async (event) => {
    if (event.type === "accepted") throw Error("disk full");
    await append(event);
  };
  await assert.rejects(
    new BenchmarkRunner(store, key, { fetchImpl: fake.fetch }).run(),
    /disk full/,
  );
  const generated = store.events.filter(
    (e) =>
      e.type === "request_start" &&
      e.run_id &&
      e.payload.path === "chat/completions",
  );
  assert.equal(generated.length, 1);
  assert.ok(
    store.events.some((e) => e.type === "provider_response" && e.run_id),
  );
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("invalid provider output is retained without being scored", async () => {
  const fake = fakeOpenRouter();
  let phase = "";
  const fetchImpl = async (url, options) => {
    if (phase.includes("fractal") && url.endsWith("chat/completions")) {
      const bad = completion("a");
      bad.choices[0].logprobs = null;
      return {
        ok: true,
        status: 200,
        json: async () => bad,
        headers: { get: () => null },
      };
    }
    return fake.fetch(url, options);
  };
  const store = new MemoryBenchmarkStore(
    manifest({ ...settings(), repetitions: 1 }),
  );
  await assert.rejects(
    new BenchmarkRunner(store, key, {
      fetchImpl,
      onStatus: (s) => {
        phase = s;
      },
    }).run(),
    /log probabilities/,
  );
  const run = [...collectRuns(store.manifest, store.events).runs.values()][0];
  assert.equal(run.accepted.length, 0);
  assert.equal(run.nodes.length, 1);
  assert.ok(
    store.events.some((e) => e.type === "provider_response" && e.run_id),
  );
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
test("Unicode bytes survive all methods; changed settings, ancestry, and budgets fail import", async () => {
  const fake = fakeOpenRouter();
  const fetchImpl = async (url, options) => {
    if (
      url.endsWith("chat/completions") &&
      !JSON.parse(options.body).messages[0].content.startsWith(
        "Copy the following text exactly",
      )
    )
      return {
        ok: true,
        status: 200,
        json: async () => completion("🙂", -0.2, "stop"),
        headers: { get: () => null },
      };
    return fake.fetch(url, options);
  };
  const { store } = await runBenchmark(
    { ...settings(), repetitions: 1 },
    { fetchImpl },
  );
  const archive = parseBenchmark(exportBenchmark(store.manifest, store.events));
  assert.ok(
    processBenchmark(archive.manifest, archive.events).tokens.every(
      (t) => t.bytes.join(",") === "240,159,153,130",
    ),
  );
  for (const corrupt of [
    (e) => {
      e.find((r) => r.type === "run_start").payload.config.temperature = 0.25;
    },
    (e) => {
      e.find(
        (r) =>
          r.type === "run_start" && r.payload.method === "independent_tokens",
      ).payload.token_budget++;
    },
    (e) => {
      e.find((r) => r.type === "generation").payload.nodes[0].parent = 1;
    },
  ]) {
    const events = structuredClone(archive.events);
    corrupt(events);
    assert.throws(() => exportBenchmark(archive.manifest, events));
  }
});

test("mean objectives and prompt embeddings remain aligned for every method", async () => {
  const { store, runs } = await runBenchmark({
    ...settings({ objective: "mean", embedding_input: "prompt" }),
    repetitions: 1,
  });
  for (const r of runs.values())
    for (const n of r.nodes.slice(1)) {
      const p = r.nodes[n.parent];
      assert.ok(
        Math.abs(
          n.reward - (n.logp / n.tokens - (p.tokens ? p.logp / p.tokens : 0)),
        ) < 1e-8,
      );
    }
  for (const event of store.events.filter(
    (e) =>
      e.type === "request_start" && e.run_id && e.payload.path === "embeddings",
  ))
    assert.ok(
      event.payload.request.input.every((text) =>
        text.startsWith(store.manifest.settings.config.prompt + "\n\n"),
      ),
    );
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});

test("resume refuses a changed provider before generating another method", async () => {
  const fake = fakeOpenRouter(),
    store = new MemoryBenchmarkStore(
      manifest({ ...settings(), repetitions: 1 }),
    );
  let runner;
  const append = store.append.bind(store);
  store.append = async (event) => {
    await append(event);
    if (event.type === "run_end") runner.stop();
  };
  runner = new BenchmarkRunner(store, key, { fetchImpl: fake.fetch });
  await assert.rejects(runner.run(), /Stopped/);
  const before = store.events.filter((e) => e.type === "run_start").length;
  const fetchImpl = async (url, options) => {
    const response = await fake.fetch(url, options);
    if (!url.endsWith("/endpoints")) return response;
    const data = await response.json();
    data.data.endpoints[0].tag = "changed";
    return { ...response, json: async () => data };
  };
  await assert.rejects(
    new BenchmarkRunner(store, key, { fetchImpl }).run({
      retryIncomplete: true,
    }),
    /No available endpoint/,
  );
  assert.equal(
    store.events.filter((e) => e.type === "run_start").length,
    before,
  );
  parseBenchmark(exportBenchmark(store.manifest, store.events));
});
