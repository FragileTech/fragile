import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  computeComparison,
  endpointWeights,
  sourceRuns,
  describe,
  distance,
  diversity,
  pairedDifference,
  pca,
  traceComparison,
} from "../../web/llm/comparison-metrics.js";
import { DEFAULTS } from "../../web/llm/config.js";
import { BenchmarkRunner } from "../../web/llm/benchmark.js";
import {
  manifest,
  fractalRecording,
  collectRuns,
} from "../../web/llm/benchmark-data.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import {
  createReport,
  exportReport,
  parseReport,
  processReport,
  verifyReportIdentities,
} from "../../web/llm/comparison-report.js";
import {
  CRITERIA,
  DEFAULT_GRADING_MODEL,
  DEFAULT_RUBRIC,
  gradingCandidates,
  prepareJudge,
  validateGrade,
  judgeBody,
  gradesForSession,
  GradingController,
  digest,
} from "../../web/llm/grading.js";
import { main } from "../../tools/llm-benchmark.mjs";
import { fakeOpenRouter } from "./fixtures.mjs";
const node = (id, parent, text, status = 0, embedding = [1, 0]) => ({
  id,
  parent,
  text,
  status,
  embedding,
  tokens: Array.from(text).length,
  logp: -Array.from(text).length,
  token_data: Array.from(text.slice(parent > 0 ? 1 : 0)).map((t) => ({
    text: t,
    bytes: [...new TextEncoder().encode(t)],
    logprob: -1,
  })),
  reward: -1,
});
function source(algorithm = "wave") {
  return {
    kind: "recording",
    record: {
      version: 2,
      config: { ...DEFAULTS, algorithm, objective: "total" },
      nodes: [
        node(0, null, ""),
        node(1, 0, "a"),
        node(2, 0, "b", 1, [0, 1]),
        node(3, 1, "ab", 2),
        node(4, 0, "b", 1, [0, 1]),
      ],
      snapshots: [
        {
          step: 1,
          node_count: 3,
          walkers: [
            { node: 1, leaf: true },
            { node: 2, leaf: true },
          ],
        },
        {
          step: 2,
          node_count: 5,
          walkers: [
            { node: 1, leaf: false },
            { node: 3, leaf: true },
            { node: 3, leaf: true },
            { node: 0, leaf: true },
          ],
        },
      ],
      requests: [],
    },
  };
}
test("archived endpoints preserve independently identical answers; retained slots keep clone weights", () => {
  const s = source(),
    run = sourceRuns(s)[0];
  assert.deepEqual(
    [...endpointWeights(run, "archive")],
    [
      [2, 1],
      [3, 1],
      [4, 1],
    ],
  );
  const d = computeComparison(s);
  assert.equal(d.groups[0].total, 3);
  assert.equal(d.groups[0].unique, 2);
  assert.equal(d.groups[1].total, 2);
  assert.equal(d.groups[1].traces[0].weight, 2);
  assert.equal(d.groups[1].diversity.pairs[0].value, 0);
  assert.equal(d.summaries[0].difference.value, null);
  assert.deepEqual(d.methods, ["fractal"]);
  assert.deepEqual(
    [...endpointWeights(sourceRuns(source("graph"))[0], "retained")],
    [[3, 2]],
  );
});
test("EOS filter, partial preview, root-relative rewards and separate likelihoods", () => {
  const s = source();
  assert.equal(computeComparison(s, { status: "eos" }).groups[0].total, 2);
  const partial = source();
  partial.record.nodes = partial.record.nodes.slice(0, 2);
  partial.record.snapshots = [{ node_count: 2, walkers: [{ node: 1 }] }];
  const d = computeComparison(partial);
  assert.equal(d.partialPreview, true);
  assert.equal(d.rows[0].mean, -1);
  assert.equal(d.rows[0].nll, 1);
  const full = computeComparison(s, { metric: "reward" });
  assert.equal(full.groups[0].stats.mean, -4 / 3);
  assert.equal(full.rows.find((r) => r.node_id === 3).reward, -2);
});
test("weighted quantiles, exact distances and missing data are mechanical", () => {
  const stats = describe(
    [
      { value: 0, weight: 1 },
      { value: 2, weight: 3 },
    ],
    "value",
  );
  assert.equal(stats.mean, 1.5);
  assert.equal(stats.median, 2);
  assert.equal(stats.q25, 1.5);
  assert.equal(distance([1, 0], [0, 1]), 1);
  assert.equal(distance([1, 0], [0, 1], "l2"), Math.SQRT2);
  assert.equal(distance([0, 0], [1, 0]), null);
  assert.equal(distance([1], [1, 2]), null);
  assert.equal(describe([{ value: null, weight: 2 }], "value").mean, null);
});
test("pair sampling is uniform, deterministic and bounded; PCA handles degenerate/incompatible embeddings", () => {
  const nodes = Array.from({ length: 400 }, (_, id) =>
    node(id, 0, "x", 1, [id + 1, 1]),
  );
  const rows = nodes.map((n) => ({
    key: String(n.id),
    node_id: n.id,
    weight: 1,
    run_id: "x",
    method: "fractal",
  }));
  const a = diversity(structuredClone(rows), nodes, "cosine", 7),
    b = diversity(structuredClone(rows), nodes, "cosine", 7);
  assert.equal(a.pairs.length, 50000);
  assert.equal(a.total, 79800);
  assert.deepEqual(a, b);
  assert.equal(new Set(a.pairs.map((p) => `${p.a}/${p.b}`)).size, 50000);
  const run = { id: "x", nodes, config: DEFAULTS };
  assert.equal(pca(rows, [run], "cosine").points.length, 400);
  for (const n of nodes) n.embedding = [1, 1];
  assert.match(pca(rows, [run], "cosine").reason, /no variation/);
  nodes[0].embedding = [1, 1, 1];
  assert.match(pca(rows, [run], "cosine").reason, /Incompatible/);
});
test("paired summaries resample trials and omit single-trial uncertainty", () => {
  const a = [
      { trial: 0, value: 1 },
      { trial: 1, value: 5 },
    ],
    b = [
      { trial: 0, value: 0 },
      { trial: 1, value: 2 },
    ];
  assert.deepEqual(pairedDifference(a, b), pairedDifference(a, b));
  assert.equal(pairedDifference(a, b).value, 2);
  assert.equal(pairedDifference(a, b).trials, 2);
  assert.equal(pairedDifference(a.slice(0, 1), b).interval, null);
  assert.equal(pairedDifference(a, []).value, null);
});
test("trace pins preserve Unicode bytes and distinguish shared ancestry from matching text", () => {
  const s = source();
  s.record.nodes.push(node(5, 1, "ac", 1));
  let compared = traceComparison(sourceRuns(s), ["current/3", "current/5"]);
  assert.equal(compared.shared_ancestry_tokens, 1);
  assert.equal(compared.matching_text_characters, 1);
  compared = traceComparison(sourceRuns(s), ["current/2", "current/4"]);
  assert.equal(compared.shared_ancestry_tokens, 0);
  assert.equal(compared.matching_text_characters, 1);
  s.record.nodes.push(node(6, 0, "🌱", 1));
  compared = traceComparison(sourceRuns(s), ["current/6"]);
  assert.deepEqual(compared.traces[0].tokens[0].bytes, [240, 159, 140, 177]);
});
async function benchmark(algorithm = "wave") {
  const store = new MemoryBenchmarkStore(
    manifest({
      comparison: "both",
      repetitions: 2,
      config: {
        algorithm,
        walkers: 2,
        chunk_tokens: 2,
        sequence_tokens: 4,
        iterations: 3,
        max_walkers: 8,
      },
    }),
  );
  await new BenchmarkRunner(store, "not-saved", {
    fetchImpl: fakeOpenRouter().fetch,
  }).run();
  return { kind: "benchmark", manifest: store.manifest, events: store.events };
}
for (const algorithm of ["wave", "graph"])
  test(`${algorithm} benchmark groups, exact boundary work, failed-attempt exclusion and offline reports`, async () => {
    const s = await benchmark(algorithm),
      d = computeComparison(s);
    assert.equal(d.summaries.length, 8);
    assert.equal(d.trials.length, 2);
    assert.ok(d.histories.every((h) => h.generated_tokens > 0));
    for (const run of sourceRuns(s)) {
      const history = d.histories.filter((h) => h.run_id === run.id);
      assert.equal(history.at(-1).generated_tokens, run.generated_tokens);
      assert.ok(
        history.every(
          (h, i) => !i || h.generated_tokens >= history[i - 1].generated_tokens,
        ),
      );
    }
    const interrupted = structuredClone(s);
    const end = interrupted.events.findIndex((e) => e.type === "run_end");
    interrupted.events = interrupted.events.slice(0, end);
    assert.equal(computeComparison(interrupted).rows.length, 0);
    assert.ok(
      computeComparison({ ...interrupted, live: true }, { status: "all" }).rows
        .length > 0,
    );
    assert.ok(
      computeComparison(interrupted, { attempt: "0:fractal:1", status: "all" })
        .rows.length > 0,
    );
    const report = createReport(s);
    const parsed = parseReport(exportReport(report));
    assert.deepEqual(processReport(parsed), processReport(report));
    assert.ok(!exportReport(report).includes("not-saved"));
    const recording = fractalRecording(
      [...collectRuns(s.manifest, s.events).runs.values()][0],
    );
    const standalone = createReport({ kind: "recording", record: recording });
    assert.ok(
      processReport(parseReport(exportReport(standalone))).nodes.length,
    );
    const corrupt = structuredClone(report);
    corrupt.source.events.find(
      (e) => e.type === "generation",
    ).payload.nodes[0].tokens = 999;
    assert.throws(() => exportReport(corrupt));
    report.api_key = "secret";
    assert.throws(() => exportReport(report), /credentials/);
  });
test("CLI comparison processing works offline and matches shared calculations", async () => {
  const report = createReport(await benchmark()),
    dir = await mkdtemp(join(tmpdir(), "fgcompare-"));
  try {
    const input = join(dir, "report.fgllmcompare"),
      out = join(dir, "processed");
    await writeFile(input, exportReport(report));
    await main(["process", "--input", input, "--output", out], { env: {} });
    const rows = (await readFile(join(out, "metrics.jsonl"), "utf8"))
      .trim()
      .split("\n")
      .map(JSON.parse);
    assert.deepEqual(rows, processReport(report).metrics);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
function judgeFetch({ invalid = false, finish = "stop", wait } = {}) {
  const calls = [];
  return {
    calls,
    fetch: async (url, options = {}) => {
      let data;
      if (url.endsWith("/endpoints"))
        data = {
          data: {
            endpoints: [
              {
                tag: "judge-endpoint",
                supported_parameters: [
                  "structured_outputs",
                  "response_format",
                  "temperature",
                  "max_tokens",
                ],
                pricing: { prompt: "0.00001", completion: "0.00002" },
              },
            ],
          },
        };
      else {
        const body = JSON.parse(options.body);
        calls.push(body);
        await wait?.();
        data = {
          id: "grade-request",
          provider: "Judge",
          model: body.model,
          choices: [
            {
              finish_reason: finish,
              message: {
                content: invalid
                  ? "bad-json"
                  : JSON.stringify({
                      scores: Object.fromEntries(CRITERIA.map((k) => [k, 3])),
                      explanation: "Evidence supports the scores.",
                    }),
              },
            },
          ],
          usage: { total_tokens: 20, cost: 0.001 },
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => data,
        headers: { get: () => null },
      };
    },
  };
}
const profile = { model: "judge/model", rubric: DEFAULT_RUBRIC, reference: "" };
test("Gemini alias resolves a stable concrete model and pins its active standard endpoint", async () => {
  const paths = [];
  const supported_parameters = [
    "structured_outputs",
    "response_format",
    "temperature",
    "max_tokens",
  ];
  const fetchImpl = async (url) => {
    paths.push(url);
    const data = url.endsWith("/models")
      ? [
          { id: "google/gemini-3.7-flash", created: 1 },
          { id: "google/gemini-3.8-flash", created: 2 },
          { id: "google/gemini-4-flash-preview", created: 3 },
          { id: "google/gemini-4-flash-lite", created: 4 },
        ]
      : url.includes("gemini-flash-latest")
        ? { endpoints: [] }
        : {
            endpoints: [
              { tag: "a-inactive", status: 1, supported_parameters },
              { tag: "google-ai-studio/flex", status: 0, supported_parameters },
              { tag: "google-ai-studio", status: 0, supported_parameters },
            ],
          };
    return new Response(JSON.stringify({ data }), { status: 200 });
  };
  const prepared = await prepareJudge(
    "test-key",
    { ...profile, model: DEFAULT_GRADING_MODEL },
    { fetchImpl },
  );
  assert.equal(prepared.requested_model, DEFAULT_GRADING_MODEL);
  assert.equal(prepared.profile.model, "google/gemini-3.8-flash");
  assert.equal(prepared.profile.provider, "google-ai-studio");
  assert.equal(paths.length, 2);
  assert.ok(paths[1].endsWith("google/gemini-3.8-flash/endpoints"));
  await assert.rejects(
    prepareJudge("test-key", profile, {
      fetchImpl: async () =>
        new Response(
          JSON.stringify({
            data: {
              endpoints: [
                { tag: "broken", supported_parameters: ["structured_outputs"] },
              ],
            },
          }),
        ),
    }),
    /No active endpoint/,
  );
});
test("latest Gemini Flash alias is the default benchmark grading model", () =>
  assert.equal(DEFAULT_GRADING_MODEL, "~google/gemini-flash-latest"));
async function sessionFor(s, fake) {
  const prepared = await prepareJudge("secret", profile, {
    fetchImpl: fake.fetch,
  });
  return {
    id: prepared.id,
    profile: prepared.profile,
    candidates: await gradingCandidates(s),
    results: {},
    requests: [],
    request_starts: [],
  };
}
test("judge selection is independent, method-blind, deduplicated, validated and reusable", async () => {
  const s = source(),
    fake = judgeFetch(),
    session = await sessionFor(s, fake);
  assert.equal(session.candidates.length, 2);
  const body = judgeBody(session.profile, session.candidates[0]);
  assert.equal(body.model, "judge/model");
  assert.equal(body.temperature, 0);
  assert.equal(body.max_tokens, 1024);
  assert.equal(body.logprobs, undefined);
  assert.equal(body.messages.length, 2);
  assert.ok(!body.messages[1].content.includes("fractal"));
  assert.equal(body.provider.only[0], "judge-endpoint");
  let writes = 0;
  await new GradingController("secret", session, {
    fetchImpl: fake.fetch,
    persist: async () => writes++,
  }).run();
  assert.equal(fake.calls.length, 2);
  assert.ok(writes > 4);
  assert.equal(Object.keys(gradesForSession(session)).length, 3);
  await new GradingController("secret", session, {
    fetchImpl: fake.fetch,
    persist: async () => {},
  }).run();
  assert.equal(fake.calls.length, 2);
  const incomplete = validateGrade({
    scores: { correctness: null, relevance: 4, completeness: 4, clarity: 4 },
    explanation: "Unknown truth.",
  });
  assert.equal(incomplete.overall, null);
  assert.throws(() =>
    validateGrade({ scores: { correctness: 5 }, explanation: "bad" }),
  );
  const changed = await prepareJudge(
    "secret",
    { ...profile, reference: "reference" },
    { fetchImpl: fake.fetch },
  );
  assert.notEqual(session.id, changed.id);
});
test("invalid or truncated grades stay failed; explicit retry does not repeat valid grades", async () => {
  const fake = judgeFetch({ invalid: true }),
    session = await sessionFor(source(), fake);
  await new GradingController("secret", session, {
    fetchImpl: fake.fetch,
    persist: async () => {},
  }).run(1);
  assert.equal(Object.values(session.results)[0].status, "failed");
  assert.ok(Object.values(session.results)[0].raw);
  const good = judgeFetch();
  await new GradingController("secret", session, {
    fetchImpl: good.fetch,
    persist: async () => {},
  }).run();
  assert.equal(good.calls.length, 2);
  assert.equal(session.previous_results.length, 1);
  const truncated = judgeFetch({ finish: "length" }),
    other = await sessionFor(source(), truncated);
  await new GradingController("secret", other, {
    fetchImpl: truncated.fetch,
    persist: async () => {},
  }).run();
  assert.ok(Object.values(other.results).every((r) => r.status === "failed"));
});
test("grading pause, cancellation and storage failure prevent further scheduling", async () => {
  const fake = judgeFetch(),
    session = await sessionFor(source(), fake),
    controller = new GradingController("secret", session, {
      fetchImpl: fake.fetch,
      persist: async () => {},
    });
  controller.pause();
  const pending = controller.run();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(fake.calls.length, 0);
  controller.stop();
  await pending;
  assert.equal(fake.calls.length, 0);
  const fail = new GradingController("secret", session, {
    fetchImpl: fake.fetch,
    persist: async () => {
      throw Error("disk full");
    },
  });
  await assert.rejects(fail.run(), /disk full/);
  assert.equal(fake.calls.length, 0);
});
test("grading report identities and source answer associations are validated", async () => {
  const s = await benchmark(),
    report = createReport(s),
    fake = judgeFetch(),
    session = await sessionFor(s, fake);
  report.evaluations.push(session);
  report.selected_evaluation = session.id;
  await new GradingController("secret", session, {
    fetchImpl: fake.fetch,
    persist: async () => {},
  }).run(1);
  await verifyReportIdentities(parseReport(exportReport(report)));
  assert.equal(processReport(report).grading_sessions.length, 1);
  assert.ok(processReport(report).grades.length > 0);
  const corrupt = structuredClone(report);
  corrupt.evaluations[0].candidates[0].answer = "changed";
  assert.throws(() => exportReport(corrupt), /source answer/);
  corrupt.evaluations[0].candidates[0].answer = session.candidates[0].answer;
  corrupt.evaluations[0].id = await digest("wrong");
  await assert.rejects(
    verifyReportIdentities(parseReport(exportReport(corrupt))),
    /identity mismatch/,
  );
});

test("distribution weights trials equally and never plots missing measurements as zero", () => {
  const s = source(),
    data = computeComparison(s, { metric: "grade" });
  assert.equal(data.groups[0].stats.mean, null);
  assert.ok(
    data.distributions.archive.selected.hist.every(
      (series) => series.points.length === 0,
    ),
  );
  const zero = computeComparison(s, {
    metric: "greedy_distance",
    baseline: "fractal",
  });
  assert.equal(zero.summaries[0].difference.value, null);
});
test("explicit grading retries are restricted to the selected candidates", async () => {
  const fake = judgeFetch(),
    session = await sessionFor(source(), fake);
  const selected = session.candidates[1].id;
  await new GradingController("secret", session, {
    fetchImpl: fake.fetch,
    persist: async () => {},
  }).run(100, [selected]);
  assert.deepEqual(Object.keys(session.results), [selected]);
  assert.equal(fake.calls.length, 1);
});
test("incremental event reads and durable notifications retain journal order", async () => {
  const input = manifest({
      comparison: "tokens",
      repetitions: 1,
      config: {
        walkers: 2,
        chunk_tokens: 2,
        sequence_tokens: 4,
        iterations: 2,
      },
    }),
    store = new MemoryBenchmarkStore(input),
    committed = [];
  await new BenchmarkRunner(store, "secret", {
    fetchImpl: fakeOpenRouter().fetch,
    onCommitted: async (event) => {
      assert.equal(store.events.at(-1).seq, event.seq);
      committed.push(event.seq);
    },
  }).run();
  assert.deepEqual(
    committed,
    store.events.map((e) => e.seq),
  );
  assert.deepEqual(await store.readEvents(10), store.events.slice(10));
});
