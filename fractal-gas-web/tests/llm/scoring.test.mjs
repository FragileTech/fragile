import test from "node:test";
import assert from "node:assert/strict";
import {
  configuration,
  recordedConfiguration,
  objective,
  selectedScore,
  bestNode,
} from "../../web/llm/config.js";
import {
  TogetherScorer,
  parsePromptScore,
  scoringPrefix,
  validateXed,
  SCORING_FORMAT,
  loadScoringTokenizer,
} from "../../web/llm/scoring.js";
import { Recording, importRecording } from "../../web/llm/recording.js";
import { TokenEnvironment } from "../../web/llm/environment.js";
import { NativeLlm } from "../../web/llm/native.js";
import { BenchmarkRunner } from "../../web/llm/benchmark.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import {
  manifest,
  collectRuns,
  exportBenchmark,
  parseBenchmark,
  processBenchmark,
} from "../../web/llm/benchmark-data.js";
import { completion, fakeOpenRouter } from "./fixtures.mjs";
import { parseCompletion } from "../../web/llm/openrouter.js";
import { AnalysisIndex } from "../../web/llm/analysis-data.js";

const config = (overrides = {}) =>
  configuration({
    objective: "xed",
    walkers: 2,
    chunk_tokens: 2,
    sequence_tokens: 6,
    iterations: 8,
    max_walkers: 12,
    concurrency: 2,
    ...overrides,
  });
function echoed(body) {
  const prefixEnd =
    body.prompt.lastIndexOf("</think>\n\n") + "</think>\n\n".length;
  const prefix = body.prompt.slice(0, prefixEnd),
    answer = body.prompt.slice(prefixEnd);
  // The scorer deliberately uses one codepoint/token, independently of generator tokens.
  const chars = [...answer];
  const lp = body.prompt.startsWith(scoringPrefix("")) ? -2 : -1;
  return {
    model: body.model,
    prompt: [
      {
        text: body.prompt,
        logprobs: {
          tokens: [prefix, ...chars],
          token_ids: [1, ...chars.map((c) => c.codePointAt(0) + 10)],
          token_logprobs: [null, ...chars.map(() => lp)],
        },
      },
    ],
    choices: [{ text: "ignored", finish_reason: "length" }],
    usage: {
      prompt_tokens: chars.length + 10,
      completion_tokens: body.max_tokens,
      total_tokens: chars.length + 10 + body.max_tokens,
    },
  };
}
function mockScorer(options = {}) {
  const calls = [];
  let active = 0,
    peak = 0;
  const fetchImpl = async (_url, init) => {
    const body = JSON.parse(init.body);
    calls.push(body);
    active++;
    peak = Math.max(peak, active);
    try {
      await new Promise((r) => setTimeout(r, 1));
      if (options.fallback && body.max_tokens === 0)
        return {
          ok: false,
          status: 400,
          json: async () => ({
            error: { message: "max_tokens must be positive" },
          }),
        };
      if (options.fail?.(body)) throw Error("scorer network failure");
      return {
        ok: true,
        status: 200,
        json: async () => options.response?.(body) ?? echoed(body),
      };
    } finally {
      active--;
    }
  };
  const scorer = new TogetherScorer("test-together-secret", {
    fetchImpl,
    concurrency: 2,
    ...options,
  });
  return {
    scorer,
    fetchImpl,
    calls,
    get peak() {
      return peak;
    },
  };
}
test("default, beam endpoints, signed XED and empty-sequence utility", () => {
  const n = {
    tokens: 4,
    logp: -8,
    xed: { tokens: 2, conditional_logp: -2, baseline_logp: -6 },
  };
  assert.equal(configuration().objective, "beam");
  assert.equal(recordedConfiguration().objective, "total");
  assert.equal(objective(n, configuration()), -8 / 4 ** 0.6);
  assert.equal(objective(n, config({ objective: "beam", beam_alpha: 0 })), -8);
  assert.equal(objective(n, config({ objective: "beam", beam_alpha: 1 })), -2);
  assert.equal(selectedScore(n, config()), 2);
  assert.equal(objective(n, config({ xed_direction: "minimize" })), -2);
  assert.equal(objective({ tokens: 0, logp: 0 }, config()), 0);
  for (const beam_alpha of [-1, 3, NaN, Infinity])
    assert.throws(() => config({ beam_alpha }));
  assert.throws(() => config({ xed_direction: "other" }));
  const a = { ...n, id: 1, status: 1 },
    b = { ...n, id: 2, status: 1, xed: { ...n.xed, baseline_logp: -10 } };
  assert.equal(bestNode([a, b], config()).id, 2);
  assert.equal(bestNode([a, b], config({ xed_direction: "minimize" })).id, 1);
});
test("capability fallback, Unicode, caching and globally bounded paired evaluations", async () => {
  const f = mockScorer({ fallback: true }),
    c = config();
  const meta = await f.scorer.prepare(c);
  assert.equal(meta.max_tokens, 1);
  const before = f.calls.length;
  const [a, b] = await Promise.all([
    f.scorer.score(c, "Café ☀ 蓝"),
    f.scorer.score(c, "Café ☀ 蓝"),
  ]);
  assert.deepEqual(a, b);
  assert.equal(f.calls.length, before + 2);
  assert.equal(a.tokens, 8);
  assert.equal(a.conditional_logp - a.baseline_logp, 8);
  await Promise.all(["one", "two", "three"].map((t) => f.scorer.score(c, t)));
  assert.ok(f.peak <= 2);
  assert.equal(await f.scorer.score(c, ""), null);
});
test("reject missing probabilities, boundary merges, model and token-ID mismatches", async () => {
  const c = config(),
    prefix = scoringPrefix(c.prompt);
  assert.throws(() => parsePromptScore({ choices: [] }, prefix, "hi"), /echo/);
  const p = echoed({
    prompt: prefix + "hi",
    model: c.scoring_model,
    max_tokens: 0,
  });
  p.prompt[0].logprobs.tokens = [prefix + "h", "i"];
  p.prompt[0].logprobs.token_ids = [1, 2];
  p.prompt[0].logprobs.token_logprobs = [-1, -1];
  assert.throws(() => parsePromptScore(p, prefix, "hi"), /boundary/);
  const f = mockScorer({
    response: (body) => {
      const p = echoed(body);
      if (body.prompt.startsWith(scoringPrefix("")))
        p.prompt[0].logprobs.token_ids[1]++;
      return p;
    },
  });
  await assert.rejects(f.scorer.score(c, "hi"), /different answer tokens/);
  const g = mockScorer({
    response: (body) => ({ ...echoed(body), model: "different" }),
  });
  await assert.rejects(g.scorer.prepare(c), /different model/);
  const h = mockScorer(),
    xed = await h.scorer.score(c, "hi");
  xed.baseline_logp = 0;
  assert.throws(
    () => validateXed({ tokens: 1, text: "hi", xed }, c),
    /match recorded/,
  );
});
for (const algorithm of ["wave", "graph"])
  for (const mode of ["mean", "beam", "xed"]) {
    test(`${algorithm}/${mode}: native utility, clone ancestry, EOS archive and round-trip`, async () => {
      const c = config({
        algorithm,
        objective: mode,
        xed_direction: "minimize",
        beam_alpha: 0.4,
        walkers: 4,
        sequence_tokens: 20,
        iterations: 40,
      });
      const f = mockScorer(),
        record = new Recording(c);
      record.metadata({ dimensions: 2 });
      let count = 0;
      const api = {
        async generate(_c, prefix, n) {
          return {
            ...parseCompletion(
              completion(
                "a".repeat(n),
                -0.3,
                ++count === 2 || count > 4 ? "stop" : "length",
              ),
              n,
            ),
            logical_request_id: `r${count}`,
          };
        },
        async embed(_m, texts) {
          return texts.map((t) => [1, t.length]);
        },
      };
      const env = new TokenEnvironment(c, api, record, f.scorer);
      const native = await NativeLlm.create({ ...c, dimensions: 2 }, (r) =>
        env.transition(r),
      );
      try {
        while (!record.data.run.stop_reason) env.commit(await native.advance());
        const imported = importRecording(record.export());
        assert.equal(imported.run.stop_reason, "eos_target");
        assert.ok(
          imported.run.generated_tokens <= c.walkers * c.sequence_tokens,
        );
        for (const s of imported.snapshots)
          for (const w of s.walkers)
            if (Number.isInteger(w.node) && imported.nodes[w.node])
              assert.ok(
                Math.abs(w.score - objective(imported.nodes[w.node], c)) < 1e-5,
              );
        for (const n of imported.nodes.slice(1)) {
          const parent = imported.nodes[n.parent];
          assert.equal(n.reward, objective(n, c) - objective(parent, c));
          if (mode === "xed") validateXed(n, c);
        }
        const index = new AnalysisIndex(imported);
        assert.equal(
          index.value(1, "objective", 1).value,
          selectedScore(imported.nodes[1], c),
        );
        if (mode !== "xed") assert.equal(f.calls.length, 0);
      } finally {
        native.close();
      }
    });
  }
test("scoring failure preserves accepted generation and the last committed boundary", async () => {
  const c = config(),
    record = new Recording(c);
  record.metadata({ dimensions: 2 });
  const f = mockScorer({ fail: () => true });
  const api = {
    async generate() {
      return parseCompletion(completion("ab", -1, "stop"), 2);
    },
    async embed() {
      throw Error("must not embed");
    },
  };
  const env = new TokenEnvironment(c, api, record, f.scorer);
  await assert.rejects(
    env.transition([{ source: 0, action: 1, duration: 2 }]),
    /network failure/,
  );
  assert.equal(record.data.attempts.length, 1);
  assert.equal(record.data.run.generated_tokens, 2);
  assert.equal(record.data.nodes.length, 1);
  assert.equal(record.data.snapshots.length, 0);
  importRecording(record.export());
});
test("XED benchmark archives both scoring terms and keeps scoring out of generation budget", async () => {
  const c = config(),
    fake = fakeOpenRouter(),
    scoring = mockScorer();
  const store = new MemoryBenchmarkStore(
    manifest({ config: c, comparison: "both", repetitions: 1 }),
  );
  const runner = new BenchmarkRunner(store, "openrouter-secret", {
    togetherKey: "together-secret",
    fetchImpl: (url, init) =>
      String(url).includes("api.together.ai")
        ? scoring.fetchImpl(url, init)
        : fake.fetch(url, init),
  });
  await runner.run();
  const text = exportBenchmark(store.manifest, store.events),
    parsed = parseBenchmark(text);
  assert.ok(!text.includes("together-secret"));
  assert.ok(!text.includes("openrouter-secret"));
  const runs = collectRuns(parsed.manifest, parsed.events).runs;
  for (const r of runs.values()) {
    assert.ok(r.requests.some((x) => x.path === "scoring/completions"));
    assert.equal(
      r.generated_tokens,
      r.accepted.reduce((n, a) => n + a.result.token_data.length, 0),
    );
    for (const n of r.nodes.slice(1)) if (n.tokens) validateXed(n, c);
  }
  const report = processBenchmark(parsed.manifest, parsed.events);
  assert.ok(
    report.runs.every((r) => r.scoring_usage.prompt_tokens.known_total > 0),
  );
});

test("actual Together echo shape uses pinned IDs and reconstructs split Unicode", async () => {
  const tokenizer = await loadScoringTokenizer(),
    prefix = scoringPrefix("why?"),
    answer = "Café ☀ — 蓝色";
  const encoded = tokenizer.encode(prefix + answer, {
    add_special_tokens: false,
  });
  const response = {
    choices: [
      {
        text: prefix + answer + " ignored",
        logprobs: {
          tokens: [
            ...encoded.ids.map((id) =>
              tokenizer.decode([id], { skip_special_tokens: false }),
            ),
            " ignored",
          ],
          token_logprobs: [...encoded.ids.map((_, i) => (i ? -1 : null)), -2],
        },
      },
    ],
    usage: { prompt_tokens: encoded.ids.length },
  };
  const result = parsePromptScore(response, prefix, answer, tokenizer);
  assert.equal(
    new TextDecoder().decode(Uint8Array.from(result.flatMap((t) => t.bytes))),
    answer,
  );
  assert.ok(result.every((t) => Number.isInteger(t.id)));
  assert.ok(
    result.some((t) =>
      new TextDecoder().decode(Uint8Array.from(t.bytes)).includes("�"),
    ),
  );
  response.usage.prompt_tokens--;
  assert.throws(
    () => parsePromptScore(response, prefix, answer, tokenizer),
    /pinned/,
  );
});

test("legacy missing objectives retain total scoring in versions 1, 2 and 3", () => {
  for (const version of [1, 2, 3]) {
    const data = new Recording(config({ objective: "total" })).data;
    data.version = version;
    delete data.config.objective;
    delete data.config.beam_alpha;
    delete data.config.xed_direction;
    delete data.config.scoring_model;
    const imported = importRecording(JSON.stringify(data));
    assert.equal(imported.config.objective, "total");
  }
});
test("cancelled queued scoring never dispatches or fabricates request provenance", async () => {
  const abort = new AbortController(),
    starts = [],
    requests = [];
  let dispatched = 0;
  const scorer = new TogetherScorer("cancel-key", {
    concurrency: 1,
    signal: abort.signal,
    onRequestStart: (r) => starts.push(r),
    onRequest: (r) => requests.push(r),
    fetchImpl: async () => {
      dispatched++;
      abort.abort(Error("Stopped"));
      throw Error("Stopped");
    },
  });
  const results = await Promise.allSettled([
    scorer.score(config(), "first"),
    scorer.score(config(), "second"),
  ]);
  assert.ok(results.every((r) => r.status === "rejected"));
  assert.equal(dispatched, 1);
  assert.equal(starts.length, 1);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].logical_request_id, starts[0].logical_request_id);
  assert.equal(scorer.running, 0);
  assert.equal(scorer.queue.length, 0);
});
