import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { main } from "../../tools/llm-benchmark.mjs";
import {
  createReport,
  exportReport,
  parseReport,
  processReport,
} from "../../web/llm/comparison-report.js";
import test from "node:test";
import assert from "node:assert/strict";
import { game, gameScoringFetch } from "./game-fixtures.mjs";
import { completion, fakeOpenRouter } from "./fixtures.mjs";
import { configuration, objective, bestNode } from "../../web/llm/config.js";
import { generateGame, gameQuestion } from "../../web/llm/games.js";
import {
  TogetherScorer,
  validateGameScore,
  scoringPrefix,
} from "../../web/llm/scoring.js";
import { TokenEnvironment } from "../../web/llm/environment.js";
import { Recording, importRecording } from "../../web/llm/recording.js";
import { NativeLlm } from "../../web/llm/native.js";
import { parseCompletion } from "../../web/llm/openrouter.js";
import { pairManifests, runGamePair } from "../../web/llm/game-benchmark.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import {
  collectRuns,
  exportBenchmark,
  parseBenchmark,
  processBenchmark,
} from "../../web/llm/benchmark-data.js";
import {
  computeComparison,
  endpointWeights,
  traceComparison,
} from "../../web/llm/comparison-metrics.js";

const config = (overrides = {}) =>
  configuration({
    objective: "xent_game",
    game,
    walkers: 2,
    chunk_tokens: 2,
    sequence_tokens: 6,
    iterations: 8,
    max_walkers: 8,
    concurrency: 2,
    ...overrides,
  });
const setup = () => {
  const f = gameScoringFetch();
  return { ...f, scorer: new TogetherScorer("secret", { fetchImpl: f.fetch }) };
};

test("game creation freezes validated content and records separate creation evidence", async () => {
  let body;
  const fetchImpl = async (_url, options) => {
    body = JSON.parse(options.body);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        model: "test",
        choices: [
          {
            finish_reason: "stop",
            message: {
              content: JSON.stringify({
                title: "secret",
                background: game.background,
                target: game.target,
              }),
            },
          },
        ],
        usage: { completion_tokens: 30 },
      }),
    };
  };
  const spec = await generateGame("secret", "A detective game", config(), {
    fetchImpl,
  });
  assert.equal(spec.title, "[redacted]");
  assert.equal(spec.target, game.target);
  assert.equal(spec.creation.requests[0].phase, "game_creation");
  assert.equal(body.response_format.json_schema.strict, true);
  assert.deepEqual(body.reasoning, { enabled: false });
  const c = config({ game: spec });
  spec.target = "changed";
  assert.equal(c.game.target, game.target);
  assert.ok(c.prompt.includes(game.target));
  assert.throws(() => config({ game: null }), /Generate or import/);
  for (const value of ["not json", JSON.stringify({ title: "bad" })])
    await assert.rejects(
      generateGame("key", "brief", config(), {
        fetchImpl: async () => ({
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ finish_reason: "stop", message: { content: value } }],
          }),
        }),
      }),
      /invalid/,
    );
  await assert.rejects(
    generateGame("key", "brief", config(), {
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ finish_reason: "length" }] }),
      }),
    }),
    /did not finish/,
  );
});

test("fixed Unicode target, opposite signs, one shared baseline, and exact context cache", async () => {
  const { scorer, calls } = setup(),
    c = config();
  const metadata = await scorer.prepareGame(c);
  const x = await scorer.scoreGame(c, "ab");
  assert.equal(x.tokens, [...game.target].length);
  const n = { tokens: 2, text: "ab", game_score: x };
  assert.equal(objective(n, c), 3);
  assert.equal(objective(n, config({ game_mode: "surprising" })), -3);
  assert.equal(objective({ tokens: 0 }, c), 0);
  assert.deepEqual(await scorer.scoreGame(c, "ab"), x);
  assert.deepEqual(
    await scorer.scoreGame(config({ game_mode: "surprising" }), "ab"),
    x,
  );
  const baselinePrompt = scoringPrefix(gameQuestion(game, "")) + game.target;
  assert.equal(calls.filter((b) => b.prompt === baselinePrompt).length, 1);
  assert.equal(
    calls.filter(
      (b) => b.prompt === scoringPrefix(gameQuestion(game, "ab")) + game.target,
    ).length,
    1,
  );
  assert.equal(await scorer.scoreGame(c, ""), null);
  assert.equal(
    objective({ tokens: 1, game_score: metadata.baseline_score }, c),
    0,
  );
  validateGameScore(n, c, metadata);
  for (const mutate of [
    (s) => s.conditional[0].id++,
    (s) => s.baseline_logp++,
    (s) => (s.context = "wrong"),
    (s) => (s.target = "changed"),
  ]) {
    const bad = structuredClone(n);
    mutate(bad.game_score);
    assert.throws(() => validateGameScore(bad, c, metadata));
  }
  const resumed = setup();
  await resumed.scorer.prepareGame(c, metadata);
  assert.equal(
    resumed.calls.filter((b) => b.prompt === baselinePrompt).length,
    0,
  );
});

for (const algorithm of ["wave", "graph"])
  for (const game_mode of ["unsurprising", "surprising"])
    test(`${algorithm}/${game_mode}: native scores, preserved prefixes, and offline round-trip`, async () => {
      const c = config({ algorithm, game_mode }),
        { scorer } = setup();
      const record = new Recording(c);
      record.metadata({ dimensions: 2, scoring: await scorer.prepareGame(c) });
      const api = {
        async generate(_c, prefix, count) {
          return parseCompletion(
            completion("a".repeat(count), -0.3, prefix ? "stop" : "length"),
            count,
          );
        },
        async embed(_m, texts) {
          return texts.map((t) => [1, t.length]);
        },
      };
      const env = new TokenEnvironment(c, api, record, scorer);
      const native = await NativeLlm.create({ ...c, dimensions: 2 }, (r) =>
        env.transition(r),
      );
      try {
        while (!record.data.run.stop_reason) env.commit(await native.advance());
        const saved = importRecording(record.export());
        for (const n of saved.nodes.slice(1)) {
          validateGameScore(n, c, saved.metadata.scoring);
          assert.equal(
            n.reward,
            objective(n, c) - objective(saved.nodes[n.parent], c),
          );
        }
        for (const s of saved.snapshots)
          for (const w of s.walkers)
            if (Number.isInteger(w.node) && saved.nodes[w.node])
              assert.ok(
                Math.abs(w.score - objective(saved.nodes[w.node], c)) < 1e-5,
              );
        const winner = bestNode(saved.nodes, c);
        assert.equal(winner.tokens, 4);
        assert.equal(winner.status, 1);
        const data = computeComparison({ kind: "recording", record: saved });
        assert.equal(data.filters.metric, "reward");
        assert.equal(data.game_leaders[0].score, objective(winner, c));
        const measured = data.histories
          .map((h) => h.best_reward)
          .filter((value) => value !== null);
        assert.ok(measured.every((value, i) => !i || value >= measured[i - 1]));
        assert.equal(
          new Set(data.rows.map((r) => r.text)).size,
          data.rows.length,
        );
        assert.ok(
          endpointWeights({ ...saved, method: "fractal" }, "archive").has(
            winner.id,
          ),
        );
        const trace = traceComparison(
          [{ ...saved, id: "r", method: "fractal" }],
          [`r/${winner.id}`],
        );
        assert.equal(trace.traces[0].tokens[0].reward, null);
        assert.equal(
          trace.traces[0].tokens.at(-1).reward,
          objective(winner, c),
        );
        const bad = structuredClone(saved);
        bad.nodes[1].game_score.baseline[0].logprob -= 1;
        assert.throws(() => importRecording(JSON.stringify(bad)));
      } finally {
        native.close();
      }
    });

test("score failure retains paid generation without committing an unscored prefix", async () => {
  const c = config(),
    record = new Recording(c),
    { scorer } = setup();
  record.metadata({ dimensions: 2, scoring: await scorer.prepareGame(c) });
  scorer.scoreGame = async () => {
    throw Error("scoring unavailable");
  };
  const env = new TokenEnvironment(
    c,
    {
      generate: async () => parseCompletion(completion("ab"), 2),
      embed: async () => {
        throw Error("must not embed");
      },
    },
    record,
    scorer,
  );
  await assert.rejects(
    env.transition([{ source: 0, action: 1, duration: 2 }]),
    /scoring unavailable/,
  );
  assert.equal(record.data.attempts.length, 1);
  assert.equal(record.data.run.generated_tokens, 2);
  assert.equal(record.data.nodes.length, 1);
  importRecording(record.export());
});

test("paired benchmarks share baseline and route, match generation budgets, resume and process offline", async () => {
  const headers = pairManifests({
    config: config(),
    comparison: "tokens",
    repetitions: 1,
  });
  const stores = headers.map((h) => new MemoryBenchmarkStore(h));
  const fake = fakeOpenRouter(),
    scoring = gameScoringFetch();
  const options = {
    togetherKey: "together-secret",
    fetchImpl: (url, init) =>
      String(url).includes("api.together.ai")
        ? scoring.fetch(url, init)
        : fake.fetch(url, init),
  };
  await runGamePair(stores, "router-secret", options);
  const metadata = stores.map(
    (s) => collectRuns(s.manifest, s.events).metadata,
  );
  assert.deepEqual(metadata[0].scoring, metadata[1].scoring);
  assert.deepEqual(metadata[0].provider, metadata[1].provider);
  const baselinePrompt = scoringPrefix(gameQuestion(game, "")) + game.target;
  assert.equal(
    scoring.calls.filter((b) => b.prompt === baselinePrompt).length,
    1,
  );
  for (const store of stores) {
    const text = exportBenchmark(store.manifest, store.events);
    assert.ok(
      !text.includes("router-secret") && !text.includes("together-secret"),
    );
    const parsed = parseBenchmark(text),
      { runs } = collectRuns(parsed.manifest, parsed.events);
    assert.equal(runs.size, 3);
    const all = [...runs.values()];
    assert.equal(
      all.find((r) => r.method === "fractal").generated_tokens,
      all.find((r) => r.method === "independent_tokens").generated_tokens,
    );
    assert.ok(all.every((r) => r.status === "completed"));
    const source = {
      kind: "benchmark",
      manifest: parsed.manifest,
      events: parsed.events,
    };
    const report = parseReport(exportReport(createReport(source)));
    assert.equal(report.view.metric, "reward");
    assert.equal(report.view.status, "eos");
    assert.equal(
      report.source.manifest.settings.config.game.target,
      game.target,
    );
    assert.ok(processReport(report).metrics.length > 0);
    const processed = processBenchmark(parsed.manifest, parsed.events);
    assert.ok(
      processed.nodes.some(
        (n) => n.target_nll != null && n.objective_score != null,
      ),
    );
  }
  const counts = stores.map((s) => s.events.length),
    requests = fake.requests.length;
  await runGamePair(stores, "router-secret", {
    ...options,
    retryIncomplete: true,
  });
  assert.deepEqual(
    stores.map((s) => s.events.length),
    counts,
  );
  assert.equal(fake.requests.length, requests);
});

test("stopped pair resumes unfinished methods without repeating completed work", async () => {
  const stores = pairManifests({
    config: config(),
    comparison: "tokens",
    repetitions: 1,
  }).map((h) => new MemoryBenchmarkStore(h));
  const fake = fakeOpenRouter(),
    scoring = gameScoringFetch();
  const options = {
    togetherKey: "together-secret",
    fetchImpl: (url, init) =>
      String(url).includes("api.together.ai")
        ? scoring.fetch(url, init)
        : fake.fetch(url, init),
  };
  const abort = new AbortController();
  await assert.rejects(
    runGamePair(stores, "router-secret", {
      ...options,
      signal: abort.signal,
      onCommitted: (event) => {
        if (event.type === "session_end") abort.abort(Error("Stopped"));
      },
    }),
    /Stopped/,
  );
  const firstEvents = stores[0].events.length;
  assert.equal(stores[1].events.length, 0);
  await runGamePair(stores, "router-secret", {
    ...options,
    retryIncomplete: true,
  });
  assert.equal(stores[0].events.length, firstEvents);
  assert.equal(collectRuns(stores[1].manifest, stores[1].events).runs.size, 3);
});

test("exported game configuration runs and processes through the existing CLI", async () => {
  const dir = await mkdtemp(join(tmpdir(), "fgllm-game-cli-"));
  try {
    await writeFile(
      join(dir, "game.json"),
      JSON.stringify({
        config: config(),
        comparison: "tokens",
        repetitions: 1,
      }),
    );
    const fake = fakeOpenRouter(),
      scoring = gameScoringFetch();
    const options = {
      env: {
        OPENROUTER_API_KEY: "router-secret",
        TOGETHER_API_KEY: "together-secret",
        INIT_CWD: dir,
      },
      runnerOptions: {
        onStatus: () => {},
        fetchImpl: (url, init) =>
          String(url).includes("api.together.ai")
            ? scoring.fetch(url, init)
            : fake.fetch(url, init),
      },
    };
    await main(["run", "--config", "game.json", "--output", "run"], options);
    const calls = fake.requests.length;
    await main(["resume", "--output", "run"], options);
    assert.equal(fake.requests.length, calls);
    await main(
      ["export", "--output", "run", "--file", "game.fgllmbench"],
      options,
    );
    await main(
      ["process", "--input", "game.fgllmbench", "--output", "processed"],
      { env: { INIT_CWD: dir } },
    );
    const nodes = (
      await readFile(join(dir, "processed", "nodes.jsonl"), "utf8")
    )
      .trim()
      .split("\n")
      .map(JSON.parse);
    assert.ok(nodes.some((n) => n.target_nll === 1 && n.objective_score === 3));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("paired pause preserves the boundary until continued", async () => {
  const stores = pairManifests({
    config: config(),
    comparison: "tokens",
    repetitions: 1,
  }).map((h) => new MemoryBenchmarkStore(h));
  const fake = fakeOpenRouter(),
    scoring = gameScoringFetch();
  let runner,
    paused = false,
    resume,
    pauseError;
  await runGamePair(stores, "router-secret", {
    togetherKey: "together-secret",
    fetchImpl: (url, init) =>
      String(url).includes("api.together.ai")
        ? scoring.fetch(url, init)
        : fake.fetch(url, init),
    onRunner: (r) => {
      runner = r;
    },
    onCommitted: (event) => {
      if (event.type !== "run_start" || paused) return;
      paused = true;
      runner.pause();
      const count = fake.requests.length;
      resume = new Promise((resolve) =>
        setTimeout(() => {
          try {
            assert.equal(fake.requests.length, count);
          } catch (error) {
            pauseError = error;
          }
          runner.continue();
          resolve();
        }, 25),
      );
    },
  });
  await resume;
  assert.equal(paused, true);
  if (pauseError) throw pauseError;
});
