import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  probabilities,
  objective,
  fitDavidson,
  predictPair,
  RANKING_CRITERIA,
  random,
} from "../../web/llm/ranking-math.js";
import {
  createRankingSession,
  rankingCandidates,
  samplingPlan,
  pairwiseBody,
  pairComplete,
  resultKey,
  applyRankingEvent,
} from "../../web/llm/ranking-data.js";
import { MemoryRankingStore } from "../../web/llm/ranking-store.js";
import {
  RankingController,
  createIndependentAudit,
  runIndependentAudit,
} from "../../web/llm/ranking-runner.js";
import {
  analyzeRanking,
  methodAudit,
  trainingObservations,
  validationDiagnostics,
} from "../../web/llm/ranking-statistics.js";
import {
  validateRankingSession,
  verifyRankingIdentities,
} from "../../web/llm/ranking-validation.js";
import {
  createReport,
  parseReport,
  exportReport,
  validateSource,
  processReport,
} from "../../web/llm/comparison-report.js";
import {
  rankingSource,
  rankingJudge,
  preparedJudge,
} from "./ranking-fixtures.mjs";
import { main } from "../../tools/llm-benchmark.mjs";
import { writeFile } from "node:fs/promises";
import { BenchmarkRunner } from "../../web/llm/benchmark.js";
import { manifest } from "../../web/llm/benchmark-data.js";
import { MemoryBenchmarkStore } from "../../web/llm/benchmark-store.js";
import { fakeOpenRouter } from "./fixtures.mjs";

test("Davidson probabilities respect ties, orientation reversal and analytic derivatives", () => {
  assert.deepEqual(probabilities(0, 0, 0, 0), [1 / 3, 1 / 3, 1 / 3]);
  const p = probabilities(1, -1, 0.3, 0.4),
    q = probabilities(-1, 1, 0.3, -0.4);
  assert.ok(Math.abs(p[0] - q[1]) < 1e-12);
  assert.ok(Math.abs(p[2] - q[2]) < 1e-12);
  const theta = new Float64Array([0.3, -0.3, 0.2, 0.4]),
    observations = [
      { i: 0, j: 1, y: 2, weight: 0.5 },
      { i: 1, j: 0, y: 0, weight: 0.5 },
    ],
    fit = objective(theta, observations, 2),
    eps = 1e-5;
  for (let i = 0; i < theta.length; i++) {
    const a = theta.slice(),
      b = theta.slice();
    a[i] += eps;
    b[i] -= eps;
    const fa = objective(a, observations, 2),
      fb = objective(b, observations, 2);
    assert.ok(
      Math.abs((fa.value - fb.value) / (2 * eps) - fit.gradient[i]) < 1e-7,
    );
    for (let j = 0; j < theta.length; j++)
      assert.ok(
        Math.abs(
          (fa.gradient[j] - fb.gradient[j]) / (2 * eps) - fit.hessian[i][j],
        ) < 1e-7,
      );
  }
});
test("ratings are order-invariant, centered and do not bridge disconnected evidence", () => {
  const obs = Array.from({ length: 30 }, () => [
    { i: 0, j: 1, y: 0, weight: 0.5 },
    { i: 1, j: 0, y: 1, weight: 0.5 },
  ]).flat();
  obs.push({ i: 2, j: 3, y: 2, weight: 0.5 });
  const a = fitDavidson(["a", "b", "c", "d", "unseen"], obs, { draws: 300 }),
    b = fitDavidson(["a", "b", "c", "d", "unseen"], [...obs].reverse(), {
      draws: 0,
    });
  assert.equal(a.converged, true);
  assert.ok(Math.abs(a.theta.slice(0, 5).reduce((a, b) => a + b, 0)) < 1e-8);
  a.theta.forEach((v, i) => assert.ok(Math.abs(v - b.theta[i]) < 1e-7));
  assert.ok(a.ratings[0].elo > a.ratings[1].elo);
  assert.equal(predictPair(a, "a", "c"), null);
  assert.equal(a.ratings[4].elo, null);
  assert.equal(a.components, 2);
  assert.ok(predictPair(a, "a", "b").win > 0.8);
});
test("known-strength simulations recover ordering, calibration and broad interval coverage", () => {
  const rng = random(42),
    ids = ["a", "b", "c", "d"],
    truth = [-1, -0.3, 0.3, 1];
  let covered = 0,
    total = 0,
    brier = 0;
  for (let run = 0; run < 16; run++) {
    const obs = [];
    for (let k = 0; k < 160; k++) {
      const i = Math.floor(rng() * 4),
        j = (i + 1 + Math.floor(rng() * 3)) % 4,
        p = probabilities(truth[i], truth[j], -0.4, 0.2),
        u = rng();
      obs.push({ i, j, y: u < p[0] ? 0 : u < p[0] + p[1] ? 1 : 2, weight: 1 });
    }
    const fit = fitDavidson(ids, obs, {
      draws: 300,
      sensitivity: false,
      seed: run,
    });
    assert.equal(fit.converged, true);
    for (let i = 0; i < 4; i++) {
      total++;
      const target = 1500 + (400 / Math.LN10) * truth[i];
      if (
        fit.ratings[i].interval[0] <= target &&
        fit.ratings[i].interval[1] >= target
      )
        covered++;
    }
    const pred = predictPair(fit, "d", "a"),
      target = probabilities(1, -1, -0.4, 0);
    brier += (pred.win - target[0]) ** 2;
  }
  assert.ok(covered / total >= 0.8, `coverage ${covered}/${total}`);
  assert.ok(brier / 16 < 0.025);
});
test("sampling is reproducible, representative and keeps holdouts outside training", async () => {
  const source = rankingSource(100),
    s = await createRankingSession(source, { budget: 240 }, preparedJudge),
    again = await createRankingSession(source, { budget: 240 }, preparedJudge);
  assert.deepEqual(s.plan, again.plan);
  assert.ok(s.plan.cohort.length < 100);
  assert.equal(s.plan.allocation.training, 96);
  const training = new Set(s.plan.training_candidates.map((p) => p.id));
  for (const p of [...s.plan.validation, ...s.plan.audit_pairs])
    assert.equal(training.has(p.id), false);
  assert.ok(s.plan.initial.length >= s.plan.cohort.length - 1);
  assert.equal(s.candidates.length, 100);
  validateRankingSession(s, validateSource);
  await verifyRankingIdentities(s);
});
test("a minimal budget prioritizes connected ranking over unavailable validation", async () => {
  const session = await createRankingSession(
    rankingSource(2),
    { budget: 2 },
    preparedJudge,
  );
  assert.equal(session.plan.initial.length, 1);
  assert.equal(session.plan.validation.length, 0);
  const fake = rankingJudge();
  await new RankingController(
    "secret",
    session,
    new MemoryRankingStore(session),
    { fetchImpl: fake.fetch },
  ).run();
  assert.equal(fake.calls.length, 2);
  assert.equal(session.status, "completed");
  const data = analyzeRanking(session, { draws: 100 });
  assert.equal(data.fits.overall.components, 1);
  assert.equal(data.validation[4].log_loss, null);
});
test("pairwise runner blinds methods, freezes training, persists both orders and preserves valid results", async () => {
  const s = await createRankingSession(
      rankingSource(),
      { budget: 80 },
      preparedJudge,
    ),
    fake = rankingJudge({ delay: 1 }),
    store = new MemoryRankingStore(s);
  await new RankingController("secret", s, store, {
    fetchImpl: fake.fetch,
  }).run();
  assert.equal(s.status, "completed");
  assert.ok(fake.peak <= 2);
  assert.equal(fake.calls.length, s.attempts.length);
  assert.ok(s.attempts.length <= 80);
  assert.ok(s.frozen_training);
  for (const body of fake.calls) {
    assert.equal(body.temperature, 0);
    assert.equal(body.max_tokens, 2048);
    assert.equal(body.logprobs, undefined);
    const candidate = JSON.parse(body.messages[1].content);
    assert.deepEqual(Object.keys(candidate), [
      "prompt",
      "candidate_A",
      "candidate_B",
      "rubric",
      "reference",
    ]);
  }
  assert.ok(s.pairs.every((p) => pairComplete(s, p)));
  const calls = fake.calls.length;
  await new RankingController("secret", s, store, {
    fetchImpl: fake.fetch,
  }).run({ retry: true });
  assert.equal(fake.calls.length, calls);
  const obs = trainingObservations(s, "overall");
  assert.equal(
    obs.reduce((s, o) => s + o.weight, 0),
    s.pairs.filter((p) => p.partition === "training").length,
  );
  assert.deepEqual(await store.read(), s);
  validateRankingSession(s, validateSource);
  const data = analyzeRanking(s, { draws: 200 });
  assert.ok(data.validation[0].held_out);
  assert.ok(data.ratings.some((r) => r.elo !== null));
  const changed = structuredClone(s),
    holdout = changed.plan.validation[0];
  if (holdout) {
    for (const o of [0, 1])
      changed.results[resultKey(holdout, o)].verdict.overall.verdict = "tie";
    assert.deepEqual(trainingObservations(changed, "overall"), obs);
  }
});
test("concurrent transport retries cannot exceed the persisted POST cap", async () => {
  const s = await createRankingSession(
      rankingSource(4),
      { budget: 4 },
      preparedJudge,
    ),
    fake = rankingJudge({ failures: 100 }),
    store = new MemoryRankingStore(s);
  await new RankingController("secret", s, store, {
    fetchImpl: fake.fetch,
    sleep: async () => {},
  }).run();
  assert.equal(fake.calls.length, 4);
  assert.equal(s.attempts.length, 4);
  assert.equal(s.status, "budget_exhausted");
  validateRankingSession(s, validateSource);
});
test("invalid output, explicit retries and storage failure retain partial evidence", async () => {
  const s = await createRankingSession(
      rankingSource(5),
      { budget: 100 },
      preparedJudge,
    ),
    bad = rankingJudge({ invalid: true }),
    store = new MemoryRankingStore(s);
  await new RankingController("secret", s, store, {
    fetchImpl: bad.fetch,
  }).run();
  assert.equal(s.status, "needs_retry");
  assert.equal(s.frozen_training, null);
  const good = rankingJudge();
  await new RankingController("secret", s, store, {
    fetchImpl: good.fetch,
  }).run({ retry: true });
  assert.equal(s.status, "completed");
  assert.ok(
    s.events.some(
      (e) => e.type === "result" && e.payload.result.status === "failed",
    ),
  );
  const broken = await createRankingSession(
      rankingSource(4),
      { budget: 40 },
      preparedJudge,
    ),
    base = new MemoryRankingStore(broken),
    fake = rankingJudge();
  const failing = {
    append: async (e) => {
      if (e.type === "result" && e.payload.result.status === "received")
        throw Error("quota");
      await base.append(e);
    },
  };
  await assert.rejects(
    new RankingController("secret", broken, failing, {
      fetchImpl: fake.fetch,
    }).run(),
    /quota/,
  );
  assert.ok(Object.values(broken.results).some((r) => r.raw));
  assert.ok(fake.calls.length <= 2);
  assert.ok(base.events.length < broken.events.length);
});
test("cancellation can resume without repeating completed judgments", async () => {
  const s = await createRankingSession(
      rankingSource(8),
      { budget: 80 },
      preparedJudge,
    ),
    store = new MemoryRankingStore(s),
    fake = rankingJudge({ delay: 2 });
  let controller;
  controller = new RankingController("secret", s, store, {
    fetchImpl: fake.fetch,
    onStatus: () => {
      if (
        Object.values(s.results).filter((r) => r.status === "valid").length ===
        2
      )
        controller.stop();
    },
  });
  await controller.run();
  assert.equal(s.status, "stopped");
  const valid = Object.fromEntries(
    Object.entries(s.results).filter(([, r]) => r.status === "valid"),
  );
  await new RankingController("secret", s, store, {
    fetchImpl: fake.fetch,
  }).run({ retry: true });
  for (const [key, r] of Object.entries(valid))
    assert.deepEqual(s.results[key], r);
  assert.equal(s.status, "completed");
});
test("independent model audits are separate and repeat neither valid primary nor audit judgments", async () => {
  const s = await createRankingSession(
      rankingSource(6),
      { budget: 80 },
      preparedJudge,
    ),
    store = new MemoryRankingStore(s),
    fake = rankingJudge(),
    primary = new RankingController("secret", s, store, {
      fetchImpl: fake.fetch,
    });
  await primary.run();
  const prior = structuredClone(s.results),
    audit = createIndependentAudit(s, {
      profile: { ...s.profile, model: "judge/independent" },
      budget: 80,
    });
  await primary.emit("audit", audit);
  await runIndependentAudit(primary, audit, "secret", {
    fetchImpl: fake.fetch,
  });
  assert.deepEqual(s.results, prior);
  const finished = s.audits[0],
    count = fake.calls.length;
  await runIndependentAudit(primary, finished, "secret", {
    fetchImpl: fake.fetch,
  });
  assert.equal(fake.calls.length, count);
  assert.equal(s.audits[0].status, "completed");
  validateRankingSession(s, validateSource);
  const corrupt = structuredClone(s),
    changed = structuredClone(s.audits[0]);
  delete changed.results[Object.keys(changed.results)[0]];
  applyRankingEvent(corrupt, {
    seq: corrupt.seq + 1,
    time: Date.now(),
    type: "audit",
    payload: changed,
  });
  assert.throws(
    () => validateRankingSession(corrupt, validateSource),
    /Valid independent audit judgment changed/,
  );
});
test("v1/v2 reports round trip, reject corrupt ranking evidence and process offline", async () => {
  const source = rankingSource(5),
    s = await createRankingSession(source, { budget: 40 }, preparedJudge),
    fake = rankingJudge();
  await new RankingController("secret", s, new MemoryRankingStore(s), {
    fetchImpl: fake.fetch,
  }).run();
  const report = createReport(source);
  report.rankings.push(s);
  report.selected_ranking = s.id;
  report.evaluation_mode = "pairwise";
  const text = exportReport(report),
    parsed = parseReport(text);
  assert.equal(parsed.version, 2);
  assert.ok(!text.includes('"secret"'));
  assert.deepEqual(processReport(parsed), processReport(report));
  const old = createReport(source);
  old.version = 1;
  delete old.rankings;
  assert.deepEqual(parseReport(JSON.stringify(old)).rankings, []);
  const bad = structuredClone(report);
  bad.rankings[0].plan.cohort.pop();
  assert.throws(() => parseReport(JSON.stringify(bad)), /sampling plan/);
  const badWeight = structuredClone(report);
  badWeight.rankings[0].candidates[0].occurrences[0].weight += 1;
  assert.throws(
    () => parseReport(JSON.stringify(badWeight)),
    /occurrence weights/,
  );
  const dir = await mkdtemp(join(tmpdir(), "ranking-cli-"));
  try {
    await writeFile(join(dir, "r.fgllmcompare"), text);
    await main(
      [
        "process",
        "--input",
        join(dir, "r.fgllmcompare"),
        "--output",
        join(dir, "out"),
      ],
      { env: {} },
    );
    const rows = (
      await readFile(join(dir, "out/ranking_ratings.jsonl"), "utf8")
    )
      .trim()
      .split("\n")
      .map(JSON.parse);
    assert.deepEqual(rows, processReport(report).ranking_ratings);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
for (const algorithm of ["wave", "graph"])
  test(`${algorithm}: exhaustive randomized method audits match full occurrence populations across trials`, async () => {
    const sourceConfig = rankingSource(2).record.config,
      benchmark = new MemoryBenchmarkStore(
        manifest({
          comparison: "both",
          repetitions: 2,
          config: {
            ...sourceConfig,
            algorithm,
            walkers: 2,
            chunk_tokens: 2,
            sequence_tokens: 4,
            iterations: 2,
            max_walkers: 8,
          },
        }),
      );
    const generator = fakeOpenRouter();
    await new BenchmarkRunner(benchmark, "fixture", {
      fetchImpl: async (url, options = {}) => {
        const response = await generator.fetch(url, options),
          body = options.body ? JSON.parse(options.body) : null;
        if (
          url.endsWith("chat/completions") &&
          body?.messages?.[0]?.content === sourceConfig.prompt
        ) {
          const data = await response.json();
          data.choices[0].finish_reason = "stop";
          return new Response(JSON.stringify(data), { status: 200 });
        }
        return response;
      },
    }).run();
    const source = {
        kind: "benchmark",
        manifest: benchmark.manifest,
        events: benchmark.events,
      },
      s = await createRankingSession(source, { budget: 2000 }, preparedJudge),
      fake = rankingJudge();
    const store = new MemoryRankingStore(s);
    await new RankingController("secret", s, store, {
      fetchImpl: fake.fetch,
    }).run();
    validateRankingSession(s, validateSource);
    const candidates = new Map(s.candidates.map((c) => [c.id, c])),
      rows = methodAudit(s);
    for (const r of rows) {
      const expectations = [];
      for (const stratum of s.plan.strata.filter(
        (x) => x.baseline === r.baseline && x.pool === r.pool,
      )) {
        assert.equal(stratum.samples.length, stratum.size);
        assert.equal(
          new Set(stratum.samples.map((x) => x.index)).size,
          stratum.size,
        );
        const f = s.groups.find(
            (g) =>
              g.trial === stratum.trial &&
              g.method === "fractal" &&
              g.pool === r.pool,
          ),
          b = s.groups.find(
            (g) =>
              g.trial === stratum.trial &&
              g.method === r.baseline &&
              g.pool === r.pool,
          );
        let total = 0;
        for (const a of f.entries)
          for (const other of b.entries) {
            const x = candidates.get(a.id).answer,
              y = candidates.get(other.id).answer;
            total += a.weight * other.weight * (x === y ? 0.5 : x > y ? 1 : 0);
          }
        expectations.push(total / (f.total * b.total));
      }
      const expected =
        expectations.reduce((a, b) => a + b, 0) / expectations.length;
      assert.ok(Math.abs(r.preference_share - expected) < 1e-10);
      assert.ok(Math.abs(r.interval[0] - expected) < 1e-10);
      assert.equal(r.coverage, 1);
      assert.equal(r.trials, 2);
      assert.ok(r.generalization_interval);
    }
    assert.equal(rows.length, 30);
    const trial = await rankingCandidates(source, "1");
    assert.ok(trial.groups.every((g) => g.trial === 1));
    // Remove one presentation to verify abstention/failure is not treated as a loss or tie.
    const missing = structuredClone(s),
      p = missing.plan.audit_pairs.find((p) => p.a !== p.b);
    delete missing.results[resultKey(p, 0)];
    const partial = methodAudit(missing);
    assert.ok(partial.some((r) => r.coverage < 1));
    assert.ok(partial.some((r) => r.missing_bounds[1] > r.missing_bounds[0]));
  });
test("bounded audit intervals control false positives on sampled finite null populations", () => {
  let falsePositives = 0;
  const rng = random(771);
  for (let repetition = 0; repetition < 500; repetition++) {
    const population = Array.from({ length: 100 }, (_, i) => i),
      sample = [];
    for (let n = 0; n < 25; n++)
      sample.push(
        population.splice(Math.floor(rng() * population.length), 1)[0],
      );
    const pairs = sample.map((i) => ({ id: `a:${i}`, a: "a", b: String(i) })),
      results = {};
    for (let j = 0; j < pairs.length; j++)
      for (const o of [0, 1])
        results[resultKey(pairs[j], o)] = {
          status: "valid",
          verdict: Object.fromEntries(
            RANKING_CRITERIA.map((k) => [
              k,
              {
                verdict: sample[j] < 50 !== Boolean(o) ? "A" : "B",
                explanation: "Null finite population",
              },
            ]),
          ),
        };
    const s = {
      config: { seed: 7 },
      status: "completed",
      results,
      plan: {
        audit_pairs: pairs,
        strata: [
          {
            trial: 0,
            pool: "archive",
            baseline: "independent_population",
            size: 100,
            samples: pairs.map((p, index) => ({
              pair_id: p.id,
              fractal: "a",
              index,
            })),
          },
        ],
      },
    };
    const r = methodAudit(s)[0];
    if (r.interval[0] > 0.5 || r.interval[1] < 0.5) falsePositives++;
  }
  assert.ok(falsePositives / 500 <= 0.05);
});
test("explicit ties, abstentions and cycles are distinguished by diagnostics", async () => {
  const s = await createRankingSession(
      rankingSource(3),
      { budget: 80 },
      preparedJudge,
    ),
    fake = rankingJudge();
  await new RankingController("secret", s, new MemoryRankingStore(s), {
    fetchImpl: fake.fetch,
  }).run();
  for (const p of s.pairs)
    for (const o of [0, 1])
      s.results[resultKey(p, o)].verdict.clarity.verdict = "cannot_assess";
  const data = analyzeRanking(s, { draws: 100 });
  assert.equal(data.fits.clarity.unobserved, s.plan.cohort.length);
  assert.equal(data.fits.clarity.evidence_weight, 0);
  for (const p of s.pairs)
    for (const o of [0, 1])
      s.results[resultKey(p, o)].verdict.relevance.verdict = "tie";
  const tied = analyzeRanking(s, { draws: 100 });
  assert.ok(
    tied.fits.relevance.ratings.every(
      (r) => r.elo === null || Math.abs(r.elo - 1500) < 1e-8,
    ),
  );
  const ids = s.plan.cohort;
  const edges = new Set([
    `${ids[0]}:${ids[1]}`,
    `${ids[1]}:${ids[2]}`,
    `${ids[2]}:${ids[0]}`,
  ]);
  for (const p of s.pairs)
    for (const o of [0, 1])
      s.results[resultKey(p, o)].verdict.overall.verdict =
        edges.has(`${p.a}:${p.b}`) !== Boolean(o) ? "A" : "B";
  const cyclic = analyzeRanking(s, { draws: 100 }).validation.find(
    (r) => r.criterion === "overall",
  );
  assert.equal(cyclic.cycles, 1);
  assert.equal(cyclic.directed_triangles, 1);
});
test("position bias is estimated separately from equal answer quality", () => {
  const observations = [],
    p = probabilities(0, 0, -0.2, 1.2);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      if (i !== j)
        for (let y = 0; y < 3; y++)
          observations.push({ i, j, y, weight: 50 * p[y] });
  const fit = fitDavidson(["a", "b", "c", "d"], observations, {
    draws: 100,
    sensitivity: false,
  });
  assert.ok(fit.position_effect > 1.1 && fit.position_effect < 1.2);
  assert.ok(fit.ratings.every((r) => Math.abs(r.elo - 1500) < 1e-7));
});
