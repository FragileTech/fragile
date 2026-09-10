import {
  rankingConfig,
  samplingPlan,
  validateVerdict,
  pairId,
  applyRankingEvent,
} from "./ranking-data.js";
import { rankingHeader } from "./ranking-store.js";
import { sourceRuns, endpointWeights } from "./comparison-metrics.js";
import { digest, gradingProfile } from "./grading.js";
import { MODEL_VERSION, PRIORS } from "./ranking-math.js";

const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export function validateRankingSession(s, validateSource) {
  if (
    !s ||
    s.version !== 1 ||
    typeof s.id !== "string" ||
    !/^[\w-]{1,100}$/.test(s.id) ||
    !Number.isFinite(s.created_at) ||
    !Array.isArray(s.candidates) ||
    !Array.isArray(s.groups) ||
    !Array.isArray(s.events) ||
    s.events.length > 100000
  )
    throw Error("Invalid ranking session");
  validateSource(s.source);
  rankingConfig(s.config);
  gradingProfile(s.profile);
  if (
    s.profile.overall !== s.config.overall ||
    !equal(s.profile.rubric, s.config.rubric) ||
    s.profile.reference !== s.config.reference ||
    s.model?.version !== MODEL_VERSION ||
    !equal(s.model.priors, PRIORS) ||
    s.model.draws !== 2000
  )
    throw Error("Unsupported ranking configuration or model");
  const ids = new Set(),
    candidates = new Map();
  for (const c of s.candidates) {
    if (
      typeof c.id !== "string" ||
      !/^[0-9a-f]{64}$/.test(c.id) ||
      ids.has(c.id) ||
      typeof c.prompt !== "string" ||
      typeof c.answer !== "string" ||
      !Array.isArray(c.occurrences) ||
      !c.occurrences.length
    )
      throw Error("Invalid ranking candidate");
    ids.add(c.id);
    candidates.set(JSON.stringify([c.prompt, c.answer]), c);
  }
  if (candidates.size !== s.candidates.length)
    throw Error("Duplicate ranking answer identities");
  const expectedGroups = [],
    occurrences = new Map(s.candidates.map((c) => [c.id, []]));
  for (const run of sourceRuns(s.source)) {
    if (
      (!run.standalone && run.status !== "completed") ||
      (s.config.trial !== "all" && run.trial !== Number(s.config.trial))
    )
      continue;
    for (const pool of ["archive", "retained"]) {
      const entries = [];
      for (const [id, weight] of endpointWeights(run, pool)) {
        const n = run.nodes[id];
        if (!n.tokens || ![1, 2].includes(n.status)) continue;
        const c = candidates.get(JSON.stringify([run.config.prompt, n.text]));
        if (!c) throw Error("Ranking source endpoint missing");
        entries.push({ id: c.id, weight });
        occurrences.get(c.id).push({
          key: `${run.id}/${id}`,
          run_id: run.id,
          trial: run.trial,
          method: run.method,
          pool,
          weight,
        });
      }
      if (entries.length)
        expectedGroups.push({
          id: `${run.trial}:${run.method}:${pool}`,
          trial: run.trial,
          method: run.method,
          pool,
          entries,
          total: entries.reduce((a, b) => a + b.weight, 0),
        });
    }
  }
  for (const c of s.candidates)
    if (!equal(c.occurrences, occurrences.get(c.id)))
      throw Error("Ranking occurrence weights do not match the source");
  if (!equal(expectedGroups, s.groups))
    throw Error("Ranking populations do not match the source");
  if (!equal(samplingPlan(s.candidates, s.groups, s.config), s.plan))
    throw Error("Ranking sampling plan was modified");
  const replay = rankingHeader(s),
    available = new Map(
      [
        ...s.plan.training_candidates,
        ...s.plan.audit_pairs,
        ...s.plan.validation,
      ].map((p) => [p.id, p]),
    );
  for (const e of s.events) {
    if (!Number.isFinite(e.time))
      throw Error("Invalid ranking event timestamp");
    const p = e.payload;
    if (e.type === "schedule") {
      if (!Array.isArray(p)) throw Error("Invalid pair schedule");
      for (const pair of p) {
        if (!equal(available.get(pair.id), pair))
          throw Error("Pair outside frozen sampling plan");
        if (pair.partition !== "training" && !replay.frozen_training)
          throw Error("Held-out evidence scheduled before training freeze");
        if (pair.partition === "training" && replay.frozen_training)
          throw Error("Training pair scheduled after freeze");
      }
    } else if (e.type === "result") {
      const match = replay.pairs.find(
        (pair) => p.key === `${pair.id}/0` || p.key === `${pair.id}/1`,
      );
      if (
        !match ||
        !["running", "received", "valid", "failed", "interrupted"].includes(
          p.result?.status,
        )
      )
        throw Error("Invalid pairwise result identity or status");
      if (match.partition === "training" && replay.frozen_training)
        throw Error("Training judgment changed after validation freeze");
      if (p.result.status === "valid") validateVerdict(p.result.verdict);
    } else if (e.type === "attempt") {
      if (
        replay.attempts.length >= s.config.budget ||
        p.path !== "chat/completions" ||
        ![0, 1].includes(p.orientation) ||
        !replay.pairs.some((pair) => pair.id === p.pair_id) ||
        !Number.isInteger(p.attempt) ||
        p.attempt < 0 ||
        p.attempt > 2
      )
        throw Error("Invalid or over-budget provider attempt");
    } else if (e.type === "freeze") {
      const keys = replay.pairs
        .filter((p) => p.partition === "training")
        .flatMap((p) => [`${p.id}/0`, `${p.id}/1`])
        .filter((key) => replay.results[key]?.status === "valid");
      if (
        !equal(p.keys, keys) ||
        p.at_seq !== replay.seq ||
        p.model_version !== MODEL_VERSION
      )
        throw Error("Invalid training freeze");
    } else if (e.type === "status") {
      if (
        ![
          "running",
          "completed",
          "budget_exhausted",
          "stopped",
          "failed",
          "needs_retry",
          "storage_error",
        ].includes(p)
      )
        throw Error("Invalid ranking execution status");
    } else if (e.type === "audit") validateAudit(p, replay);
    else if (e.type !== "request") throw Error("Unsupported ranking event");
    applyRankingEvent(replay, e);
  }
  for (const key of [
    "seq",
    "pairs",
    "results",
    "requests",
    "attempts",
    "frozen_training",
    "audits",
    "status",
  ])
    if (!equal(replay[key], s[key]))
      throw Error(`Ranking journal mismatch: ${key}`);
  return s;
}
function validateAudit(a, s) {
  if (
    !a ||
    typeof a.id !== "string" ||
    !["human", "model"].includes(a.kind) ||
    !Array.isArray(a.pairs) ||
    a.pairs.length > 20 ||
    !Array.isArray(a.requests) ||
    !Array.isArray(a.attempts) ||
    !Number.isSafeInteger(a.budget) ||
    a.budget < 2 ||
    a.attempts.length > a.budget
  )
    throw Error("Invalid independent audit");
  if (a.kind === "model") gradingProfile(a.profile);
  for (const p of a.pairs)
    if (p.id !== pairId(p.a, p.b) || !s.pairs.some((x) => equal(x, p)))
      throw Error("Audit pair outside primary evidence");
  const old = s.audits.find((x) => x.id === a.id);
  if (old) {
    for (const key of [
      "kind",
      "created_at",
      "profile",
      "endpoint",
      "pairs",
      "budget",
    ])
      if (!equal(old[key], a[key]))
        throw Error("Frozen independent audit configuration changed");
    for (const key of ["attempts", "requests"])
      if (!equal(old[key], a[key].slice(0, old[key].length)))
        throw Error("Independent audit request evidence changed");
    for (const [key, result] of Object.entries(old.results))
      if (result.status === "valid" && !equal(result, a.results[key]))
        throw Error("Valid independent audit judgment changed");
  }
  for (const [key, r] of Object.entries(a.results)) {
    if (!a.pairs.some((p) => key === `${p.id}/0` || key === `${p.id}/1`))
      throw Error("Unknown independent audit judgment");
    if (
      !["running", "received", "valid", "failed", "interrupted"].includes(
        r.status,
      )
    )
      throw Error("Invalid independent audit result status");
    if (r.status === "valid") validateVerdict(r.verdict);
    if (old?.results[key]?.status === "valid" && !equal(old.results[key], r))
      throw Error("Valid independent audit judgment changed");
  }
}
export async function verifyRankingIdentities(s) {
  if (s.source_digest !== (await digest(s.source)))
    throw Error("Ranking source snapshot identity mismatch");
  for (const c of s.candidates)
    if (c.id !== (await digest([c.prompt, c.answer])))
      throw Error("Ranking candidate identity mismatch");
  return s;
}
