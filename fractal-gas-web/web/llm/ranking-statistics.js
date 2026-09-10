import {
  RANKING_CRITERIA,
  fitDavidson,
  predictPair,
  probabilities,
  random,
  quantile,
} from "./ranking-math.js";
import { resultKey, pairComplete } from "./ranking-data.js";

export function trainingObservations(session, criterion) {
  const ids = session.plan.cohort,
    index = new Map(ids.map((id, i) => [id, i])),
    frozen = session.frozen_training
      ? new Set(session.frozen_training.keys)
      : null,
    observations = [];
  for (const pair of session.pairs.filter((p) => p.partition === "training"))
    for (const orientation of [0, 1]) {
      const key = resultKey(pair, orientation),
        result = session.results[key];
      if (frozen && !frozen.has(key)) continue;
      const verdict =
        result?.status === "valid" ? result.verdict[criterion].verdict : null;
      if (
        !["A", "B", "tie"].includes(verdict) ||
        !index.has(pair.a) ||
        !index.has(pair.b)
      )
        continue;
      observations.push({
        i: index.get(orientation ? pair.b : pair.a),
        j: index.get(orientation ? pair.a : pair.b),
        y: ["A", "B", "tie"].indexOf(verdict),
        weight: 0.5,
        pair_id: pair.id,
        orientation,
      });
    }
  return observations;
}
export function fitRanking(session, { draws = 2000, sensitivity = true } = {}) {
  return Object.fromEntries(
    RANKING_CRITERIA.map((k) => [
      k,
      fitDavidson(session.plan.cohort, trainingObservations(session, k), {
        draws,
        sensitivity,
        seed: `${session.config.seed}:${k}`,
        priors: session.model.priors,
      }),
    ]),
  );
}
export function canonicalOutcome(session, pair, orientation, criterion) {
  if (pair.a === pair.b) return "tie"; // Identical prompt/answer content is an exact equality, not another judge vote.
  const result = session.results[resultKey(pair, orientation)];
  if (result?.status !== "valid") return null;
  const verdict = result.verdict[criterion].verdict;
  if (verdict === "cannot_assess") return null;
  if (verdict === "tie") return "tie";
  return (verdict === "A") !== Boolean(orientation) ? "win" : "loss";
}
export function methodAudit(session) {
  const pairs = new Map(session.plan.audit_pairs.map((p) => [p.id, p])),
    rows = [];
  const contrasts = [
      ...new Set(session.plan.strata.map((s) => `${s.pool}:${s.baseline}`)),
    ],
    family = Math.max(1, contrasts.length * RANKING_CRITERIA.length);
  for (const contrast of contrasts)
    for (const criterion of RANKING_CRITERIA) {
      const strata = session.plan.strata.filter(
          (s) => `${s.pool}:${s.baseline}` === contrast,
        ),
        trials = [];
      for (const s of strata) {
        let win = 0,
          loss = 0,
          tie = 0,
          known = 0,
          score = 0;
        for (const sample of s.samples) {
          const p = pairs.get(sample.pair_id);
          for (const orientation of [0, 1]) {
            let outcome = canonicalOutcome(session, p, orientation, criterion);
            if (sample.fractal !== p.a && outcome && outcome !== "tie")
              outcome = outcome === "win" ? "loss" : "win";
            if (outcome) {
              known += 0.5;
              if (outcome === "win") {
                win += 0.5;
                score += 0.5;
              } else if (outcome === "loss") loss += 0.5;
              else {
                tie += 0.5;
                score += 0.25;
              }
            }
          }
        }
        const n = s.samples.length;
        trials.push({
          trial: s.trial,
          n,
          population_pairs: s.size,
          win,
          loss,
          tie,
          known,
          coverage: n ? known / n : null,
          preference_share: known ? score / known : null,
          lower: n ? score / n : 0,
          upper: n ? (score + n - known) / n : 1,
        });
      }
      const count = trials.length,
        covered = trials.every((t) => t.n > 0),
        allKnown = trials.every((t) => t.coverage === 1),
        lower = trials.reduce((s, t) => s + t.lower, 0) / count,
        upper = trials.reduce((s, t) => s + t.upper, 0) / count;
      // Hoeffding also holds for uniform sampling without replacement from a fixed
      // bounded population. Exhaustive strata have no sampling error. The target is
      // this saved source under its frozen judge protocol, not human correctness.
      const varianceProxy = trials.reduce(
          (s, t) =>
            s +
            (t.n === t.population_pairs
              ? 0
              : t.n
                ? 1 / (count * count * t.n)
                : Infinity),
          0,
        ),
        radius = covered
          ? Math.sqrt(0.5 * Math.log((2 * family) / 0.05) * varianceProxy)
          : null;
      let generalization = null;
      if (count > 1 && allKnown) {
        const rng = random(
            `${session.config.seed}:${contrast}:${criterion}:bootstrap`,
          ),
          values = [];
        for (let b = 0; b < 2000; b++) {
          let sum = 0;
          for (let i = 0; i < count; i++)
            sum += trials[Math.floor(rng() * count)].preference_share;
          values.push(sum / count);
        }
        generalization = [quantile(values, 0.025), quantile(values, 0.975)];
      }
      const [pool, baseline] = contrast.split(":");
      rows.push({
        criterion,
        pool,
        baseline,
        trials: count,
        samples: trials.reduce((s, t) => s + t.n, 0),
        preference_share:
          covered && trials.every((t) => t.preference_share !== null)
            ? trials.reduce((s, t) => s + t.preference_share, 0) / count
            : null,
        win: covered
          ? trials.reduce((s, t) => s + t.win / t.n, 0) / count
          : null,
        loss: covered
          ? trials.reduce((s, t) => s + t.loss / t.n, 0) / count
          : null,
        tie: covered
          ? trials.reduce((s, t) => s + t.tie / t.n, 0) / count
          : null,
        coverage: covered
          ? trials.reduce((s, t) => s + t.coverage, 0) / count
          : null,
        missing_bounds: [lower, upper],
        interval:
          radius === null
            ? null
            : [Math.max(0, lower - radius), Math.min(1, upper + radius)],
        family_size: family,
        confidence: 0.95,
        interval_basis:
          "Simultaneous fixed-sample finite-source bounds; conditional on the frozen judge protocol",
        generalization_interval: generalization,
        generalization_basis: generalization
          ? "Approximate trial-cluster bootstrap; small trial counts are unstable"
          : null,
        trial_values: trials,
        evidence:
          session.status === "completed" ||
          session.status === "budget_exhausted"
            ? "Observed fixed-budget audit"
            : "Provisional audit; fixed-budget conclusions unavailable until execution ends",
      });
    }
  return rows;
}
export function validationDiagnostics(session, fits) {
  const rows = [];
  for (const k of RANKING_CRITERIA) {
    const fit = fits[k],
      observations = trainingObservations(session, k),
      counts = [1, 1, 1];
    for (const o of observations) counts[o.y] += o.weight;
    const total = counts.reduce((a, b) => a + b, 0),
      constant = counts.map((c) => c / total);
    let loss = 0,
      brier = 0,
      baselineLoss = 0,
      baselineBrier = 0,
      weight = 0,
      unknown = 0,
      orderDisagreements = 0,
      swappedPairs = 0;
    const bins = Array.from({ length: 5 }, (_, i) => ({
      low: i / 5,
      high: (i + 1) / 5,
      count: 0,
      predicted: 0,
      observed: 0,
    }));
    for (const pair of session.plan.validation) {
      const a = canonicalOutcome(session, pair, 0, k),
        b = canonicalOutcome(session, pair, 1, k);
      if (a && b) {
        swappedPairs++;
        if (a !== b) orderDisagreements++;
      }
      for (const orientation of [0, 1]) {
        const result = session.results[resultKey(pair, orientation)],
          v = result?.status === "valid" ? result.verdict[k].verdict : null;
        const predicted = predictPair(
          fit,
          orientation ? pair.b : pair.a,
          orientation ? pair.a : pair.b,
          { presented: true },
        );
        if (!predicted || !["A", "B", "tie"].includes(v)) {
          unknown += 0.5;
          continue;
        }
        const p = [predicted.win, predicted.loss, predicted.tie],
          y = ["A", "B", "tie"].indexOf(v);
        weight += 0.5;
        loss -= 0.5 * Math.log(Math.max(1e-15, p[y]));
        baselineLoss -= 0.5 * Math.log(constant[y]);
        baselineBrier +=
          0.5 *
          constant.reduce((sum, v, j) => sum + (v - Number(j === y)) ** 2, 0);
        brier += 0.5 * p.reduce((s, v, j) => s + (v - Number(j === y)) ** 2, 0);
        for (let j = 0; j < 3; j++) {
          const bin = bins[Math.min(4, Math.floor(p[j] * 5))];
          bin.count += 0.5;
          bin.predicted += 0.5 * p[j];
          bin.observed += 0.5 * Number(j === y);
        }
      }
    }
    let allDisagreements = 0,
      allSwaps = 0;
    const directed = new Set();
    for (const p of session.pairs) {
      const a = canonicalOutcome(session, p, 0, k),
        b = canonicalOutcome(session, p, 1, k);
      if (a && b) {
        allSwaps++;
        if (a !== b) allDisagreements++;
        if (a === b && a !== "tie")
          directed.add(a === "win" ? `${p.a}:${p.b}` : `${p.b}:${p.a}`);
      }
    }
    let cycles = 0,
      triangles = 0;
    const ids = session.plan.cohort;
    for (let i = 0; i < ids.length; i++)
      for (let j = i + 1; j < ids.length; j++)
        for (let l = j + 1; l < ids.length; l++) {
          const [a, b, c] = [ids[i], ids[j], ids[l]],
            ab = directed.has(`${a}:${b}`),
            ba = directed.has(`${b}:${a}`),
            bc = directed.has(`${b}:${c}`),
            cb = directed.has(`${c}:${b}`),
            ca = directed.has(`${c}:${a}`),
            ac = directed.has(`${a}:${c}`);
          if ((ab || ba) && (bc || cb) && (ca || ac)) {
            triangles++;
            if ((ab && bc && ca) || (ba && cb && ac)) cycles++;
          }
        }
    rows.push({
      criterion: k,
      pairs: session.plan.validation.length,
      assessed_weight: weight,
      unavailable_weight: unknown,
      log_loss: weight ? loss / weight : null,
      brier: weight ? brier / weight : null,
      constant_log_loss: weight ? baselineLoss / weight : null,
      constant_brier: weight ? baselineBrier / weight : null,
      calibration: bins.map((b) => ({
        ...b,
        predicted: b.count ? b.predicted / b.count : null,
        observed: b.count ? b.observed / b.count : null,
      })),
      order_disagreements: allDisagreements,
      swapped_pairs: allSwaps,
      validation_order_disagreements: orderDisagreements,
      directed_triangles: triangles,
      cycles,
      position_effect: fit.position_effect,
      components: fit.components,
      unobserved: fit.unobserved,
      converged: fit.converged,
      prior_sensitive: fit.ratings.filter((r) => r.prior_sensitive).length,
      held_out: !!session.frozen_training,
      interpretation:
        "Judge assessment; model-based intervals do not guarantee factual correctness",
    });
  }
  return rows;
}
export function auditAgreement(session, audit) {
  const rows = [];
  for (const k of RANKING_CRITERIA) {
    let compared = 0,
      agree = 0;
    for (const pair of audit.pairs) {
      for (const o of [0, 1]) {
        const primary = canonicalOutcome(session, pair, o, k),
          secondary = canonicalOutcome(
            { ...audit, candidates: session.candidates },
            pair,
            o,
            k,
          );
        if (primary && secondary) {
          compared += 0.5;
          if (primary === secondary) agree += 0.5;
        }
      }
    }
    rows.push({
      audit_id: audit.id,
      kind: audit.kind,
      criterion: k,
      compared_pairs: compared,
      agreement: compared ? agree / compared : null,
    });
  }
  return rows;
}
export function analyzeRanking(session, { fits: cachedFits, ...options } = {}) {
  const fits = cachedFits ?? fitRanking(session, options);
  return {
    fits,
    ratings: RANKING_CRITERIA.flatMap((k) =>
      fits[k].ratings.map((r) => ({
        ...r,
        criterion: k,
        evidence: r.opponents ? "Model prediction" : "Insufficient evidence",
      })),
    ),
    methods: methodAudit(session),
    validation: validationDiagnostics(session, fits),
    audits: session.audits.flatMap((a) => auditAgreement(session, a)),
  };
}
export function rankingTables(s) {
  const data = analyzeRanking(s),
    link = { ranking_session_id: s.id };
  return {
    ranking_sessions: [
      {
        ...link,
        created_at: s.created_at,
        config: s.config,
        profile: s.profile,
        status: s.status,
        source_digest: s.source_digest,
        parent_id: s.parent_id,
        model: s.model,
        allocation: s.plan.allocation,
        notes: s.plan.notes,
      },
    ],
    ranking_candidates: s.candidates.map((c) => ({
      ...link,
      ...c,
      ranked: s.plan.cohort.includes(c.id),
    })),
    ranking_pairs: s.pairs.map((p) => ({ ...link, ...p })),
    ranking_judgments: Object.entries(s.results).map(([key, r]) => ({
      ...link,
      key,
      ...r,
    })),
    ranking_requests: s.requests.map((r) => ({ ...link, ...r })),
    ranking_attempts: s.attempts.map((r) => ({ ...link, ...r })),
    ranking_ratings: data.ratings.map((r) => ({ ...link, ...r })),
    ranking_methods: data.methods.map((r) => ({ ...link, ...r })),
    ranking_validation: data.validation.map((r) => ({ ...link, ...r })),
    ranking_audits: s.audits.map((a) => ({
      ...link,
      ...a,
      agreement: auditAgreement(s, a),
    })),
    ranking_fits: Object.entries(data.fits).map(
      ([criterion, { covariance, ratings, ...f }]) => ({
        ...link,
        criterion,
        ...f,
      }),
    ),
  };
}
