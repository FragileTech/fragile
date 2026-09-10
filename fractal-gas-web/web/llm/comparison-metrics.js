import { objective } from "./config.js";
import { collectRuns, usageTotals } from "./benchmark-data.js";

export const METHOD_STYLE = {
  fractal: { label: "Fractal", color: "#d0a9e2", marker: "circle" },
  independent_population: {
    label: "Independent population",
    color: "#7ef5df",
    marker: "square",
  },
  independent_tokens: {
    label: "Independent token budget",
    color: "#efc87b",
    marker: "diamond",
  },
  temperature_zero: {
    label: "Temperature zero",
    color: "#ff729b",
    marker: "triangle",
  },
};
export const METRICS = {
  mean: "Mean token log likelihood",
  reward: "Full-trace reward",
  logp: "Total log likelihood",
  nll: "Total NLL",
  mean_nll: "Mean token NLL",
  tokens: "Generated length",
  nearest: "Nearest-neighbor distance",
  greedy_distance: "Distance to temperature-zero answer",
  grade: "Judge overall score",
  correctness: "Judge correctness",
  relevance: "Judge relevance",
  completeness: "Judge completeness",
  clarity: "Judge clarity",
};
export const DEFAULT_FILTERS = {
  trial: "all",
  method: "all",
  status: "full",
  pool: "both",
  attempt: "default",
  metric: "mean",
  baseline: "independent_tokens",
  distance: "cosine",
};
export const finite = Number.isFinite;
export const maxValue = (a) =>
  a.length ? a.reduce((m, x) => Math.max(m, x), -Infinity) : null;
export const minValue = (a) =>
  a.length ? a.reduce((m, x) => Math.min(m, x), Infinity) : null;
export const mean = (a) =>
  a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
export function rng(seed = 7) {
  let state =
    typeof seed === "number"
      ? seed
      : [...seed].reduce(
          (n, c) => Math.imul(n ^ c.charCodeAt(0), 16777619),
          2166136261,
        );
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function quantile(values, p) {
  if (!values.length) return null;
  const a = [...values].sort((x, y) => x - y),
    x = p * (a.length - 1),
    lo = Math.floor(x);
  return a[lo] + (a[Math.ceil(x)] - a[lo]) * (x - lo);
}
export function describe(rows, key) {
  const a = rows.filter((r) => finite(r[key])).sort((x, y) => x[key] - y[key]);
  const n = a.reduce((s, r) => s + r.weight, 0);
  if (!n)
    return {
      count: 0,
      mean: null,
      median: null,
      sd: null,
      min: null,
      max: null,
      q25: null,
      q75: null,
    };
  const avg = a.reduce((s, r) => s + r[key] * r.weight, 0) / n;
  const at = (i) => {
    let seen = 0;
    for (const r of a) {
      seen += r.weight;
      if (i < seen) return r[key];
    }
    return a.at(-1)[key];
  };
  const q = (p) => {
    const x = p * (n - 1);
    return at(Math.floor(x)) * (1 - (x % 1)) + at(Math.ceil(x)) * (x % 1);
  };
  return {
    count: n,
    mean: avg,
    median: q(0.5),
    sd: Math.sqrt(
      a.reduce((s, r) => s + r.weight * (r[key] - avg) ** 2, 0) / n,
    ),
    min: a[0][key],
    max: a.at(-1)[key],
    q25: q(0.25),
    q75: q(0.75),
  };
}
export function sourceRuns(source) {
  if (!source) return [];
  if (source.kind === "benchmark") {
    const runs = [...collectRuns(source.manifest, source.events).runs.values()];
    const work = new Map(),
      boundary = new Map(),
      byId = new Map(runs.map((r) => [r.id, r]));
    for (const e of source.events) {
      if (e.type === "accepted")
        work.set(
          e.run_id,
          (work.get(e.run_id) ?? 0) +
            (e.payload.result?.token_data?.length ?? 0),
        );
      if (e.type === "generation") {
        const run = byId.get(e.run_id),
          index = boundary.get(e.run_id) ?? 0;
        run.snapshots[index] = {
          ...run.snapshots[index],
          generated_work: work.get(e.run_id) ?? 0,
        };
        boundary.set(e.run_id, index + 1);
      }
    }
    return runs;
  }
  const r = source.record;
  return r
    ? [
        {
          id: "current",
          method: "fractal",
          trial: 0,
          attempt: 1,
          config: r.config,
          metadata: r.metadata,
          nodes: r.nodes,
          snapshots: r.snapshots,
          requests: r.requests ?? [],
          accepted: r.attempts ?? [],
          status: r.run?.stop_reason ? "completed" : "live",
          stop_reason: r.run?.stop_reason ?? null,
          run: r.run,
          generated_tokens: r.run?.generated_tokens,
          standalone: true,
        },
      ]
    : [];
}
export function endpointWeights(run, pool, snapshot = run.snapshots.at(-1)) {
  const weights = new Map(),
    add = (id) => {
      if (id != null && id > 0 && run.nodes[id])
        weights.set(id, (weights.get(id) ?? 0) + 1);
    };
  if (pool === "retained" && run.method === "fractal") {
    for (const w of snapshot?.walkers ?? [])
      if (run.config.algorithm !== "graph" || w.leaf) add(w.node);
  } else if (run.method !== "fractal") {
    const latest = new Map((run.trajectories ?? []).map((t) => [t.id, 0]));
    for (const n of run.nodes.slice(
      1,
      snapshot?.node_count ?? run.nodes.length,
    ))
      latest.set(n.trajectory_id, n.id);
    for (const id of latest.values()) add(id);
  } else {
    const nodes = run.nodes.slice(0, snapshot?.node_count ?? run.nodes.length);
    const parents = new Set(nodes.slice(1).map((n) => n.parent));
    for (const n of nodes.slice(1)) if (!parents.has(n.id)) add(n.id);
  }
  return weights;
}
export function distance(a, b, metric = "cosine") {
  if (
    !a?.length ||
    a.length !== b?.length ||
    a.some((x) => !finite(x)) ||
    b.some((x) => !finite(x))
  )
    return null;
  let dot = 0,
    aa = 0,
    bb = 0,
    d = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aa += a[i] ** 2;
    bb += b[i] ** 2;
    d += (a[i] - b[i]) ** 2;
  }
  if (!aa || !bb) return null;
  return metric === "l2"
    ? Math.sqrt(d)
    : Math.max(0, Math.min(2, 1 - dot / Math.sqrt(aa * bb)));
}
function pairAt(k, n) {
  let lo = 0,
    hi = n - 1;
  const before = (i) => (i * (2 * n - i - 1)) / 2;
  while (lo + 1 < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (before(mid) <= k) lo = mid;
    else hi = mid;
  }
  return [lo, lo + 1 + k - before(lo)];
}
export function diversity(rows, nodes, metric, seed, limit = 50000) {
  const valid = rows.filter(
    (r) =>
      distance(
        nodes[r.node_id]?.embedding,
        nodes[r.node_id]?.embedding,
        metric,
      ) !== null,
  );
  const dims = new Set(valid.map((r) => nodes[r.node_id].embedding.length));
  if (dims.size > 1)
    return {
      pairs: [],
      total: 0,
      sampled: false,
      coverage: 0,
      reason: "Incompatible embedding dimensions",
    };
  const slots = valid.flatMap((r) => Array(r.weight).fill(r));
  const total = (slots.length * (slots.length - 1)) / 2,
    count = Math.min(limit, total),
    random = rng(seed);
  const indices = new Set();
  if (total <= limit) for (let i = 0; i < total; i++) indices.add(i);
  else while (indices.size < count) indices.add(Math.floor(random() * total));
  const nearest = new Map(),
    pairs = [];
  for (const k of indices) {
    const [i, j] = pairAt(k, slots.length),
      a = slots[i],
      b = slots[j];
    const value = distance(
      nodes[a.node_id].embedding,
      nodes[b.node_id].embedding,
      metric,
    );
    pairs.push({ value, a: a.key, b: b.key });
    for (const r of [a, b])
      nearest.set(r.key, Math.min(nearest.get(r.key) ?? Infinity, value));
  }
  for (const r of rows) r.nearest = nearest.get(r.key) ?? null;
  return {
    pairs,
    total,
    sampled: total > limit,
    coverage: slots.length,
    reason: !total ? "At least two embedded answers are needed" : null,
  };
}
export function pairedDifference(left, right, seed = 7) {
  const pairs = left
    .map((a) => [a, right.find((b) => b.trial === a.trial)])
    .filter(([a, b]) => b && finite(a.value) && finite(b.value));
  const deltas = pairs.map(([a, b]) => a.value - b.value),
    value = mean(deltas);
  if (deltas.length < 2)
    return { value, trials: deltas.length, interval: null };
  const random = rng(seed),
    samples = Array.from({ length: 1000 }, () =>
      mean(deltas.map(() => deltas[Math.floor(random() * deltas.length)])),
    );
  return {
    value,
    trials: deltas.length,
    interval: [quantile(samples, 0.025), quantile(samples, 0.975)],
  };
}
export function pca(rows, runs, metric) {
  const unique = [...new Map(rows.map((r) => [r.key, r])).values()];
  const eligible = unique.filter((r) => {
    const run = runs.find((x) => x.id === r.run_id);
    return (
      distance(
        run.nodes[r.node_id].embedding,
        run.nodes[r.node_id].embedding,
        metric,
      ) !== null
    );
  });
  const signatures = new Set(
    eligible.map((r) => {
      const run = runs.find((x) => x.id === r.run_id);
      return `${run.config.embedding_model}:${run.config.embedding_input}:${run.nodes[r.node_id].embedding?.length}`;
    }),
  );
  if (signatures.size > 1)
    return {
      points: [],
      reason: "Incompatible embedding models, inputs or dimensions",
    };
  const input = eligible
    .map((r) => ({
      r,
      v: runs.find((x) => x.id === r.run_id).nodes[r.node_id].embedding,
    }))
    .filter((x) => distance(x.v, x.v, metric) !== null);
  if (input.length < 2)
    return { points: [], reason: "At least two embedded answers are needed" };
  const d = input[0].v.length,
    means = new Float64Array(d);
  const vectors = input.map(({ v }) => {
    const norm =
      metric === "cosine" ? Math.sqrt(v.reduce((s, x) => s + x * x, 0)) : 1;
    return Float64Array.from(v, (x) => x / norm);
  });
  for (const v of vectors)
    for (let j = 0; j < d; j++) means[j] += v[j] / vectors.length;
  for (const v of vectors) for (let j = 0; j < d; j++) v[j] -= means[j];
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0),
    random = rng("pca");
  const axes = [];
  for (let axis = 0; axis < 2; axis++) {
    let v = Float64Array.from({ length: d }, () => random() - 0.5);
    for (let step = 0; step < 32; step++) {
      const next = new Float64Array(d);
      for (const row of vectors) {
        const p = dot(row, v);
        for (let j = 0; j < d; j++) next[j] += row[j] * p;
      }
      for (const prior of axes) {
        const p = dot(next, prior);
        for (let j = 0; j < d; j++) next[j] -= prior[j] * p;
      }
      const norm = Math.sqrt(dot(next, next));
      if (norm < 1e-12) {
        v.fill(0);
        break;
      }
      v = next.map((x) => x / norm);
    }
    axes.push(v);
  }
  if (!axes[0].some((x) => x !== 0))
    return { points: [], reason: "Embeddings have no variation" };
  return {
    points: input.map(({ r }, i) => ({
      key: r.key,
      method: r.method,
      x: dot(vectors[i], axes[0]),
      y: dot(vectors[i], axes[1]),
    })),
    reason: null,
  };
}
export function traceChain(run, id) {
  const chain = [];
  for (let node = run.nodes[id]; node?.id; node = run.nodes[node.parent])
    chain.push(node);
  return chain.reverse();
}
export function traceComparison(runs, keys) {
  const selected = keys
    .map((key) => {
      const run = runs.find((r) => key.startsWith(r.id + "/"));
      if (!run) return null;
      const id = Number(key.slice(run.id.length + 1));
      const node = run.nodes[id];
      return node ? { run, node, chain: traceChain(run, id) } : null;
    })
    .filter(Boolean);
  const traces = selected.map(({ run, node, chain }) => {
    let logp = 0,
      index = 0;
    const tokens = chain.flatMap((chunk) =>
      chunk.token_data.map((t) => {
        logp += t.logprob;
        index++;
        return {
          ...t,
          index,
          chunk: chunk.id,
          logp,
          reward:
            run.config.objective === "xed"
              ? index === chunk.tokens
                ? objective(chunk, run.config)
                : null
              : objective({ logp, tokens: index }, run.config),
        };
      }),
    );
    return {
      key: `${run.id}/${node.id}`,
      method: run.method,
      text: node.text,
      node,
      tokens,
      chunks: chain.map((c) => ({
        id: c.id,
        tokens: c.tokens,
        reward: c.reward,
      })),
    };
  });
  let shared = 0,
    matching = 0;
  if (selected.length === 2) {
    const [a, b] = selected;
    if (a.run.id === b.run.id)
      for (
        let i = 0;
        i < Math.min(a.chain.length, b.chain.length) &&
        a.chain[i].id === b.chain[i].id;
        i++
      )
        shared = a.chain[i].tokens;
    const left = Array.from(a.node.text),
      right = Array.from(b.node.text);
    while (
      matching < Math.min(left.length, right.length) &&
      left[matching] === right[matching]
    )
      matching++;
  }
  return {
    traces,
    shared_ancestry_tokens: shared,
    matching_text_characters: matching,
  };
}
export function computeComparison(source, input = {}, gradeLookup = {}) {
  const filters = { ...DEFAULT_FILTERS, ...input },
    allRuns = sourceRuns(source);
  const runs = allRuns.filter(
    (r) =>
      (filters.trial === "all" || r.trial === Number(filters.trial)) &&
      (filters.method === "all" || r.method === filters.method) &&
      (filters.attempt === "default"
        ? r.status === "completed" ||
          r.standalone ||
          (source.live &&
            r.id === allRuns.at(-1)?.id &&
            r.status === "interrupted")
        : filters.attempt === r.id),
  );
  const groups = [],
    rows = [];
  const pools =
    filters.pool === "both" ? ["archive", "retained"] : [filters.pool];
  for (const run of runs)
    for (const pool of pools) {
      const weights = endpointWeights(run, pool),
        endpoints = [...weights].map(([id, weight]) => ({
          n: run.nodes[id],
          weight,
        }));
      const outcomes = { eos: 0, capped: 0, partial: 0, empty: 0 };
      for (const { n, weight } of endpoints)
        outcomes[
          n.tokens === 0
            ? "empty"
            : n.status === 1
              ? "eos"
              : n.status === 2
                ? "capped"
                : "partial"
        ] += weight;
      let chosen = endpoints.filter(
        ({ n }) =>
          n.tokens > 0 &&
          (filters.status === "all" ||
            (filters.status === "full" && n.status > 0) ||
            (filters.status === "eos" && n.status === 1) ||
            (filters.status === "partial" && n.status === 0)),
      );
      const partialPreview =
        !chosen.length && run.standalone && filters.status === "full";
      if (partialPreview)
        chosen = endpoints.filter(({ n }) => n.tokens > 0 && n.status === 0);
      const traces = chosen.map(({ n, weight }) => ({
        key: `${run.id}/${n.id}`,
        run_id: run.id,
        node_id: n.id,
        trial: run.trial,
        method: run.method,
        pool,
        weight,
        status: n.status,
        text: n.text.slice(0, 120),
        tokens: n.tokens,
        logp: n.logp,
        mean: n.logp / n.tokens,
        reward: objective(n, run.config),
        nll: 0 - n.logp,
        mean_nll: (0 - n.logp) / n.tokens,
        grade: gradeLookup[`${run.id}/${n.id}`]?.overall ?? null,
        ...Object.fromEntries(
          ["correctness", "relevance", "completeness", "clarity"].map((k) => [
            k,
            gradeLookup[`${run.id}/${n.id}`]?.scores[k] ?? null,
          ]),
        ),
        nearest: null,
        greedy_distance: null,
      }));
      const div = diversity(
        traces,
        run.nodes,
        filters.distance,
        `${run.id}:${pool}:${filters.distance}`,
      );
      const zero = allRuns.find(
        (r) =>
          r.trial === run.trial &&
          r.method === "temperature_zero" &&
          r.status === "completed",
      );
      if (
        zero &&
        zero.config.embedding_model === run.config.embedding_model &&
        zero.config.embedding_input === run.config.embedding_input
      ) {
        const last = zero.nodes.at(-1);
        for (const row of traces)
          row.greedy_distance = distance(
            run.nodes[row.node_id].embedding,
            last?.embedding,
            filters.distance,
          );
      }
      const weight = traces.reduce((s, r) => s + r.weight, 0),
        unique = new Set(chosen.map(({ n }) => n.text)).size;
      const concentration = weight
        ? traces.reduce((s, r) => s + (r.weight / weight) ** 2, 0)
        : null;
      groups.push({
        run_id: run.id,
        trial: run.trial,
        method: run.method,
        pool,
        outcomes,
        partialPreview,
        traces,
        diversity: div,
        unique,
        uniqueness: weight ? unique / weight : null,
        concentration,
        stats: describe(traces, filters.metric),
        total: weight,
      });
      rows.push(...traces);
    }
  const summaries = [];
  for (const method of Object.keys(METHOD_STYLE))
    for (const pool of pools) {
      const selected = groups.filter(
        (g) => g.method === method && g.pool === pool,
      );
      if (!selected.length) continue;
      const measured = selected.filter((g) => g.stats.count);
      const value = (key) =>
        mean(measured.map((g) => g.stats[key]).filter(finite));
      const baseline = groups.filter(
        (g) => g.method === filters.baseline && g.pool === pool,
      );
      summaries.push({
        method,
        pool,
        trials: selected.length,
        measured_trials: measured.length,
        count: selected.reduce((s, g) => s + g.total, 0),
        measured_count: selected.reduce((s, g) => s + g.stats.count, 0),
        embedding_count: selected.reduce((s, g) => s + g.diversity.coverage, 0),
        mean: value("mean"),
        median: value("median"),
        sd: value("sd"),
        q25: value("q25"),
        q75: value("q75"),
        unique: selected.reduce((s, g) => s + g.unique, 0),
        uniqueness: mean(selected.map((g) => g.uniqueness).filter(finite)),
        concentration: mean(
          selected.map((g) => g.concentration).filter(finite),
        ),
        diversity: mean(
          selected
            .map((g) => mean(g.diversity.pairs.map((p) => p.value)))
            .filter(finite),
        ),
        graded: selected.reduce(
          (s, g) =>
            s +
            g.traces
              .filter((r) => finite(r.grade))
              .reduce((n, r) => n + r.weight, 0),
          0,
        ),
        difference:
          method === filters.baseline
            ? {
                value: measured.length ? 0 : null,
                trials: measured.length,
                interval: null,
              }
            : pairedDifference(
                selected.map((g) => ({ trial: g.trial, value: g.stats.mean })),
                baseline.map((g) => ({ trial: g.trial, value: g.stats.mean })),
                method + pool,
              ),
      });
    }
  const histories = [],
    efficiency = [];
  for (const run of runs) {
    let count = 1,
      committedTokens = 0;
    for (const s of run.snapshots) {
      for (; count < s.node_count; count++)
        committedTokens += run.nodes[count].token_data.length;
      const full = run.nodes
        .slice(1, s.node_count)
        .filter((n) => n.tokens > 0 && n.status > 0);
      const graded = full
        .map((n) => gradeLookup[`${run.id}/${n.id}`]?.overall)
        .filter(finite);
      const weights = endpointWeights(run, "retained", s),
        total = [...weights.values()].reduce((a, b) => a + b, 0);
      histories.push({
        method: run.method,
        run_id: run.id,
        trial: run.trial,
        step: s.step ?? histories.length + 1,
        generated_tokens:
          s.generated_work ?? s.run?.generated_tokens ?? committedTokens,
        work_basis:
          s.run || s.generated_work != null
            ? "All accepted generation"
            : "Committed generation",
        best_mean: maxValue(full.map((n) => n.logp / n.tokens)),
        best_reward: maxValue(full.map((n) => objective(n, run.config))),
        best_grade: maxValue(graded),
        graded: graded.length,
        completed: full.length,
        eos: full.filter((n) => n.status === 1).length,
        capped: full.filter((n) => n.status === 2).length,
        clones: s.decisions ? s.decisions.filter((d) => d.cloned).length : null,
        clone_rate: s.decisions?.length
          ? s.decisions.filter((d) => d.cloned).length / s.decisions.length
          : null,
        concentration: total
          ? [...weights.values()].reduce((a, w) => a + (w / total) ** 2, 0)
          : null,
        distinct: weights.size,
        retained: total,
      });
    }
    const generation = run.requests.filter(
        (r) => r.path === "chat/completions",
      ),
      embedding = run.requests.filter((r) => r.path === "embeddings"),
      scoring = run.requests.filter((r) => r.path === "scoring/completions");
    efficiency.push({
      run_id: run.id,
      method: run.method,
      trial: run.trial,
      status: run.status,
      stop_reason: run.stop_reason ?? "Not recorded",
      generated_tokens:
        run.generated_tokens ??
        run.run?.generated_tokens ??
        (run.accepted.length
          ? run.accepted.reduce(
              (sum, a) =>
                sum +
                (a.result?.token_data?.length ?? a.token_data?.length ?? 0),
              0,
            )
          : committedTokens),
      generation: usageTotals(generation),
      scoring: usageTotals(scoring),
      scoring_requests: scoring.length,
      scoring_request_ms: scoring.reduce((n, r) => n + (r.elapsed_ms ?? 0), 0),
      embedding: usageTotals(embedding),
      generation_requests: generation.length,
      embedding_requests: embedding.length,
      generation_request_ms:
        generation.length && generation.every((r) => finite(r.elapsed_ms))
          ? generation.reduce((a, r) => a + r.elapsed_ms, 0)
          : null,
      embedding_request_ms:
        embedding.length && embedding.every((r) => finite(r.elapsed_ms))
          ? embedding.reduce((a, r) => a + r.elapsed_ms, 0)
          : null,
      elapsed_ms:
        finite(run.ended_at) && finite(run.started_at)
          ? run.ended_at - run.started_at
          : null,
      completion_target: run.run?.completion_target ?? null,
    });
  }
  const plottedDistributions = Object.fromEntries(
    pools.map((pool) => {
      const selected = groups.filter((g) => g.pool === pool);
      return [
        pool,
        {
          selected: distributions(selected, filters.metric),
          pairs: distributions(selected, null, { distance: true }),
          nearest: distributions(selected, "nearest"),
          grade: distributions(selected, "grade"),
        },
      ];
    }),
  );
  return {
    distributions: plottedDistributions,
    filters,
    rows,
    groups,
    summaries,
    histories,
    efficiency,
    projection: pca(rows, runs, filters.distance),
    attempts: allRuns.map((r) => ({
      id: r.id,
      trial: r.trial,
      method: r.method,
      attempt: r.attempt,
      status: r.status,
      stop_reason: r.stop_reason,
    })),
    partialPreview: groups.some((g) => g.partialPreview),
    methods: [...new Set(runs.map((r) => r.method))],
    trials: [...new Set(allRuns.map((r) => r.trial))],
  };
}

export function distributions(groups, key, { distance = false } = {}) {
  const selected = groups.map((g) => ({
    ...g,
    values: distance
      ? g.diversity.pairs.map((p) => ({
          value: p.value,
          weight: 1,
          keys: [p.a, p.b],
        }))
      : g.traces
          .filter((r) => finite(r[key]))
          .map((r) => ({ value: r[key], weight: r.weight, keys: [r.key] })),
  }));
  const values = selected.flatMap((g) => g.values.map((v) => v.value));
  if (!values.length) return { hist: [], cdf: [], domain: [0, 1] };
  let lo = minValue(values),
    hi = maxValue(values);
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const n = 20,
    width = (hi - lo) / n,
    hist = [],
    cdf = [];
  for (const method of [...new Set(selected.map((g) => g.method))]) {
    const trials = selected.filter(
        (g) => g.method === method && g.values.length,
      ),
      bins = Array.from({ length: n }, (_, i) => ({
        x: lo + (i + 0.5) * width,
        y: 0,
        range: [lo + i * width, lo + (i + 1) * width],
        keys: [],
      })),
      points = [];
    const masses = [];
    for (const g of trials) {
      const total = g.values.reduce((s, v) => s + v.weight, 0);
      for (const v of g.values) {
        const b = bins[Math.min(n - 1, Math.floor((v.value - lo) / width))];
        const mass = v.weight / total / trials.length;
        b.y += mass;
        b.keys.push(...v.keys);
        masses.push({ ...v, mass });
      }
    }
    let sum = 0;
    if (!distance)
      for (const v of masses.sort((a, b) => a.value - b.value)) {
        sum += v.mass;
        points.push({ x: v.value, y: sum, keys: v.keys });
      }
    for (const bin of bins) bin.keys = [...new Set(bin.keys)];
    hist.push({ method, points: trials.length ? bins : [] });
    cdf.push({ method, points });
  }
  return { hist, cdf, domain: [lo, hi] };
}
