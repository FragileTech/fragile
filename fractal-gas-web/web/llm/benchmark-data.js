import { validateXed } from "./scoring.js";
import { configuration, recordedConfiguration, objective } from "./config.js";
import { initialRun } from "./run-control.js";
import { importRecording, RECORDING_LIMIT } from "./recording.js";
import { parseCompletion } from "./openrouter.js";

export const BENCHMARK_VERSION = 1;
export const METHODS = [
  "fractal",
  "independent_population",
  "independent_tokens",
  "temperature_zero",
];
export function benchmarkConfiguration(input = {}) {
  const comparison = input.comparison ?? "tokens";
  const repetitions = Number(input.repetitions ?? 1);
  if (!["population", "tokens", "both"].includes(comparison))
    throw Error("Invalid benchmark comparison");
  if (
    !Number.isSafeInteger(repetitions) ||
    repetitions < 1 ||
    repetitions > 10000
  )
    throw Error("Repetitions must be between 1 and 10000");
  return { config: configuration(input.config), comparison, repetitions };
}
export function methodOrder(comparison) {
  return [
    "fractal",
    ...(comparison !== "tokens" ? ["independent_population"] : []),
    ...(comparison !== "population" ? ["independent_tokens"] : []),
    "temperature_zero",
  ];
}
export function manifest(input, id = crypto.randomUUID()) {
  const settings = benchmarkConfiguration(input);
  return {
    format: "fgllmbench",
    version: BENCHMARK_VERSION,
    id,
    created_at: Date.now(),
    settings,
    method_order: methodOrder(settings.comparison),
  };
}
export function validateManifest(value) {
  if (
    value?.format !== "fgllmbench" ||
    value.version !== BENCHMARK_VERSION ||
    typeof value.id !== "string" ||
    !/^[\w-]{1,100}$/.test(value.id) ||
    !Number.isFinite(value.created_at)
  )
    throw Error("Unsupported benchmark archive");
  const settings = benchmarkConfiguration({
    ...value.settings,
    config: recordedConfiguration(value.settings?.config),
  });
  if (
    JSON.stringify(value.method_order) !==
    JSON.stringify(methodOrder(settings.comparison))
  )
    throw Error("Invalid benchmark method order");
  return {
    format: value.format,
    version: value.version,
    id: value.id,
    created_at: value.created_at,
    settings,
    method_order: value.method_order,
  };
}
const TYPES = new Set([
  "session_start",
  "session_end",
  "preflight",
  "run_start",
  "run_end",
  "request_start",
  "request",
  "provider_response",
  "accepted",
  "generation",
  "trajectory_start",
  "error",
]);
export function validateEvent(event, header, seq) {
  if (
    !event ||
    event.seq !== seq ||
    event.benchmark_id !== header.id ||
    !TYPES.has(event.type) ||
    !Number.isFinite(event.time) ||
    !event.payload ||
    typeof event.payload !== "object" ||
    Array.isArray(event.payload)
  )
    throw Error(`Invalid benchmark event ${seq}`);
  if (
    event.run_id !== undefined &&
    (typeof event.run_id !== "string" ||
      !/^\d+:[a-z_]+:\d+$/.test(event.run_id))
  )
    throw Error("Invalid benchmark run identity");
  return event;
}
export class Journal {
  constructor(
    store,
    { redact = (value) => value, onCommitted = () => {} } = {},
  ) {
    this.store = store;
    this.manifest = store.manifest;
    this.seq = store.lastSeq ?? 0;
    this.tail = Promise.resolve();
    this.redact = redact;
    this.onCommitted = onCommitted;
  }
  append(type, payload, context = {}) {
    const copy = this.redact(structuredClone({ type, payload, ...context }));
    const job = this.tail.then(async () => {
      const event = {
        ...copy,
        seq: this.seq + 1,
        benchmark_id: this.manifest.id,
        time: Date.now(),
      };
      validateEvent(event, this.manifest, event.seq);
      await this.store.append(event);
      this.seq = event.seq;
      await this.onCommitted(event);
      return event;
    });
    this.tail = job;
    // Callers still receive the rejection; retaining a failed tail prevents later writes.
    job.catch(() => {});
    return job;
  }
}
export function collectRuns(header, events) {
  const runs = new Map();
  let metadata = null;
  for (const e of events) {
    if (e.type === "preflight") metadata = e.payload;
    if (e.type === "run_start") {
      if (runs.has(e.run_id)) throw Error("Duplicate benchmark run");
      const p = e.payload;
      if (
        !Number.isInteger(p.trial) ||
        p.trial < 0 ||
        p.trial >= header.settings.repetitions ||
        !header.method_order.includes(p.method) ||
        !Number.isInteger(p.attempt) ||
        p.attempt < 1 ||
        e.run_id !== `${p.trial}:${p.method}:${p.attempt}`
      )
        throw Error("Invalid benchmark run");
      const expected = {
        ...header.settings.config,
        seed: (header.settings.config.seed + p.trial) % 2147483648,
        ...(p.method === "temperature_zero" ? { temperature: 0 } : {}),
      };
      if (
        !p.config ||
        Object.keys(expected).some(
          (k) => expected[k] !== recordedConfiguration(p.config)[k],
        ) ||
        !metadata ||
        JSON.stringify(p.metadata) !== JSON.stringify(metadata)
      )
        throw Error("Run settings do not match the benchmark");
      runs.set(e.run_id, {
        id: e.run_id,
        ...p,
        config: recordedConfiguration(p.config),
        started_at: e.time,
        status: "interrupted",
        nodes: [p.root],
        snapshots: [],
        requests: [],
        accepted: [],
        trajectories: [],
        errors: [],
      });
    } else if (e.run_id) {
      const run = runs.get(e.run_id);
      if (!run) throw Error("Event references an unknown run");
      if (run.ended_at !== undefined)
        throw Error("Event follows a finished run");
      if (e.type === "generation") {
        run.nodes.push(...e.payload.nodes);
        run.snapshots.push(e.payload.snapshot);
      } else if (e.type === "accepted") run.accepted.push(e.payload);
      else if (e.type === "request") run.requests.push(e.payload);
      else if (e.type === "trajectory_start") run.trajectories.push(e.payload);
      else if (e.type === "error") run.errors.push(e.payload);
      else if (e.type === "run_end") {
        if (
          !["completed", "incomplete", "failed", "stopped"].includes(
            e.payload.status,
          )
        )
          throw Error("Invalid run status");
        Object.assign(run, e.payload, { ended_at: e.time });
      }
    }
  }
  return { runs, metadata };
}
export function fractalRecording(run) {
  if (run.method !== "fractal")
    throw Error("Only Fractal runs have .fgllm recordings");
  const recording = {
    format: "fgllm",
    version: run.recording_version ?? 2,
    engine: "fgllm-1",
    config: run.config,
    metadata: run.metadata,
    nodes: run.nodes,
    snapshots: run.snapshots,
    requests: run.requests,
    attempts: run.accepted.map((p) => ({ ...p.transition, ...p.result })),
    errors: run.errors,
  };
  if (recording.version === 3) {
    recording.run = {
      ...(run.snapshots.at(-1)?.run ?? initialRun(run.config)),
      generated_tokens: run.accepted.reduce(
        (sum, a) => sum + a.result.token_data.length,
        0,
      ),
    };
    if (
      run.run &&
      Object.keys(recording.run).some(
        (key) =>
          JSON.stringify(run.run[key]) !== JSON.stringify(recording.run[key]),
      )
    )
      throw Error("Fractal run summary does not match its recording");
    if (
      run.status === "completed" &&
      (run.stop_reason !== recording.run.stop_reason ||
        run.completion_success !== (recording.run.stop_reason === "eos_target"))
    )
      throw Error("Invalid Fractal completion outcome");
  }
  // Journaled responses can outlive a failed append to the bounded live recording.
  // Preserve all of them in .fgllmbench while keeping this optional legacy export valid.
  if (
    new TextEncoder().encode(JSON.stringify(recording)).length > RECORDING_LIMIT
  ) {
    recording.requests = [];
    recording.attempts = [];
    recording.errors = [
      {
        message:
          "Request provenance remains in the benchmark archive; omitted here to respect the .fgllm size limit.",
      },
    ];
  }
  return recording;
}
function validateBaseline(run) {
  const c = recordedConfiguration(run.config),
    root = run.nodes[0];
  if (
    root?.id !== 0 ||
    root.parent !== null ||
    root.text !== "" ||
    root.tokens !== 0 ||
    root.logp !== 0 ||
    root.status !== 0
  )
    throw Error("Invalid baseline root");
  const trajectories = new Map();
  for (const t of run.trajectories) {
    if (!Number.isSafeInteger(t.id) || t.id < 0 || trajectories.has(t.id))
      throw Error("Invalid independent trajectory");
    trajectories.set(t.id, 0);
  }
  for (let i = 1; i < run.nodes.length; i++) {
    const n = run.nodes[i],
      parent = run.nodes[n.parent];
    if (c.objective === "xed") validateXed(n, c);
    if (
      n.id !== i ||
      !Number.isInteger(n.parent) ||
      n.parent < 0 ||
      n.parent >= i ||
      !parent ||
      parent.status !== 0 ||
      trajectories.get(n.trajectory_id) !== n.parent ||
      typeof n.text !== "string" ||
      !n.text.startsWith(parent.text) ||
      !Array.isArray(n.token_data) ||
      !Number.isInteger(n.duration) ||
      n.duration < 1 ||
      n.duration > c.chunk_tokens
    )
      throw Error("Invalid independent prefix chain");
    const result = parseCompletion(
      {
        choices: [
          {
            message: { content: n.text.slice(parent.text.length) },
            logprobs: {
              content: n.token_data.map((t) => ({
                token: t.text,
                bytes: t.bytes,
                logprob: t.logprob,
              })),
            },
            finish_reason: n.finish_reason,
          },
        ],
      },
      Math.min(n.duration, c.sequence_tokens - parent.tokens),
    );
    if (
      n.tokens !== parent.tokens + result.token_data.length ||
      n.actual_tokens !== result.token_data.length ||
      Math.abs(n.logp - parent.logp - result.logp) > 1e-8 ||
      !Number.isFinite(n.logp) ||
      n.status !==
        (result.finish_reason === "stop"
          ? 1
          : n.tokens === c.sequence_tokens
            ? 2
            : 0) ||
      !Number.isFinite(n.reward) ||
      Math.abs(n.reward - objective(n, c) + objective(parent, c)) > 1e-8 ||
      !Array.isArray(n.embedding) ||
      n.embedding.length !== run.metadata.dimensions ||
      n.embedding.some((x) => !Number.isFinite(x)) ||
      (n.tokens > 0 && !n.embedding.some((x) => x !== 0))
    )
      throw Error("Invalid baseline generation data");
    trajectories.set(n.trajectory_id, n.id);
  }
  let count = 1;
  for (const [index, s] of run.snapshots.entries()) {
    if (
      s.step !== index + 1 ||
      !Number.isInteger(s.node_count) ||
      s.node_count < count ||
      s.node_count > run.nodes.length ||
      !Array.isArray(s.trajectories) ||
      s.trajectories.some(
        (t) =>
          !trajectories.has(t.id) ||
          !Number.isInteger(t.node) ||
          t.node < 0 ||
          t.node >= s.node_count ||
          (t.node > 0 && run.nodes[t.node].trajectory_id !== t.id),
      )
    )
      throw Error("Invalid baseline snapshot");
    for (let i = count; i < s.node_count; i++)
      if (run.nodes[i].birth_step !== s.step)
        throw Error("Invalid baseline generation step");
    count = s.node_count;
  }
  if (count !== run.nodes.length) throw Error("Uncommitted baseline nodes");
}
export function validateArchive(header, events) {
  header = validateManifest(header);
  events.forEach((e, i) => validateEvent(e, header, i + 1));
  const { runs } = collectRuns(header, events);
  const completed = new Set();
  const budgets = new Map();
  const requestIds = new Map();
  let provider;
  for (const e of events) {
    if (e.type === "preflight") {
      const current = e.payload.provider;
      if (
        !current ||
        typeof current.tag !== "string" ||
        typeof current.name !== "string" ||
        (provider &&
          (provider.tag !== current.tag || provider.name !== current.name))
      )
        throw Error("Benchmark provider changed");
      provider = current;
    }
    if (e.type === "request_start") {
      const id = e.payload.logical_request_id;
      if (typeof id !== "string" || requestIds.has(id))
        throw Error("Invalid logical request identity");
      requestIds.set(id, e);
    }
    if (
      ["request", "provider_response"].includes(e.type) &&
      e.payload.logical_request_id
    ) {
      const start = requestIds.get(e.payload.logical_request_id);
      if (!start || start.run_id !== e.run_id)
        throw Error("Invalid request attribution");
    }
  }
  for (const run of runs.values()) {
    if (run.method === "fractal")
      importRecording(JSON.stringify(fractalRecording(run)));
    else validateBaseline(run);
    const tokens = run.accepted.reduce(
      (n, a) => n + a.result.token_data.length,
      0,
    );
    for (const a of run.accepted) {
      const r = a.result;
      if (
        !Number.isInteger(a.count) ||
        a.count < 1 ||
        a.count > run.config.chunk_tokens ||
        !a.transition ||
        !Number.isInteger(a.transition.source) ||
        a.transition.source < 0 ||
        !run.nodes[a.transition.source] ||
        !requestIds.has(r.logical_request_id) ||
        requestIds.get(r.logical_request_id).run_id !== run.id
      )
        throw Error("Invalid accepted response attribution");
      const parsed = parseCompletion(
        {
          choices: [
            {
              message: { content: r.text },
              logprobs: {
                content: r.token_data.map((t) => ({
                  token: t.text,
                  bytes: t.bytes,
                  logprob: t.logprob,
                })),
              },
              finish_reason: r.finish_reason,
            },
          ],
          usage: r.usage,
        },
        a.count,
      );
      if (Math.abs(parsed.logp - r.logp) > 1e-8)
        throw Error("Invalid accepted response");
    }
    const accepted = new Map(
      run.accepted.map((a) => [a.result.logical_request_id, a]),
    );
    if (accepted.size !== run.accepted.length)
      throw Error("Duplicate accepted response");
    for (const n of run.nodes.slice(1)) {
      const a = accepted.get(n.logical_request_id);
      if (
        !a ||
        a.transition.source !== n.parent ||
        a.result.text !== n.text.slice(run.nodes[n.parent].text.length) ||
        JSON.stringify(a.result.token_data) !== JSON.stringify(n.token_data)
      )
        throw Error("Node does not match its accepted response");
    }
    if (run.generated_tokens !== undefined && run.generated_tokens !== tokens)
      throw Error("Invalid generated-token total");
    if (
      run.method === "independent_tokens" &&
      (!Number.isSafeInteger(run.token_budget) ||
        run.token_budget < 0 ||
        tokens > run.token_budget ||
        run.token_budget !== budgets.get(run.trial) ||
        (run.status === "completed" && tokens !== run.token_budget))
    )
      throw Error("Invalid matched token budget");
    if (run.status === "completed") {
      const key = `${run.trial}:${run.method}`;
      if (completed.has(key)) throw Error("Duplicate completed method");
      completed.add(key);
      if (run.method === "fractal") budgets.set(run.trial, tokens);
    }
  }
  return { manifest: header, events };
}
export function parseBenchmark(text) {
  const lines = text.split("\n").filter((line) => line.trim());
  if (!lines.length) throw Error("Empty benchmark archive");
  const [header, ...events] = lines.map((line) => JSON.parse(line));
  return validateArchive(header, events);
}
export function exportBenchmark(header, events) {
  validateArchive(header, events);
  return [header, ...events].map((row) => JSON.stringify(row) + "\n").join("");
}
export function usageTotals(requests) {
  const total = {};
  for (const field of [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
  ]) {
    const relevant = requests.filter(
      (r) =>
        r.path === "chat/completions" ||
        r.path === "embeddings" ||
        r.path === "scoring/completions",
    );
    const known = relevant.filter((r) => Number.isFinite(r.usage?.[field]));
    total[field] = {
      known_total: known.reduce((n, r) => n + r.usage[field], 0),
      measured_requests: known.length,
      missing_requests: relevant.length - known.length,
      total:
        relevant.length && known.length === relevant.length
          ? known.reduce((n, r) => n + r.usage[field], 0)
          : null,
    };
  }
  return total;
}
export function processBenchmark(header, events) {
  validateArchive(header, events);
  const { runs } = collectRuns(header, events);
  const tables = {
    runs: [],
    trajectories: [],
    nodes: [],
    tokens: [],
    requests: [],
    snapshots: [],
    clone_decisions: [],
  };
  for (const e of events)
    if (
      ["request_start", "request", "provider_response", "accepted"].includes(
        e.type,
      )
    )
      tables.requests.push({
        benchmark_id: header.id,
        run_id: e.run_id ?? null,
        event: e.type,
        seq: e.seq,
        time: e.time,
        ...e.payload,
      });
  for (const run of runs.values()) {
    const link = {
      benchmark_id: header.id,
      run_id: run.id,
      trial: run.trial,
      method: run.method,
      attempt: run.attempt,
    };
    tables.runs.push({
      ...link,
      config: run.config,
      metadata: run.metadata,
      status: run.status,
      stop_reason: run.stop_reason ?? "interrupted",
      started_at: run.started_at,
      ended_at: run.ended_at ?? null,
      token_budget: run.token_budget,
      completion_success: run.completion_success ?? null,
      run:
        run.method === "fractal" ? (fractalRecording(run).run ?? null) : null,
      generated_tokens: run.accepted.reduce(
        (n, a) => n + a.result.token_data.length,
        0,
      ),
      scoring_usage: usageTotals(
        run.requests.filter((r) => r.path === "scoring/completions"),
      ),
      generation_usage: usageTotals(
        run.requests.filter((r) => r.path === "chat/completions"),
      ),
      embedding_usage: usageTotals(
        run.requests.filter((r) => r.path === "embeddings"),
      ),
    });
    for (const n of run.nodes) {
      const id = `${run.id}:${n.id}`,
        { token_data, ...node } = n;
      tables.nodes.push({
        ...link,
        ...node,
        node_id: id,
        parent_id: n.parent === null ? null : `${run.id}:${n.parent}`,
        total_nll: 0 - n.logp,
        mean_nll: n.tokens ? (0 - n.logp) / n.tokens : null,
      });
      token_data.forEach((t, i) =>
        tables.tokens.push({
          ...link,
          node_id: id,
          chunk_index: i,
          sequence_index: run.nodes[n.parent].tokens + i,
          ...t,
        }),
      );
    }
    for (const s of run.snapshots) {
      const { decisions, ...snapshot } = s;
      tables.snapshots.push({ ...link, ...snapshot });
      for (const d of decisions ?? [])
        tables.clone_decisions.push({ ...link, step: s.step, ...d });
    }
    const parents = new Set(run.nodes.slice(1).map((n) => n.parent));
    const endpoints =
      run.method === "fractal"
        ? run.nodes.slice(1).filter((n) => !parents.has(n.id))
        : run.trajectories.map(
            (t) =>
              run.nodes.findLast((n) => n.trajectory_id === t.id) ?? {
                ...run.nodes[0],
                trajectory_id: t.id,
              },
          );
    for (const n of endpoints)
      tables.trajectories.push({
        ...link,
        trajectory_id: n.trajectory_id ?? n.id,
        node_id: `${run.id}:${n.id}`,
        text: n.text,
        tokens: n.tokens,
        logp: n.logp,
        total_nll: 0 - n.logp,
        mean_nll: n.tokens ? (0 - n.logp) / n.tokens : null,
        status:
          n.status === 1 ? "finished" : n.status === 2 ? "capped" : "partial",
        final_population_multiplicity:
          run.method === "fractal"
            ? (run.snapshots.at(-1)?.walkers ?? []).filter(
                (w) => w.node === n.id,
              ).length
            : null,
      });
  }
  return tables;
}
