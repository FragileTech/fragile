import { configuration, recordedConfiguration, objective } from "./config.js";
import { validateXed } from "./scoring.js";
import { parseCompletion } from "./openrouter.js";
import { initialRun, stoppingReason } from "./run-control.js";
export const RECORDING_LIMIT = 64 * 1024 * 1024;
const bytes = (value) => new TextEncoder().encode(JSON.stringify(value)).length;
export class Recording {
  constructor(config) {
    this.data = {
      format: "fgllm",
      version: 3,
      engine: "fgllm-1",
      config: configuration(config),
      metadata: {},
      nodes: [
        {
          id: 0,
          parent: null,
          text: "",
          tokens: 0,
          logp: 0,
          status: 0,
          embedding: [],
          token_data: [],
        },
      ],
      snapshots: [],
      requests: [],
      attempts: [],
      errors: [],
    };
    this.data.run = initialRun(this.data.config);
    this.size = bytes(this.data);
  }
  append(kind, value) {
    const copy = structuredClone(value),
      n = bytes(copy) + 1;
    const generated =
      this.data.run.generated_tokens +
      (kind === "attempts" ? copy.token_data.length : 0);
    const growth =
      String(generated).length - String(this.data.run.generated_tokens).length;
    if (generated > this.data.run.token_budget)
      throw new Error("Generated-token budget exceeded");
    if (this.size + n + growth > RECORDING_LIMIT)
      throw new Error("Recording reached 64 MiB. Export this run and reset.");
    this.data[kind].push(copy);
    this.data.run.generated_tokens = generated;
    this.size += n + growth;
  }
  metadata(value) {
    const n = bytes(value) - bytes(this.data.metadata);
    if (this.size + n > RECORDING_LIMIT)
      throw new Error("Recording reached 64 MiB");
    this.data.metadata = structuredClone(value);
    this.size += n;
  }
  commit(nodes, snapshot) {
    if (this.data.run.stop_reason)
      throw new Error("Run has ended; reset to generate again");
    const allNodes = [...this.data.nodes, ...nodes];
    const run = {
      ...this.data.run,
      eos_node_ids: allNodes
        .filter((node) => node.status === 1)
        .map((node) => node.id),
    };
    run.stop_reason = stoppingReason(
      this.data.config,
      run,
      snapshot,
      allNodes,
      this.data.snapshots.length + 1,
    );
    snapshot = { ...snapshot, run: structuredClone(run) };
    const n =
      bytes(nodes) +
      bytes(snapshot) +
      nodes.length +
      1 +
      bytes(run) -
      bytes(this.data.run);
    if (this.size + n > RECORDING_LIMIT)
      throw new Error("Recording reached 64 MiB. Export this run and reset.");
    this.data.nodes.push(...structuredClone(nodes));
    this.data.snapshots.push(structuredClone(snapshot));
    this.data.run = run;
    this.size += n;
  }
  export() {
    const text = JSON.stringify(this.data);
    if (new TextEncoder().encode(text).length > RECORDING_LIMIT)
      throw new Error("Recording too large");
    return text;
  }
}
export function importRecording(text) {
  if (new TextEncoder().encode(text).length > RECORDING_LIMIT)
    throw new Error("Recording exceeds 64 MiB");
  const data = JSON.parse(text);
  if (
    data?.format !== "fgllm" ||
    ![1, 2, 3].includes(data.version) ||
    data.engine !== "fgllm-1"
  )
    throw new Error("Unsupported LLM recording");
  const config = recordedConfiguration(data.config);
  if (
    !Array.isArray(data.nodes) ||
    !data.nodes.length ||
    !Array.isArray(data.snapshots)
  )
    throw new Error("Invalid trace data");
  const dimensions = data.metadata?.dimensions;
  if (
    data.nodes.length > 1 &&
    (!Number.isInteger(dimensions) || dimensions < 1 || dimensions > 65536)
  )
    throw new Error("Invalid recorded dimensions");
  for (const kind of ["requests", "attempts", "errors"])
    if (!Array.isArray(data[kind])) throw new Error("Invalid request archive");
  for (let i = 0; i < data.nodes.length; i++) {
    const n = data.nodes[i];
    if (
      n.id !== i ||
      typeof n.text !== "string" ||
      !Number.isInteger(n.tokens) ||
      n.tokens < 0 ||
      !Number.isFinite(n.logp) ||
      n.logp > 0 ||
      ![0, 1, 2].includes(n.status) ||
      !Array.isArray(n.token_data) ||
      !Array.isArray(n.embedding)
    )
      throw new Error("Invalid sequence record");
    if (config.objective === "xed") validateXed(n, config);
    if (i === 0) {
      if (
        n.parent !== null ||
        n.tokens !== 0 ||
        n.logp !== 0 ||
        n.text !== "" ||
        n.status !== 0 ||
        n.token_data.length ||
        n.embedding.length
      )
        throw new Error("Invalid root");
      continue;
    }
    if (!Number.isInteger(n.parent) || n.parent < 0 || n.parent >= i)
      throw new Error("Invalid trace ancestry");
    const parent = data.nodes[n.parent];
    if (
      data.version === 3 &&
      (!Number.isInteger(n.requested_tokens) ||
        n.requested_tokens < 1 ||
        n.requested_tokens >
          Math.min(n.duration, config.sequence_tokens - parent.tokens))
    )
      throw new Error("Invalid requested-token allowance");
    if (
      n.tokens !== parent.tokens + n.token_data.length ||
      n.tokens > config.sequence_tokens ||
      !n.text.startsWith(parent.text) ||
      parent.status !== 0 ||
      !Number.isInteger(n.duration) ||
      n.duration < 1 ||
      n.duration > 4096 ||
      !Number.isInteger(n.action) ||
      n.action < 0 ||
      n.action >= 2 ** 24 ||
      n.actual_tokens !== n.token_data.length
    )
      throw new Error("Invalid token counts or prefix");
    if (
      n.token_data.some(
        (t) =>
          typeof t.text !== "string" ||
          !Number.isFinite(t.logprob) ||
          t.logprob > 0 ||
          t.logprob <= -9999,
      )
    )
      throw new Error("Invalid token probabilities");
    parseCompletion(
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
      n.requested_tokens ??
        Math.min(n.duration, config.sequence_tokens - parent.tokens),
    );
    const expectedStatus =
      n.finish_reason === "stop"
        ? 1
        : n.tokens === config.sequence_tokens
          ? 2
          : 0;
    if (n.status !== expectedStatus)
      throw new Error("Invalid sequence termination");
    if (
      data.version >= 2 &&
      (!Number.isFinite(n.reward) ||
        Math.abs(
          n.reward - (objective(n, config) - objective(parent, config)),
        ) > 1e-9 ||
        !Number.isInteger(n.birth_step) ||
        n.birth_step < 1 ||
        n.birth_step > data.snapshots.length)
    )
      throw new Error("Invalid generation reward or iteration");
    const sum = parent.logp + n.token_data.reduce((x, t) => x + t.logprob, 0);
    if (Math.abs(sum - n.logp) > 1e-8 * Math.max(1, Math.abs(sum)))
      throw new Error("Invalid cumulative log probability");
    if (
      n.embedding.length !== dimensions ||
      n.embedding.some((x) => !Number.isFinite(x)) ||
      (n.tokens > 0 && !n.embedding.some((x) => x !== 0))
    )
      throw new Error("Invalid recorded embedding");
  }
  let lastCount = 1;
  for (const [snapshotIndex, s] of data.snapshots.entries()) {
    const priorCount = lastCount;
    if (
      !Number.isInteger(s.node_count) ||
      s.node_count < lastCount ||
      s.node_count > data.nodes.length ||
      !Array.isArray(s.walkers) ||
      !s.walkers.length ||
      s.walkers.length > 4096
    )
      throw new Error("Invalid recorded population");
    if (
      (s.step !== undefined && s.step !== snapshotIndex + 1) ||
      !Number.isInteger(s.iteration) ||
      s.iteration < 0
    )
      throw new Error("Invalid recorded iteration");
    if (data.version >= 2) {
      const prior =
        data.snapshots[snapshotIndex - 1]?.walkers ??
        (config.algorithm === "wave"
          ? s.walkers.map((w) => ({ ...w, node: 0 }))
          : []);
      if (!Array.isArray(s.decisions) || s.decisions.length !== prior.length)
        throw new Error("Missing decision diagnostics");
      for (const [slot, d] of s.decisions.entries()) {
        const source = (id) =>
          id === null || (Number.isInteger(id) && id >= 0 && id < priorCount);
        if (
          !Number.isInteger(d.companion_slot) ||
          !Number.isInteger(d.donor_slot) ||
          d.slot !== slot ||
          d.iteration !== s.iteration ||
          !source(d.evaluated) ||
          !source(d.companion) ||
          !source(d.donor) ||
          d.evaluated !== prior[slot]?.node ||
          d.companion !== prior[d.companion_slot]?.node ||
          d.donor !== prior[d.donor_slot]?.node ||
          d.result !== s.walkers[slot]?.node
        )
          throw new Error("Invalid decision sequence attribution");
        for (const key of [
          "distance",
          "distance_norm",
          "reward_norm",
          "other",
          "fitness",
          "donor_fitness",
          "clone_score",
          "draw",
        ])
          if (!Number.isFinite(d[key]))
            throw new Error("Invalid decision metric");
        if (
          d.distance < 0 ||
          d.draw < 0 ||
          d.draw > 1 ||
          d.distance_norm < 0 ||
          d.reward_norm < 0
        )
          throw new Error("Invalid decision range");
        for (const key of [
          "alive",
          "normalization_leaf",
          "leaf",
          "donor_protected",
          "best_protected",
          "elite_protected",
          "invalid_donor",
          "wanted",
          "cloned",
        ])
          if (typeof d[key] !== "boolean")
            throw new Error("Invalid decision outcome");
        if (
          d.cloned !== s.walkers[slot].cloned ||
          d.fitness !== s.walkers[slot].fitness ||
          d.donor_fitness !== s.decisions[d.donor_slot]?.fitness ||
          d.wanted !== (d.clone_score > d.draw || !d.alive) ||
          d.cloned !==
            (d.wanted &&
              d.leaf &&
              !d.donor_protected &&
              !d.best_protected &&
              !d.elite_protected &&
              !d.invalid_donor)
        )
          throw new Error("Inconsistent decision outcome");
      }
    }
    if (data.version >= 2)
      for (let i = priorCount; i < s.node_count; i++)
        if (data.nodes[i].birth_step !== snapshotIndex + 1)
          throw new Error("Invalid generation iteration");
    lastCount = s.node_count;
    for (const [slot, w] of s.walkers.entries())
      if (
        w.slot !== slot ||
        !Number.isInteger(w.parentSlot) ||
        w.parentSlot < 0 ||
        w.parentSlot >= s.walkers.length ||
        typeof w.alive !== "boolean" ||
        typeof w.leaf !== "boolean" ||
        typeof w.cloned !== "boolean" ||
        !Number.isFinite(w.score) ||
        !Number.isFinite(w.fitness) ||
        (w.node !== null &&
          (!Number.isInteger(w.node) || w.node < 0 || w.node >= s.node_count))
      )
        throw new Error("Missing population node");
  }
  if (data.version === 3) {
    if (lastCount !== data.nodes.length)
      throw new Error("Uncommitted generation nodes");
    let priorRun = null;
    for (const [i, snapshot] of data.snapshots.entries()) {
      if (priorRun?.stop_reason)
        throw new Error("Snapshot follows an ended run");
      const nodes = data.nodes.slice(0, snapshot.node_count);
      validateRun(
        snapshot.run,
        config,
        nodes,
        nodes.reduce((total, node) => total + node.token_data.length, 0),
      );
      if (
        snapshot.run.stop_reason !==
        stoppingReason(config, snapshot.run, snapshot, nodes, i + 1)
      )
        throw new Error("Invalid run stopping reason");
      priorRun = snapshot.run;
    }
    const generated = data.attempts.reduce((total, attempt) => {
      if (!Array.isArray(attempt.token_data))
        throw new Error("Invalid accepted token archive");
      return total + attempt.token_data.length;
    }, 0);
    // Large benchmark exports may omit request provenance, which remains in
    // their journal. The committed node total is still a verified lower bound.
    validateRun(
      data.run,
      config,
      data.nodes,
      data.attempts.length ? generated : undefined,
    );
    if (data.run.stop_reason !== (priorRun?.stop_reason ?? null))
      throw new Error("Run ending does not match its committed boundary");
  }
  // Explicitly return known fields, never treating imported settings as executable code.
  return {
    format: data.format,
    version: data.version,
    engine: data.engine,
    config,
    metadata: data.metadata,
    nodes: data.nodes,
    snapshots: data.snapshots,
    requests: data.requests,
    attempts: data.attempts,
    errors: data.errors,
    ...(data.version === 3 ? { run: data.run } : {}),
  };
}
function validateRun(run, config, nodes, generated) {
  const expected = initialRun(config);
  const ids = nodes.filter((n) => n.status === 1).map((n) => n.id);
  const committed = nodes.reduce((sum, n) => sum + n.token_data.length, 0);
  if (
    !run ||
    run.completion_target !== expected.completion_target ||
    run.token_budget !== expected.token_budget ||
    !Number.isSafeInteger(run.generated_tokens) ||
    run.generated_tokens < committed ||
    run.generated_tokens > run.token_budget ||
    (generated !== undefined && run.generated_tokens !== generated) ||
    JSON.stringify(run.eos_node_ids) !== JSON.stringify(ids) ||
    ![
      null,
      "eos_target",
      "token_budget",
      "no_active_branches",
      "iteration_limit",
    ].includes(run.stop_reason)
  )
    throw new Error("Invalid recorded run progress");
}
export { objective };
