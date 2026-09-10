import { embeddingText, objective } from "./config.js";
import { mapConcurrent } from "./openrouter.js";
import { RunControl } from "./run-control.js";
export class TokenEnvironment {
  constructor(config, api, recording, scorer = null) {
    this.scorer = scorer;
    this.config = Object.freeze({ ...config });
    this.api = api;
    this.recording = recording;
    this.cache = new Map();
    this.pending = [];
    this.control = new RunControl(recording);
  }
  async transition(requests) {
    const config = this.config,
      data = this.recording.data;
    const jobs = new Map(),
      generated = [];
    const rows = await mapConcurrent(
      requests,
      config.concurrency,
      async (request) => {
        const parent = data.nodes[request.source];
        if (!parent) throw new Error("Unknown source sequence");
        if (parent.status) return { node: parent, skipped: false };
        const key = JSON.stringify([
          request.source,
          request.action,
          request.duration,
        ]);
        if (this.cache.has(key))
          return { node: this.cache.get(key), skipped: false };
        if (jobs.has(key)) return jobs.get(key);
        const job = (async () => {
          const limit = Math.min(
            request.duration,
            config.sequence_tokens - parent.tokens,
          );
          if (limit < 1) throw new Error("Cannot extend a capped sequence");
          const count = this.control.reserve(limit);
          if (!count) return { node: parent, skipped: true };
          let result;
          try {
            result = await this.api.generate(
              config,
              parent.text,
              count,
              request,
            );
            this.recording.append("attempts", { ...request, count, ...result });
          } finally {
            this.control.settle(count, result);
          }
          const text = parent.text + result.text,
            tokens = parent.tokens + result.token_data.length;
          const node = {
            parent: parent.id,
            text,
            tokens,
            logp: parent.logp + result.logp,
            status:
              result.finish_reason === "stop"
                ? 1
                : tokens >= config.sequence_tokens
                  ? 2
                  : 0,
            token_data: result.token_data,
            request_id: result.request_id,
            logical_request_id: result.logical_request_id ?? null,
            finish_reason: result.finish_reason,
            action: request.action,
            duration: request.duration,
            requested_tokens: count,
            actual_tokens: result.token_data.length,
          };
          if (config.objective === "xed")
            node.xed = await this.scorer.score(config, text, {
              source: parent.id,
            });
          node.reward = objective(node, config) - objective(parent, config);
          node.birth_step = data.snapshots.length + 1;
          // A no-content model stop is a valid termination, but never an eligible
          // empty solution. Reuse the parent observation in that case.
          generated.push({ key, node });
          return { node, skipped: false };
        })();
        jobs.set(key, job);
        return job;
      },
      this.api.signal,
    );
    // Stable IDs follow source batch order, independent of response arrival order.
    const unique = [
      ...new Set(rows.map((r) => r.node).filter((n) => n.id === undefined)),
    ];
    const texts = unique
      .filter((n) => n.text)
      .map((n) => embeddingText(config, n.text));
    const embeddings = texts.length
      ? await this.api.embed(config.embedding_model, texts)
      : [];
    let e = 0;
    for (const node of unique) {
      node.id = data.nodes.length + this.pending.length;
      node.embedding = node.text
        ? embeddings[e++]
        : new Array(this.api.dimensions).fill(0);
      this.pending.push(node);
    }
    for (const { key, node } of generated) this.cache.set(key, node);
    this.api.signal?.throwIfAborted();
    return rows.map(({ node: n, skipped }) => ({
      id: n.id,
      tokens: n.tokens,
      logp: n.logp,
      utility: objective(n, config),
      status: n.status,
      skipped,
      embedding: n.embedding.length
        ? n.embedding
        : new Array(this.api.dimensions).fill(0),
    }));
  }
  commit(snapshot) {
    this.recording.commit(this.pending, {
      ...snapshot,
      step: this.recording.data.snapshots.length + 1,
      node_count: this.recording.data.nodes.length + this.pending.length,
    });
    this.pending = [];
  }
}
