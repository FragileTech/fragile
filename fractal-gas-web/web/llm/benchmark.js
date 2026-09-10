import { TogetherScorer } from "./scoring.js";
import { objective, embeddingText } from "./config.js";
import { OpenRouter, mapConcurrent } from "./openrouter.js";
import { Recording } from "./recording.js";
import { TokenEnvironment } from "./environment.js";
import { formatRunProgress } from "./run-control.js";
import { Journal, collectRuns } from "./benchmark-data.js";

const rootNode = () => ({
  id: 0,
  parent: null,
  text: "",
  tokens: 0,
  logp: 0,
  status: 0,
  embedding: [],
  token_data: [],
});
function redactor(key) {
  return (value) =>
    JSON.parse(
      JSON.stringify(value, (_k, v) =>
        typeof v === "string" ? v.replaceAll(key, "[redacted]") : v,
      ),
    );
}
export class BenchmarkRunner {
  constructor(
    store,
    key,
    {
      fetchImpl,
      togetherKey,
      nativeFactory = async (c, t) =>
        (await import("./native.js")).NativeLlm.create(c, t),
      onStatus = () => {},
      onCommitted = () => {},
    } = {},
  ) {
    if (!key) throw Error("Enter your OpenRouter API key");
    this.journal = new Journal(store, {
      redact: (v) => redactor(key)(togetherKey ? redactor(togetherKey)(v) : v),
      onCommitted,
    });
    this.togetherKey = togetherKey;
    this.scoringFetch = fetchImpl;
    this.store = store;
    this.nativeFactory = nativeFactory;
    this.onStatus = onStatus;
    this.abort = new AbortController();
    this.paused = false;
    this.running = false;
    this.context = { phase: "preflight", session_id: crypto.randomUUID() };
    this.api = new OpenRouter(key, {
      signal: this.abort.signal,
      fetchImpl,
      onRequestStart: (r) => this.emit("request_start", r),
      onGenerationResponse: (r) => this.emit("provider_response", r),
      onRequest: async (r) => {
        await this.emit("request", r);
        this.record?.append("requests", r);
      },
    });
  }
  emit(type, payload) {
    return this.journal.append(type, payload, this.context);
  }
  pause() {
    this.paused = true;
    this.onStatus("Pausing after current generation boundary…");
  }
  continue() {
    this.paused = false;
    this.wake?.();
  }
  stop() {
    this.abort.abort(new Error("Stopped"));
    this.wake?.();
  }
  async boundary() {
    this.abort.signal.throwIfAborted();
    while (this.paused) {
      this.onStatus("Paused — progress saved");
      await new Promise((resolve) => {
        this.wake = resolve;
      });
      this.abort.signal.throwIfAborted();
    }
    this.wake = null;
  }
  async generate(config, prefix, count, transition) {
    const result = await this.api.generate(config, prefix, count, {
      transition,
    });
    await this.emit("accepted", { transition, count, result });
    this.generated += result.token_data.length;
    return result;
  }
  async run({ retryIncomplete = false } = {}) {
    if (this.running) throw Error("Benchmark is already running");
    this.running = true;
    let active = false;
    try {
      const events = await this.store.readEvents();
      const { runs, metadata: savedMetadata } = collectRuns(
        this.store.manifest,
        events,
      );
      const { settings, method_order } = this.store.manifest;
      const pending = [];
      for (let trial = 0; trial < settings.repetitions; trial++)
        for (const method of method_order) {
          const previous = [...runs.values()].filter(
            (r) => r.trial === trial && r.method === method,
          );
          if (previous.some((r) => r.status === "completed")) continue;
          if (previous.length && !retryIncomplete)
            throw Error(
              "Unfinished method found. Explicitly retry it to continue.",
            );
          pending.push({
            trial,
            method,
            attempt: 1 + previous.reduce((n, r) => Math.max(n, r.attempt), 0),
          });
        }
      if (!pending.length) {
        this.onStatus("Completed — all methods are saved");
        return;
      }
      // Readiness checks are outside every run and never enter matched token budgets.
      await this.emit("session_start", { retry_incomplete: retryIncomplete });
      this.onStatus("Checking the shared model route…");
      if (settings.config.objective === "xed") {
        this.scorer = new TogetherScorer(this.togetherKey, {
          signal: this.abort.signal,
          concurrency: settings.config.concurrency,
          fetchImpl: this.scoringFetch,
          onRequestStart: (r) => this.emit("request_start", r),
          onRequest: async (r) => {
            await this.emit("request", r);
            this.record?.append("requests", r);
          },
        });
      }
      const scoring = this.scorer
        ? await this.scorer.prepare(settings.config)
        : null;
      const metadata = await this.api.prepare(settings.config, {
        pinnedProvider: savedMetadata?.provider,
      });
      if (scoring) metadata.scoring = scoring;
      const zeroConfig = {
        ...settings.config,
        temperature: 0,
        prompt:
          "Continue this list with one word per numbered line:\n1. amber\n2. blue\n3.",
      };
      metadata.temperature_zero_probe = await this.api.generate(
        zeroConfig,
        "",
        8,
      );
      await this.emit("preflight", metadata);
      const budgets = new Map(
        [...runs.values()]
          .filter((r) => r.method === "fractal" && r.status === "completed")
          .map((r) => [r.trial, r.generated_tokens]),
      );
      for (const p of pending) {
        await this.boundary();
        const config = {
          ...settings.config,
          seed: (settings.config.seed + p.trial) % 2147483648,
          ...(p.method === "temperature_zero" ? { temperature: 0 } : {}),
        };
        this.context = {
          phase: "generation",
          session_id: this.context.session_id,
          run_id: `${p.trial}:${p.method}:${p.attempt}`,
        };
        this.record = null;
        this.generated = 0;
        // Reset caches between methods so measured embedding work is attributable.
        this.api.embeddingCache = new Map();
        this.scorer?.cache.clear();
        const tokenBudget =
          p.method === "independent_tokens" ? budgets.get(p.trial) : null;
        if (
          p.method === "independent_tokens" &&
          !Number.isSafeInteger(tokenBudget)
        )
          throw Error("Missing completed Fractal budget");
        await this.emit("run_start", {
          ...p,
          config,
          metadata,
          root: rootNode(),
          token_budget: tokenBudget,
          ...(p.method === "fractal" ? { recording_version: 3 } : {}),
        });
        active = true;
        this.onStatus(
          `Trial ${p.trial + 1}/${settings.repetitions} · ${p.method} · attempt ${p.attempt}`,
        );
        const result =
          p.method === "fractal"
            ? await this.fractal(config, metadata)
            : await this.independent(config, p.method, tokenBudget);
        await this.emit("run_end", {
          ...result,
          generated_tokens: this.generated,
        });
        active = false;
        if (result.status !== "completed")
          throw Error(`Method incomplete: ${result.stop_reason}`);
        if (p.method === "fractal") budgets.set(p.trial, this.generated);
      }
      this.context = {
        phase: "preflight",
        session_id: this.context.session_id,
      };
      await this.emit("session_end", { status: "completed" });
      this.onStatus("Completed — benchmark saved");
    } catch (error) {
      const status = this.abort.signal.aborted ? "stopped" : "failed";
      try {
        if (active) {
          await this.emit("error", { message: error.message });
          await this.emit("run_end", {
            status,
            stop_reason: error.message,
            generated_tokens: this.generated,
            ...(this.record
              ? {
                  run: {
                    ...this.record.data.run,
                    generated_tokens: this.generated,
                  },
                }
              : {}),
          });
        }
        this.context = {
          phase: "preflight",
          session_id: this.context.session_id,
        };
        await this.emit("session_end", { status, message: error.message });
      } catch {
        /* A failed storage write latches the journal; previously saved events survive. */
      }
      throw error;
    } finally {
      this.record = null;
      this.running = false;
    }
  }
  async fractal(config, metadata) {
    const record = (this.record = new Recording(config));
    record.metadata(metadata);
    // The original environment still performs transition caching and atomic population commits.
    const adapter = {
      signal: this.abort.signal,
      dimensions: metadata.dimensions,
      generate: (c, prefix, count, transition) =>
        this.generate(c, prefix, count, transition),
      embed: (model, texts) => this.api.embed(model, texts),
    };
    const env = new TokenEnvironment(config, adapter, record, this.scorer);
    const native = await this.nativeFactory(
      { ...config, dimensions: metadata.dimensions },
      (r) => env.transition(r),
    );
    try {
      for (let step = 0; step < config.iterations; step++) {
        await this.boundary();
        const before = record.data.nodes.length;
        const snapshot = await native.advance();
        this.abort.signal.throwIfAborted();
        env.commit({ ...snapshot, time: Date.now() });
        await this.emit("generation", {
          nodes: record.data.nodes.slice(before),
          snapshot: record.data.snapshots.at(-1),
        });
        this.onStatus(`Fractal · ${formatRunProgress(record.data.run)}`);
        if (record.data.run.stop_reason)
          return {
            status: "completed",
            stop_reason: record.data.run.stop_reason,
            completion_success: record.data.run.stop_reason === "eos_target",
            run: record.data.run,
          };
      }
      throw Error("Fractal run did not record its iteration limit");
    } finally {
      native.close();
    }
  }
  async independent(config, method, budget) {
    const nodes = [rootNode()],
      trajectories = [];
    const width = method === "temperature_zero" ? 1 : config.walkers;
    let active = [],
      step = 0,
      emptyRounds = 0;
    const start = async () => {
      const t = { id: trajectories.length, node: 0 };
      await this.emit("trajectory_start", { id: t.id });
      trajectories.push(t);
      return t;
    };
    if (budget === 0)
      return { status: "completed", stop_reason: "token_budget" };
    for (let i = 0; i < width; i++) active.push(await start());
    while (active.length) {
      await this.boundary();
      let remaining = budget === null ? Infinity : budget - this.generated;
      if (remaining === 0)
        return { status: "completed", stop_reason: "token_budget" };
      const jobs = [];
      // Reserve the entire wave before dispatch; unused reservations return next round.
      for (const t of active) {
        const parent = nodes[t.node];
        const count = Math.min(
          config.chunk_tokens,
          config.sequence_tokens - parent.tokens,
          remaining,
        );
        if (count > 0) {
          jobs.push({ t, parent, count });
          remaining -= count;
        }
      }
      const before = this.generated;
      const results = await mapConcurrent(
        jobs,
        config.concurrency,
        async ({ t, parent, count }) => {
          const transition = {
            source: parent.id,
            trajectory_id: t.id,
            duration: count,
            step: step + 1,
          };
          const result = await this.generate(
            config,
            parent.text,
            count,
            transition,
          );
          const tokens = parent.tokens + result.token_data.length;
          const n = {
            parent: parent.id,
            trajectory_id: t.id,
            text: parent.text + result.text,
            tokens,
            logp: parent.logp + result.logp,
            status:
              result.finish_reason === "stop"
                ? 1
                : tokens === config.sequence_tokens
                  ? 2
                  : 0,
            token_data: result.token_data,
            request_id: result.request_id,
            logical_request_id: result.logical_request_id,
            finish_reason: result.finish_reason,
            duration: count,
            actual_tokens: result.token_data.length,
            birth_step: step + 1,
          };
          if (config.objective === "xed")
            n.xed = await this.scorer.score(config, n.text, {
              source: parent.id,
            });
          n.reward = objective(n, config) - objective(parent, config);
          return n;
        },
        this.abort.signal,
      );
      const vectors = await this.api.embed(
        config.embedding_model,
        results.filter((n) => n.text).map((n) => embeddingText(config, n.text)),
      );
      let embedding = 0;
      for (const [i, n] of results.entries()) {
        n.id = nodes.length + i;
        n.embedding = n.text
          ? vectors[embedding++]
          : new Array(this.api.dimensions).fill(0);
      }
      this.abort.signal.throwIfAborted();
      const updated = new Map(jobs.map((job, i) => [job.t.id, results[i].id]));
      const snapshot = {
        step: ++step,
        node_count: nodes.length + results.length,
        trajectories: trajectories.map((t) => ({
          id: t.id,
          node: updated.get(t.id) ?? t.node,
        })),
      };
      await this.emit("generation", { nodes: results, snapshot });
      nodes.push(...results);
      for (const t of trajectories) t.node = updated.get(t.id) ?? t.node;
      emptyRounds = before === this.generated ? emptyRounds + 1 : 0;
      active = active.filter((t) => nodes[t.node].status === 0);
      if (budget !== null) {
        if (this.generated === budget)
          return { status: "completed", stop_reason: "token_budget" };
        if (emptyRounds >= 3)
          return { status: "incomplete", stop_reason: "no_token_progress" };
        while (active.length < width) active.push(await start());
      }
    }
    return { status: "completed", stop_reason: "all_terminated" };
  }
}
