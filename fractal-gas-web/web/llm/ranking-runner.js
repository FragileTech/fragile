import { OpenRouter, mapConcurrent } from "./openrouter.js";
import { prepareJudge } from "./grading.js";
import {
  applyRankingEvent,
  pairwiseBody,
  resultKey,
  pairComplete,
  validateVerdict,
  nextTrainingBatch,
} from "./ranking-data.js";
import { fitRanking } from "./ranking-statistics.js";
import { random, shuffle } from "./ranking-math.js";

export class RankingController {
  constructor(
    key,
    session,
    store,
    {
      fetchImpl,
      sleep,
      onStatus = () => {},
      fit = async (s) => fitRanking(s, { draws: 0, sensitivity: false }),
    } = {},
  ) {
    this.session = session;
    this.store = store;
    this.onStatus = onStatus;
    this.fit = fit;
    this.abort = new AbortController();
    this.tail = Promise.resolve();
    this.paused = false;
    this.key = key;
    this.received = new Map();
    this.api = new OpenRouter(key, {
      fetchImpl,
      sleep,
      signal: this.abort.signal,
      onProviderAttempt: (e) =>
        this.emit("attempt", this.clean(e), () => {
          if (session.attempts.length >= session.config.budget)
            throw Error("Ranking request budget exhausted");
        }),
      onProviderResponse: (e) => {
        this.received.set(
          `${e.pair_id}/${e.orientation}`,
          this.clean(e.response),
        );
      },
      onRequest: (e) =>
        this.emit("request", {
          ...this.clean(e),
          response:
            typeof e.status === "number"
              ? (this.received.get(`${e.pair_id}/${e.orientation}`) ?? null)
              : null,
        }),
    });
    this.fetchImpl = fetchImpl;
  }
  clean(value) {
    return typeof value === "string"
      ? value.replaceAll(this.key, "[redacted]")
      : Array.isArray(value)
        ? value.map((v) => this.clean(v))
        : value && typeof value === "object"
          ? Object.fromEntries(
              Object.entries(value).map(([k, v]) => [k, this.clean(v)]),
            )
          : value;
  }
  emit(type, payload, guard) {
    const task = this.tail.then(async () => {
      if (this.storageError) throw this.storageError;
      guard?.();
      const e = {
        seq: this.session.seq + 1,
        time: Date.now(),
        type,
        payload: structuredClone(payload),
      };
      try {
        await this.store.append(e);
      } catch (error) {
        applyRankingEvent(this.session, e);
        this.storageError = error;
        this.stop();
        throw error;
      }
      applyRankingEvent(this.session, e);
      this.onStatus(this.session);
      return e;
    });
    this.tail = task.catch(() => {});
    return task;
  }
  pause() {
    this.paused = true;
    this.onStatus(this.session, "Paused after in-flight requests");
  }
  continue() {
    this.paused = false;
    this.wake?.();
  }
  stop() {
    this.abort.abort(Error("Ranking stopped"));
    this.continue();
  }
  async gate() {
    while (this.paused && !this.abort.signal.aborted)
      await new Promise((resolve) => {
        const prior = this.wake;
        this.wake = () => {
          prior?.();
          resolve();
        };
      });
    this.abort.signal.throwIfAborted();
  }
  async evaluate(pairs, { retry = false } = {}) {
    const work = [];
    for (const pair of pairs)
      for (const orientation of [0, 1]) {
        const key = resultKey(pair, orientation),
          r = this.session.results[key];
        if (r?.status === "valid") continue;
        if (r && !retry) continue;
        work.push({ pair, orientation, key });
      }
    await mapConcurrent(
      work,
      2,
      async ({ pair, orientation, key }) => {
        await this.gate();
        if (pair.a === pair.b) {
          await this.emit("result", {
            key,
            result: {
              status: "valid",
              basis: "exact_text_identity",
              verdict: Object.fromEntries(
                [
                  "correctness",
                  "relevance",
                  "completeness",
                  "clarity",
                  "overall",
                ].map((k) => [
                  k,
                  {
                    verdict: "tie",
                    explanation:
                      "Identical prompt/answer content; no judge request.",
                  },
                ]),
              ),
            },
          });
          return;
        }
        if (this.session.attempts.length >= this.session.config.budget) return;
        const started_at = Date.now();
        let raw = null;
        try {
          await this.emit("result", {
            key,
            result: { status: "running", started_at },
          });
          raw = this.clean(
            await this.api.request(
              "chat/completions",
              pairwiseBody(this.session, pair, orientation),
              {
                phase: "pairwise_ranking",
                ranking_session_id: this.session.id,
                pair_id: pair.id,
                orientation,
              },
            ),
          );
          await this.emit("result", {
            key,
            result: {
              status: "received",
              raw,
              started_at,
              ended_at: Date.now(),
            },
          });
          const choice = raw.choices?.[0];
          if (
            raw.choices?.length !== 1 ||
            choice.finish_reason !== "stop" ||
            typeof choice.message?.content !== "string"
          )
            throw Error("Pairwise judge response was missing or truncated");
          const verdict = validateVerdict(JSON.parse(choice.message.content));
          await this.emit("result", {
            key,
            result: {
              status: "valid",
              verdict,
              raw,
              started_at,
              ended_at: Date.now(),
            },
          });
        } catch (e) {
          if (this.storageError) {
            const received = this.received.get(key);
            if (received && this.session.results[key]?.status !== "valid")
              applyRankingEvent(this.session, {
                seq: this.session.seq + 1,
                time: Date.now(),
                type: "result",
                payload: {
                  key,
                  result: {
                    status: "received",
                    raw: received,
                    started_at,
                    ended_at: Date.now(),
                    storage_error: this.clean(this.storageError.message),
                  },
                },
              });
            throw e;
          }
          if (this.session.results[key]?.status !== "valid")
            await this.emit("result", {
              key,
              result: {
                status: this.abort.signal.aborted ? "interrupted" : "failed",
                error: this.clean(e.message),
                raw,
                started_at,
                ended_at: Date.now(),
              },
            });
        }
      },
      this.abort.signal,
    );
  }
  async run({ retry = false } = {}) {
    const execute = async () => {
      const s = this.session;
      if (s.status === "completed") return s;
      // Public route revalidation is independent from the paid POST budget.
      const checked = await prepareJudge(this.key, s.profile, {
        fetchImpl: this.fetchImpl,
        signal: this.abort.signal,
      });
      if (
        checked.profile.model !== s.profile.model ||
        checked.profile.provider !== s.profile.provider
      )
        throw Error("Saved ranking route changed; start a separate session");
      await this.emit("status", "running");
      try {
        if (!s.frozen_training) {
          if (!s.pairs.some((p) => p.partition === "training"))
            await this.emit("schedule", s.plan.initial);
          await this.evaluate(
            s.pairs.filter((p) => p.partition === "training"),
            { retry },
          );
          while (s.attempts.length < s.config.budget) {
            await this.gate();
            const batch = nextTrainingBatch(s, await this.fit(s));
            if (!batch.length) break;
            await this.emit("schedule", batch);
            await this.evaluate(batch);
          }
          if (
            s.pairs
              .filter((p) => p.partition === "training")
              .some((p) => !pairComplete(s, p)) &&
            s.attempts.length < s.config.budget
          ) {
            await this.emit("status", "needs_retry");
            return s;
          }
          const finalFit = await this.fit(s);
          await this.gate();
          await this.emit("freeze", {
            keys: s.pairs
              .filter((p) => p.partition === "training")
              .flatMap((p) => [0, 1].map((o) => resultKey(p, o)))
              .filter((key) => s.results[key]?.status === "valid"),
            at_seq: s.seq,
            model_version: s.model.version,
            fits: Object.fromEntries(
              Object.entries(finalFit).map(([criterion, fit]) => [
                criterion,
                {
                  theta: fit.theta,
                  converged: fit.converged,
                  iterations: fit.iterations,
                },
              ]),
            ),
          });
        }
        // Holdouts are not requested until training evidence has been frozen durably.
        const heldout = [...s.plan.audit_pairs, ...s.plan.validation];
        await this.emit("schedule", heldout);
        await this.evaluate(heldout, { retry });
        await this.emit(
          "status",
          s.pairs.every((p) => pairComplete(s, p))
            ? "completed"
            : s.attempts.length >= s.config.budget
              ? "budget_exhausted"
              : "needs_retry",
        );
      } catch (e) {
        if (!this.storageError)
          await this.emit(
            "status",
            this.abort.signal.aborted ? "stopped" : "failed",
          );
        if (!this.abort.signal.aborted || this.storageError) throw e;
      }
      return s;
    };
    return this.store.lock ? this.store.lock(execute) : execute();
  }
}

export function createIndependentAudit(
  session,
  { kind = "model", profile = null, endpoint = null, budget = 80 } = {},
) {
  if (!["model", "human"].includes(kind))
    throw Error("Invalid independent audit kind");
  if (kind === "model" && (!profile || profile.model === session.profile.model))
    throw Error("Select an independent second judge model");
  if (!Number.isSafeInteger(budget) || budget < 2 || budget > 10000)
    throw Error("Invalid audit request budget");
  const pairs = shuffle(
    session.pairs.filter((p) => pairComplete(session, p) && p.a !== p.b),
    random(`${session.config.seed}:independent-audit`),
  ).slice(0, 20);
  if (!pairs.length) throw Error("Complete primary pairs before auditing");
  return {
    id: crypto.randomUUID(),
    kind,
    created_at: Date.now(),
    profile,
    endpoint,
    pairs,
    results: {},
    attempts: [],
    requests: [],
    status: "ready",
    budget,
  };
}
export async function runIndependentAudit(
  primary,
  audit,
  key,
  { fetchImpl, onStatus = () => {}, onController = () => {}, signal } = {},
) {
  const temporary = {
    ...primary.session,
    ...structuredClone(audit),
    config: { ...primary.session.config, budget: audit.budget },
    seq: 0,
    events: [],
    audits: [],
    frozen_training: null,
  };
  const store = {
    append: async (event) => {
      const next = structuredClone(temporary);
      applyRankingEvent(next, event);
      const saved = {
        ...audit,
        results: next.results,
        attempts: next.attempts,
        requests: next.requests,
        status: next.status,
      };
      await primary.emit("audit", saved);
      onStatus(saved);
    },
  };
  const runner = new RankingController(key, temporary, store, { fetchImpl });
  onController(runner);
  const abort = () => runner.stop();
  signal?.addEventListener("abort", abort, { once: true });
  try {
    signal?.throwIfAborted();
    const checked = await prepareJudge(key, audit.profile, {
      fetchImpl,
      signal: runner.abort.signal,
    });
    if (
      checked.profile.model !== audit.profile.model ||
      checked.profile.provider !== audit.profile.provider
    )
      throw Error("Independent audit route changed");
    await runner.emit("status", "running");
    await runner.evaluate(audit.pairs, { retry: true });
    await runner.emit(
      "status",
      audit.pairs.every((p) => pairComplete(temporary, p))
        ? "completed"
        : temporary.attempts.length >= audit.budget
          ? "budget_exhausted"
          : "needs_retry",
    );
  } catch (error) {
    if (!runner.storageError)
      await runner.emit(
        "status",
        runner.abort.signal.aborted ? "stopped" : "failed",
      );
    throw error;
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}
