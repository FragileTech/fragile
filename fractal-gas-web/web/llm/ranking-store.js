import { transact } from "./benchmark-store.js";
import { applyRankingEvent } from "./ranking-data.js";
export function rankingHeader(session) {
  const h = structuredClone(session);
  h.events = [];
  h.seq = 0;
  h.status = "ready";
  h.pairs = [];
  h.results = {};
  h.requests = [];
  h.attempts = [];
  h.frozen_training = null;
  h.audits = [];
  return h;
}
export class MemoryRankingStore {
  constructor(session) {
    this.header = rankingHeader(session);
    this.events = structuredClone(session.events);
    this.lastSeq = session.seq;
  }
  async append(event) {
    if (event.seq !== this.lastSeq + 1)
      throw Error("Concurrent ranking writer");
    this.events.push(structuredClone(event));
    this.lastSeq = event.seq;
  }
  async read() {
    const s = structuredClone(this.header);
    for (const e of this.events) applyRankingEvent(s, e);
    return s;
  }
}
export class BrowserRankingStore {
  constructor(id, seq = 0) {
    this.id = id;
    this.lastSeq = seq;
  }
  static async create(session, reportId) {
    await transact(
      ["ranking_sessions", "ranking_events"],
      "readwrite",
      (tx) => {
        tx.objectStore("ranking_sessions").add({
          id: session.id,
          report_id: reportId,
          header: rankingHeader(session),
          lastSeq: session.seq,
          created_at: session.created_at,
        });
        for (const e of session.events)
          tx.objectStore("ranking_events").add({
            ...e,
            session_id: session.id,
          });
      },
    );
    return new BrowserRankingStore(session.id, session.seq);
  }
  static async open(id) {
    const row = await transact(
      ["ranking_sessions"],
      "readonly",
      (tx, result) => {
        tx.objectStore("ranking_sessions").get(id).onsuccess = (e) =>
          result(e.target.result);
      },
    );
    if (!row) throw Error("Ranking session not found");
    return new BrowserRankingStore(id, row.lastSeq);
  }
  static async list(reportId) {
    return transact(["ranking_sessions"], "readonly", (tx, result) => {
      tx.objectStore("ranking_sessions").getAll().onsuccess = (e) =>
        result(
          e.target.result
            .filter((r) => r.report_id === reportId)
            .map(({ header, ...r }) => r),
        );
    });
  }
  async append(event) {
    await transact(
      ["ranking_sessions", "ranking_events"],
      "readwrite",
      (tx) => {
        const table = tx.objectStore("ranking_sessions");
        table.get(this.id).onsuccess = (e) => {
          const row = e.target.result;
          if (!row || row.lastSeq !== event.seq - 1) {
            tx.abort();
            return;
          }
          tx.objectStore("ranking_events").add({
            ...event,
            session_id: this.id,
          });
          table.put({ ...row, lastSeq: event.seq });
        };
      },
    );
    this.lastSeq = event.seq;
  }
  async read() {
    return transact(
      ["ranking_sessions", "ranking_events"],
      "readonly",
      (tx, result) => {
        let header, events;
        const finish = () => {
          if (!header || !events) return;
          const s = structuredClone(header);
          for (const { session_id, ...e } of events) applyRankingEvent(s, e);
          result(s);
        };
        tx.objectStore("ranking_sessions").get(this.id).onsuccess = (e) => {
          header = e.target.result?.header;
          finish();
        };
        tx
          .objectStore("ranking_events")
          .getAll(
            IDBKeyRange.bound([this.id, 0], [this.id, Number.MAX_SAFE_INTEGER]),
          ).onsuccess = (e) => {
          events = e.target.result;
          finish();
        };
      },
    );
  }
  async lock(fn) {
    if (!navigator.locks)
      throw Error(
        "This browser does not support the writer lock required for ranking",
      );
    return navigator.locks.request(
      `llm-ranking:${this.id}`,
      { ifAvailable: true },
      async (lock) => {
        if (!lock) throw Error("This ranking is running in another tab");
        const current = await BrowserRankingStore.open(this.id);
        if (current.lastSeq !== this.lastSeq)
          throw Error(
            "Ranking changed in another tab; reload the saved session",
          );
        return fn();
      },
    );
  }
}
