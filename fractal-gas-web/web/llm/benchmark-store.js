import { validateManifest, validateArchive } from "./benchmark-data.js";

export class MemoryBenchmarkStore {
  constructor(header, events = []) {
    this.manifest = validateManifest(header);
    this.events = structuredClone(events);
    this.lastSeq = events.length;
  }
  async append(event) {
    if (event.seq !== this.lastSeq + 1)
      throw Error("Concurrent benchmark writer");
    this.events.push(structuredClone(event));
    this.lastSeq = event.seq;
  }
  async readEvents(afterSeq = 0) {
    return structuredClone(this.events.filter((e) => e.seq > afterSeq));
  }
}
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("fragile-llm-benchmarks", 4);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("ranking_sessions"))
        db.createObjectStore("ranking_sessions", { keyPath: "id" });
      if (!db.objectStoreNames.contains("ranking_events"))
        db.createObjectStore("ranking_events", {
          keyPath: ["session_id", "seq"],
        });
      if (!db.objectStoreNames.contains("benchmarks"))
        db.createObjectStore("benchmarks", { keyPath: "id" });
      if (!db.objectStoreNames.contains("events"))
        db.createObjectStore("events", { keyPath: ["benchmark_id", "seq"] });
      if (!db.objectStoreNames.contains("comparisons"))
        db.createObjectStore("comparisons", { keyPath: "id" });
      if (!db.objectStoreNames.contains("comparison_index")) {
        const index = db.createObjectStore("comparison_index", {
          keyPath: "id",
        });
        request.transaction.objectStore("comparisons").openCursor().onsuccess =
          (e) => {
            const cursor = e.target.result;
            if (!cursor) return;
            const { id, created_at, source } = cursor.value;
            index.put({ id, created_at, source: { kind: source.kind } });
            cursor.continue();
          };
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () =>
      reject(Error("Close other LLM Lab tabs to open benchmark storage"));
  });
}
export async function transact(names, mode, fn) {
  const db = await openDB();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(names, mode);
      let result;
      tx.oncomplete = () => resolve(result);
      tx.onabort = tx.onerror = () =>
        reject(tx.error ?? Error("Benchmark storage transaction failed"));
      try {
        fn(tx, (value) => {
          result = value;
        });
      } catch (error) {
        tx.abort();
        reject(error);
      }
    });
  } finally {
    db.close();
  }
}
export class BrowserBenchmarkStore {
  constructor(row) {
    this.manifest = row.manifest;
    this.lastSeq = row.lastSeq;
  }
  static async create(header, events = []) {
    header = validateManifest(header);
    validateArchive(header, events);
    const row = {
      id: header.id,
      manifest: header,
      lastSeq: events.length,
      updated_at: Date.now(),
    };
    await transact(["benchmarks", "events"], "readwrite", (tx) => {
      tx.objectStore("benchmarks").add(row);
      for (const event of events) tx.objectStore("events").add(event);
    });
    return new BrowserBenchmarkStore(row);
  }
  static async list() {
    return transact(["benchmarks"], "readonly", (tx, result) => {
      tx.objectStore("benchmarks").getAll().onsuccess = (e) =>
        result(e.target.result);
    });
  }
  static async open(id) {
    const row = await transact(["benchmarks"], "readonly", (tx, result) => {
      tx.objectStore("benchmarks").get(id).onsuccess = (e) =>
        result(e.target.result);
    });
    if (!row) throw Error("Saved benchmark not found");
    return new BrowserBenchmarkStore(row);
  }
  async append(event) {
    await transact(["benchmarks", "events"], "readwrite", (tx) => {
      const table = tx.objectStore("benchmarks");
      table.get(this.manifest.id).onsuccess = (e) => {
        const row = e.target.result;
        if (!row || row.lastSeq !== event.seq - 1) {
          tx.abort();
          return;
        }
        tx.objectStore("events").add(event);
        table.put({ ...row, lastSeq: event.seq, updated_at: Date.now() });
      };
    });
    this.lastSeq = event.seq;
  }
  async readEvents(afterSeq = 0) {
    return transact(["events"], "readonly", (tx, result) => {
      tx
        .objectStore("events")
        .getAll(
          IDBKeyRange.bound(
            [this.manifest.id, afterSeq + 1],
            [this.manifest.id, Number.MAX_SAFE_INTEGER],
          ),
        ).onsuccess = (e) => result(e.target.result);
    });
  }
}
export class BrowserComparisonStore {
  static async save(report) {
    return transact(["comparisons", "comparison_index"], "readwrite", (tx) => {
      tx.objectStore("comparisons").put(report);
      tx.objectStore("comparison_index").put({
        id: report.id,
        created_at: report.created_at,
        source: { kind: report.source.kind },
      });
    });
  }
  static async list() {
    return transact(["comparison_index"], "readonly", (tx, result) => {
      tx.objectStore("comparison_index").getAll().onsuccess = (e) =>
        result(e.target.result);
    });
  }
  static async open(id) {
    return transact(["comparisons"], "readonly", (tx, result) => {
      tx.objectStore("comparisons").get(id).onsuccess = (e) =>
        result(e.target.result);
    });
  }
}
