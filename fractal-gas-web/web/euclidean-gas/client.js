export class GasClient {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    this.nextId = 1;
    this.pending = new Map();
    this.worker.onmessage = ({ data }) => {
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);
      data.ok
        ? entry.resolve(data.result)
        : entry.reject(new Error(data.error));
    };
    this.worker.onerror = (event) => {
      this.fail(
        new Error(
          event.message || "Rust worker failed; reset the page to restart it.",
        ),
      );
    };
  }
  request(type, payload, transfers = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, payload }, transfers);
    });
  }
  fail(error) {
    for (const entry of this.pending.values()) entry.reject(error);
    this.pending.clear();
  }
  dispose() {
    this.worker.terminate();
    this.fail(new Error("Worker disposed"));
  }
}
