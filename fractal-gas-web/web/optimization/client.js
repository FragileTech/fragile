export class EngineClient {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    this.pending = new Map();
    this.next = 1;
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
    };
    this.worker.onerror = () => {
      for (const p of this.pending.values())
        p.reject(
          new Error(
            "Optimization engine could not load. Run make optimization-web and reload.",
          ),
        );
      this.pending.clear();
    };
  }
  request(type, fields = {}) {
    return new Promise((resolve, reject) => {
      const id = this.next++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(
        { id, type, ...fields },
        fields.positions ? [fields.positions.buffer] : [],
      );
    });
  }
  dispose() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error("Engine closed"));
    this.pending.clear();
  }
}
