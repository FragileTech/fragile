// A replacement is not committed until the previous run is durable and the
// replacement worker has initialized successfully. Failed candidates are closed.
export class ConfigurationTransition {
  busy = false;
  async run({ quiesce, save, prepare, commit }) {
    if (this.busy)
      throw new Error("A configuration change is already in progress.");
    this.busy = true;
    let candidate;
    try {
      // Native initialization validates the complete candidate before the
      // running session is paused or any device writes are requested.
      candidate = await prepare();
      await quiesce();
      await save();
      await commit(candidate);
    } catch (error) {
      candidate?.worker?.terminate();
      throw error;
    } finally {
      this.busy = false;
    }
  }
}

export function quiesceWorker(worker, requestId) {
  if (!worker) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      worker.removeEventListener("message", receive);
    };
    const receive = ({ data }) => {
      if (data.type === "quiesced" && data.requestId === requestId) {
        cleanup();
        resolve();
      }
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(
        new Error("The simulation did not pause. Your run has been retained."),
      );
    }, 30000);
    worker.addEventListener("message", receive);
    worker.postMessage({ type: "quiesce", requestId });
  });
}

export function prepareWorker(message) {
  const worker = new Worker(
    new URL("./simulation-worker.js", import.meta.url),
    { type: "module" },
  );
  const messages = [];
  return new Promise((resolve, reject) => {
    const fail = (error) => {
      clearTimeout(timer);
      worker.terminate();
      reject(error);
    };
    const timer = setTimeout(
      () =>
        fail(
          new Error(
            "Preparing the new world timed out. The previous run is still available.",
          ),
        ),
      45000,
    );
    worker.onerror = (e) => fail(new Error(e.message));
    worker.onmessage = (event) => {
      messages.push(event);
      if (event.data.type === "error") fail(new Error(event.data.message));
      if (
        event.data.type ===
        (message.checkpoint ? "checkpoint-restored" : "ready")
      ) {
        clearTimeout(timer);
        resolve({ worker, messages });
      }
    };
    worker.postMessage(message);
  });
}
