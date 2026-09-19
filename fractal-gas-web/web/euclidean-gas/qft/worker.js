// Promise-RPC worker of the QFT Simulator. Every scientific operation is a call
// into the compiled Rust spectroscopy session; this file only routes messages.
export const MESSAGE_TYPES = Object.freeze([
  "defaults",
  "capabilities",
  "create",
  "advance",
  "snapshot",
  "analyze",
  "presentation",
  "evidence",
  "checkpoint",
  "restore",
  "import_evidence",
  "import_archive",
  "dispose",
]);
// `SpectroscopySession::advance` accepts 1..=64 updates per call.
export const MAX_ADVANCE = 64;

let compiled;
export function loadWasm() {
  if (!compiled)
    compiled = import("../engine/cpu/gas.js")
      .then(async (module) => {
        await module.default();
        return module;
      })
      .catch((error) => {
        compiled = null;
        throw error;
      });
  return compiled;
}

function bytes(value, what) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  throw new Error(what + " must be binary CBOR data.");
}

// Adapter from the message vocabulary to the wasm API. `load` is injectable so
// the adapter can be exercised without a compiled module.
export function createWasmEngine(load = loadWasm) {
  let session = null,
    imported = null;
  const release = () => {
    session?.free?.();
    session = null;
    imported = null;
  };
  const live = () => {
    if (!session)
      throw new Error("Create or restore a spectroscopy session first.");
    return session;
  };
  return {
    async defaults() {
      return (await load()).spectroscopy_defaults();
    },
    async capabilities(request) {
      return (await load()).spectroscopy_capabilities(JSON.stringify(request));
    },
    async create(request) {
      const wasm = await load();
      release();
      session = await wasm.SpectroscopyExperiment.create(
        JSON.stringify(request),
      );
      return session.snapshot();
    },
    async advance(steps) {
      return live().advance(steps);
    },
    async snapshot() {
      return live().snapshot();
    },
    async analyze(analysis) {
      const json = JSON.stringify(analysis);
      if (session) return session.analyze(json);
      if (imported?.kind === "evidence")
        return (await load()).spectroscopy_analyze(imported.bytes, json);
      if (imported?.kind === "archive")
        return (await load()).spectroscopy_archive(
          JSON.stringify({ ...imported.config, analysis }),
          imported.bytes,
        );
      throw new Error(
        "Create a session or import evidence before requesting an analysis.",
      );
    },
    async presentation(analysis) {
      return live().presentation(JSON.stringify(analysis));
    },
    async evidence() {
      return live().evidence();
    },
    async checkpoint() {
      return live().checkpoint();
    },
    async restore(checkpoint) {
      const wasm = await load();
      const data = bytes(checkpoint, "A session checkpoint");
      release();
      session = await wasm.SpectroscopyExperiment.restore(data);
      return session.snapshot();
    },
    async import_evidence({ bytes: evidence, analysis }) {
      const wasm = await load();
      const data = bytes(evidence, "Spectroscopy evidence");
      const report = await wasm.spectroscopy_analyze(
        data,
        JSON.stringify(analysis),
      );
      release();
      imported = { kind: "evidence", bytes: data };
      return report;
    },
    async import_archive({ bytes: archive, config }) {
      const wasm = await load();
      const data = bytes(archive, "A run archive");
      const report = await wasm.spectroscopy_archive(
        JSON.stringify(config),
        data,
      );
      release();
      imported = { kind: "archive", bytes: data, config };
      return report;
    },
    async dispose() {
      release();
      return null;
    },
  };
}

// Message type -> engine call. The engine is any object with the thirteen
// async methods of MESSAGE_TYPES.
export function createDispatcher(engine) {
  return async function dispatch(type, payload) {
    if (!MESSAGE_TYPES.includes(type))
      throw new Error("Unknown spectroscopy operation: " + String(type));
    if (typeof engine[type] !== "function")
      throw new Error("The engine does not implement " + type + ".");
    if (type === "advance") {
      const steps = payload?.steps;
      if (!Number.isInteger(steps) || steps < 1 || steps > MAX_ADVANCE)
        throw new Error(
          "advance takes an integer number of steps between 1 and " +
            MAX_ADVANCE +
            ".",
        );
      return engine.advance(steps);
    }
    return engine[type](payload);
  };
}

// One request at a time, in arrival order; a failure never blocks the queue.
export function createQueue(dispatch) {
  let queue = Promise.resolve();
  return function handle({ id, type, payload }) {
    const reply = queue.then(async () => {
      try {
        return { id, result: await dispatch(type, payload) };
      } catch (error) {
        return { id, error: String(error?.message || error) };
      }
    });
    queue = reply;
    return reply;
  };
}

export function transferables(reply) {
  const result = reply?.result;
  return result instanceof Uint8Array &&
    result.byteOffset === 0 &&
    result.byteLength === result.buffer.byteLength &&
    result.buffer instanceof ArrayBuffer
    ? [result.buffer]
    : [];
}

export function serve(scope, engine) {
  const handle = createQueue(createDispatcher(engine));
  scope.onmessage = ({ data }) =>
    handle(data).then((reply) =>
      scope.postMessage(reply, transferables(reply)),
    );
  return scope;
}

// Main-thread side of the protocol: an engine-shaped object over a Worker.
export function createClient(worker, onCrash = () => {}) {
  let serial = 0;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    const waiter = pending.get(data.id);
    if (!waiter) return;
    pending.delete(data.id);
    if (data.error) waiter.reject(new Error(data.error));
    else waiter.resolve(data.result);
  };
  worker.onerror = (event) => {
    const error = new Error(
      event?.message || "Could not load the spectroscopy worker. Reload.",
    );
    pending.forEach((waiter) => waiter.reject(error));
    pending.clear();
    onCrash(error);
  };
  const request = (type, payload) =>
    new Promise((resolve, reject) => {
      const id = ++serial;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, type, payload });
    });
  const client = Object.fromEntries(
    MESSAGE_TYPES.map((type) => [type, (payload) => request(type, payload)]),
  );
  client.advance = (steps) => request("advance", { steps });
  return client;
}

if (
  typeof WorkerGlobalScope !== "undefined" &&
  self instanceof WorkerGlobalScope
)
  serve(self, createWasmEngine());
