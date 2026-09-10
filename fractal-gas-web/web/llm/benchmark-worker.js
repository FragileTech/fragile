import { BenchmarkRunner } from "./benchmark.js";
import { manifest } from "./benchmark-data.js";
import { BrowserBenchmarkStore } from "./benchmark-store.js";
let runner,
  busy = false,
  pauseRequested = false,
  stopRequested = false;
const send = (type, value = {}) => postMessage({ type, ...value });
onmessage = async ({ data }) => {
  if (data.type === "pause") {
    pauseRequested = true;
    runner?.pause();
    return;
  }
  if (data.type === "continue") {
    pauseRequested = false;
    runner?.continue();
    return;
  }
  if (data.type === "stop") {
    stopRequested = true;
    runner?.stop();
    return;
  }
  if (!["start", "resume"].includes(data.type) || busy) return;
  busy = true;
  try {
    if (!navigator.locks)
      throw Error(
        "Benchmark execution requires browser storage locks (use localhost or HTTPS)",
      );
    const header = data.type === "start" ? manifest(data.settings) : null;
    const id = header?.id ?? data.id;
    await navigator.locks.request(
      `fgllmbench:${id}`,
      { ifAvailable: true },
      async (lock) => {
        if (!lock)
          throw Error("This benchmark is already running in another tab");
        const store = header
          ? await BrowserBenchmarkStore.create(header)
          : await BrowserBenchmarkStore.open(id);
        send("saved", { id });
        runner = new BenchmarkRunner(store, data.key, {
          togetherKey: data.togetherKey,
          onStatus: (status) => send("status", { status }),
          onCommitted: (event) => {
            if (
              ["generation", "run_start", "run_end", "session_end"].includes(
                event.type,
              )
            )
              send("changed", { id, seq: event.seq });
          },
        });
        if (pauseRequested) runner.pause();
        if (stopRequested) runner.stop();
        await runner.run({ retryIncomplete: data.retryIncomplete ?? false });
      },
    );
  } catch (error) {
    send("status", {
      status: String(error.message)
        .replaceAll(data.key || "\0", "[redacted]")
        .replaceAll(data.togetherKey || "\0", "[redacted]"),
    });
  } finally {
    runner = null;
    busy = pauseRequested = stopRequested = false;
    send("idle");
  }
};
