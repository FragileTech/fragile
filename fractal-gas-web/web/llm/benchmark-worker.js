import { pairManifests, runGamePair } from "./game-benchmark.js";
import { BenchmarkRunner } from "./benchmark.js";
import { manifest } from "./benchmark-data.js";
import { BrowserBenchmarkStore } from "./benchmark-store.js";
let runner,
  busy = false,
  pauseRequested = false,
  stopRequested = false,
  pairAbort;
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
    pairAbort?.abort(new Error("Stopped"));
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
    const headers =
      data.type === "start" && data.pair ? pairManifests(data.settings) : null;
    const header =
      headers?.[0] ?? (data.type === "start" ? manifest(data.settings) : null);
    const existing = !header ? await BrowserBenchmarkStore.open(data.id) : null;
    const pair = header?.pair ?? existing?.manifest.pair;
    const id = header?.id ?? data.id;
    pairAbort = new AbortController();
    await navigator.locks.request(
      `fgllmbench:${pair?.id ?? id}`,
      { ifAvailable: true },
      async (lock) => {
        if (!lock)
          throw Error("This benchmark is already running in another tab");
        if (pair) {
          const expected =
            headers ?? pairManifests(existing.manifest.settings, pair);
          const entries = await BrowserBenchmarkStore.list();
          const stores = [];
          for (const h of expected)
            stores.push(
              entries.some((e) => e.id === h.id)
                ? await BrowserBenchmarkStore.open(h.id)
                : await BrowserBenchmarkStore.create(h),
            );
          let activeId;
          await runGamePair(stores, data.key, {
            togetherKey: data.togetherKey,
            signal: pairAbort.signal,
            retryIncomplete: data.retryIncomplete ?? false,
            onRunner: (next, h) => {
              runner = next;
              activeId = h.id;
              send("saved", { id: activeId });
              if (pauseRequested) runner.pause();
              if (stopRequested) runner.stop();
            },
            onStatus: (status) => send("status", { status }),
            onCommitted: (event) => {
              if (
                ["generation", "run_start", "run_end", "session_end"].includes(
                  event.type,
                )
              )
                send("changed", { id: activeId, seq: event.seq });
            },
          });
          send("status", { status: "Completed — both game benchmarks saved" });
          return;
        }
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
    pairAbort = null;
    busy = pauseRequested = stopRequested = false;
    send("idle");
  }
};
