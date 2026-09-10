import {
  benchmarkConfiguration,
  parseBenchmark,
  exportBenchmark,
  collectRuns,
} from "./benchmark-data.js";
import { BrowserBenchmarkStore } from "./benchmark-store.js";
import { formatRunProgress } from "./run-control.js";
export function initBenchmark({
  getConfig,
  getKey,
  getTogetherKey = () => "",
  onBusy,
  onSource = () => {},
  onChanged = () => {},
}) {
  const $ = (id) => document.getElementById(`benchmark-${id}`);
  let worker,
    running = false,
    blocked = false,
    selected = "",
    pausing = false;
  const status = (text) => {
    $("status").textContent = text;
  };
  function controls() {
    $("start").disabled = blocked || running;
    $("comparison").disabled = $("repetitions").disabled = running;
    $("pause").disabled = !running || pausing;
    $("continue").disabled = !running || !pausing;
    $("stop").disabled = !running;
    $("saved").disabled = $("import").disabled = running;
    $("export").disabled = !selected;
    $("retry").disabled = !selected || running || blocked;
  }
  async function describe() {
    if (!selected || running) return;
    const store = await BrowserBenchmarkStore.open(selected);
    const runs = [
      ...collectRuns(store.manifest, await store.readEvents()).runs.values(),
    ];
    const complete = runs.filter((r) => r.status === "completed").length;
    const total =
      store.manifest.method_order.length * store.manifest.settings.repetitions;
    const progress = runs.findLast((r) => r.method === "fractal")?.run;
    status(
      `Saved benchmark · ${complete}/${total} methods completed. ${complete === total ? "Ready to export." : "Retry unfinished to continue; partial attempts are retained."}${progress ? ` Fractal: ${formatRunProgress(progress)}.` : ""}`,
    );
  }
  async function refresh() {
    const entries = await BrowserBenchmarkStore.list();
    $("saved").replaceChildren(
      new Option("Current Fractal run", ""),
      ...entries
        .sort((a, b) => b.updated_at - a.updated_at)
        .map(
          (e) =>
            new Option(
              `${new Date(e.manifest.created_at).toLocaleString()} · ${e.manifest.settings.config.algorithm} · ${e.manifest.settings.comparison} · ${e.id.slice(0, 8)}`,
              e.id,
            ),
        ),
    );
    $("saved").value = selected;
    controls();
  }
  function getWorker() {
    if (worker) return worker;
    worker = new Worker(new URL("benchmark-worker.js", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      if (data.type === "saved") {
        selected = data.id;
        onSource(selected, true);
        refresh().catch((e) => status(e.message));
      }
      if (data.type === "changed") onChanged(data.id, running);
      if (data.type === "status") status(data.status);
      if (data.type === "idle") {
        running = pausing = false;
        onBusy(false);
        onChanged(selected, false);
        refresh().catch((e) => status(e.message));
      }
    };
    worker.onerror = (e) => {
      status(
        e.message || "Benchmark worker failed; saved progress is retained",
      );
      running = pausing = false;
      onBusy(false);
      worker.terminate();
      worker = null;
      controls();
    };
    return worker;
  }
  function begin(retry) {
    try {
      if (running || blocked) return;
      const key = getKey();
      if (!key) throw Error("Enter your OpenRouter API key");
      const message = retry
        ? { type: "resume", id: selected, retryIncomplete: true }
        : {
            type: "start",
            settings: benchmarkConfiguration({
              config: getConfig(),
              comparison: $("comparison").value,
              repetitions: $("repetitions").value,
            }),
          };
      getWorker().postMessage({
        ...message,
        key,
        togetherKey: getTogetherKey(),
      });
      running = true;
      pausing = false;
      onBusy(true);
      controls();
      status("Starting benchmark…");
    } catch (e) {
      status(e.message);
    }
  }
  $("start").onclick = () => begin(false);
  $("retry").onclick = () => begin(true);
  $("pause").onclick = () => {
    pausing = true;
    worker?.postMessage({ type: "pause" });
    controls();
  };
  $("continue").onclick = () => {
    pausing = false;
    worker?.postMessage({ type: "continue" });
    controls();
    status("Continuing benchmark…");
  };
  $("stop").onclick = () => {
    worker?.postMessage({ type: "stop" });
    status("Stopping pending requests; retaining saved progress…");
  };
  $("saved").onchange = () => {
    selected = $("saved").value;
    controls();
    onSource(selected, running);
    describe().catch((e) => status(e.message));
  };
  $("export").onclick = async () => {
    try {
      const store = await BrowserBenchmarkStore.open(selected);
      const text = exportBenchmark(store.manifest, await store.readEvents());
      const url = URL.createObjectURL(
        new Blob([text], { type: "application/x-ndjson" }),
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = `llm-benchmark-${selected}.fgllmbench`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      status(e.message);
    }
  };
  $("import").onclick = () => $("file").click();
  $("file").onchange = async () => {
    try {
      const file = $("file").files[0];
      if (!file) return;
      const archive = parseBenchmark(await file.text());
      const existing = (await BrowserBenchmarkStore.list()).find(
        (e) => e.id === archive.manifest.id,
      );
      if (existing) {
        selected = existing.id;
        onSource(selected, false);
        await refresh();
        status(
          "This benchmark is already saved. Its local data was preserved.",
        );
        return;
      }
      await BrowserBenchmarkStore.create(archive.manifest, archive.events);
      selected = archive.manifest.id;
      onSource(selected, false);
      await refresh();
      await describe();
    } catch (e) {
      status(e.message);
    } finally {
      $("file").value = "";
    }
  };
  refresh().catch((e) => status(`Browser storage unavailable: ${e.message}`));
  controls();
  return {
    selectSource(id) {
      if (running)
        throw Error(
          "Wait for the running benchmark before selecting another source",
        );
      selected = id;
      $("saved").value = id;
      controls();
      onSource(selected, false);
      describe().catch((e) => status(e.message));
    },
    clearSelection() {
      selected = "";
      $("saved").value = "";
      controls();
    },
    setBlocked(value) {
      blocked = value;
      controls();
    },
  };
}
