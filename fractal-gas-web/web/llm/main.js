import { generateGame, gameSpec, externalScore } from "./games.js";
import {
  DEFAULTS,
  configuration,
  bestNode,
  selectedScore,
  objectiveLabel,
} from "./config.js";
import { AnalysisView, renderTokens } from "./analysis.js";
import { importRecording } from "./recording.js";
import { ComparisonView } from "./comparison-view.js";
import { initBenchmark } from "./benchmark-ui.js";
import { formatRunProgress } from "./run-control.js";
const $ = (id) => document.getElementById(id),
  form = $("config"),
  fields = form.elements;
for (const [key, value] of Object.entries(DEFAULTS))
  if (fields.namedItem(key)) fields.namedItem(key).value = value;
let worker,
  record,
  started = false,
  busy = false,
  failed = false,
  offline = false,
  selected = null,
  latest = true;
let activeTab = "generation";
let currentRecordingId = 0;
let benchmarkBusy = false,
  selectedGame = null,
  creatingGame = false,
  gameAbort;
const getConfiguration = () =>
  configuration({
    ...Object.fromEntries(
      Object.keys(DEFAULTS).map((key) => [
        key,
        fields.namedItem(key)?.value ?? DEFAULTS[key],
      ]),
    ),
    game: selectedGame,
  });
function scoringControls() {
  const mode = fields.namedItem("objective").value;
  $("beam-settings").hidden = mode !== "beam";
  const game = mode === "xent_game";
  $("xed-settings").hidden = !externalScore({ objective: mode });
  $("xed-direction-setting").hidden = game;
  $("game-settings").hidden = !game;
  $("prompt-setting").hidden = game;
  $("benchmark-pair").hidden = !game;
  $("game-preview").hidden = !selectedGame;
  for (const key of ["title", "background", "target"])
    $("game-" + key).textContent = selectedGame?.[key] ?? "";
  const creationUsage =
    selectedGame?.creation?.requests?.filter((r) => r.usage) ?? [];
  $("game-cost").textContent = creationUsage.length
    ? `Game creation: ${creationUsage.reduce((s, r) => s + (r.usage.prompt_tokens ?? 0), 0)} input / ${creationUsage.reduce((s, r) => s + (r.usage.completion_tokens ?? 0), 0)} output tokens. Separate from sampling budgets.`
    : "";
  $("game-export").disabled = !selectedGame;
  $("game-benchmark").disabled = !selectedGame;
  $("objective-hint").textContent = game
    ? "The judge scores the fixed target after your context. Higher signed game score is better; units are nats per target token."
    : mode === "xed"
      ? "Maximize favors answers made more likely by the question; minimize favors the reverse. Two extra scoring evaluations per new prefix. This measures likelihood contrast, not correctness."
      : mode === "beam"
        ? "Total log probability ÷ token count^α. Higher is better; α = 1 matches negative mean Xent."
        : "Negative mean Xent = mean token log probability. Higher is more predictable, not necessarily more correct.";
}
fields.namedItem("objective").addEventListener("change", scoringControls);
scoringControls();
function graphControls() {
  const graph = fields.namedItem("algorithm").value === "graph";
  $("freeze-prefix-setting").hidden = !graph;
  $("freeze-prefix-hint").hidden = !graph;
}
fields.namedItem("algorithm").addEventListener("change", graphControls);
graphControls();
const comparison = new ComparisonView($("comparison-workspace"), {
  evaluationHost: $("evaluation-panel"),
  getKey: () => $("api-key").value.trim(),
  setKey: (value) => {
    $("api-key").value = value;
  },
  getConfig: getConfiguration,
  onEvaluation: () => switchTab("evaluation"),
  onBenchmark: () => switchTab("benchmark"),
  onBenchmarkSource: (id) => benchmarkUI.selectSource(id),
  onGeneration: () => switchTab("generation"),
  onCurrentSource: () => benchmarkUI.clearSelection(),
});
const benchmarkUI = initBenchmark({
  getTogetherKey: () => $("together-key").value.trim(),
  onSource: (id, live) => comparison.selectBenchmark(id, live),
  onChanged: (id, live) => comparison.refreshBenchmark(id, live),
  getConfig: getConfiguration,
  getKey: () => $("api-key").value.trim(),
  onBusy: (value) => {
    benchmarkBusy = value;
    controls();
  },
});
$("game-generate").onclick = async () => {
  creatingGame = true;
  gameAbort = new AbortController();
  controls();
  $("game-status").textContent = "Generating a fixed-target game…";
  try {
    const game = await generateGame(
      $("api-key").value.trim(),
      $("game-brief").value,
      {
        model: fields.namedItem("model").value,
        temperature: Number(fields.namedItem("temperature").value),
      },
      { signal: gameAbort.signal },
    );
    selectedGame = game;
    $("game-status").textContent =
      "Game saved in this session. Run either mode or benchmark both.";
  } catch (error) {
    $("game-status").textContent = error.message;
  } finally {
    creatingGame = false;
    gameAbort = null;
    controls();
  }
};
$("game-benchmark").onclick = () => {
  switchTab("benchmark");
  benchmarkUI.startPair();
};
$("game-export").onclick = () => {
  try {
    const value = {
      config: getConfiguration(),
      comparison: "tokens",
      repetitions: 1,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `xent-${selectedGame.id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    $("game-status").textContent = error.message;
  }
};
$("game-import").onclick = () => $("game-file").click();
$("game-file").onchange = async () => {
  try {
    const file = $("game-file").files[0];
    if (!file) return;
    if (file.size > 1024 * 1024) throw Error("Game settings exceed 1 MiB");
    const input = JSON.parse(await file.text());
    const game = gameSpec(input.config?.game ?? input);
    const config = input.config ? configuration(input.config) : null;
    selectedGame = game;
    if (config)
      for (const [key, value] of Object.entries(config))
        if (fields.namedItem(key)) fields.namedItem(key).value = value;
    graphControls();
    fields.namedItem("objective").value = "xent_game";
    $("game-brief").value = game.creation?.brief ?? "";
    $("game-status").textContent = "Imported fixed-target game.";
    controls();
  } catch (error) {
    $("game-status").textContent = error.message;
  } finally {
    $("game-file").value = "";
  }
};
const analysis = new AnalysisView($("analysis-panel"), {
  onSelect: (id) => {
    selected = id;
    render();
  },
  onStep: (step) => {
    $("iteration").value = step;
    render();
  },
  onFollow: (follow) => {
    latest = follow;
    render();
  },
  onStatus: (text) => status(text),
});
function switchTab(tab) {
  activeTab = tab;
  for (const name of ["generation", "analysis", "benchmark", "evaluation"]) {
    const active = name === tab;
    $(`tab-${name}`).setAttribute("aria-selected", String(active));
    $(`tab-${name}`).tabIndex = active ? 0 : -1;
    $(`${name}-panel`).hidden = !active;
  }
  $("llm-main").classList.toggle("analyzing", tab !== "generation");
  $("generation-settings").hidden = tab !== "generation";
  for (const id of ["generation-toolbar", "status", "stats"])
    $(id).hidden = ["benchmark", "evaluation"].includes(tab);
  comparison.setPanel(tab);
  comparison.setActive(["benchmark", "evaluation"].includes(tab));
  analysis.setActive(tab === "analysis");
  render();
}
for (const name of ["generation", "analysis", "benchmark", "evaluation"]) {
  $(`tab-${name}`).onclick = () => switchTab(name);
  $(`tab-${name}`).onkeydown = (e) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
      e.preventDefault();
      const names = ["generation", "analysis", "benchmark", "evaluation"];
      const next =
        e.key === "Home"
          ? names[0]
          : e.key === "End"
            ? names.at(-1)
            : names[
                (names.indexOf(name) +
                  (e.key === "ArrowRight" ? 1 : names.length - 1)) %
                  names.length
              ];
      switchTab(next);
      $(`tab-${next}`).focus();
    }
  };
}
function status(text) {
  $("status").textContent = text;
}
function controls() {
  scoringControls();
  benchmarkUI.setBlocked(busy || creatingGame);
  $("settings").disabled =
    started || busy || offline || benchmarkBusy || creatingGame;
  $("run").disabled = $("step").disabled =
    busy ||
    failed ||
    offline ||
    benchmarkBusy ||
    creatingGame ||
    !!record?.run?.stop_reason;
  $("pause").disabled = !busy;
  $("stop").disabled =
    !creatingGame && ((!busy && !started) || !!record?.run?.stop_reason);
  $("reset").disabled = busy || benchmarkBusy || creatingGame;
  $("import").disabled = busy || benchmarkBusy || creatingGame;
  $("export").disabled = !record;
}
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("worker.js", import.meta.url), {
    type: "module",
  });
  worker.onmessage = ({ data }) => {
    if (data.type === "update") {
      record = data.data;
      status(data.status);
      render();
    }
    if (data.type === "idle") {
      busy = false;
      failed = data.failed;
      controls();
    }
    if (data.type === "error") {
      status(data.message);
      controls();
    }
    if (data.type === "catalogs") {
      for (const [id, entries] of [
        [
          "generation-models",
          data.models.filter((m) =>
            m.supported_parameters?.includes("logprobs"),
          ),
        ],
        ["embedding-models", data.embeddings],
      ]) {
        $(id).replaceChildren(
          ...entries.map((m) => {
            const o = document.createElement("option");
            o.value = m.id;
            o.label = m.name;
            return o;
          }),
        );
      }
      status(
        "Model lists refreshed. Generation support is checked before each run.",
      );
    }
  };
  worker.onerror = (e) => {
    busy = false;
    failed = true;
    status(
      e.message || "Worker failed to load. Build the LLM engine and reload.",
    );
    controls();
  };
  return worker;
}
function begin(single) {
  try {
    if (!started) {
      const config = getConfiguration();
      const key = $("api-key").value.trim();
      if (!key) throw new Error("Enter your OpenRouter API key");
      getWorker().postMessage({
        type: "start",
        single,
        config,
        key,
        togetherKey: $("together-key").value.trim(),
      });
      currentRecordingId++;
      started = true;
    } else getWorker().postMessage({ type: single ? "step" : "run" });
    busy = true;
    controls();
  } catch (e) {
    status(e.message);
  }
}
$("run").onclick = () => begin(false);
$("step").onclick = () => begin(true);
$("pause").onclick = () => {
  worker?.postMessage({ type: "pause" });
  status("Pausing after the current iteration…");
};
$("stop").onclick = () => {
  if (creatingGame) {
    gameAbort?.abort(new Error("Stopped"));
    return;
  }
  worker?.postMessage({ type: "stop" });
  failed = true;
  status("Stopping pending requests…");
  controls();
};
$("reset").onclick = () => {
  worker?.terminate();
  worker = null;
  record = null;
  currentRecordingId++;
  started = busy = failed = offline = false;
  selected = null;
  latest = true;
  switchTab("generation");
  status("Ready for a new run.");
  render();
};
$("models").onclick = () => {
  const key = $("api-key").value.trim();
  if (!key) {
    status("Enter your OpenRouter API key to refresh models.");
    return;
  }
  getWorker().postMessage({ type: "catalogs", key });
  status("Loading available models…");
};
$("export").onclick = () => {
  if (!record) return;
  const blob = new Blob([JSON.stringify(record)], { type: "application/json" });
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = `llm-${record.config.algorithm}-${Date.now()}.fgllm`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("import").onclick = () => $("import-file").click();
$("import-file").onchange = async () => {
  try {
    const file = $("import-file").files[0];
    if (!file) return;
    if (file.size > 64 * 1024 * 1024)
      throw new Error("Recording exceeds 64 MiB");
    const imported = importRecording(await file.text());
    worker?.terminate();
    worker = null;
    analysis.update(null, 0, null, true);
    record = imported;
    selectedGame = record.config.game;
    currentRecordingId++;
    latest = true;
    failed = false;
    offline = true;
    started = false;
    selected = null;
    for (const [key, value] of Object.entries(record.config))
      if (fields.namedItem(key)) fields.namedItem(key).value = value;
    graphControls();
    switchTab("analysis");
    status("Imported recording. Inspect offline; reset to start a new run.");
    render();
  } catch (e) {
    status(e.message);
  } finally {
    $("import-file").value = "";
  }
};
$("iteration").oninput = () => {
  latest = Number($("iteration").value) === record.snapshots.length;
  render();
};
$("trace-filter").onchange = () => render();
const number = (n) => Number(n).toFixed(3),
  state = (n) =>
    n.status === 1 ? "Finished" : n.status === 2 ? "Capped" : "Partial";
function render() {
  controls();
  comparison.updateCurrent(record, currentRecordingId);
  const snapshots = record?.snapshots ?? [],
    count = snapshots.length;
  $("iteration").disabled = !count;
  $("iteration").max = Math.max(1, count);
  if (latest) $("iteration").value = Math.max(1, count);
  const step = count ? Number($("iteration").value) : 0,
    snapshot = snapshots[step - 1];
  $("iteration-label").value = `${step} / ${count}`;
  const nodes = (record?.nodes ?? []).slice(
    0,
    snapshot?.node_count ?? record?.nodes.length ?? 0,
  );
  const best = bestNode(nodes, record?.config);
  $("show-best").disabled = !best;
  $("show-best").onclick = () => {
    selected = best.id;
    render();
  };
  $("best-title").textContent = best?.game_score
    ? "Best sampled context"
    : best
      ? `Best ${state(best).toLowerCase()} trace`
      : "Best trace";
  $("best-text").textContent =
    best?.text || "Your best continuation will appear here.";
  $("best-score").textContent = best
    ? `${objectiveLabel(record.config)}: ${number(selectedScore(best, record.config))} · ${best.tokens} tokens · total NLL ${number(-best.logp)} · mean NLL ${number(-best.logp / best.tokens)} · trace ${best.id}`
    : fields.namedItem("objective").value === "xent_game"
      ? "Every scored nonempty context can lead. Higher game score is better."
      : "Finished answers are ranked by the selected objective. Before completion, compare the deepest prefixes.";
  if (best?.game_score) {
    const x = best.game_score;
    $("best-score").textContent =
      `Make it ${record.config.game_mode}: ${number(selectedScore(best, record.config))} · target surprise ${number(-x.conditional_logp / x.tokens)} · baseline ${number(-x.baseline_logp / x.tokens)} nats/target token · ${best.tokens} generated tokens · trace ${best.id}`;
  }
  const usage = (record?.requests ?? []).reduce(
    (v, r) => ({
      prompt: v.prompt + (r.usage?.prompt_tokens ?? 0),
      completion: v.completion + (r.usage?.completion_tokens ?? 0),
      cost: v.cost + (r.usage?.cost ?? 0),
      hasCost: v.hasCost || Number.isFinite(r.usage?.cost),
    }),
    { prompt: 0, completion: 0, cost: 0, hasCost: false },
  );
  const scoringUsage = (record?.requests ?? []).filter(
    (r) => r.path === "scoring/completions" && r.status !== "started",
  );
  const scoringText = scoringUsage.length
    ? ` · Scoring: ${scoringUsage.length} requests, ${scoringUsage.reduce((n, r) => n + (r.usage?.prompt_tokens ?? 0), 0)} input / ${scoringUsage.reduce((n, r) => n + (r.usage?.completion_tokens ?? 0), 0)} output tokens`
    : "";
  const progress = snapshot?.run ?? record?.run;
  const progressText = progress ? `${formatRunProgress(progress)} · ` : "";
  $("stats").textContent = record
    ? `${count} steps · ${progressText}${record.nodes.length - 1} generated branches · ${snapshot?.walkers.filter((w) => w.cloned).length ?? 0} clones this step · ${usage.completion} output / ${usage.prompt} input tokens · ${usage.hasCost ? `$${usage.cost.toFixed(5)} reported` : "Cost not reported"} · ${record.metadata?.provider?.name ?? "Checking provider"}${scoringText}`
    : "Each step extends the sequences, measures their differences, and selects branches to continue.";
  let list = nodes.filter((n) => n.tokens > 0);
  if ($("trace-filter").value === "population") {
    const ids = new Set(snapshot?.walkers.map((w) => w.node));
    list = list.filter((n) => ids.has(n.id));
  }
  if ($("trace-filter").value === "finished")
    list = list.filter((n) => n.status !== 0);
  const total = list.length;
  list = list.slice(-500).reverse();
  $("trace-count").textContent =
    `${total} sequences${total > 500 ? " · showing the latest 500" : ""}`;
  if (selected !== null && !nodes[selected]) selected = best?.id ?? null;
  if (selected === null && analysis.unusedSlot == null)
    selected = best?.id ?? null;
  analysis.update(record, step, selected, latest);
  if (activeTab === "analysis") return;
  $("traces").replaceChildren(
    ...list.map((n) => {
      const tr = document.createElement("tr");
      if (n.id === selected) tr.className = "selected";
      const first = document.createElement("td"),
        button = document.createElement("button");
      button.textContent = `#${n.id}`;
      button.onclick = () => {
        selected = n.id;
        render();
      };
      first.append(button);
      tr.append(first);
      for (const value of [n.tokens, number(-n.logp), state(n)]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      return tr;
    }),
  );
  const node = nodes[selected];
  $("trace-title").textContent = node ? `Trace #${node.id}` : "Trace inspector";
  $("trace-score").textContent = node
    ? `${node.tokens} tokens · total NLL ${number(-node.logp)} · mean NLL ${number(-node.logp / Math.max(1, node.tokens))} · ${state(node)}`
    : "Select a sequence to inspect its ancestry and token probabilities.";
  if (node?.game_score) {
    const x = node.game_score;
    $("trace-score").textContent =
      `${objectiveLabel(record.config)}: ${number(selectedScore(node, record.config))} · target surprise ${number(-x.conditional_logp / x.tokens)} · baseline ${number(-x.baseline_logp / x.tokens)} · ${node.tokens} generated tokens · ${state(node)}`;
  }
  $("trace-text").textContent = node?.text ?? "";
  const chain = [];
  let cursor = node;
  while (cursor && cursor.id) {
    chain.unshift(cursor);
    cursor = nodes[cursor.parent];
  }
  $("ancestry").replaceChildren(
    ...chain.map((n) => {
      const b = document.createElement("button");
      b.textContent = `#${n.id}`;
      b.onclick = () => {
        selected = n.id;
        render();
      };
      return b;
    }),
  );
  renderTokens(
    $("tokens"),
    chain.flatMap((n) => n.token_data),
  );
  $("population").textContent = JSON.stringify(
    snapshot?.walkers ?? [],
    null,
    2,
  );
  $("requests").textContent = JSON.stringify(
    {
      metadata: record?.metadata,
      requests: record?.requests,
      errors: record?.errors,
    },
    null,
    2,
  );
}
render();
