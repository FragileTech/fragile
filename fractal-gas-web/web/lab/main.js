import { configureRocks, rockOptions } from "./rock-scene.js";
import {
  configureVehicleCount,
  configureVehicleType,
  vehicleCount,
  vehicleType,
} from "./vehicle-scene.js";
import {
  RewardSettings,
  withRewards,
  coefficientValues,
} from "./reward-settings.js";
import { ActionSettings, withActionMultipliers } from "./action-settings.js";
import { treePoseDim, treeWidth } from "./actions.js";
import { bytesOf } from "./motion.js";
import { ExperimentPanel } from "./experiment-panel.js";
import { ControllerSettings } from "./controller-settings.js";
import { scenePresentation, sceneReadout } from "./scene-presentation.js";
import { renderCircuitPreview } from "./circuit-preview.js";
import { StoragePanel } from "./storage-panel.js";
import { PhysicsInspector } from "./physics-inspector.js";
import { ReplayPanel } from "./replay-panel.js";
import { createSceneEditor } from "./scene-editor.js";
import { installManualControl } from "./manual-control.js";
import { LabRenderer } from "./renderer.js";
import { installStyleControls } from "./style-controls.js";
import { installAnimationControls } from "./animation-controls.js";
import { installActionGuideControls } from "./action-guides.js";
import { initHelp } from "../help.js";
import { Recording, exportRecording, importRecording } from "./archive.js";
const $ = (id) => document.getElementById(id),
  copy = (value) => structuredClone(value);
const renderer = new LabRenderer($("world"), {
  isEditing: () => !$("editor").hidden,
});
installStyleControls();
installAnimationControls();
installActionGuideControls();
const vehicleCounts = new Map();
const vehicleTypes = new Map();
const miningOptions = new Map();
let agentCatalog = {},
  presetRequest = 0;
let worker,
  currentScene,
  currentInfo,
  currentChannels,
  currentState,
  revision = 0,
  running = false,
  record = new Recording();
let selectedRecord = 0,
  ready = false;
let importPending,
  checkpointPending,
  clean = false,
  oldLayers,
  initialized = false,
  lastLiveFrame,
  lastDiagnostics,
  playbackDecision;
const replay = new ReplayPanel({
  stop,
  error,
  animationChanged({ active, playing, speed }) {
    renderer.setAnimationPlayback({
      playing: active ? playing : running,
      speed: active ? speed : 1,
    });
  },
  show(frame, { seek = false } = {}) {
    currentState = frame.state;
    renderer.update(frame.state, frame.action);
    if (seek) renderer.setAnimationPlayback({ seek: true });
    const bits = new Uint32Array(
      frame.state.buffer,
      frame.state.byteOffset,
      frame.state.length,
    );
    const metrics = new Float32Array(16);
    metrics[5] = bits[4];
    metrics[6] = bits[5];
    metrics[7] = bits[6];
    updateFrame({ ...frame, metrics, missed: 0 });
    $("run-state").textContent = "WORLD REPLAY";
  },
  live() {
    if (lastLiveFrame) {
      currentState = lastLiveFrame.state;
      renderer.update(currentState, lastLiveFrame.action);
      updateFrame(lastLiveFrame);
      updateDiagnostics(lastDiagnostics);
      $("run-state").textContent = "PAUSED";
    }
  },
  resume(rows, configuration) {
    if (configuration?.scene) {
      applyRewardUpdate(configuration.scene, coefficientValues(configuration.settings), rows, configuration.root);
    } else worker.postMessage({ type: "resume-motion", rows });
  },
  async decision(number) {
    if (number === playbackDecision) return;
    playbackDecision = number;
    let index = record.entries.findIndex((e) => e.decision === number);
    if (index < 0 && replay.recording?.loadObject) {
      try {
        const entry = await replay.recording.loadObject("tree", number);
        if (playbackDecision !== number) return;
        if (entry) {
          record.append(entry);
          index = record.entries.length - 1;
        }
      } catch (e) {
        error(e);
      }
    }
    if (index >= 0) {
      selectedRecord = index;
      renderer.diagnostics(record.entries[index].tree);
      updateDiagnostics(record.entries[index]);
      updateRecordUI();
    } else {
      renderer.clearDiagnostics();
      updateDiagnostics();
    }
  },
});
const editor = createSceneEditor({
  renderer,
  getState: () => currentState,
  getInfo: () => currentInfo,
  getChannels: () => currentChannels,
  applyAction: (action) => {
    stop();
    replay.playback.live();
    worker.postMessage({ type: "manual", action, frames: 1 });
  },
  isReady: () => ready,
  loadScene,
  stop,
  status,
  error,
  download,
  upload,
  slug,
  onSelection: () => updateCargoReadout(),
  onWorldClick(point) {
    const tree = record.entries[selectedRecord]?.tree;
    if (!tree || !renderer.layers.tree) return;
    let best = 1.5,
      node;
    const width = treeWidth(tree);
    for (let i = 0; i < tree.meta.length / 5; i++)
      for (let c = 0; c < treePoseDim(tree); c += 2) {
        const distance = Math.hypot(
          point[0] - tree.values[i * width + 3 + c],
          point[1] - tree.values[i * width + 4 + c],
        );
        if (distance < best) {
          best = distance;
          node = tree.meta[i * 5];
        }
      }
    if (node) {
      $("node").value = node;
      $("hint").textContent =
        `NODE ${node} SELECTED · REPLAY BRANCH TO RESTORE THIS FUTURE`;
    }
    return;
  },
});
const storage = new StoragePanel({
  getRecording: () => replay.recording,
  getScene: () => currentScene,
  getSettings: () => ({ ...settings(), seed: +$("seed").value }),
  async loadRun(motion) {
    const last = motion.length
      ? await motion.getFrame(motion.length - 1)
      : undefined;
    importPending = {
      scene: motion.scene,
      settings: motion.settings,
      motion,
      recording: new Recording(),
      lastRows: last ? await motion.getRows(motion.length - 1) : undefined,
      lastDecision: last?.decision || 0,
    };
    const configuration = motion.rewardConfiguration();
    importPending.root = configuration.root;
    applySettings(configuration.settings);
    loadScene(configuration.scene);
  },
  saveCheckpoint() {
    if (!ready) {
      status("Wait for the world to load.");
      return;
    }
    stop();
    worker.postMessage({ type: "checkpoint" });
    status("Saving planner at an iteration boundary…");
  },
  async loadCheckpoint(data) {
    checkpointPending = data;
    applySettings(data.settings);
    loadScene(data.scene);
  },
  upload,
  download,
  status,
  error,
});
new ExperimentPanel({
  getScene: () => currentScene,
  getSettings: settings,
  getRoot: () => ({ snapshot: replay.recording.root, rows: currentRows() }),
  stop,
  download,
  error,
});
const controllerSettings = new ControllerSettings(
  $("algorithm-settings"),
  $("algorithm"),
  () => loadScene(currentScene),
);
const actionSettings = new ActionSettings(
  $("agent-action-settings"),
  $("apply-action-settings"),
  (values) => {
    if (!ready) return;
    try {
      actionSettings.setEnabled(false);
      const next = withActionMultipliers(currentScene, values);
      editor.clearHistory();
      loadScene(next);
    } catch (e) {
      actionSettings.setEnabled(true);
      error(e);
    }
  },
);
let rewardChangePending;
let appliedCoefficients = coefficientValues();
function applyRewardUpdate(scene, coefficients, rows, root, replayBranch) {
  if (!ready || rewardChangePending) return;
  rewardChangePending = { scene, coefficients };
  rewardSettings.setEnabled(false);
  status("Applying reward settings at the current world state…");
  worker.postMessage({ type: "update-rewards", scene, coefficients, rows, root, replayBranch });
}
const rewardSettings = new RewardSettings($("reward-terms"), (values) => {
  if (!ready || rewardChangePending) return;
  try {
    applyRewardUpdate(withRewards(currentScene, values), coefficientValues(values));
  } catch (e) {
    rewardChangePending = undefined;
    error(e);
  }
});
rewardSettings.setEnabled(false);
const inspector = new PhysicsInspector({
  renderer,
  getState: () => currentState,
  getInfo: () => currentInfo,
  getSelection: () => editor.selection,
  request: () =>
    worker?.postMessage({
      type: "inspect",
      rows: currentRows(),
      action: lastLiveFrame?.action,
    }),
});
function currentRows() {
  if (!currentState || !currentInfo)
    throw new Error("Wait for the world to load");
  const rows = new Float32Array(currentInfo[3]);
  bytesOf(rows).set(bytesOf(currentState));
  return rows;
}
function applySettings(values = {}) {
  appliedCoefficients = coefficientValues({
    ...appliedCoefficients,
    ...values,
  });
  for (const [key, value] of Object.entries(values)) {
    if (key in appliedCoefficients) continue;
    const node = $(key);
    if (node?.tagName === "INPUT" && node.type === "checkbox")
      node.checked = value;
    else if (node && ["INPUT", "SELECT"].includes(node.tagName))
      node.value = value;
  }
  controllerSettings.render(values);
}
function status(text = "", error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
function error(value) {
  status(value.message || String(value), true);
  stop();
}
function updateViewControls() {
  const flight = renderer.flightMode;
  const view = $("view");
  view.textContent = flight ? "Side / overhead" : "2D / 3D";
  view.title = flight
    ? "Switch between side-on and overhead flight views"
    : "Switch between top and angled views";
  view.dataset.help = flight
    ? "Switch between the horizontal side-on flight view and the overhead physics view."
    : "Switch between the top-down physics view and the angled three-dimensional presentation.";
  $("hint").textContent = flight
    ? "SCROLL TO ZOOM · SIDE / OVERHEAD TO CHANGE VIEW"
    : "SCROLL TO ZOOM · 2D / 3D TO CHANGE VIEW";
}
function updateFlightControl(scene, effective = renderer.flightMode) {
  const input = $("flight-mode"),
    label = $("flight-control");
  const override = scene.environment?.flight;
  const automatic = override == null;
  const enabled = automatic ? !!effective : override;
  input.checked = enabled;
  input.indeterminate = automatic;
  input.setAttribute("aria-checked", automatic ? "mixed" : String(enabled));
  label.dataset.mode = automatic ? "auto" : enabled ? "on" : "off";
  $("flight-mode-state").textContent = automatic
    ? "AUTO"
    : enabled
      ? "ON"
      : "OFF";
}
function rockWeight() {
  return 10 ** +$("rock-weight-slider").value;
}
function renderRockWeight() {
  const value = rockWeight();
  $("rock-weight-value").textContent = `${Number(value.toPrecision(3))}×`;
}
function settings() {
  return {
    ...controllerSettings.values(),
    algorithm: $("algorithm").value || "fmc",
    walkers: +$("walkers").value,
    horizon: +$("horizon").value,
    frames: +$("frames").value,
    ...appliedCoefficients,
    noise: +$("noise").value,
    elites: +$("elites").value,
    inertial: $("inertial").checked,
    recording: +$("recording").value,
  };
}
function stop() {
  running = false;
  renderer.setAnimationPlayback({ playing: false });
  if (ready) worker?.postMessage({ type: "run", value: false });
  $("run").textContent = "▶ Run experiment";
}
function loadScene(scene, autoStep = false, continuation = undefined) {
  ++presetRequest;
  rewardChangePending = undefined;
  $("flight-mode").disabled = true;
  rewardSettings.render(scene, appliedCoefficients);
  rewardSettings.setEnabled(false);
  const isCircuit = scene.environment?.kind === "circuit";
  $("track-control").hidden = !isCircuit;
  if (isCircuit) {
    $("scenario").value = "racing";
    const trackId = scene.circuit?.id;
    $("track").querySelector('option[value=""]')?.remove();
    if ([...$("track").options].some((option) => option.value === trackId)) {
      $("track").value = trackId;
    } else {
      $("track").add(new Option("Current / imported circuit", "", true, true));
    }
  }
  const rocks = rockOptions(scene);
  $("rock-controls").hidden = !rocks;
  if (rocks) {
    $("hook-stiffness-field").hidden = !scene.tethers?.length;
    const stiffness = scene.tethers?.[0]?.stiffness ?? 25;
    $("hook-stiffness").value = stiffness;
    $("hook-stiffness-slider").value = Math.log10(stiffness + 1);
    $("rock-size").value = rocks.scale;
    $("rock-weight-slider").value = Math.log10(rocks.weight);
    renderRockWeight();
    $("rock-count").value = rocks.count;
    $("rock-count-field").hidden =
      scene.rock_options?.collaborative ??
      scene.name === "Collaborative mining";
  }
  $("ants-vehicle-count").value = vehicleCount(scene);
  $("ants-vehicle-type").value = vehicleType(scene) || "";
  $("ants-vehicle-type").disabled = vehicleCount(scene) === 0;
  if (worker) {
    worker.postMessage({ type: "close" });
    const previous = worker;
    setTimeout(() => previous.terminate(), 250);
  }
  ready = false;
  replay.recording?.flush?.().catch(error);
  replay.attach(undefined);
  currentState = lastLiveFrame = lastDiagnostics = playbackDecision = undefined;
  updateDiagnostics();
  running = false;
  currentScene = copy(scene);
  renderer.setAnimationPlayback({ playing: false, seek: true });
  editor.setScene(scene);
  actionSettings.render(scene);
  actionSettings.setEnabled(false);
  $("focus").textContent = "Follow agent";
  record = new Recording();
  $("run").disabled = $("step").disabled = true;
  $("run").textContent = "▶ Run experiment";
  $("run-state").textContent = "LOADING";
  $("description").textContent =
    scene.description || "Custom continuous-control experiment.";
  renderCircuitPreview($("circuit-preview"), scene);
  $("scene-title").textContent = (
    scene.name || "Custom experiment"
  ).toUpperCase();
  $("viewport-task").textContent = scenePresentation(scene).task_label;
  $("record-count").textContent = "No decisions recorded";
  $("timeline").max = 0;
  status("Preparing scene and planning workers…");
  const id = ++revision;
  worker = new Worker(new URL("./simulation-worker.js", import.meta.url), {
    type: "module",
  });
  worker.onerror = (event) => {
    if (id === revision) error(event.message);
  };
  worker.onmessage = ({ data }) => {
    if (id !== revision) return;
    if (data.type === "error") {
      rewardChangePending = undefined;
      rewardSettings.setEnabled(ready);
      actionSettings.setEnabled(ready);
      error(data.message);
      return;
    }
    if (data.type === "rewards-error") {
      rewardChangePending = undefined;
      rewardSettings.setEnabled(ready);
      status(data.message, true);
      return;
    }
    if (data.type === "rewards-updated") {
      rewardChangePending = undefined;
      currentScene = copy(data.scene);
      editor.updateRewards(currentScene);
      appliedCoefficients = data.coefficients;
      rewardSettings.render(currentScene, appliedCoefficients);
      rewardSettings.setEnabled(true);
      replay.recording.addRewardChange({
        scene: currentScene, settings: { ...data.settings, seed: +$("seed").value },
        coefficients: data.coefficients, root: data.root, tick: data.tick, decision: data.decisions,
      });
      lastDiagnostics = undefined;
      playbackDecision = undefined;
      updateDiagnostics();
      if (!replay.active) renderer.diagnostics();
      status("Reward settings applied. World and recording preserved.");
      return;
    }
    if (data.type === "ready") {
      ready = true;
      currentInfo = data.info;
      currentChannels = data.channels;
      actionSettings.render(currentScene, currentChannels);
      $("controller-label").textContent = settings().algorithm.toUpperCase();
      try {
        renderer.load(currentScene, data.info, data.channels);
        updateViewControls();
        updateFlightControl(currentScene);
        $("flight-mode").disabled = false;
        replay.attach(
          importPending?.motion ||
            storage.create(
              data.info,
              data.root,
              currentScene.physics?.dt || 1 / 60,
              data.channels,
            ),
        );
      } catch (e) {
        error(e);
        return;
      }
      $("run").disabled = $("step").disabled = false;
      rewardSettings.setEnabled(true);
      actionSettings.setEnabled(true);
      $("backend").textContent =
        `${data.threads} ${data.threads === 1 ? "THREAD" : "THREADS"} / WEBASSEMBLY`;
      $("state-size").textContent = `${data.info[4] * 4} BYTES / WORLD`;
      editor.refreshChannels();
      status();
      if (checkpointPending) {
        worker.postMessage({
          ...checkpointPending,
          type: "restore-checkpoint",
        });
        checkpointPending = undefined;
      }
      if (importPending) {
        record = importPending.recording;
        const hasMotion = importPending.motion?.length;
        importPending = undefined;
        updateRecordUI();
        showRecord(record.entries.length - 1);
        if (hasMotion) replay.playback.seek(0);
      } else if (autoStep) {
        status("Growing the first search tree…");
        worker.postMessage({ type: "step" });
      } else if (continuation?.running) {
        running = true;
        renderer.setAnimationPlayback({ playing: true, speed: 1 });
        $("run").textContent = "Ⅱ Pause experiment";
        worker.postMessage({ type: "run", value: true });
      }
      return;
    }
    if (data.type === "motion") {
      try {
        replay.append(data);
      } catch (e) {
        stop();
        error(e);
      }
      return;
    }
    if (data.type === "checkpoint") {
      storage.downloadCheckpoint(data);
      replay.recording?.saveObject?.("checkpoint", data.decisions, data);
      status("Planner checkpoint saved.");
      return;
    }
    if (data.type === "checkpoint-restored") {
      status(
        data.wave
          ? "Wave restored. Advance Wave continues the saved population."
          : "Planner restored. Step continues the saved search or trajectory.",
      );
      return;
    }
    if (data.type === "inspection") {
      inspector.update(data.vectors);
      return;
    }
    if (data.type === "frame") {
      inspector.profile(data.profile);
      lastLiveFrame = data;
      if (replay.active) return;
      currentState = data.state;
      renderer.update(data.state, data.action);
      if ($("manual").checked) renderer.pulseAnimation();
      updateFrame(data);
      if (ready && !data.running)
        $("run-state").textContent = data.replay ? "REPLAY" : "PAUSED";
      return;
    }
    if (data.type === "diagnostics") {
      status();
      playbackDecision = undefined;
      if (!replay.active) renderer.diagnostics(data.tree, data.cloud);
      lastDiagnostics = data;
      if (!replay.active) updateDiagnostics(data);
      if (data.tree.meta.length) {
        try {
          record.append(
            {
              tree: data.tree,
              decision: data.decision,
              action: data.action,
              risk: data.risk,
              riskSamples: data.riskSamples,
              riskFrames: data.riskFrames,
              metrics: data.metrics,
              elapsed: data.elapsed,
            },
            !replay.recording?.id && $("archive-all").checked,
          );
        } catch (e) {
          stop();
          error(e);
        }
        replay.recording?.saveObject?.(
          "tree",
          data.decision,
          record.entries[record.entries.length - 1],
        );
        selectedRecord = record.entries.length - 1;
        updateRecordUI();
      }
      if (data.wave) $("run-state").textContent = "WAVE";
    }
    if (data.type === "late")
      $("latency").textContent =
        `${data.elapsed.toFixed(0)} MS · DEADLINE MISSED`;
    if (data.type === "snapshot")
      download(data.bytes, `${slug()}.fgcs`, "application/octet-stream");
  };
  worker.postMessage({
    type: "init",
    scene: copy(scene),
    settings: settings(),
    revision: id,
    seed: +$("seed").value,
    mode: $("clock").value,
    threads: +$("threads").value,
    recordingDecision: importPending?.lastDecision || 0,
    recordingInfo: importPending?.motion?.info,
    recordingRoot: importPending?.root || importPending?.motion?.root,
    recordingLast: importPending?.lastRows,
    continuationRows: continuation?.rows,
    continuationInfo: continuation?.info,
  });
}
function updateCargoReadout(
  label = scenePresentation(currentScene).score.label,
) {
  if (currentInfo?.[15] && currentState) {
    const ids = [
      ...new Set((currentChannels || []).map((channel) => channel.body)),
    ];
    const selected =
      editor.selection?.key === "bodies" ? editor.selection.i : ids[0];
    const c = ids.indexOf(selected),
      at = currentInfo[15] + Math.max(0, c) * 4;
    let delivered = 0;
    for (let i = 0; i < ids.length; i++)
      delivered += currentState[currentInfo[15] + 4 * i + 2];
    $("score-note").textContent =
      `${label} · ${delivered.toFixed(1)} units delivered · ${new Uint32Array(currentState.buffer, currentState.byteOffset)[4]} loads`;
    $("cargo-status").hidden = false;
    $("cargo-status").textContent =
      c < 0
        ? "Select a collecting vehicle"
        : `Vehicle ${selected + 1} · Cargo ${currentState[at].toFixed(1)} / ${currentScene.cargo.capacity ?? 5} · ${currentState[at + 1] ? "Return / unload" : "Collecting"}`;
  } else $("cargo-status").hidden = true;
}
function updateFrame(data) {
  const progress = data.trajectoryProgress;
  $("trajectory-progress").hidden = settings().algorithm !== "wave-jump";
  $("trajectory-progress").textContent = progress
    ? `Wave Jump · ${progress.executionMode || "full path"} · depth ${progress.searchDepth ?? "—"} · action ${progress.index + 1}/${progress.total} · ${progress.remaining} frames left · path reward ${progress.reward.toFixed(3)}`
    : settings().consensus_prefix
      ? "Wave Jump · search for shared prefix → execute → search"
      : "Wave Jump · search → execute full path → search";
  const m = data.metrics,
    dt = currentScene.physics?.dt || 1 / 60;
  $("time").innerHTML = `${(data.tick * dt).toFixed(2)} <small>s</small>`;
  $("tick").textContent = `TICK ${String(data.tick).padStart(6, "0")}`;
  const readout = sceneReadout(currentScene, m);
  $("score").textContent = String(readout.score);
  $("score-note").textContent = readout.label;
  updateCargoReadout(readout.label);
  $("footer-stats").textContent =
    `${currentInfo?.[1] || 0} BODIES · ${currentInfo?.[12] ?? (currentInfo?.[2] || 0) * 2} ACTION DIMENSIONS · ${data.missed} MISSED DEADLINES`;
  if (m[3] > 0 && running) {
    stop();
    status("Episode ended. Reset to start another run.");
  }
  if (data.running) $("run-state").textContent = "RUNNING";
}
function updateDiagnostics(data) {
  if (!data?.metrics) {
    for (const id of ["dead", "risk", "pruned", "used", "clone", "latency"])
      $(id).textContent = "—";
    $("risk-note").textContent = "NO PLANNING RECORD";
    return;
  }
  const m = data.metrics;
  $("dead").textContent = `${(m[9] * 100).toFixed(1)}%`;
  $("risk").textContent =
    data.risk == null ? "—" : `${(data.risk * 100).toFixed(1)}%`;
  $("risk-note").textContent =
    data.risk == null
      ? "NOT EVALUATED IN WAVE MODE"
      : `${data.riskSamples} SAMPLES · ${data.riskFrames} FRAMES`;
  $("pruned").textContent = Math.round(m[14]).toLocaleString();
  $("clone").textContent = `${(m[10] * 100).toFixed(0)}% CLONED`;
  $("used").textContent =
    `${Math.min(100, (data.budgetUsed ?? m[8] / +$("horizon").value) * 100).toFixed(0)}%`;
  $("latency").textContent =
    `${m[8]} ITERATIONS · ${data.elapsed.toFixed(0)} MS` +
    (data.selectedReward == null
      ? ""
      : ` · PATH REWARD ${data.selectedReward.toFixed(3)}${data.executionMode ? ` · ${data.executionMode}` : ""}`);
}
function updateRecordUI() {
  $("record-count").textContent =
    `${record.entries.length} decisions · ${(record.bytes / 1024).toFixed(0)} KiB`;
  $("timeline").max = Math.max(0, record.entries.length - 1);
  $("timeline").value = selectedRecord;
  const entry = record.entries[selectedRecord];
  if (entry?.tree.meta.length)
    $("node").value = entry.tree.meta[entry.tree.meta.length - 5];
}
function showRecord(i) {
  selectedRecord = Math.max(0, Math.min(record.entries.length - 1, i));
  const e = record.entries[selectedRecord];
  if (!e) return;
  renderer.diagnostics(e.tree);
  updateDiagnostics(e);
  updateRecordUI();
  $("run-state").textContent = "RECORD";
  $("record-count").textContent =
    `Decision ${e.decision} · ${e.tree.meta.length / 5} nodes · ${(record.bytes / 1024).toFixed(0)} KiB`;
}
function slug() {
  return (currentScene?.name || "fractal-control")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-");
}
function download(data, name, type = "application/json") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function upload(accept, callback) {
  $("file").accept = accept;
  $("file").value = "";
  $("file").onchange = async () => {
    try {
      const file = $("file").files[0];
      if (file) {
        if (file.size > 1024 * 1024 * 1024)
          throw new Error("File exceeds 1 GiB");
        await callback(file);
      }
    } catch (e) {
      error(e);
    }
  };
  $("file").click();
}
async function preset() {
  const request = ++presetRequest,
    scenario = $("scenario").value,
    sceneId = scenario === "racing" ? $("track").value || "racing" : scenario;
  let applying = false;
  stop();
  ready = false;
  $("run").disabled = $("step").disabled = true;
  try {
    const response = await fetch(`./scenarios/${sceneId}.json`);
    if (!response.ok) throw new Error("Unable to load scenario");
    const template = await response.json();
    if (request !== presetRequest) return;
    let scene = vehicleTypes.has(scenario)
      ? configureVehicleType(template, vehicleTypes.get(scenario), agentCatalog)
      : template;
    scene = configureVehicleCount(
      scene,
      vehicleCounts.get(scenario) ?? vehicleCount(template),
    );
    if (rockOptions(scene))
      scene = configureRocks(
        scene,
        miningOptions.get(scenario) ?? rockOptions(scene),
      );
    $("toy").textContent = String($("scenario").selectedIndex + 1).padStart(
      2,
      "0",
    );
    editor.clearHistory();
    applying = true;
    loadScene(scene, !initialized);
    initialized = true;
  } catch (e) {
    if (!applying && request !== presetRequest) return;
    error(e);
  }
}
$("scenario").onchange = preset;
$("track").onchange = () => {
  $("scenario").value = "racing";
  preset();
};
$("ants-vehicle-type").onchange = () => {
  if (!currentScene) return;
  const input = $("ants-vehicle-type");
  try {
    const scene = configureVehicleType(currentScene, input.value, agentCatalog);
    vehicleTypes.set($("scenario").value, input.value);
    vehicleCounts.set($("scenario").value, vehicleCount(scene));
    editor.clearHistory();
    loadScene(scene);
  } catch (e) {
    input.value = vehicleType(currentScene) || "";
    error(e);
  }
};
$("ants-vehicle-count").onchange = () => {
  const input = $("ants-vehicle-count");
  if (!input.checkValidity()) {
    input.reportValidity();
    return;
  }
  try {
    const count = +input.value;
    const scene = configureVehicleCount(currentScene, count);
    vehicleCounts.set($("scenario").value, count);
    editor.clearHistory();
    loadScene(scene);
  } catch (e) {
    input.value = vehicleCount(currentScene);
    error(e);
  }
};
$("flight-mode").onchange = () => {
  if (!currentScene) return;
  const scene = copy(currentScene);
  scene.environment = {
    ...(scene.environment || {}),
    flight: $("flight-mode").checked,
  };
  editor.clearHistory();
  loadScene(scene);
};
$("hook-stiffness-slider").oninput = () => {
  const value = +$("hook-stiffness-slider").value;
  $("hook-stiffness").value =
    value === 6 ? 1000000 : Math.round(10 ** value - 1);
};
$("hook-stiffness").oninput = () => {
  $("hook-stiffness-slider").value = Math.log10(
    Math.max(0, +$("hook-stiffness").value) + 1,
  );
};
$("rock-weight-slider").oninput = renderRockWeight;
$("apply-rocks").onclick = () => {
  if (!currentScene || !rockOptions(currentScene)) return;
  for (const id of ["rock-size", "rock-count"]) {
    if (!$(id).reportValidity()) return;
  }
  try {
    if (currentScene.tethers?.length && !$("hook-stiffness").reportValidity())
      return;
    const options = {
      scale: +$("rock-size").value,
      count: +$("rock-count").value,
      weight: rockWeight(),
      ...(currentScene.tethers?.length
        ? { stiffness: +$("hook-stiffness").value }
        : {}),
    };
    const scene = configureRocks(currentScene, options);
    miningOptions.set($("scenario").value, options);
    editor.clearHistory();
    loadScene(scene);
  } catch (e) {
    error(e);
  }
};
$("run").onclick = () => {
  replay.playback.live();
  running = !running;
  renderer.setAnimationPlayback({ playing: running, speed: 1 });
  $("run").textContent = running ? "Ⅱ Pause experiment" : "▶ Run experiment";
  worker.postMessage({ type: "run", value: running });
  status();
};
$("step").onclick = () => {
  replay.playback.live();
  stop();
  status("Planning one action…");
  worker.postMessage({ type: "step" });
};
$("reset").onclick = () => loadScene(currentScene);
$("wave").onclick = () => {
  replay.playback.live();
  stop();
  worker.postMessage({ type: "wave" });
};
for (const id of [
  "algorithm",
  "walkers",
  "horizon",
  "frames",
  "seed",
  "noise",
  "elites",
  "threads",
  "inertial",
  "recording",
  "clock",
])
  $(id).onchange = () => {
    if (!$(id).checkValidity()) {
      $(id).reportValidity();
      return;
    }
    if (id === "walkers") {
      $("elites").max = $("walkers").value;
      $("elites").value = Math.min(+$("elites").value, +$("walkers").value);
    }
    if (id === "algorithm") controllerSettings.render();
    loadScene(currentScene);
  };
for (const name of ["tree", "cloud", "geometry", "tethers"])
  $(`layer-${name}`).onchange = () =>
    renderer.setLayers({ [name]: $(`layer-${name}`).checked });
$("clean").onclick = () => {
  clean = !clean;
  if (clean) {
    oldLayers = { ...renderer.layers };
    renderer.setLayers({
      tree: false,
      cloud: false,
      geometry: false,
      tethers: false,
    });
  } else renderer.setLayers(oldLayers);
  $("clean").textContent = clean ? "Show diagnostics" : "Clean view";
};
$("world").addEventListener("camerachange", () => {
  $("focus").textContent =
    renderer.followBody == null ? "Follow agent" : "Whole arena";
});
$("reset-view").onclick = () => renderer.focus(null);
$("focus").onclick = () => {
  const body =
    renderer.followBody == null
      ? editor.selection?.key === "bodies"
        ? editor.selection.i
        : renderer.controlled[0]
      : null;
  renderer.focus(body ?? null);
  $("focus").textContent = body == null ? "Follow agent" : "Whole arena";
};
$("view").onclick = () => {
  renderer.top = !renderer.top;
  renderer.resize();
};
$("timeline").oninput = () => {
  stop();
  showRecord(+$("timeline").value);
};
$("replay").onclick = () => {
  const entry = record.entries[selectedRecord];
  if (entry) {
    replay.playback.live();
    stop();
    const configuration = replay.recording?.rewardConfigurationForRoot(entry.tree.root, entry.decision);
    if (configuration?.scene) {
      applyRewardUpdate(configuration.scene, coefficientValues(configuration.settings), undefined,
        configuration.root, { tree: entry.tree, node: +$("node").value });
      return;
    }
    worker.postMessage({
      type: "replay",
      tree: entry.tree,
      node: +$("node").value,
    });
  }
};
$("save-state").onclick = () => worker.postMessage({ type: "snapshot" });
$("load-state").onclick = () =>
  upload(".fgcs", async (file) => {
    replay.playback.live();
    stop();
    worker.postMessage({
      type: "restore",
      bytes: new Uint8Array(await file.arrayBuffer()),
    });
  });
$("export-run").onclick = async () => {
  try {
    if (replay.recording?.exportFile) {
      download(
        await replay.recording.exportFile(),
        `${slug()}.fgcrec`,
        "application/octet-stream",
      );
      return;
    }
    if (record.entries.length || replay.recording?.length)
      download(
        exportRecording(
          currentScene,
          { ...settings(), seed: +$("seed").value },
          record.entries,
          replay.recording,
        ),
        `${slug()}.fgclab`,
      );
    else status("Run or step the controller to record movement first.");
  } catch (e) {
    error(e);
  }
};
$("import-run").onclick = () =>
  upload(".fgclab,.fgcrec", async (file) => {
    if (file.name.endsWith(".fgcrec")) {
      const motion = await storage.import(file);
      importPending = {
        scene: motion.scene,
        settings: motion.settings,
        motion,
        recording: new Recording(),
        lastRows: await motion.getRows(motion.length - 1),
        lastDecision: (await motion.getFrame(motion.length - 1)).decision,
      };
    } else {
      importPending = importRecording(await file.text());
      if (importPending.motion?.length) {
        importPending.lastRows = importPending.motion.rows(
          importPending.motion.length - 1,
        );
        importPending.lastDecision = importPending.motion.frame(
          importPending.motion.length - 1,
        ).decision;
      }
    }
    const configuration = importPending.motion?.rewardConfiguration();
    importPending.root = configuration?.root;
    applySettings(configuration?.settings || importPending.settings);
    loadScene(configuration?.scene || importPending.scene);
  });
installManualControl({
  isReady: () => ready,
  channels: () => currentChannels,
  selectedBody: () =>
    editor.selection?.key === "bodies" ? editor.selection.i : undefined,
  apply(action) {
    replay.playback.live();
    stop();
    worker.postMessage({ type: "manual", action, frames: 2 });
  },
});

// Static controls are annotated in the HTML; controller and scene-editor
// controls call initHelp again when they replace their dynamic fields.
initHelp();

async function loadPresets() {
  try {
    const catalogResponse = await fetch("./agent-catalog.json");
    if (!catalogResponse.ok) throw new Error("Unable to load vehicle catalog");
    agentCatalog = await catalogResponse.json();
    const response = await fetch("./scenario-catalog.json");
    if (!response.ok) throw new Error("Unable to load environment catalog");
    const entries = await response.json();
    if (
      !Array.isArray(entries) ||
      !entries.length ||
      entries.some(
        (e) => !/^[a-z][a-z0-9_-]*$/.test(e.id) || typeof e.label !== "string",
      ) ||
      new Set(entries.map((e) => e.id)).size !== entries.length
    )
      throw new Error("Invalid environment catalog");
    $("scenario").replaceChildren(
      ...entries.map(
        (e, i) =>
          new Option(`${String(i + 1).padStart(2, "0")} · ${e.label}`, e.id),
      ),
    );
    const tracks = entries.find((entry) => entry.id === "racing")?.tracks || [];
    if (
      !tracks.length ||
      tracks.some(
        (track) =>
          !/^[a-z][a-z0-9_-]*$/.test(track.id) ||
          typeof track.label !== "string",
      ) ||
      new Set(tracks.map((track) => track.id)).size !== tracks.length
    )
      throw new Error("Invalid racing track catalog");
    $("track").replaceChildren(
      ...tracks.map((track) => new Option(track.label, track.id)),
    );
    $("scenario").disabled = false;
    await preset();
  } catch (e) {
    error(e);
  }
}
loadPresets();
