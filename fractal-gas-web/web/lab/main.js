import { mountWorkspace } from "./workspace.js";
import {
  WorkspaceState,
  readPreference,
  savePreference,
} from "./workspace-state.js";
import {
  ConfigurationTransition,
  quiesceWorker,
  prepareWorker,
} from "./configuration-transition.js";
import { RunSession } from "./run-session.js";
const workspaceUI = mountWorkspace();
const workspace = new WorkspaceState();
const transitions = new ConfigurationTransition();
let runSession, driveControl, nextParent, presetLoading;
import { configureRocks, rockOptions } from "./rock-scene.js";
import { rockLiftBudget } from "./rock-lift.js";
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
  rewardValues,
} from "./reward-settings.js";
import { ActionSettings, withActionMultipliers } from "./action-settings.js";
import { resolveBodies } from "./agent-types.js";
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
  currentBodies,
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
    updateSelection();
    $("run-state").textContent = "WORLD REPLAY";
  },
  live() {
    playbackDecision = undefined;
    if (lastLiveFrame) {
      currentState = lastLiveFrame.state;
      renderer.update(currentState, lastLiveFrame.action);
      updateFrame(lastLiveFrame);
      updateDiagnostics(lastDiagnostics);
      updateDecision(lastDiagnostics);
      updateSelection();
      $("run-state").textContent = "PAUSED";
    }
  },
  async resume(rows, configuration) {
    const source = replay.recording;
    nextParent = { run: source.id, frame: replay.playback.cursor };
    return loadScene(configuration?.scene || source.scene, false, {
      settings: configuration?.settings || source.settings,
      rows,
      info: source.info,
    });
  },
  async decision(number) {
    if (number === playbackDecision) return;
    playbackDecision = number;
    renderer.clearDiagnostics();
    updateDiagnostics();
    updateDecision();
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
      updateDecision(record.entries[index]);
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
    if (workspace.mode !== "drive") {
      status(
        "Use Drive mode to apply actuator commands. Apply scene edits first.",
      );
      return;
    }
    stop();
    replay.playback.live();
    worker.postMessage({ type: "manual", action, frames: 1 });
  },
  isReady: () => ready,
  loadScene: stageScene,
  stop,
  status,
  error,
  download,
  upload,
  slug,
  onSelection: () => {
    updateCargoReadout();
    updateSelection();
    driveControl?.refresh();
  },
  onWorldClick(point) {
    if (workspace.timeline !== "decisions") {
      editor.selectAt(point);
      updateSelection();
      return;
    }
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
  getSettings: () => ({ ...settings() }),
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
    await loadScene(configuration.scene, false, {
      settings: configuration.settings,
    });
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
    await loadScene(data.scene, false, { settings: data.settings });
  },
  upload,
  download,
  status,
  error,
});
new ExperimentPanel({
  getScene: () => currentScene,
  getSettings: settings,
  getRoot: () => ({
    // The submitted scene uses active rewards; the selected rows supply only
    // physical state, including when viewing an older reward boundary.
    snapshot: replay.recording.rewardConfiguration().root,
    rows: currentRows(),
  }),
  stop,
  download,
  error,
});
const controllerSettings = new ControllerSettings(
  $("algorithm-settings"),
  $("algorithm"),
  () => stageSettings(),
);
const actionSettings = new ActionSettings(
  $("agent-action-settings"),
  $("apply-action-settings"),
  (values) => {
    if (!ready) return;
    try {
      const next = withActionMultipliers(workspace.draft.scene, values);
      stageScene(next);
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
  worker.postMessage({
    type: "update-rewards",
    scene,
    coefficients,
    rows,
    root,
    replayBranch,
  });
}
const rewardSettings = new RewardSettings($("reward-terms"), (values) => {
  if (!ready || rewardChangePending) return;
  try {
    stageRewards(values);
    applyConfiguration();
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
  request: () => {
    if (replay.active || workspace.readOnly) {
      renderer.inspectVectors([]);
      $("physics-readout").textContent =
        "Historical forces and contacts: Not recorded. Velocity and commands are shown in the inspector.";
      return;
    }
    worker?.postMessage({
      type: "inspect",
      rows: currentRows(),
      action: lastLiveFrame?.action,
    });
  },
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
  renderRockLift();
}
function renderRockLift() {
  const scene = workspace.draft?.scene;
  const budget = scene && rockLiftBudget(scene, rockWeight());
  $("rock-lift-controls").hidden = !budget;
  if (!budget) return;
  $("fit-rock-lift").disabled =
    !budget.suggestedWeight || rockWeight() <= budget.suggestedWeight;
  $("rock-lift-budget").textContent =
    budget.unavailable ||
    `Upward thrust: ${budget.thrust.toFixed(1)} N. Weight with the largest rock: ${budget.load.toFixed(1)} N. ` +
      (budget.canLift
        ? "Thrust exceeds weight."
        : "Too heavy to lift at this thrust.") +
      (budget.suggestedWeight
        ? ""
        : " Increase thrust before choosing a lighter flight load.");
}
function draftSettings() {
  return {
    ...controllerSettings.values(),
    algorithm: $("algorithm").value || "fmc",
    walkers: +$("walkers").value,
    horizon: +$("horizon").value,
    frames: +$("frames").value,
    seed: +$("seed").value,
    clock: $("clock").value,
    threads: +$("threads").value,
    ...appliedCoefficients,
    noise: +$("noise").value,
    elites: +$("elites").value,
    inertial: $("inertial").checked,
    recording: +$("recording").value,
  };
}
function settings() {
  return workspace.active?.settings || draftSettings();
}
function stop() {
  running = false;
  renderer.setAnimationPlayback({ playing: false });
  if (ready) worker?.postMessage({ type: "run", value: false });
  $("run").textContent = workspace.mode === "drive" ? "Start driving" : "Run";
  driveControl?.clear();
}
function commitScene(
  scene,
  autoStep = false,
  continuation = undefined,
  prepared,
  configuration,
) {
  workspace.commit(scene, configuration);
  currentBodies = resolveBodies(scene);
  workspace.readOnly = !!importPending;
  applySettings(configuration);
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
  worker = prepared.worker;
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
      workspace.commit(currentScene, {
        ...settings(),
        ...data.settings,
        ...data.coefficients,
      });
      editor.updateRewards(currentScene);
      appliedCoefficients = data.coefficients;
      rewardSettings.render(currentScene, appliedCoefficients);
      rewardSettings.setEnabled(true);
      replay.recording.addRewardChange({
        scene: currentScene,
        settings: { ...data.settings, seed: settings().seed },
        coefficients: data.coefficients,
        root: data.root,
        tick: data.tick,
        decision: data.decisions,
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
      if (nextParent && !importPending) replay.recording.parent = nextParent;
      nextParent = undefined;
      if (replay.recording) {
        replay.recording.readOnly = workspace.readOnly;
        $("run-name").value = replay.recording.name || currentScene.name;
      }
      driveControl?.refresh();
      updateSelection();
      status();
      checkpointPending = undefined;
      if (importPending) {
        record = importPending.recording;
        const hasMotion = importPending.motion?.length;
        importPending = undefined;
        updateRecordUI();
        showRecord(record.entries.length - 1);
        if (hasMotion) replay.playback.seek(0);
      } else if (autoStep) {
        // New worlds always start paused at tick zero.
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
      if (!replay.active && !workspace.readOnly) inspector.update(data.vectors);
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
      updateSelection();
      if (data.driveSlow)
        status("Driving is slower than real time on this device.");
      if (ready && !data.running)
        $("run-state").textContent = data.replay ? "REPLAY" : "PAUSED";
      return;
    }
    if (data.type === "diagnostics") {
      status();
      playbackDecision = undefined;
      if (!replay.active) renderer.diagnostics(data.tree, data.cloud);
      lastDiagnostics = data;
      if (!replay.active) {
        updateDecision(data);
        updateDiagnostics(data);
      }
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
              selectedReward: data.selectedReward,
              executionMode: data.executionMode,
              settings: copy(settings()),
              rewards: copy(currentScene.rewards),
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
  for (const event of prepared.messages) worker.onmessage(event);
  prepared.messages.length = 0;
  workspace.mode = "inspect";
  $("manual").checked = false;
  $("editor").hidden = true;
  refreshMode();
}
async function loadScene(scene, autoStep = false, continuation) {
  if (transitions.busy) return false;
  try {
    const configuration = continuation?.settings
      ? { ...settings(), ...copy(continuation.settings) }
      : {
          ...draftSettings(),
          ...coefficientValues(workspace.draft?.settings || draftSettings()),
        };
    const request = {
      type: "init",
      scene: copy(scene),
      settings: configuration,
      revision: revision + 1,
      seed: configuration.seed,
      mode: configuration.clock,
      threads: configuration.threads,
      recordingDecision: importPending?.lastDecision || 0,
      recordingInfo: importPending?.motion?.info,
      recordingRoot: importPending?.root || importPending?.motion?.root,
      recordingLast: importPending?.lastRows,
      continuationRows: continuation?.rows,
      continuationInfo: continuation?.info,
      snapshot: continuation?.snapshot,
      replayBranch: continuation?.replayBranch,
      checkpoint: checkpointPending,
    };
    document.querySelector("main").inert = true;
    document.querySelector(".workspace-toolbar").inert = true;
    await transitions.run({
      quiesce: async () => {
        stop();
        await quiesceWorker(worker, crypto.randomUUID());
      },
      save: () => runSession?.preserve(),
      prepare: () => prepareWorker(request),
      commit: (prepared) =>
        commitScene(scene, false, continuation, prepared, configuration),
    });
    return true;
  } catch (e) {
    importPending = checkpointPending = nextParent = undefined;
    error(e);
    return false;
  } finally {
    document.querySelector("main").inert = false;
    document.querySelector(".workspace-toolbar").inert = false;
    refreshMode();
  }
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
  const horizon =
    data.settings?.horizon ??
    (data === lastDiagnostics && !replay.active
      ? settings().horizon
      : undefined);
  const budget = data.budgetUsed ?? (horizon ? m[8] / horizon : undefined);
  $("used").textContent = Number.isFinite(budget)
    ? `${Math.min(100, budget * 100).toFixed(0)}%`
    : "Not recorded";
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
  updateDecision(e);
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
  presetLoading = request;
  document.body.dataset.loadingPreset = "true";
  $("apply-configuration").disabled = true;
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
    if (!initialized) {
      await loadScene(scene);
      initialized = true;
    } else {
      stageScene(scene);
      renderSetupDraft(scene);
    }
  } catch (e) {
    if (!applying && request !== presetRequest) return;
    error(e);
  } finally {
    if (presetLoading === request) {
      presetLoading = undefined;
      delete document.body.dataset.loadingPreset;
      $("apply-configuration").disabled = false;
    }
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
    const scene = configureVehicleType(
      workspace.draft.scene,
      input.value,
      agentCatalog,
    );
    vehicleTypes.set($("scenario").value, input.value);
    vehicleCounts.set($("scenario").value, vehicleCount(scene));

    stageScene(scene);
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
    const scene = configureVehicleCount(workspace.draft.scene, count);
    vehicleCounts.set($("scenario").value, count);

    stageScene(scene);
  } catch (e) {
    input.value = vehicleCount(currentScene);
    error(e);
  }
};
$("flight-mode").onchange = () => {
  if (!currentScene) return;
  const scene = copy(workspace.draft.scene);
  scene.environment = {
    ...(scene.environment || {}),
    flight: $("flight-mode").checked,
  };

  stageScene(scene);
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
$("fit-rock-lift").onclick = () => {
  const budget = rockLiftBudget(workspace.draft.scene);
  if (!budget?.suggestedWeight || rockWeight() <= budget.suggestedWeight)
    return;
  $("rock-weight-slider").value = Math.log10(budget.suggestedWeight);
  renderRockWeight();
  $("apply-rocks").click();
};
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
    const scene = configureRocks(workspace.draft.scene, options);
    miningOptions.set($("scenario").value, options);

    stageScene(scene);
  } catch (e) {
    error(e);
  }
};
$("run").onclick = () => {
  if (workspace.readOnly) {
    status("Create a run from a recorded frame to continue.");
    return;
  }
  if (workspace.mode === "edit") return;
  replay.playback.live();
  running = !running;
  renderer.setAnimationPlayback({ playing: running, speed: 1 });
  $("run").textContent = running
    ? "Pause"
    : workspace.mode === "drive"
      ? "Start driving"
      : "Run";
  worker.postMessage({ type: "run", value: running });
  if (!running) driveControl?.clear();
  status();
};
$("step").onclick = () => {
  if (workspace.readOnly || workspace.mode === "edit") return;
  if (workspace.mode === "drive") {
    driveControl.step();
    return;
  }
  replay.playback.live();
  stop();
  status("Planning one action…");
  worker.postMessage({ type: "step" });
};
$("reset").onclick = () =>
  workspace.dirty ? applyConfiguration() : loadScene(currentScene);
$("wave").onclick = () => {
  if (workspace.readOnly || workspace.mode !== "inspect") return;
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
    stageSettings();
  };
for (const name of ["tree", "cloud", "geometry", "tethers"]) {
  const input = $(`layer-${name}`);
  input.checked = readPreference(`layer.${name}`, input.checked);
  renderer.setLayers({ [name]: input.checked });
  input.onchange = () => {
    renderer.setLayers({ [name]: input.checked });
    savePreference(`layer.${name}`, input.checked);
  };
}
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
$("replay").onclick = async () => {
  const entry = record.entries[selectedRecord];
  if (!entry) return;
  const configuration = replay.recording.rewardConfigurationForRoot(
    entry.tree.root,
    entry.decision,
  );
  nextParent = {
    run: replay.recording.id,
    decision: entry.decision,
    node: +$("node").value,
  };
  await loadScene(configuration.scene, false, {
    settings: configuration.settings,
    replayBranch: { tree: entry.tree, node: +$("node").value },
  });
};
$("save-state").onclick = () => worker.postMessage({ type: "snapshot" });
$("load-state").onclick = () =>
  upload(".fgcs", async (file) => {
    await loadScene(currentScene, false, {
      settings: settings(),
      snapshot: new Uint8Array(await file.arrayBuffer()),
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
          { ...settings() },
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
    await loadScene(configuration?.scene || importPending.scene, false, {
      settings: configuration?.settings || importPending.settings,
    });
  });
driveControl = installManualControl({
  isReady: () => ready && !workspace.readOnly && workspace.mode === "drive",
  channels: () => currentChannels || [],
  selectedBody: () =>
    editor.selection?.key === "bodies"
      ? editor.selection.i
      : renderer.controlled[0],
  apply: (action) => worker?.postMessage({ type: "drive-action", action }),
  step: (action) => {
    stop();
    worker?.postMessage({ type: "manual", action, frames: 1 });
  },
  pause: stop,
});
installWorkspaceActions();

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
    const defaults = draftSettings();
    await preset();
    await workspaceUI.tasks(entries, async (id) => {
      applySettings(defaults);
      vehicleTypes.delete(id);
      vehicleCounts.delete(id);
      miningOptions.delete(id);
      $("scenario").value = id;
      await preset();
      await applyConfiguration();
    });
  } catch (e) {
    error(e);
  }
}
loadPresets();

function stageScene(scene) {
  if (!workspace.draft) return;
  workspace.draft.scene = copy(scene);
  if (workspace.mode === "edit") editor.setDraftScene(scene);
  workspace.changed();
  renderRockLift();
}
function stageSettings() {
  if (!workspace.draft) return;
  workspace.draft.settings = {
    ...draftSettings(),
    ...coefficientValues(workspace.draft.settings),
  };
  workspace.changed();
}
function stageRewards(values) {
  if (!workspace.draft) return;
  workspace.draft.scene = withRewards(workspace.draft.scene, values);
  Object.assign(workspace.draft.settings, coefficientValues(values));
  workspace.changed();
}
function readRewardDraft() {
  return Object.fromEntries(
    [...rewardSettings.inputs].map(([key, { number }]) => [
      key,
      Number(number.value),
    ]),
  );
}
function renderSetupDraft(scene) {
  $("ants-vehicle-count").value = vehicleCount(scene);
  $("ants-vehicle-type").value = vehicleType(scene) || "";
  updateFlightControl(scene);
  const rocks = rockOptions(scene);
  $("rock-controls").hidden = !rocks;
  if (rocks) {
    $("rock-size").value = rocks.scale;
    $("rock-count").value = rocks.count;
    $("rock-weight-slider").value = Math.log10(rocks.weight);
    $("hook-stiffness").value = scene.tethers?.[0]?.stiffness ?? 25;
    $("hook-stiffness-slider").value = Math.log10(
      +$("hook-stiffness").value + 1,
    );
    renderRockWeight();
  }
  $("track-control").hidden = scene.environment?.kind !== "circuit";
  $("description").textContent = scene.description || scene.name;
  actionSettings.render(scene);
  rewardSettings.render(scene, workspace.draft.settings);
}
function refreshPending() {
  const changes = workspace.changes;
  $("pending-settings").hidden = !changes.length;
  $("pending-count").textContent =
    `${changes.length} pending ${changes.length === 1 ? "change" : "changes"}`;
  $("pending-diff").replaceChildren(
    ...changes.map((change) => {
      const li = document.createElement("li");
      const format = (value) =>
        typeof value === "object"
          ? `${Array.isArray(value) ? value.length + " items" : "modified"}`
          : String(value ?? "default");
      li.textContent = `${change.path.replace(/^(scene|settings)\./, "").replaceAll("_", " ")}: ${format(change.before)} → ${format(change.after)}`;
      return li;
    }),
  );
  const rewardOnly = changes.every((c) =>
    /^(scene\.rewards\.|scene\.rewards$|scene\.cargo\.full_reward$|settings\.(reward_coef|distance_coef)$)/.test(
      c.path,
    ),
  );
  $("apply-configuration").textContent = rewardOnly
    ? "Apply to current run"
    : "Apply and restart";
  rewardSettings.apply.textContent = rewardOnly
    ? "Apply to current run"
    : "Apply and restart";
  if (changes.length) $("reset").textContent = "Apply and restart";
  else $("reset").textContent = "Restart";
}
async function applyConfiguration() {
  if (!workspace.active || transitions.busy || presetLoading) return false;
  for (const input of document.querySelectorAll(
    ".settings-panel input:not([type=range]),.settings-panel select",
  )) {
    if (!input.disabled && !input.checkValidity()) {
      input.reportValidity();
      status("Correct the highlighted setting before applying.", true);
      return false;
    }
  }
  stageSettings();
  const changes = workspace.changes;
  if (!changes.length) return true;
  const rewardOnly = changes.every((c) =>
    /^(scene\.rewards\.|scene\.rewards$|scene\.cargo\.full_reward$|settings\.(reward_coef|distance_coef)$)/.test(
      c.path,
    ),
  );
  if (rewardOnly && !workspace.readOnly) {
    applyRewardUpdate(
      workspace.draft.scene,
      coefficientValues(workspace.draft.settings),
    );
    return true;
  }
  return loadScene(workspace.draft.scene);
}
function discardConfiguration() {
  workspace.discard();
  applySettings(workspace.active.settings);
  renderSetupDraft(workspace.active.scene);
  editor.setScene(workspace.active.scene);
  renderer.clearDraft?.();
  refreshPending();
}
function refreshMode() {
  for (const mode of ["inspect", "edit", "drive"])
    $("mode-" + mode).setAttribute(
      "aria-pressed",
      String(workspace.mode === mode),
    );
  $("drive-controls").hidden = workspace.mode !== "drive";
  $("manual").checked = workspace.mode === "drive";
  $("step").textContent =
    workspace.mode === "drive"
      ? "Step physics frame"
      : settings().algorithm === "wave-jump"
        ? "Execute trajectory"
        : "Step action";
  $("run").textContent = running
    ? "Pause"
    : workspace.mode === "drive"
      ? "Start driving"
      : "Run";
  $("run").disabled = $("step").disabled =
    !ready || workspace.readOnly || workspace.mode === "edit";
  $("wave").disabled =
    !ready || workspace.readOnly || workspace.mode !== "inspect";
  $("run-name").disabled = workspace.readOnly;
  $("mode-hint").textContent =
    workspace.mode === "edit"
      ? "Editing a draft · Apply and restart commits changes"
      : workspace.mode === "drive"
        ? "Release keys to coast · Pause stops physics"
        : "Click a vehicle to inspect it · Drag to pan";
  $("playback-status").textContent =
    replay.active || workspace.readOnly ? "Replay" : "Live";
}
async function setMode(mode) {
  if (workspace.mode === "edit" && mode !== "edit" && workspace.dirty) {
    const decision = await new Promise((resolve) => {
      const dialog = $("leave-editor");
      let choice = "cancel";
      for (const key of ["apply", "discard", "cancel"])
        $("leave-" + key).onclick = () => {
          choice = key;
          dialog.close();
        };
      dialog.onclose = () => resolve(choice);
      dialog.showModal();
    });
    if (decision === "cancel") return;
    if (decision === "apply" && !(await applyConfiguration())) return;
    if (decision === "discard") discardConfiguration();
  }
  if (workspace.readOnly && mode !== "inspect") {
    status("Create a run from a recorded frame before editing or driving.");
    return;
  }
  stop();
  workspace.mode = mode;
  if (mode === "inspect") document.body.classList.remove("inspector-open");
  $("editor").hidden = mode !== "edit";
  document.body.classList.toggle("editing", mode === "edit");
  if (mode === "edit") editor.setDraftScene(workspace.draft.scene);
  else renderer.clearDraft();
  if (mode === "drive") {
    if (
      !renderer.controlled.includes(editor.selection?.i) ||
      editor.selection?.key !== "bodies"
    )
      editor.selectBody(renderer.controlled[0]);
    replay.playback.live();
    document.body.classList.add("inspector-open");
    $("world").focus();
  }
  worker?.postMessage({ type: "drive-mode", enabled: mode === "drive" });
  driveControl?.refresh();
  refreshMode();
}
function updateSelection() {
  if (!currentState || !currentInfo) return;
  const body =
    editor.selection?.key === "bodies"
      ? editor.selection.i
      : renderer.controlled[0];
  if (body == null || body >= currentInfo[1]) {
    $("selected-body").textContent = "Select a vehicle in the world.";
    $("selection-details").replaceChildren();
    return;
  }
  const definition = currentBodies[body];
  const B = currentInfo[1],
    action = replay.active ? renderer.action : lastLiveFrame?.action;
  $("selected-body").textContent =
    definition.name ||
    `Vehicle ${body + 1}${definition.agent_type ? " · " + definition.agent_type : ""}`;
  const selectedChannels = (currentChannels || [])
    .map((c, i) => ({ ...c, value: action?.[i] }))
    .filter((c) => c.body === body);
  const values = {
    Position: `${currentState[8 + body].toFixed(2)}, ${currentState[8 + B + body].toFixed(2)} m`,
    Velocity: `${currentState[8 + 2 * B + body].toFixed(2)}, ${currentState[8 + 3 * B + body].toFixed(2)} m/s`,
    "Angular velocity": `${currentState[8 + 5 * B + body].toFixed(2)} rad/s`,
    "Current commands":
      selectedChannels
        .map(
          (c) =>
            `${c.name}: ${c.value == null ? "Not recorded" : c.value.toFixed(2)}`,
        )
        .join(" · ") || "Uncontrolled body",
    Task: scenePresentation(currentScene).task_label,
    Destination: currentScene.bases?.length
      ? currentScene.bases
          .map((b) => `Base at ${b.position.join(", ")}`)
          .join(" · ")
      : "Not recorded",
    Mass: definition.mass == null ? "Scene default" : `${definition.mass} kg`,
  };
  const controlled = [
    ...new Set((currentChannels || []).map((c) => c.body)),
  ].indexOf(body);
  if (currentInfo[15] && controlled >= 0) {
    const at = currentInfo[15] + controlled * 4;
    values.Cargo = `${currentState[at].toFixed(1)} / ${currentScene.cargo.capacity ?? 5}`;
    values["Task state"] = currentState[at + 1]
      ? "Return / unload"
      : "Collecting";
  }
  $("selection-details").replaceChildren(
    ...Object.entries(values).flatMap(([name, value]) => {
      const dt = document.createElement("dt"),
        dd = document.createElement("dd");
      dt.textContent = name;
      dd.textContent = value;
      return [dt, dd];
    }),
  );
  if (workspace.mode !== "edit")
    renderer.select([currentState[8 + body], currentState[8 + B + body]]);
  $("execution-status").textContent = running
    ? workspace.mode === "drive"
      ? "Driving"
      : "Running"
    : "Paused";
  refreshMode();
}
function updateDecision(data) {
  if (!data) {
    $("decision-details").textContent = "No decision selected.";
    return;
  }
  const parts = [
    `Decision ${data.decision ?? "—"}`,
    `Selected path reward: ${data.selectedReward ?? "Not recorded"}`,
    `Action: ${data.action ? Array.from(data.action, (v) => v.toFixed(3)).join(", ") : "Not recorded"}`,
    `Outcome: ${data.executionMode || "Not recorded"}`,
    `Reward weights: ${data.rewards ? JSON.stringify(data.rewards) : "Not recorded"}`,
    "Individual reward contributions: Not recorded",
  ];
  $("decision-details").textContent = parts.join("\n");
}
function installWorkspaceActions() {
  workspace.addEventListener("change", refreshPending);
  runSession = new RunSession({
    getRecording: () => replay.recording,
    attach: (r) => replay.attach(r),
    getEntries: () => record.entries,
    download,
    status: (message) => {
      $("save-status").textContent = message;
    },
  });
  $("export-scene").onclick = () =>
    download(JSON.stringify(currentScene, null, 2), `${slug()}.json`);
  $("run-name").onchange = () => {
    if (!replay.recording || workspace.readOnly) return;
    replay.recording.name = $("run-name").value.trim() || currentScene.name;
    runSession.save().catch(error);
  };
  $("flush-run").onclick = () => runSession.save().catch(error);
  $("apply-configuration").onclick = applyConfiguration;
  $("discard-configuration").onclick = discardConfiguration;
  $("apply-editor").onclick = applyConfiguration;
  $("discard-editor").onclick = discardConfiguration;
  for (const mode of ["inspect", "edit", "drive"])
    $("mode-" + mode).onclick = () => setMode(mode);
  $("edit").onclick = () =>
    setMode(workspace.mode === "edit" ? "inspect" : "edit");
  $("close-editor").onclick = () => setMode("inspect");
  $("world").tabIndex = 0;
  document.addEventListener("lab-timeline", ({ detail }) => {
    workspace.timeline = detail;
  });
  document.addEventListener("lab-modal", () => {
    if (workspace.mode === "drive") stop();
  });
  $("reward-terms").addEventListener("input", () => {
    try {
      stageRewards(readRewardDraft());
    } catch {
      /* incomplete draft */
    }
  });
  $("reward-terms").addEventListener("click", (e) => {
    if (e.target.textContent === "Reset defaults")
      stageRewards(readRewardDraft());
  });
  $("agent-action-settings").addEventListener("input", () => {
    const values = {};
    for (const input of actionSettings.inputs)
      (values[input.dataset.agentType] ||= {})[input.dataset.channel] =
        +input.value;
    stageScene(withActionMultipliers(workspace.draft.scene, values));
  });
  for (const id of [
    "rock-size",
    "rock-count",
    "rock-weight-slider",
    "hook-stiffness",
    "hook-stiffness-slider",
  ])
    $(id).addEventListener("change", () => $("apply-rocks").click());
  $("apply-rocks").textContent = "Update draft";
  $("apply-action-settings").textContent = "Update draft";
  $("recording").addEventListener("change", stageSettings);
  $("persistent-recording").addEventListener("change", stageSettings);
  document.querySelectorAll(".controls [data-help]").forEach((node) => {
    node.dataset.help = node.dataset.help.replace(
      /Changing[^.]*restarts[^.]*\./g,
      "Apply and restart commits your changes.",
    );
  });
}
