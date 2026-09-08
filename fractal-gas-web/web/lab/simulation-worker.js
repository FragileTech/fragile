import { loadNative, NativeEngine } from "./native.js";
import { WorldCapture } from "./motion.js";
import { acceptPlan, branchActions } from "./timing.js";
import { TrajectoryCursor } from "./trajectory.js";
import { prepareRewardEngines } from "./live-rewards.js";
let updating = false,
  plannerThreads,
  stagedPlanner;
const deferredMessages = [];
let trajectoryCursor,
  trajectoryRoot,
  selectedReward,
  executionMode,
  searchDepth;
const isJump = () => settings?.algorithm === "wave-jump";
function clearTrajectory() {
  trajectoryCursor =
    trajectoryRoot =
    selectedReward =
    executionMode =
    searchDepth =
      undefined;
}
let engine,
  predict,
  planner,
  scene,
  settings,
  revision = 0,
  seed = 7;
let running = false,
  busy = false,
  ready = false,
  single = false,
  mode = "reproducible";
let tick = 0,
  target = 0,
  pending,
  action,
  timer,
  nextTime = 0,
  missed = 0,
  decisions = 0;
let waveStarted = false,
  capture;
const sendError = (error) => {
  running = single = false;
  postMessage({ type: "error", message: String(error.message || error) });
};
function publish(extra = {}) {
  const state = engine.states(),
    metrics = engine.metrics();
  tick = new Uint32Array(state.buffer)[0];
  postMessage(
    {
      type: "frame",
      state,
      metrics,
      tick,
      missed,
      decisions,
      action,
      running,
      trajectoryProgress: trajectoryCursor
        ? {
            index: trajectoryCursor.index,
            total: trajectoryCursor.trajectory.length,
            remaining: trajectoryCursor.remaining,
            reward: selectedReward,
            executionMode,
            searchDepth,
          }
        : undefined,
      profile: engine.profile(),
      ...extra,
    },
    [state.buffer, metrics.buffer],
  );
}
function request() {
  if (updating) return;
  if (busy || trajectoryCursor || !ready || (!running && !single)) return;
  if (isJump() && engine.metrics()[3] > 0) {
    running = single = false;
    publish();
    return;
  }
  let root = engine.snapshot();
  target = tick;
  if (mode === "realtime" && !single && !isJump()) {
    predict.restore(root);
    predict.step(action, settings.frames);
    root = predict.snapshot();
    target = new DataView(root.buffer, root.byteOffset).getUint32(32, true);
  }
  busy = true;
  planner.postMessage({
    type: "plan",
    root,
    target,
    revision,
    seed: (seed + decisions) >>> 0,
    budget:
      mode === "realtime" && !single && !isJump()
        ? Math.max(
            1,
            settings.frames * (scene.physics?.dt || 1 / 60) * 1000 - 20,
          )
        : 0,
  });
}
function commit(result) {
  action = result.action;
  if (result.trajectory) {
    trajectoryCursor = new TrajectoryCursor(result.trajectory, engine.channels);
    trajectoryRoot = result.root;
    selectedReward = result.selectedReward;
    executionMode = result.executionMode;
    searchDepth = result.searchDepth;
    nextTime = performance.now();
  }
  decisions++;
  postMessage({ ...result, type: "diagnostics", decision: decisions });
}
function trajectoryTick() {
  if (busy || (!running && !single) || !trajectoryCursor) return;
  const paced = mode === "realtime" && !single;
  const now = performance.now(),
    dt = (scene.physics?.dt || 1 / 60) * 1000;
  for (let i = 0; i < (paced ? 4 : 32) && trajectoryCursor; i++) {
    if (paced && now < nextTime) break;
    if (engine.metrics()[3] > 0) {
      clearTrajectory();
      running = single = false;
      break;
    }
    action = trajectoryCursor.action;
    capture.step(action, 1, decisions);
    trajectoryCursor.advance();
    nextTime += dt;
    if (trajectoryCursor.done || engine.metrics()[3] > 0) {
      const dead = engine.metrics()[3] > 0;
      clearTrajectory();
      single = false;
      if (dead) running = false;
      break;
    }
  }
  if (paced && nextTime < now - 4 * dt) nextTime = now + dt;
  publish();
  request();
}
function realtimeTick() {
  if (updating) return;
  if (isJump()) {
    try {
      trajectoryTick();
    } catch (error) {
      sendError(error);
    }
    return;
  }
  if (!running || mode !== "realtime") return;
  try {
    const now = performance.now(),
      dt = (scene.physics?.dt || 1 / 60) * 1000;
    let steps = 0;
    while (now >= nextTime && steps++ < 4) {
      if ((tick === target && busy) || (pending && tick >= pending.target)) {
        if (pending && acceptPlan(pending, tick, revision, engine.snapshot()))
          commit(pending);
        else {
          action = engine.neutralAction();
          missed++;
        }
        pending = undefined;
        // Keep the physical clock moving even if a planning iteration overruns.
        target = -1;
      }
      capture.step(action, 1, decisions);
      tick = new Uint32Array(engine.states().buffer)[0];
      nextTime += dt;
    }
    if (steps > 4) nextTime = now + dt;
    if (!busy && !pending) request();
    publish();
  } catch (error) {
    sendError(error);
  }
}
self.onmessage = async ({ data }) => {
  try {
    if (updating && data.type !== "close") {
      if (data.type === "run") running = data.value;
      else if (data.type !== "update-rewards") deferredMessages.push(data);
      return;
    }
    if (data.type === "init") {
      scene = data.scene;
      settings = data.settings;
      revision = data.revision;
      seed = data.seed;
      plannerThreads = data.threads;
      decisions = data.recordingDecision || 0;
      mode = data.mode;
      const module = await loadNative(false);
      engine = new NativeEngine(module, scene);
      predict = new NativeEngine(module, scene);
      engine.reset(seed);
      if (data.continuationRows) {
        if (
          !data.continuationInfo ||
          data.continuationInfo.some((v, i) => v !== engine.info[i])
        )
          throw new Error("Reward update changed the world layout");
        engine.restoreRows(data.continuationRows);
      }
      if (data.recordingRoot) {
        if (
          !data.recordingInfo ||
          data.recordingInfo.some((value, i) => value !== engine.info[i])
        )
          throw new Error("Recording scene layout mismatch");
        engine.restore(data.recordingRoot);
        engine.restoreRows(data.recordingLast);
      }
      capture = new WorldCapture(engine, (message, transfer) =>
        postMessage(message, transfer),
      );
      action = engine.neutralAction();
      planner = new Worker(new URL("./planner-worker.js", import.meta.url), {
        type: "module",
      });
      planner.onerror = (e) => {
        if (!updating) sendError(e.message);
      };
      const handlePlanner = ({ data: result, currentTarget }) => {
        if (updating || (currentTarget && currentTarget !== planner)) return;
        try {
          if (result.type === "error") {
            busy = false;
            sendError(result.message);
            return;
          }
          if (result.type === "ready") {
            ready = true;
            postMessage({
              type: "ready",
              info: engine.info,
              channels: engine.channels,
              descriptor: engine.descriptor(),
              threads: result.threads,
              root: engine.snapshot(),
            });
            capture.capture(action, decisions, "Initial world", true);
            publish();
            request();
            return;
          }
          if (result.type === "checkpoint") {
            busy = false;
            postMessage({
              ...result,
              root: engine.snapshot(),
              decisions,
              seed,
              execution: trajectoryCursor
                ? {
                    ...trajectoryCursor.checkpoint(),
                    selectedReward,
                    executionMode,
                    searchDepth,
                  }
                : undefined,
            });
            return;
          }
          if (result.type === "checkpoint-restored") {
            busy = false;
            postMessage(result);
            return;
          }
          if (result.type !== "plan") return;
          busy = false;
          if (mode === "realtime" && !single && !isJump()) {
            if (result.target >= tick && result.revision === revision)
              pending = result;
            else {
              postMessage({
                type: "late",
                elapsed: result.elapsed,
              });
              request();
            }
            return;
          }
          if (isJump()) {
            if (acceptPlan(result, tick, revision, engine.snapshot())) {
              commit(result);
              publish();
            } else request();
            return;
          }
          if (
            (running || single) &&
            acceptPlan(result, tick, revision, engine.snapshot())
          ) {
            commit(result);
            capture.step(action, settings.frames, decisions);
            single = false;
            publish();
            request();
          } else request();
        } catch (error) {
          sendError(error);
        }
      };
      planner.onmessage = handlePlanner;
      planner.postMessage({
        type: "init",
        scene,
        settings,
        revision,
        threads: data.threads,
      });
      timer = setInterval(realtimeTick, 8);
      return;
    }
    if (data.type === "update-rewards") {
      updating = true;
      let prepared, candidate;
      try {
        prepared = prepareRewardEngines(
          engine,
          scene,
          data.scene,
          data.coefficients,
          data.rows,
          data.root,
        );
        const nextSettings = { ...settings, ...prepared.coefficients };
        candidate = stagedPlanner = new Worker(
          new URL("./planner-worker.js", import.meta.url),
          {
            type: "module",
          },
        );
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(
            () => reject(new Error("Reward planner preparation timed out")),
            30000,
          );
          const finish = (error) => {
            clearTimeout(timeout);
            if (error) reject(error);
            else resolve();
          };
          candidate.onerror = (e) => finish(new Error(e.message));
          candidate.onmessage = ({ data: result }) => {
            if (result.type === "ready") finish();
            if (result.type === "error") finish(new Error(result.message));
          };
          candidate.postMessage({
            type: "init",
            scene: prepared.scene,
            settings: nextSettings,
            revision: revision + 1,
            threads: plannerThreads,
          });
        });
        const root = prepared.engine.snapshot();
        const nextAction = prepared.engine.neutralAction();
        const nextTick = new Uint32Array(prepared.engine.states().buffer)[0];
        const nextCapture = new WorldCapture(
          prepared.engine,
          (message, transfer) => postMessage(message, transfer),
        );
        const packet = nextCapture.packet(1);
        nextCapture.write(packet, 0, nextAction, decisions);
        const handler = planner.onmessage;
        planner.onmessage = null;
        planner.terminate();
        engine.dispose();
        predict.dispose();
        engine = prepared.engine;
        predict = prepared.predict;
        scene = prepared.scene;
        settings = nextSettings;
        planner = candidate;
        candidate.onmessage = handler;
        candidate.onerror = (e) => {
          if (!updating) sendError(e.message);
        };
        revision++;
        clearTrajectory();
        pending = undefined;
        waveStarted = busy = single = false;
        action = nextAction;
        capture = nextCapture;
        tick = nextTick;
        postMessage({
          type: "rewards-updated",
          scene,
          coefficients: prepared.coefficients,
          settings,
          root,
          tick,
          decisions,
        });
        postMessage(
          {
            type: "motion",
            packet,
            label: data.rows
              ? "Continued from replay · rewards restored"
              : "Reward settings changed",
          },
          [packet.buffer],
        );
        if (data.replayBranch)
          deferredMessages.unshift({ type: "replay", ...data.replayBranch });
      } catch (error) {
        candidate?.terminate();
        prepared?.engine.dispose();
        prepared?.predict.dispose();
        // A result may have arrived while preparation suspended its handler.
        busy = false;
        revision++;
        pending = undefined;
        postMessage({
          type: "rewards-error",
          message: String(error.message || error),
        });
      } finally {
        stagedPlanner = undefined;
        updating = false;
        nextTime = performance.now();
      }
      publish();
      for (const message of deferredMessages.splice(0))
        await self.onmessage({ data: message });
      request();
      return;
    }
    if (data.type === "run") {
      running = data.value;
      single = false;
      nextTime = performance.now();
      if (running) request();
      publish();
    }
    if (data.type === "step") {
      running = false;
      single = true;
      request();
    }
    if (data.type === "manual") {
      clearTrajectory();
      running = false;
      single = false;
      revision++;
      pending = undefined;
      waveStarted = false;
      action.set(data.action);
      capture.step(action, data.frames || 1, decisions);
      publish();
    }
    if (data.type === "wave") {
      clearTrajectory();
      running = false;
      single = false;
      revision++;
      pending = undefined;
      if (!waveStarted) {
        engine.begin(settings, seed);
        waveStarted = true;
      }
      engine.waveStep();
      const cloud = engine.states(true, settings.walkers),
        tree = engine.tree();
      engine.restoreRows(cloud.slice(0, engine.stride));
      postMessage({
        type: "diagnostics",
        tree,
        cloud,
        metrics: engine.metrics(),
        action: engine.action(),
        elapsed: engine.metrics()[15],
        decision: ++decisions,
        wave: true,
      });
      action = engine.action();
      capture.capture(action, decisions, "Wave selection");
      publish();
    }
    if (data.type === "inspect") {
      predict.restoreRows(data.rows || engine.states());
      postMessage({
        type: "inspection",
        vectors: predict.inspect(data.action || action),
      });
    }
    if (data.type === "checkpoint") {
      if (trajectoryCursor) {
        running = single = false;
        busy = true;
        planner.postMessage({
          type: "checkpoint",
          id: data.id,
          root: trajectoryRoot,
          seed: (seed + decisions - 1) >>> 0,
        });
        publish();
        return;
      }
      running = single = false;
      revision++;
      pending = undefined;
      if (waveStarted) {
        postMessage({
          type: "checkpoint",
          wave: true,
          checkpoint: { algorithm: "wave", bytes: engine.checkpoint() },
          root: engine.snapshot(),
          decisions,
          seed,
        });
        return;
      }
      planner.postMessage({
        type: "checkpoint",
        id: data.id,
        root: engine.snapshot(),
        seed: (seed + decisions) >>> 0,
      });
    }
    if (data.type === "restore-checkpoint") {
      clearTrajectory();
      if (data.execution) {
        if (!isJump() || data.checkpoint?.algorithm !== "wave-jump")
          throw new Error("Trajectory checkpoint type mismatch");
        trajectoryCursor = new TrajectoryCursor(
          data.execution.trajectory,
          engine.channels,
          data.execution,
        );
        trajectoryRoot = data.checkpoint.root;
        selectedReward = data.execution.selectedReward;
        executionMode = data.execution.executionMode;
        searchDepth = data.execution.searchDepth;
      }
      running = single = false;
      revision++;
      pending = undefined;
      waveStarted = false;
      engine.restore(data.root);
      action = engine.neutralAction();
      if (data.wave) {
        engine.restoreCheckpoint(data.checkpoint.bytes);
        waveStarted = true;
        decisions = data.decisions;
        seed = data.seed;
        action = engine.action();
        capture.capture(action, decisions, "Restored Wave checkpoint");
        publish();
        postMessage({ type: "checkpoint-restored", wave: true });
        return;
      }
      decisions = data.decisions;
      seed = data.seed;
      busy = true;
      planner.postMessage({
        type: "restore-checkpoint",
        checkpoint: data.checkpoint,
      });
      capture.capture(action, decisions, "Restored planner checkpoint");
      publish();
    }
    if (data.type === "snapshot")
      postMessage({ type: "snapshot", bytes: engine.snapshot() });
    if (data.type === "restore") {
      clearTrajectory();
      running = false;
      single = false;
      revision++;
      engine.restore(data.bytes);
      pending = undefined;
      action = engine.neutralAction();
      waveStarted = false;
      capture.capture(action, decisions, "Restored snapshot");
      publish();
    }
    if (data.type === "resume-motion") {
      clearTrajectory();
      running = single = false;
      revision++;
      pending = undefined;
      waveStarted = false;
      engine.restoreRows(data.rows);
      action = engine.neutralAction();
      capture.capture(action, decisions, "Continued from replay");
      publish();
    }
    if (data.type === "replay") {
      clearTrajectory();
      running = false;
      single = false;
      revision++;
      pending = undefined;
      engine.restore(data.tree.root);
      waveStarted = false;
      action = engine.neutralAction();
      capture.capture(action, decisions, `Search branch / node ${data.node}`);
      for (const edge of branchActions(data.tree, data.node))
        capture.step(edge.action, edge.frames, decisions);
      action = engine.neutralAction();
      publish({ replay: data.node });
    }
    if (data.type === "close") {
      clearInterval(timer);
      stagedPlanner?.terminate();
      planner?.terminate();
      engine?.dispose();
      predict?.dispose();
      self.close();
    }
  } catch (error) {
    sendError(error);
  }
};
