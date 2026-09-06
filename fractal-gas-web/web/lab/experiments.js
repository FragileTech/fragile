import "./controllers/index.js";
import { createController } from "./controllers/registry.js";
import { NativeEngine } from "./native.js";
import { MotionRecording, WorldCapture } from "./motion.js";
const yieldTask = () => new Promise((resolve) => setTimeout(resolve, 0));
export function validateExperiment(spec) {
  if (
    !Array.isArray(spec.seeds) ||
    !spec.seeds.length ||
    spec.seeds.length > 64 ||
    !spec.seeds.every((s) => Number.isInteger(s) && s >= 0 && s <= 0xffffffff)
  )
    throw new Error("Use 1–64 unsigned integer seeds");
  if (
    !Number.isInteger(spec.maxFrames) ||
    spec.maxFrames < 1 ||
    spec.maxFrames > 36000
  )
    throw new Error("Episode duration must be 1–36000 frames");
  if (
    !Array.isArray(spec.variants) ||
    !spec.variants.length ||
    spec.variants.length > 8
  )
    throw new Error("Use 1–8 controller variants");
  for (const s of spec.variants) {
    if (
      !Number.isInteger(s.walkers) ||
      s.walkers < 1 ||
      s.walkers > 8192 ||
      !Number.isInteger(s.horizon) ||
      s.horizon < 1 ||
      s.horizon > 4096 ||
      !Number.isInteger(s.frames) ||
      s.frames < 1 ||
      s.frames > 4096
    )
      throw new Error("Invalid experiment controller dimensions");
  }
}
const goals = new Map([
  ["deliveries", (m) => m[5]],
  ["pickups", (m) => m[6]],
  ["gates", (m) => m[7]],
  ["survival", (_, stats) => stats.frames],
  ["reward", (_, stats) => stats.reward],
]);
export function registerEvaluationMetric(name, evaluate) {
  if (goals.has(name) || typeof evaluate !== "function")
    throw new Error("Invalid evaluation metric");
  goals.set(name, evaluate);
}
export async function runEpisode({
  module,
  scene,
  settings,
  seed,
  maxFrames,
  goal,
  root,
  record = false,
  threads = 1,
  cancelled = () => false,
}) {
  const dt = scene.physics?.dt || 1 / 60,
    evaluation = goal ||
      scene.evaluation || { metric: "survival", target: maxFrames };
  if (
    !goals.has(evaluation.metric) ||
    !Number.isFinite(evaluation.target) ||
    evaluation.target <= 0
  )
    throw new Error("Invalid episode success criterion");
  const stats = {
    seed,
    settings,
    frames: 0,
    decisions: 0,
    reward: 0,
    collisions: 0,
    controlEffort: 0,
    planningMs: 0,
    success: false,
    dead: false,
    goal: evaluation,
  };
  const world = new NativeEngine(module, scene);
  let strategy;
  try {
    strategy = createController(module, scene, settings, threads);
    world.reset(seed);
    if (root) {
      world.restore(root.snapshot);
      world.restoreRows(root.rows);
    }
    const initial = world.snapshot(),
      motion = record ? new MotionRecording(world.info, initial, dt) : null;
    if (motion) motion.channels = world.channels;
    const capture = record
      ? new WorldCapture(world, ({ packet, label }) =>
          motion.append(packet, label),
        )
      : null;
    capture?.capture(world.neutralAction(), 0, "Experiment root");
    const initialMetrics = world.metrics();
    stats.dead = initialMetrics[3] > 0;
    stats.success =
      !stats.dead &&
      goals.get(evaluation.metric)(initialMetrics, stats) >= evaluation.target;
    while (stats.frames < maxFrames && !stats.dead && !stats.success) {
      if (cancelled()) throw new Error("Experiment cancelled");
      strategy.controller.begin(
        world.snapshot(),
        (seed + stats.decisions) >>> 0,
      );
      const start = performance.now();
      let done = false;
      while (!done) {
        done = strategy.controller.advance();
        if (cancelled()) throw new Error("Experiment cancelled");
        await yieldTask();
      }
      const decision = strategy.controller.result();
      stats.planningMs += performance.now() - start;
      stats.decisions++;
      const duration = Math.min(settings.frames, maxFrames - stats.frames);
      for (let frame = 0; frame < duration; ) {
        // One physical frame permits exact event times and stops at success/death.
        if (capture) capture.step(decision.action, 1, stats.decisions);
        else world.step(decision.action, 1);
        const m = world.metrics();
        stats.frames++;
        stats.reward += m[0];
        stats.collisions += m[2];
        stats.dead = m[3] > 0;
        stats.controlEffort +=
          decision.action.reduce((sum, u) => sum + u * u, 0) * dt;
        stats.success =
          !stats.dead &&
          goals.get(evaluation.metric)(m, stats) >= evaluation.target;
        frame++;
        if (
          stats.success ||
          stats.dead ||
          frame >= settings.frames ||
          stats.frames >= maxFrames
        )
          break;
      }
    }
    stats.simulationSeconds = stats.frames * dt;
    stats.completionSeconds = stats.success ? stats.simulationSeconds : null;
    stats.meanPlanningMs = stats.planningMs / Math.max(1, stats.decisions);
    stats.profile = Array.from(strategy.engine.profile());
    // Native counter of actual world-frames, accounting for terminal worlds
    // and zero-duration slots in a decaying population. Risk probes and the
    // authoritative world use separate engines and are excluded.
    stats.simulatorFrames = stats.profile[1];
    return { stats, motion, initial, final: world.snapshot() };
  } finally {
    world.dispose();
    strategy?.dispose();
  }
}
export function summarizeTrials(trials) {
  const groups = new Map();
  for (const trial of trials) {
    const key = JSON.stringify(trial.settings);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(trial);
  }
  return Array.from(groups, ([, rows]) => ({
    settings: rows[0].settings,
    episodes: rows.length,
    successRate: rows.filter((r) => r.success).length / rows.length,
    collisions: rows.reduce((s, r) => s + r.collisions, 0) / rows.length,
    controlEffort: rows.reduce((s, r) => s + r.controlEffort, 0) / rows.length,
    completionSeconds: rows.some((r) => r.success)
      ? rows
          .filter((r) => r.success)
          .reduce((s, r) => s + r.completionSeconds, 0) /
        rows.filter((r) => r.success).length
      : null,
    planningMs:
      rows.reduce((s, r) => s + r.planningMs, 0) /
      Math.max(
        1,
        rows.reduce((s, r) => s + r.decisions, 0),
      ),
    simulatorFrames:
      rows.reduce((sum, r) => sum + (r.simulatorFrames || 0), 0) / rows.length,
  }));
}
export async function runBenchmark(options) {
  validateExperiment(options.spec);
  const scenes = options.scenes || [options.scene];
  if (!scenes.length || scenes.length > 16)
    throw new Error("Use 1–16 benchmark scenes");
  const results = [];
  for (const [sceneIndex, scene] of scenes.entries())
    for (const settings of options.spec.variants)
      for (const seed of options.spec.seeds) {
        const { stats } = await runEpisode({
          ...options,
          scene,
          settings,
          seed,
          maxFrames: options.spec.maxFrames,
          goal: options.spec.goal,
          record: false,
        });
        stats.scene = scene.name || `Scene ${sceneIndex + 1}`;
        stats.sceneIndex = sceneIndex;
        results.push(stats);
        options.progress?.(
          stats,
          results.length,
          scenes.length *
            options.spec.variants.length *
            options.spec.seeds.length,
        );
      }
  return {
    version: 1,
    engine: "fractal-control-2",
    scene: scenes[0],
    scenes,
    spec: options.spec,
    results,
    summary: summarizeTrials(results),
    perScene: scenes.map((scene, index) => ({
      scene: scene.name || `Scene ${index + 1}`,
      summary: summarizeTrials(results.filter((r) => r.sceneIndex === index)),
    })),
  };
}
