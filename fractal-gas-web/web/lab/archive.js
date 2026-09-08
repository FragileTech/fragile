import { treePoseDim, treeWidth } from "./actions.js";
import { encode, decode, checksum } from "./binary.js";
import { MotionRecording } from "./motion.js";
const LIMIT = 64 * 1024 * 1024;
export function treeBytes(tree) {
  return tree.meta.byteLength + tree.values.byteLength + tree.root.byteLength;
}
export class Recording {
  entries = [];
  bytes = 0;
  append(entry, keepAll = false) {
    const size = treeBytes(entry.tree);
    if (size > LIMIT || (keepAll && this.bytes + size > LIMIT))
      throw new Error(
        "Recording reached 64 MiB. Export this run and start a new recording.",
      );
    this.entries.push(entry);
    this.bytes += size;
    while (!keepAll && (this.entries.length > 32 || this.bytes > LIMIT))
      this.bytes -= treeBytes(this.entries.shift().tree);
  }
}
export function exportRecording(scene, settings, entries, motion) {
  const frames = motion?.pack();
  return JSON.stringify({
    version: 2,
    motion: motion
      ? {
          info: motion.info,
          channels: motion.channels,
          root: encode(motion.root),
          dt: motion.dt,
          segments: motion.segments,
          events: motion.events,
          rewardChanges: motion.rewardChanges.map((change) => ({
            ...change,
            root: encode(change.root),
          })),
          frames: encode(frames),
          checksum: checksum(frames),
        }
      : undefined,
    engine: "fractal-control-1",
    scene: motion?.scene || scene,
    settings: motion?.settings || settings,
    entries: entries.map((e) => ({
      decision: e.decision,
      risk: e.risk,
      riskSamples: e.riskSamples,
      riskFrames: e.riskFrames,
      metrics: e.metrics ? Array.from(e.metrics) : undefined,
      elapsed: e.elapsed,
      action: Array.from(e.action || []),
      tree: {
        dim: e.tree.dim,
        poseDim: treePoseDim(e.tree),
        meta: encode(e.tree.meta),
        values: encode(e.tree.values),
        root: encode(e.tree.root),
      },
    })),
  });
}
export function importRecording(text) {
  if (text.length > 192 * 1024 * 1024)
    throw new Error("Recording file exceeds 192 MiB");
  const data = JSON.parse(text);
  if (
    ![1, 2].includes(data.version) ||
    data.engine !== "fractal-control-1" ||
    !Array.isArray(data.entries)
  )
    throw new Error("Unsupported recording format");
  const recording = new Recording();
  for (const e of data.entries) {
    const tree = {
      dim: e.tree.dim,
      poseDim: treePoseDim(e.tree),
      meta: decode(e.tree.meta, Uint32Array),
      values: decode(e.tree.values, Float32Array),
      root: decode(e.tree.root, Uint8Array),
    };
    if (
      !Number.isInteger(tree.dim) ||
      tree.dim < 1 ||
      tree.dim > 8192 ||
      tree.meta.length % 5 ||
      !Number.isInteger(tree.poseDim) ||
      tree.poseDim < 2 ||
      tree.poseDim > 8192 ||
      tree.poseDim % 2 ||
      tree.values.length !== (tree.meta.length / 5) * treeWidth(tree)
    )
      throw new Error("Invalid tree shape");
    if (
      e.metrics &&
      (!Array.isArray(e.metrics) ||
        e.metrics.length !== 16 ||
        !e.metrics.every(Number.isFinite))
    )
      throw new Error("Invalid planning metrics");
    recording.append({ ...e, action: Float32Array.from(e.action), tree }, true);
  }
  let motion;
  if (data.version === 2 && data.motion) {
    const m = data.motion,
      frames = decode(m.frames);
    if (checksum(frames) !== m.checksum)
      throw new Error("Motion checksum mismatch");
    motion = new MotionRecording(m.info, decode(m.root), m.dt);
    motion.append(frames);
    motion.validate();
    motion.scene = data.scene;
    motion.settings = data.settings;
    motion.restoreRewardChanges(
      (m.rewardChanges || []).map((change) => ({
        ...change,
        root: decode(change.root),
      })),
    );
    if (
      !motion.length ||
      !Array.isArray(m.segments) ||
      m.segments.some(
        (s, i) =>
          !Number.isInteger(s.frame) ||
          s.frame < 0 ||
          s.frame >= motion.length ||
          typeof s.label !== "string" ||
          s.label.length > 120 ||
          (i > 0 && s.frame <= m.segments[i - 1].frame),
      )
    )
      throw new Error("Invalid motion segments");
    motion.channels = m.channels;
    motion.segments = m.segments;
    motion.events = Array.isArray(m.events)
      ? m.events.filter(
          (e) =>
            Number.isInteger(e.frame) &&
            e.frame >= 0 &&
            e.frame < motion.length &&
            typeof e.label === "string",
        )
      : [];
  }
  return { ...data, recording, motion };
}
