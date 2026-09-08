// Thin memory adapter. Physics, state restoration and Wave all live in C++.
export const MAX_CONTROL_THREADS = 64;
export function controlThreads(value = 8) {
  if (!Number.isInteger(value) || value < 1 || value > MAX_CONTROL_THREADS)
    throw new RangeError(
      `Worker threads must be an integer from 1 to ${MAX_CONTROL_THREADS}`,
    );
  return value;
}
export async function loadNative(threaded = false, threads = 8) {
  threads = threaded ? controlThreads(threads) : 1;
  const { default: create } = await import(
    threaded ? "./engine/control-threaded.mjs" : "./engine/control.mjs"
  );
  // Prewarm exactly the requested pool before synchronous native construction.
  return create({ controlThreads: threads });
}

export class NativeEngine {
  constructor(module, scene, worlds = 1, threads = 1) {
    this.m = module;
    this.h = this.string(JSON.stringify(scene), (p) =>
      module._fgc_create(p, worlds, threads),
    );
    if (!this.h) throw new Error(module.UTF8ToString(module._fgc_error()));
    this.info = Array.from({ length: 16 }, (_, i) =>
      module._fgc_info(this.h, i),
    );
    [this.worlds, this.bodies, this.controlled, this.stride, this.words] =
      this.info;
    this.dim = this.info[12];
    this.channels = Array.from({ length: this.dim }, (_, i) => ({
      body: module._fgc_action_body(this.h, i),
      name: module.UTF8ToString(module._fgc_action_name(this.h, i)),
      low: module._fgc_action_bound(this.h, i, 0),
      high: module._fgc_action_bound(this.h, i, 1),
    }));
    this.scratch = 0;
    this.capacity = 0;
  }
  check(code) {
    if (code < 0) throw new Error(this.m.UTF8ToString(this.m._fgc_error()));
    return code;
  }
  string(value, fn) {
    const m = this.m,
      size = m.lengthBytesUTF8(value) + 1,
      p = m._malloc(size);
    if (!p) throw new Error("Native allocation failed");
    try {
      m.stringToUTF8(value, p, size);
      return fn(p);
    } finally {
      m._free(p);
    }
  }
  buffer(bytes) {
    if (bytes > this.capacity) {
      this.m._free(this.scratch);
      this.scratch = this.m._malloc(bytes);
      this.capacity = bytes;
      if (!this.scratch) throw new Error("Native allocation failed");
    }
    return this.scratch;
  }
  dispose() {
    if (this.h) this.m._fgc_destroy(this.h);
    this.m._free(this.scratch);
    this.h = this.scratch = 0;
  }
  neutralAction() {
    return Float32Array.from({ length: this.worlds * this.dim }, (_, i) =>
      Math.max(
        this.channels[i % this.dim].low,
        Math.min(this.channels[i % this.dim].high, 0),
      ),
    );
  }
  reset(seed = 7) {
    this.check(this.m._fgc_reset(this.h, seed >>> 0, 0));
  }
  states(wave = false, walkers = 1) {
    const p = wave
      ? this.m._fgc_wave_states(this.h)
      : this.m._fgc_states(this.h);
    if (!p) throw new Error(this.m.UTF8ToString(this.m._fgc_error()));
    return this.m.HEAPF32.slice(
      p / 4,
      p / 4 + this.stride * (wave ? walkers : this.worlds),
    );
  }
  restoreRows(rows) {
    const bytes = new Uint8Array(rows.buffer, rows.byteOffset, rows.byteLength),
      p = this.buffer(bytes.length);
    this.m.HEAPU8.set(bytes, p);
    this.check(this.m._fgc_set_states(this.h, p, bytes.length));
  }
  snapshot() {
    const size = this.m._fgc_snapshot_size(this.h),
      p = this.buffer(size);
    this.check(this.m._fgc_serialize(this.h, p, size));
    return this.m.HEAPU8.slice(p, p + size);
  }
  restore(bytes) {
    const p = this.buffer(bytes.length);
    this.m.HEAPU8.set(bytes, p);
    this.check(this.m._fgc_deserialize(this.h, p, bytes.length));
  }
  broadcast(bytes) {
    const p = this.buffer(bytes.length);
    this.m.HEAPU8.set(bytes, p);
    this.check(this.m._fgc_broadcast(this.h, p, bytes.length));
  }
  results() {
    const p = this.m._fgc_results(this.h) / 4;
    return this.m.HEAPF32.slice(p, p + this.worlds * 4);
  }
  profile() {
    const p = this.m._fgc_profile(this.h) / 8;
    return this.m.HEAPF64.slice(p, p + 12);
  }
  resetProfile() {
    this.check(this.m._fgc_profile_reset(this.h));
  }
  inspect(actions) {
    if (actions) {
      if (
        actions.length !== this.worlds * this.dim ||
        !actions.every(Number.isFinite)
      )
        throw new Error("Invalid inspection action");
      this.m.HEAPF32.set(actions, this.m._fgc_actions(this.h) / 4);
    }
    const n = this.check(this.m._fgc_inspect(this.h)),
      p = this.m._fgc_inspection(this.h) / 4;
    return this.m.HEAPF32.slice(p, p + n * 8);
  }
  gather(indices) {
    const p = this.buffer(indices.byteLength);
    this.m.HEAP32.set(indices, p / 4);
    this.check(this.m._fgc_gather(this.h, p, indices.length));
  }
  copyStates(out = new Float32Array(this.worlds * this.stride)) {
    if (out.length !== this.worlds * this.stride)
      throw new Error("State output shape mismatch");
    const p = this.buffer(out.byteLength);
    this.check(this.m._fgc_get_states(this.h, p, out.byteLength));
    new Uint8Array(out.buffer, out.byteOffset, out.byteLength).set(
      this.m.HEAPU8.subarray(p, p + out.byteLength),
    );
    return out;
  }
  checkpoint() {
    const size = this.m._fgc_checkpoint_size(this.h);
    if (!size) throw new Error(this.m.UTF8ToString(this.m._fgc_error()));
    const p = this.buffer(size);
    this.check(this.m._fgc_checkpoint_write(this.h, p, size));
    return this.m.HEAPU8.slice(p, p + size);
  }
  restoreCheckpoint(bytes) {
    const p = this.buffer(bytes.length);
    this.m.HEAPU8.set(bytes, p);
    this.check(this.m._fgc_checkpoint_restore(this.h, p, bytes.length));
  }
  descriptor() {
    return {
      version: 1,
      backend: "fractal-control",
      state: {
        words: this.words,
        stride: this.stride,
        bytes: this.words * 4,
        opaque: true,
      },
      channels: this.channels,
      observationDimension: this.info[11],
      capabilities: ["batch-step", "gather", "snapshot", "planner-checkpoint"],
    };
  }
  step(actions, frames = 1) {
    const m = this.m;
    if (actions.length !== this.dim * this.worlds)
      throw new Error("Action batch shape mismatch");
    m.HEAPF32.set(actions, m._fgc_actions(this.h) / 4);
    const p = m._fgc_frames(this.h) / 4;
    if (typeof frames === "number") m.HEAP32.fill(frames, p, p + this.worlds);
    else {
      if (frames.length !== this.worlds)
        throw new Error("Duration shape mismatch");
      m.HEAP32.set(frames, p);
    }
    this.check(m._fgc_step(this.h));
  }
  metrics() {
    const p = this.m._fgc_metrics(this.h) / 4;
    return this.m.HEAPF32.slice(p, p + 16);
  }
  begin(settings, seed) {
    this.string(JSON.stringify(settings), (p) =>
      this.check(this.m._fgc_plan_begin(this.h, p, seed >>> 0)),
    );
  }
  advance() {
    return !!this.check(this.m._fgc_plan_advance(this.h));
  }
  waveStep() {
    this.check(this.m._fgc_wave_step(this.h));
  }
  commonAncestor() {
    return this.check(this.m._fgc_plan_common_ancestor(this.h));
  }
  bestLeaf() {
    return this.check(this.m._fgc_plan_best_leaf(this.h));
  }
  action() {
    const p = this.m._fgc_plan_action(this.h);
    if (!p) throw new Error(this.m.UTF8ToString(this.m._fgc_error()));
    return this.m.HEAPF32.slice(p / 4, p / 4 + this.dim);
  }
  planResult() {
    const p = this.m._fgc_plan_result(this.h);
    if (!p) throw new Error(this.m.UTF8ToString(this.m._fgc_error()));
    const plan = JSON.parse(this.m.UTF8ToString(p));
    plan.trajectory = plan.trajectory.map((edge) => ({
      ...edge,
      action: Float32Array.from(edge.action),
    }));
    plan.action = plan.trajectory[0].action;
    return plan;
  }
  tree() {
    const m = this.m,
      count = this.check(m._fgc_tree_export(this.h));
    const a = m._fgc_tree_meta(this.h) / 4,
      b = m._fgc_tree_values(this.h) / 4;
    const root = m._fgc_tree_root(this.h),
      size = m._fgc_tree_root_size(this.h);
    return {
      meta: m.HEAPU32.slice(a, a + 5 * count),
      values: m.HEAPF32.slice(
        b,
        b + count * (3 + this.dim + 2 * this.controlled),
      ),
      root: m.HEAPU8.slice(root, root + size),
      dim: this.dim,
      poseDim: this.controlled * 2,
    };
  }
}
