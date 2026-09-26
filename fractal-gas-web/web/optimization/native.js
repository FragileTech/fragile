export const ENGINE_VERSION = "fgopt-11";
export class NativeOptimization {
  constructor(module) {
    this.m = module;
    this.handle = 0;
  }
  error() {
    return new Error(this.m.UTF8ToString(this.m._fgo_error()));
  }
  check(code) {
    if (code < 0) throw this.error();
    return code;
  }
  catalog() {
    return JSON.parse(this.m.UTF8ToString(this.m._fgo_catalog()));
  }
  create(config) {
    const text = JSON.stringify(config),
      size = this.m.lengthBytesUTF8(text) + 1;
    const p = this.m._malloc(size);
    if (!p) throw new Error("Unable to allocate configuration");
    let next;
    try {
      this.m.stringToUTF8(text, p, size);
      next = this.m._fgo_create(p);
    } finally {
      this.m._free(p);
    }
    if (!next) throw this.error();
    if (this.handle) this.m._fgo_destroy(this.handle);
    this.handle = next;
    const resolved = JSON.parse(this.m.UTF8ToString(this.m._fgo_config(next)));
    this.dimension = resolved.dimensions;
    return resolved;
  }
  setGeometryDiagnostics(enabled) {
    this.check(this.m._fgo_geometry_diagnostics(this.handle, enabled ? 1 : 0));
    return this.status();
  }
  status() {
    const pointer = this.m._fgo_status(this.handle);
    if (!pointer) throw this.error();
    return JSON.parse(this.m.UTF8ToString(pointer));
  }
  config() {
    const pointer = this.m._fgo_config(this.handle);
    if (!pointer) throw this.error();
    return JSON.parse(this.m.UTF8ToString(pointer));
  }
  settingsRequest(patch, preview = false) {
    const text = JSON.stringify(patch),
      size = this.m.lengthBytesUTF8(text) + 1;
    const p = this.m._malloc(size);
    if (!p) throw new Error("Settings request allocation failed");
    try {
      this.m.stringToUTF8(text, p, size);
      if (preview) {
        const result = this.m._fgo_preview_settings(this.handle, p);
        if (!result) throw this.error();
        return JSON.parse(this.m.UTF8ToString(result));
      }
      this.check(this.m._fgo_update_settings(this.handle, p));
      return this.status();
    } finally {
      this.m._free(p);
    }
  }
  previewSettings(patch) {
    return this.settingsRequest(patch, true);
  }
  updateSettings(patch) {
    return this.settingsRequest(patch);
  }
  exportBasins() {
    const pointer = this.m._fgo_export_basins(this.handle);
    if (!pointer) throw this.error();
    return this.m.UTF8ToString(pointer);
  }
  importBasins(text) {
    const size = this.m.lengthBytesUTF8(text) + 1;
    if (size > 8 * 1024 * 1024) throw new Error("Basin archive exceeds 8 MiB");
    const pointer = this.m._malloc(size);
    if (!pointer) throw new Error("Basin archive allocation failed");
    try {
      this.m.stringToUTF8(text, pointer, size);
      this.check(this.m._fgo_import_basins(this.handle, pointer));
    } finally {
      this.m._free(pointer);
    }
    return this.status();
  }
  setPopulation(walkers, policy) {
    if (!Number.isInteger(walkers) || walkers < 1 || walkers > 2147483647)
      throw new RangeError("Walker count must be a positive integer");
    const size = this.m.lengthBytesUTF8(policy) + 1,
      p = this.m._malloc(size);
    if (!p) throw new Error("Population request allocation failed");
    try {
      this.m.stringToUTF8(policy, p, size);
      this.check(this.m._fgo_set_population(this.handle, walkers, p));
    } finally {
      this.m._free(p);
    }
    return this.status();
  }
  snapshot() {
    const size = this.check(this.m._fgo_snapshot_size(this.handle));
    const pointer = this.m._fgo_snapshot(this.handle);
    if (!pointer) throw this.error();
    return this.m.HEAPF64.slice(pointer / 8, pointer / 8 + size);
  }
  step() {
    this.check(this.m._fgo_step(this.handle));
    return this.snapshot();
  }
  sample(positions, dimension) {
    if (
      !(
        positions instanceof Float32Array || positions instanceof Float64Array
      ) ||
      dimension !== this.dimension ||
      !Number.isInteger(dimension) ||
      dimension < 1 ||
      positions.length % dimension !== 0
    )
      throw new Error("Invalid objective sample dimensions");
    const count = positions.length / dimension;
    if (!count) return new Float64Array();
    const p = this.m._malloc(positions.byteLength),
      q = this.m._malloc(count * 8);
    if (!p || !q) {
      this.m._free(p);
      this.m._free(q);
      throw new Error("Surface allocation failed");
    }
    try {
      if (positions instanceof Float64Array) {
        this.m.HEAPF64.set(positions, p / 8);
        this.check(this.m._fgo_sample64(this.handle, p, count, q));
      } else {
        this.m.HEAPF32.set(positions, p / 4);
        this.check(this.m._fgo_sample(this.handle, p, count, q));
      }
      return this.m.HEAPF64.slice(q / 8, q / 8 + count);
    } finally {
      this.m._free(p);
      this.m._free(q);
    }
  }
  dispose() {
    if (this.handle) this.m._fgo_destroy(this.handle);
    this.handle = 0;
    this.dimension = 0;
  }
}
export function frameInfo(frame) {
  return {
    n: frame[1],
    d: frame[2],
    velocity: !!frame[3],
    iteration: frame[4],
    evaluations: frame[5],
    alive: frame[6],
    cloned: frame[7],
    currentBest: frame[8],
    best: frame[9],
    mean: frame[10],
    bestIndex: frame[11],
    stride: 2 * frame[2] + 8,
  };
}
export function row(frame, index) {
  const { d, stride } = frameInfo(frame);
  const offset = 12 + index * stride;
  return {
    x: frame.subarray(offset, offset + d),
    v: frame.subarray(offset + d, offset + 2 * d),
    value: frame[offset + 2 * d],
    fitness: frame[offset + 2 * d + 1],
    alive: !!frame[offset + 2 * d + 2],
    companion: frame[offset + 2 * d + 3],
    cloneCompanion: frame[offset + 2 * d + 4],
    parent: frame[offset + 2 * d + 5],
    cloned: !!frame[offset + 2 * d + 6],
    leaf: !!frame[offset + 2 * d + 7],
  };
}

// One allocation boundary for the additive population JSON APIs.
export function populationJson(module, name, handle, request) {
  const text = JSON.stringify(request),
    size = module.lengthBytesUTF8(text) + 1;
  const p = module._malloc(size);
  if (!p) throw new Error("Population request allocation failed");
  try {
    module.stringToUTF8(text, p, size);
    const result = module[name](handle, p);
    if (!result) throw new Error(module.UTF8ToString(module._fgo_error()));
    return JSON.parse(module.UTF8ToString(result));
  } finally {
    module._free(p);
  }
}
export class NativePopulationCoordinator {
  constructor(module) {
    this.m = module;
    this.handle = 0;
  }
  create(config) {
    const text = JSON.stringify(config),
      size = this.m.lengthBytesUTF8(text) + 1;
    const p = this.m._malloc(size);
    if (!p) throw new Error("Population configuration allocation failed");
    let h;
    try {
      this.m.stringToUTF8(text, p, size);
      h = this.m._fgp_create(p, 1);
    } finally {
      this.m._free(p);
    }
    if (!h) throw new Error(this.m.UTF8ToString(this.m._fgo_error()));
    this.dispose();
    this.handle = h;
    return this.request({ op: "config" });
  }
  request(request) {
    return populationJson(this.m, "_fgp_request", this.handle, request);
  }
  dispose() {
    if (this.handle) this.m._fgp_destroy(this.handle);
    this.handle = 0;
  }
}
