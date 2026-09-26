class WorkerClient {
  constructor() {
    this.worker = new Worker(
      new URL("./population-worker.js", import.meta.url),
      { type: "module" },
    );
    this.pending = new Map();
    this.next = 1;
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
    };
    this.worker.onerror = () => {
      this.dead = true;
      this.reject(new Error("Population worker failed; reset to continue."));
    };
  }
  reject(error) {
    for (const p of this.pending.values()) p.reject(error);
    this.pending.clear();
  }
  request(type, fields = {}) {
    if (this.dead)
      return Promise.reject(
        new Error("Population worker is unavailable; reset to continue."),
      );
    return new Promise((resolve, reject) => {
      const id = this.next++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, ...fields });
    });
  }
  dispose() {
    this.dead = true;
    this.worker.terminate();
    this.reject(new Error("Population closed"));
  }
}
async function scheduled(items, slots, fn) {
  const results = new Array(items.length);
  let next = 0,
    error;
  await Promise.all(
    Array.from({ length: Math.min(slots, items.length) }, async () => {
      while (next < items.length && !error) {
        const i = next++;
        try {
          results[i] = await fn(items[i], i);
        } catch (e) {
          error = new Error(`Swarm ${i + 1}: ${e.message}`, { cause: e });
        }
      }
    }),
  );
  if (error) {
    error.completed = results;
    throw error;
  }
  return results;
}
export class FractalPopulation {
  constructor() {
    this.coordinator = new WorkerClient();
    this.members = [];
    this.busy = false;
    this.failed = false;
  }
  catalog() {
    return this.coordinator.request("catalog");
  }
  command(op, fields = {}) {
    return this.coordinator.request("coordinator", {
      request: { op, ...fields },
    });
  }
  async create(config) {
    for (const member of this.members) member.dispose();
    this.members = [];
    this.config = await this.coordinator.request("coordinatorCreate", {
      config,
    });
    this.failed = false;
    try {
      this.members = this.config.members.map(() => new WorkerClient());
      const results = await scheduled(
        this.members,
        this.config.concurrency,
        (worker, i) =>
          worker.request("create", { member: this.config.members[i] }),
      );
      this.frames = results.map((r) => r.frame);
      this.reports = results.map((r) => r.report);
      this.status = await this.command("initialize", { reports: this.reports });
      return this.snapshot();
    } catch (error) {
      this.failed = true;
      for (const worker of this.members) worker.dispose();
      this.members = [];
      throw error;
    }
  }
  snapshot() {
    return { config: this.config, status: this.status, frames: this.frames };
  }
  async step() {
    if (this.busy || this.failed)
      throw new Error(
        this.failed
          ? "Population failed; reset to continue."
          : "A population round is already running.",
      );
    this.busy = true;
    let dispatched = false;
    try {
      const { archive } = await this.command("prepare");
      this.reports = await scheduled(
        this.members,
        this.config.concurrency,
        (w) => w.request("sync", { archive }),
      );
      await this.command("admit", { reports: this.reports });
      dispatched = true;
      const results = await scheduled(
        this.members,
        this.config.concurrency,
        (w) => w.request("step"),
      );
      this.frames = results.map((r) => r.frame);
      this.reports = results.map((r) => r.report);
      const { plans } = await this.command("finish", { reports: this.reports });
      try {
        await scheduled(plans, this.config.concurrency, (p) =>
          this.members[
            this.config.members.findIndex((m) => m.id === p.id)
          ].request("stage", { imports: p.imports }),
        );
      } catch (error) {
        await Promise.allSettled(this.members.map((w) => w.request("discard")));
        throw error;
      }
      const committed = await scheduled(plans, this.config.concurrency, (p) =>
        this.members[
          this.config.members.findIndex((m) => m.id === p.id)
        ].request("commit"),
      );
      plans.forEach((p, i) => {
        this.frames[this.config.members.findIndex((m) => m.id === p.id)] =
          committed[i].frame;
      });
      this.status = await this.command("commit");
      return this.snapshot();
    } catch (error) {
      if (dispatched) {
        this.failed = true;
        for (const result of error.completed ?? []) {
          if (!result?.report) continue;
          const i = this.config.members.findIndex(
            (m) => m.id === result.report.id,
          );
          this.reports[i] = result.report;
          if (result.frame) this.frames[i] = result.frame;
        }
        this.status = await this.command("fail", {
          reports: this.reports,
        }).catch(() => this.status);
      }
      throw error;
    } finally {
      this.busy = false;
    }
  }
  async updateMember(index, patch) {
    if (this.busy || this.failed)
      throw new Error("Pause at a completed round before changing a swarm.");
    // Task, membership, exchange capacity and budget changes require reset.
    const allowed = new Set([
      "walkers",
      "elites",
      "removal_policy",
      "perturbation_std",
      "distance_coef",
      "reward_coef",
      "dt_min",
      "dt_max",
      "restart_token",
    ]);
    for (const key of Object.keys(patch))
      if (!allowed.has(key)) throw new Error(`${key} changes require reset.`);
    const current = this.reports[index].settings,
      count = this.config.members[index].exchange_count;
    if (
      this.members.length > 1 &&
      (patch.walkers ?? current.walkers) <
        Math.max(patch.elites ?? current.elites, count) + count
    )
      throw new Error("Keep enough walkers for elites and imports.");
    this.busy = true;
    try {
      const result = await this.members[index].request("settings", { patch });
      this.reports[index] = result.report;
      this.frames[index] = result.frame;
      this.status = await this.command("refresh", { reports: this.reports });
      this.config.members[index].settings = result.report.settings;
      return this.snapshot();
    } finally {
      this.busy = false;
    }
  }
  sample(positions, dimension) {
    return this.members[0].request("sample", { positions, dimension });
  }
  dispose() {
    for (const w of this.members) w.dispose();
    this.members = [];
    this.coordinator.dispose();
  }
}
export const POPULATION_RECORDING_VERSION = 1;
export function encodePopulationRecording(config, frames) {
  return JSON.stringify({
    format: "fractal-populations",
    version: POPULATION_RECORDING_VERSION,
    config,
    frames: frames.map((f) => ({
      status: f.status,
      frames: f.frames.map((v) =>
        Array.from(v, (x) => (Number.isFinite(x) ? x : null)),
      ),
    })),
  });
}
export function decodePopulationRecording(text) {
  if (text.length > 64 * 1024 * 1024)
    throw new Error("Recording exceeds 64 MiB");
  const data = JSON.parse(text);
  if (
    data.format !== "fractal-populations" ||
    data.version !== 1 ||
    !Array.isArray(data.config?.members) ||
    !data.config.members.length ||
    !Array.isArray(data.frames)
  )
    throw new Error("Invalid population recording");
  for (const frame of data.frames) {
    if (
      !Array.isArray(frame.frames) ||
      frame.frames.length !== data.config.members.length
    )
      throw new Error("Incomplete recorded population");
    frame.frames = frame.frames.map((values) => {
      if (
        !Array.isArray(values) ||
        values.length < 12 ||
        !Number.isInteger(values[1]) ||
        values[1] < 1 ||
        !Number.isInteger(values[2]) ||
        values[2] < 1 ||
        values.length !== 12 + values[1] * (2 * values[2] + 8) ||
        values.some((v) => v !== null && !Number.isFinite(v))
      )
        throw new Error("Invalid recorded walker dimensions");
      return Float64Array.from(values, (v) => (v === null ? Infinity : v));
    });
  }
  return data;
}
