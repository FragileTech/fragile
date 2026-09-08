import { MotionRecording, CHUNK_FRAMES } from "../motion.js";
import { read, transaction, listRuns, deleteRun, records } from "./database.js";
import { compress, decompress, encodeObject, decodeObject } from "./codec.js";
export { listRuns, deleteRun };

export class StoredMotionRecording extends MotionRecording {
  constructor(
    info,
    root,
    dt,
    { scene, settings, id = crypto.randomUUID(), onError = () => {} } = {},
  ) {
    super(info, root, dt);
    this.limit = Number.MAX_SAFE_INTEGER;
    this.id = id;
    this.scene = scene;
    this.settings = settings;
    this.onError = onError;
    this.queue = Promise.resolve();
    this.savedFull = 0;
    this.residentLimit = 8;
    this.loading = new Map();
    this.lastRead = -1;
    this.durableFrames = 0;
    this.storedBytes = 0;
    this.closed = false;
    this.failed = null;
  }
  get residentBytes() {
    return (
      this.chunks.reduce((s, c) => s + (c?.byteLength || 0), 0) +
      this.root.length
    );
  }
  metadata(length = this.durableFrames) {
    return {
      id: this.id,
      version: 3,
      name: this.scene.name || "Experiment",
      scene: this.scene,
      settings: this.settings,
      info: this.info,
      channels: this.channels,
      root: this.root,
      dt: this.dt,
      length,
      segments: this.segments.filter((s) => s.frame < length),
      events: this.events.filter((e) => e.frame < length),
      rewardChanges: this.rewardChanges.filter((e) => e.frame < length),
      updated: Date.now(),
      storedBytes: this.storedBytes,
    };
  }
  enqueue(task) {
    if (this.failed) return;
    this.queue = this.queue.then(task).catch((error) => {
      this.failed = error;
      this.onError(error);
    });
  }
  append(packet, label) {
    if (this.failed) throw this.failed;
    if (
      this.residentBytes + packet.length >
      Math.max(
        32 * 1024 * 1024,
        this.residentLimit * CHUNK_FRAMES * this.frameBytes * 2,
      )
    )
      throw new Error(
        "Recording storage cannot keep up. Wait for writes, then continue.",
      );
    super.append(packet, label);
    const full = Math.floor(this.length / CHUNK_FRAMES);
    while (this.savedFull < full) {
      const index = this.savedFull++;
      this.persist(index, CHUNK_FRAMES);
    }
  }
  persist(index, count) {
    const bytes = this.chunks[index].slice(0, count * this.frameBytes),
      end = index * CHUNK_FRAMES + count;
    this.enqueue(async () => {
      const packed = await compress(bytes);
      const record = { run: this.id, kind: "motion", index, count, ...packed };
      const previous = await read("records", [this.id, "motion", index]);
      this.storedBytes += packed.data.length - (previous?.data.length || 0);
      await transaction(["runs", "records"], "readwrite", (tx) => {
        tx.objectStore("records").put(record);
        tx.objectStore("runs").put(
          this.metadata(Math.max(end, this.durableFrames)),
        );
      });
      this.durableFrames = Math.max(end, this.durableFrames);
      this.evict();
    });
  }
  evict() {
    const latest = Math.floor((this.length - 1) / CHUNK_FRAMES);
    let remaining = this.chunks.filter(Boolean).length;
    for (
      let i = 0;
      i < this.chunks.length && remaining > this.residentLimit;
      i++
    ) {
      if (
        i === latest ||
        i === this.lastRead ||
        i >= this.savedFull ||
        !this.chunks[i]
      )
        continue;
      this.chunks[i] = null;
      remaining--;
    }
  }
  async getFrame(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.length)
      throw new Error("Motion frame out of range");
    const chunk = Math.floor(index / CHUNK_FRAMES);
    this.lastRead = chunk;
    if (!this.chunks[chunk]) {
      if (!this.loading.has(chunk))
        this.loading.set(
          chunk,
          (async () => {
            await this.queue;
            const record = await read("records", [this.id, "motion", chunk]);
            if (!record) throw new Error("Recording chunk is missing");
            const bytes = await decompress(
              record,
              CHUNK_FRAMES * this.frameBytes,
            );
            const storage = new Uint8Array(CHUNK_FRAMES * this.frameBytes);
            storage.set(bytes);
            this.chunks[chunk] = storage;
          })().finally(() => this.loading.delete(chunk)),
        );
      await this.loading.get(chunk);
    }
    const result = super.frame(index);
    this.evict();
    return result;
  }
  async getRows(index) {
    await this.getFrame(index);
    return this.rows(index);
  }
  flush() {
    if (this.length % CHUNK_FRAMES)
      this.persist(
        Math.floor(this.length / CHUNK_FRAMES),
        this.length % CHUNK_FRAMES,
      );
    else
      this.enqueue(() =>
        transaction(["runs"], "readwrite", (tx) =>
          tx.objectStore("runs").put(this.metadata()),
        ),
      );
    return this.queue.then(() => {
      if (this.failed) throw this.failed;
    });
  }
  saveObject(kind, index, value) {
    this.enqueue(async () => {
      const packed = await compress(encodeObject(value));
      await transaction(["records"], "readwrite", (tx) =>
        tx.objectStore("records").put({ run: this.id, kind, index, ...packed }),
      );
    });
  }
  async loadObject(kind, index) {
    await this.queue;
    const r = await read("records", [this.id, kind, index]);
    return r ? decodeObject(await decompress(r)) : undefined;
  }
  static async open(id, onError) {
    const meta = await read("runs", id);
    if (!meta) throw new Error("Stored run not found");
    const record = new StoredMotionRecording(meta.info, meta.root, meta.dt, {
      ...meta,
      onError,
    });
    record.length = record.durableFrames = meta.length;
    record.channels = meta.channels;
    record.segments = meta.segments;
    record.events = meta.events || [];
    record.restoreRewardChanges(meta.rewardChanges);
    record.savedFull = Math.floor(meta.length / CHUNK_FRAMES);
    record.storedBytes = meta.storedBytes || 0;
    record.chunks = Array(Math.ceil(meta.length / CHUNK_FRAMES)).fill(null);
    if (meta.length) {
      await record.getFrame(0);
      const last = await record.getFrame(meta.length - 1);
      record.lastCounters = new Uint32Array(
        last.state.buffer,
        last.state.byteOffset,
        8,
      ).slice();
    }
    return record;
  }
  async exportFile() {
    await this.flush();
    const header = encodeObject(this.metadata()),
      head = new Uint32Array([0x52434746, 3, header.length]);
    const parts = [head, header];
    // IndexedDB cursors would keep a transaction alive during compression;
    // records are already compressed, so export holds only compressed chunks.
    for (const r of await records(this.id)) {
      const { data, ...meta } = r,
        description = encodeObject(meta);
      parts.push(
        new Uint32Array([description.length, data.byteLength]),
        description,
        data,
      );
    }
    return new Blob(parts, { type: "application/octet-stream" });
  }
}
export async function importStoredFile(file, onError) {
  if (file.size > 1024 * 1024 * 1024)
    throw new Error("Stored archive exceeds 1 GiB");
  const head = new DataView(await file.slice(0, 12).arrayBuffer());
  if (head.getUint32(0, true) !== 0x52434746 || head.getUint32(4, true) !== 3)
    throw new Error("Unsupported stored archive");
  const length = head.getUint32(8, true);
  if (length > 8 * 1024 * 1024 || length + 12 > file.size)
    throw new Error("Invalid archive header");
  const meta = decodeObject(
    new Uint8Array(await file.slice(12, 12 + length).arrayBuffer()),
  );
  const id = crypto.randomUUID();
  const record = new StoredMotionRecording(meta.info, meta.root, meta.dt, {
    ...meta,
    id,
    onError,
  });
  let at = 12 + length,
    totalFrames = 0,
    previous = -1,
    previousCount = CHUNK_FRAMES;
  try {
    while (at < file.size) {
      if (at + 8 > file.size) throw new Error("Truncated archive");
      const lengths = new DataView(await file.slice(at, at + 8).arrayBuffer());
      const a = lengths.getUint32(0, true),
        b = lengths.getUint32(4, true);
      at += 8;
      if (a > 1024 * 1024 || b > 128 * 1024 * 1024 || at + a + b > file.size)
        throw new Error("Invalid archive record");
      const r = decodeObject(
        new Uint8Array(await file.slice(at, at + a).arrayBuffer()),
      );
      at += a;
      r.data = new Uint8Array(await file.slice(at, at + b).arrayBuffer());
      at += b;
      r.run = id;
      const raw = await decompress(r);
      if (r.kind === "motion") {
        if (
          previousCount !== CHUNK_FRAMES ||
          !Number.isInteger(r.index) ||
          r.index !== previous + 1 ||
          !Number.isInteger(r.count) ||
          r.count < 1 ||
          r.count > CHUNK_FRAMES ||
          raw.length !== r.count * record.frameBytes
        )
          throw new Error("Invalid motion chunk sequence");
        const temporary = new MotionRecording(meta.info, meta.root, meta.dt);
        temporary.append(raw);
        temporary.validate();
        previous = r.index;
        previousCount = r.count;
        totalFrames += r.count;
      } else if (
        !["tree", "checkpoint"].includes(r.kind) ||
        !Number.isInteger(r.index)
      )
        throw new Error("Invalid archive record kind");
      await transaction(["records"], "readwrite", (tx) =>
        tx.objectStore("records").put(r),
      );
    }
    if (totalFrames !== meta.length)
      throw new Error("Incomplete recording archive");
    await transaction(["runs"], "readwrite", (tx) =>
      tx.objectStore("runs").put({ ...meta, id, updated: Date.now() }),
    );
    return await StoredMotionRecording.open(id, onError);
  } catch (error) {
    await deleteRun(id);
    throw error;
  }
}
