// Complete world frames, stored as packed binary rows. Visual assets and static
// scene data never enter the stream. Float metadata must be copied as bytes.
export const MOTION_LIMIT = 64 * 1024 * 1024;
export const CHUNK_FRAMES = 256;
export function bytesOf(array) {
  return new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
}
export class MotionRecording {
  constructor(info, root, dt = 1 / 60) {
    if (
      !Array.isArray(info) ||
      ![12, 13, 15].includes(info.length) ||
      !info.every(Number.isInteger) ||
      info[0] !== 1 ||
      info[1] < 1 ||
      info[1] > 4096 ||
      info[2] < 0 ||
      info[2] > info[1] ||
      info[9] < 0 ||
      info[10] < 0 ||
      info[5] !== 8 + 6 * info[1] ||
      info[6] !== info[5] + info[1] ||
      info[7] !== info[6] + info[2] ||
      info[8] !== info[7] + 2 * info[9] ||
      info[4] !== info[8] + 3 * info[10] + (info[14] ?? 0) ||
      info[4] > 100000 ||
      info[3] !== Math.ceil(info[4] / 16) * 16 ||
      !Number.isFinite(dt) ||
      dt <= 0 ||
      dt > 0.1
    )
      throw new Error("Invalid motion layout");
    if (!(root instanceof Uint8Array) || root.length !== 32 + info[4] * 4)
      throw new Error("Invalid motion root");
    const header = new DataView(root.buffer, root.byteOffset, root.byteLength);
    if (
      header.getUint32(0, true) !== 0x53434746 ||
      header.getUint32(4, true) !== 1 ||
      header.getUint32(16, true) !== 1 ||
      header.getUint32(20, true) !== info[4]
    )
      throw new Error("Invalid motion snapshot header");
    this.info = info.slice();
    this.root = root.slice();
    this.dt = dt;
    this.words = info[4];
    this.dim = info[12] ?? info[2] * 2;
    if (!Number.isInteger(this.dim) || this.dim < 0 || this.dim > 8192)
      throw new Error("Invalid motion action dimension");
    this.frameBytes = (this.words + this.dim + 1) * 4;
    this.limit = MOTION_LIMIT;
    this.events = [];
    this.length = 0;
    this.chunks = [];
    this.segments = [];
  }
  get bytes() {
    return this.length * this.frameBytes + this.root.length;
  }
  append(packet, label) {
    if (!(packet instanceof Uint8Array) || packet.length % this.frameBytes)
      throw new Error("Invalid motion frame shape");
    if (this.bytes + packet.length > this.limit)
      throw new Error(
        "World recording reached 64 MiB. Export the run and reset to record more.",
      );
    if (label && packet.length)
      this.segments.push({
        frame: this.length,
        label: String(label).slice(0, 120),
      });
    const previousLength = this.length;
    let source = 0;
    while (source < packet.length) {
      const chunkIndex = Math.floor(this.length / CHUNK_FRAMES),
        offset = (this.length % CHUNK_FRAMES) * this.frameBytes;
      if (!this.chunks[chunkIndex])
        this.chunks[chunkIndex] = new Uint8Array(
          CHUNK_FRAMES * this.frameBytes,
        );
      const count = Math.min(
        packet.length - source,
        this.chunks[chunkIndex].length - offset,
      );
      this.chunks[chunkIndex].set(
        packet.subarray(source, source + count),
        offset,
      );
      source += count;
      this.length += count / this.frameBytes;
    }
    for (let frame = previousLength; frame < this.length; frame++) {
      const row = this.frame(frame),
        bits = new Uint32Array(
          row.state.buffer,
          row.state.byteOffset,
          row.state.length,
        );
      if (this.lastCounters && !label)
        for (const [word, name] of [
          [3, "Terminal event"],
          [4, "Delivery"],
          [5, "Pickup"],
          [6, "Gate"],
        ])
          if (bits[word] > this.lastCounters[word])
            this.addEvent(frame, name, "world");
      this.lastCounters = bits.slice(0, 8);
    }
  }
  addEvent(frame, label, kind = "note") {
    if (Number.isInteger(frame) && frame >= 0 && frame < this.length)
      this.events.push({ frame, label: String(label).slice(0, 120), kind });
  }
  frame(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.length)
      throw new Error("Motion frame out of range");
    const chunk = this.chunks[Math.floor(index / CHUNK_FRAMES)],
      offset = (index % CHUNK_FRAMES) * this.frameBytes;
    const state = new Float32Array(chunk.buffer, offset, this.words);
    const action = new Float32Array(
      chunk.buffer,
      offset + this.words * 4,
      this.dim,
    );
    const view = new DataView(chunk.buffer, offset, this.frameBytes);
    return {
      state,
      action,
      tick: view.getUint32(0, true),
      decision: view.getUint32(this.frameBytes - 4, true),
    };
  }
  validate() {
    for (let i = 0; i < this.length; i++) {
      const { state, action } = this.frame(i);
      const bits = new Uint32Array(
        state.buffer,
        state.byteOffset,
        state.length,
      );
      for (let j = 8; j < this.info[5]; j++)
        if (!Number.isFinite(state[j]))
          throw new Error("Non-finite motion body state");
      for (let j = this.info[5]; j < this.info[6]; j++)
        if (bits[j] > 3) throw new Error("Invalid motion body flags");
      for (let j = this.info[7]; j < this.info[8]; j += 2)
        if (
          bits[j] > this.info[1] ||
          !Number.isFinite(state[j + 1]) ||
          state[j + 1] < 0
        )
          throw new Error("Invalid motion tether state");
      for (let j = this.info[8]; j < this.words; j++)
        if (!Number.isFinite(state[j]))
          throw new Error("Non-finite motion pickup state");
      if (bits[7] > 1 || !action.every(Number.isFinite))
        throw new Error("Invalid motion frame");
    }
  }
  rows(index) {
    const rows = new Float32Array(this.info[3]);
    bytesOf(rows).set(bytesOf(this.frame(index).state));
    return rows;
  }
  pack() {
    const result = new Uint8Array(this.length * this.frameBytes);
    let offset = 0;
    for (const chunk of this.chunks) {
      const count = Math.min(chunk.length, result.length - offset);
      result.set(chunk.subarray(0, count), offset);
      offset += count;
    }
    return result;
  }
  segment(index) {
    return (
      this.segments.findLast((s) => s.frame <= index)?.label || "Executed world"
    );
  }
}
// Worker-side capture of the authoritative world only. Wave populations and
// predictive deadline roots never pass through this recorder.
export class WorldCapture {
  constructor(engine, emit) {
    this.engine = engine;
    this.emit = emit;
  }
  capture(action, decision = 0, label, initial = false) {
    const packet = this.packet(1);
    this.write(packet, 0, action, decision);
    this.emit({ type: "motion", packet, label, initial }, [packet.buffer]);
  }
  packet(frames) {
    return new Uint8Array(
      frames * (this.engine.words + this.engine.dim + 1) * 4,
    );
  }
  write(packet, index, action, decision) {
    const size = (this.engine.words + this.engine.dim + 1) * 4,
      offset = index * size;
    packet.set(
      bytesOf(this.engine.states()).subarray(0, this.engine.words * 4),
      offset,
    );
    packet.set(bytesOf(action), offset + this.engine.words * 4);
    new DataView(packet.buffer).setUint32(offset + size - 4, decision, true);
  }
  step(action, frames, decision) {
    const packet = this.packet(frames);
    for (let i = 0; i < frames; i++) {
      this.engine.step(action, 1);
      this.write(packet, i, action, decision);
    }
    this.emit({ type: "motion", packet }, [packet.buffer]);
  }
}
