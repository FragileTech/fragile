// Client-side ring of recent walker positions. Full d-dimensional coordinates
// are kept so trails survive axis changes. Pure: no DOM, no three.js.
const PRESENT = 1,
  BROKEN = 2;
export class TrailBuffer {
  constructor({ frames = 30, maxWalkers = 96 } = {}) {
    this.frames = frames;
    this.maxWalkers = maxWalkers;
    this.capacity = (frames - 1) * (maxWalkers + 1) * 6;
    this.clear();
  }
  clear() {
    this.slots = new Map();
    this.head = -1;
    this.count = 0;
    this.lastStep = null;
    this.n = 0;
    this.d = 0;
    this.selected = -1;
  }
  slot(index) {
    if (!this.slots.has(index))
      this.slots.set(index, {
        positions: new Float64Array(this.frames * this.d),
        raw: new Float64Array(this.frames),
        flags: new Uint8Array(this.frames),
        generation: null,
      });
    return this.slots.get(index);
  }
  // A newly selected walker beyond the default set starts with an empty trail.
  track(index) {
    if (this.selected >= this.maxWalkers && this.selected !== index)
      this.slots.delete(this.selected);
    this.selected = index;
    if (this.d && index >= 0 && index < this.n) this.slot(index);
  }
  push(frame, eligible, { periodic = false, low = 0, high = 0 } = {}) {
    const field = frame.population.observations.fields.positions;
    if (!field) return;
    const n = field.rows,
      d = field.item_shape.reduce((a, b) => a * b, 1);
    if (n !== this.n || d !== this.d) {
      const selected = this.selected;
      this.clear();
      this.n = n;
      this.d = d;
      this.selected = selected;
    }
    if (this.lastStep === frame.step) return;
    const gap = this.lastStep !== null && frame.step !== this.lastStep + 1;
    this.lastStep = frame.step;
    const previous = this.head;
    this.head = (this.head + 1) % this.frames;
    this.count = Math.min(this.frames, this.count + 1);
    for (let i = 0; i < Math.min(n, this.maxWalkers); i++) this.slot(i);
    if (this.selected >= 0 && this.selected < n) this.slot(this.selected);
    const choices = frame.report?.clone_plan?.choices,
      generations = frame.population.generations,
      half = 0.5 * (high - low);
    for (const [i, entry] of this.slots) {
      if (i >= n) {
        entry.flags[this.head] = 0;
        continue;
      }
      const at = this.head * d;
      let valid = !!eligible[i],
        broken = gap;
      for (let k = 0; k < d; k++) {
        const x = field.values[i * d + k];
        entry.positions[at + k] = x;
        if (!Number.isFinite(x)) valid = false;
        else if (
          periodic &&
          previous >= 0 &&
          Math.abs(x - entry.positions[previous * d + k]) > half
        )
          broken = true;
      }
      entry.raw[this.head] = frame.population.rewards.raw[i];
      const generation = generations ? Number(generations[i]) : null;
      if (choices ? choices[i]?.accepted || choices[i]?.revival : false)
        broken = true;
      else if (
        !choices &&
        entry.generation !== null &&
        generation !== entry.generation
      )
        broken = true;
      entry.generation = generation;
      entry.flags[this.head] = valid ? PRESENT | (broken ? BROKEN : 0) : 0;
    }
  }
  // pointOf(positions, offset, raw, target) writes xyz and returns validity.
  // Returns the number of floats written to `out` (two xyz points a segment).
  segments(pointOf, out) {
    let written = 0;
    const a = [0, 0, 0],
      b = [0, 0, 0];
    for (let age = this.count - 1; age > 0; age--) {
      const from = (this.head - age + this.frames) % this.frames,
        to = (from + 1) % this.frames;
      for (const entry of this.slots.values()) {
        if (
          !(entry.flags[from] & PRESENT) ||
          !(entry.flags[to] & PRESENT) ||
          entry.flags[to] & BROKEN ||
          written + 6 > out.length
        )
          continue;
        if (
          !pointOf(entry.positions, from * this.d, entry.raw[from], a) ||
          !pointOf(entry.positions, to * this.d, entry.raw[to], b)
        )
          continue;
        out[written++] = a[0];
        out[written++] = a[1];
        out[written++] = a[2];
        out[written++] = b[0];
        out[written++] = b[1];
        out[written++] = b[2];
      }
    }
    return written;
  }
}
