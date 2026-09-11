import * as T from "../vendor/three.module.js";
import { palette } from "./primitives.js";

export const CROSSING_FLASH_MS = 600;

// Presentation-only counters; authoritative checkpoint progress stays in state.
export class CheckpointPresentation {
  constructor(scene, info, canvas, { now = () => performance.now() } = {}) {
    this.now = now;
    this.canvas = canvas;
    this.count = info[2];
    this.offset = info[6];
    this.gates = scene.task === "tandem" && this.count ? scene.gates || [] : [];
    this.group = new T.Group();
    this.group.name = "Checkpoint feedback";
    this.previous = new Uint32Array(this.count);
    this.flashes = new Float64Array(this.gates.length);
    this.flashes.fill(-Infinity);
    this.active = -1;
    this.crossed = 0;
    this.initialized = false;
    this.enabled = true;
    this.labelVisible = true;
    this.projected = new T.Vector3();
    this.amber = new T.Color(palette.gold);
    this.green = new T.Color(palette.green);
    this.dim = new T.Color(0x756a59);
    this.markers = [];
    if (!this.gates.length) return;
    this.ringGeometry = new T.RingGeometry(0.97, 1.03, 72);
    this.fillGeometry = new T.CircleGeometry(1, 72);
    for (const gate of this.gates) {
      const material = new T.MeshBasicMaterial({
        transparent: true,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
        fog: false,
        side: T.DoubleSide,
      });
      const fillMaterial = material.clone();
      const ring = new T.Mesh(this.ringGeometry, material);
      const fill = new T.Mesh(this.fillGeometry, fillMaterial);
      const radius = gate.radius ?? 1;
      ring.scale.set(radius, radius, 1);
      fill.scale.copy(ring.scale);
      ring.position.set(...gate.position, 0.15);
      fill.position.set(...gate.position, 0.14);
      ring.renderOrder = 3;
      fill.renderOrder = 2;
      this.group.add(ring, fill);
      this.markers.push({ ring, fill });
    }
    this.label = canvas.ownerDocument.createElement("div");
    this.label.className = "checkpoint-readout";
    this.label.hidden = true;
    canvas.parentElement.append(this.label);
    this.paint();
  }
  setEnabled(enabled) {
    this.enabled = !!enabled;
    if (!this.enabled) this.flashes.fill(-Infinity);
    this.paint();
  }
  update(state, { discontinuity = false } = {}) {
    if (!this.gates.length) return;
    const bits = new Uint32Array(state.buffer, state.byteOffset, state.length);
    const tick = bits[0];
    let reset = discontinuity || !this.initialized || tick < this.tick;
    let stage = Infinity;
    for (let c = 0; c < this.count; ++c) {
      const value = bits[this.offset + c];
      stage = Math.min(stage, value);
      if (value < this.previous[c]) reset = true;
    }
    if (reset) this.flashes.fill(-Infinity);
    const now = this.now();
    this.crossed = 0;
    for (let c = 0; c < this.count; ++c) {
      const value = bits[this.offset + c];
      if (!reset && this.enabled && value > this.previous[c])
        this.flashes[(value - 1) % this.gates.length] = now;
      if (value > stage) ++this.crossed;
      this.previous[c] = value;
    }
    this.active = stage % this.gates.length;
    this.tick = tick;
    this.initialized = true;
    const text = `Checkpoint ${this.active + 1} · ${this.crossed}/${this.count} crossed`;
    if (this.label.textContent !== text) this.label.textContent = text;
    this.paint(now);
  }
  paint(now = this.now()) {
    for (let i = 0; i < this.markers.length; ++i) {
      const { ring, fill } = this.markers[i];
      const active = i === this.active;
      const flash = this.enabled
        ? Math.max(0, 1 - (now - this.flashes[i]) / CROSSING_FLASH_MS)
        : 0;
      ring.material.color
        .copy(active ? this.amber : this.dim)
        .lerp(this.green, flash);
      ring.material.opacity = (active ? 0.95 : 0.3) * (1 - flash) + flash;
      fill.material.color.copy(ring.material.color);
      fill.material.opacity =
        (active ? 0.1 : 0.015) * (1 - flash) + 0.22 * flash;
    }
  }
  project(camera) {
    if (!this.label) return;
    if (!this.labelVisible || !this.initialized) {
      this.label.hidden = true;
      return;
    }
    const gate = this.gates[this.active];
    this.group.updateWorldMatrix(true, false);
    this.projected
      .set(...gate.position, 0.2)
      .applyMatrix4(this.group.matrixWorld)
      .project(camera);
    const p = this.projected;
    this.label.hidden =
      Math.abs(p.x) > 1 || Math.abs(p.y) > 1 || Math.abs(p.z) > 1;
    if (this.label.hidden) return;
    const x = ((p.x + 1) * this.canvas.clientWidth) / 2;
    const y = ((1 - p.y) * this.canvas.clientHeight) / 2;
    this.label.style.left = `${Math.max(105, Math.min(this.canvas.clientWidth - 105, x))}px`;
    this.label.style.top = `${Math.max(8, y - 38)}px`;
  }
  dispose() {
    this.ringGeometry?.dispose();
    this.fillGeometry?.dispose();
    for (const { ring, fill } of this.markers) {
      ring.material.dispose();
      fill.material.dispose();
    }
    this.label?.remove();
    this.group.clear();
    this.group.removeFromParent();
  }
}
