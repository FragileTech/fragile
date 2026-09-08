import * as T from "../vendor/three.module.js";

// A fixed GPU budget, independent of cargo capacity: 106 triangles / vehicle
// across the same three shared draws used by the original cargo presentation.
export const CARGO_SLOTS = 6;
export const CARGO_TRANSFER_SLOTS = 6;
export const CARGO_LOAD_SECONDS = 0.65;
const clamp = (value, max = 1) =>
  Math.min(max, Math.max(0, Number.isFinite(value) ? value : 0));
const smooth = (t) => t * t * (3 - 2 * t);

// Normalized vehicle coordinates (+X forward). Harvesters use their authored
// open hopper; smaller vehicles carry a compact load on the rear/central deck.
const mounts = {
  harvester: {
    center: [-0.3, 0, 0.54],
    size: [0.5, 0.37, 0.18],
    intake: [0.66, 0, 0.22],
  },
  kart: {
    center: [-0.43, 0, 0.32],
    size: [0.36, 0.31, 0.17],
    intake: [0.48, 0, 0.15],
  },
  rocket: {
    center: [-0.16, 0, 0.34],
    size: [0.38, 0.32, 0.16],
    intake: [0.42, 0, 0.23],
  },
  drone: {
    center: [0, 0, 0.49],
    size: [0.36, 0.32, 0.17],
    intake: [0.38, 0, 0.13],
  },
};

export class CargoVisuals {
  constructor(scene, info, bodyLayer, parent, style) {
    this.animationsEnabled = true;
    this.scene = scene;
    this.info = info;
    this.bodies = bodyLayer;
    this.entries = [];
    this.revision = 0;
    if (!info[15]) return;
    this.capacity = Math.max(
      1,
      clamp(scene.cargo?.capacity ?? 5, Number.MAX_SAFE_INTEGER),
    );
    this.dt = scene.physics?.dt || 1 / 60;
    const count = Math.max(1, bodyLayer.controlled.length);
    const color = style === "steampunk" ? 0xeab553 : 0xb78bff;
    this.color = new T.Color(color);
    this.segmentColor = new T.Color();
    this.zero = new T.Matrix4().makeScale(0, 0, 0);
    this.dimColor = new T.Color(style === "steampunk" ? 0x51422a : 0x303047);
    this.fullColor = new T.Color(style === "steampunk" ? 0xffda83 : 0x8fffe2);
    this.group = new T.Group();
    this.group.name = "Resource loads and transfers";
    parent.add(this.group);
    this.fill = new T.InstancedMesh(
      new T.OctahedronGeometry(1, 0),
      new T.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.24,
        metalness: 0.28,
        roughness: 0.48,
        flatShading: true,
      }),
      count * CARGO_SLOTS,
    );
    this.meter = new T.InstancedMesh(
      new T.PlaneGeometry(1, 1),
      new T.MeshBasicMaterial({ side: T.DoubleSide, toneMapped: false }),
      count * 5,
    );
    this.transfer = new T.InstancedMesh(
      new T.OctahedronGeometry(1, 0),
      new T.MeshBasicMaterial({ color, toneMapped: false }),
      count * CARGO_TRANSFER_SLOTS,
    );
    this.group.add(this.fill, this.meter, this.transfer);
    for (const mesh of [this.fill, this.meter, this.transfer]) {
      mesh.frustumCulled = false;
      for (let i = 0; i < mesh.count; i++) mesh.setMatrixAt(i, this.zero);
      mesh.count = 0;
      mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
    }
    this.pose = new T.Object3D();
    this.base = new T.Matrix4();
    this.matrix = new T.Matrix4();
    this.point = new T.Vector3();
    this.target = new T.Vector3();
    this.bodyPose = new T.Object3D();
    this.entries = bodyLayer.controlled.map((body) => {
      const kind =
        bodyLayer.bodies[body].visual?.model ||
        (scene.task === "forage" ? "kart" : "rocket");
      let mount = mounts[kind] || mounts.rocket;
      if (kind === "harvester") {
        let authored = false;
        bodyLayer.models[body].traverse((part) => {
          authored ||= part.userData.assetModel === "harvester";
        });
        mount = !authored
          ? {
              center: [-0.19, 0, 0.4],
              size: [0.42, 0.3, 0.16],
              intake: [0.63, 0, 0.15],
            }
          : style === "steampunk"
            ? {
                center: [-0.37, 0, 0.56],
                size: [0.55, 0.42, 0.18],
                intake: [0.62, 0, 0.23],
              }
            : mount;
      }
      return {
        body,
        mount,
        capacity: this.capacity,
        amount: 0,
        delivered: 0,
        collected: 0,
        status: "Empty",
        loadingAmount: 0,
        loadingFrom: 0,
        loadStarted: -Infinity,
        anchor: new T.Vector3(),
        base: new T.Matrix4(),
        zone: null,
      };
    });
  }
  setAnimationsEnabled(enabled) {
    this.animationsEnabled = !!enabled;
    if (this.transfer) this.transfer.visible = !!enabled;
    this.revision++;
    this.animate();
  }
  resetTransitions() {
    for (const entry of this.entries) {
      entry.loadStarted = -Infinity;
      entry.loadingAmount = 0;
      entry.status = this.status(entry);
    }
    this.revision++;
    this.animate();
  }
  status(entry) {
    return entry.zone && entry.amount > 0
      ? "Unloading"
      : entry.loadingAmount > 0
        ? "Loading"
        : entry.amount >= entry.capacity - 1e-5
          ? "Full"
          : entry.amount > 0
            ? "Carrying"
            : "Empty";
  }
  update(state) {
    if (!this.fill) return;
    const time =
      new Uint32Array(state.buffer, state.byteOffset, 1)[0] * this.dt;
    // Transitions use observed gains, never guessed pickup ownership. Reset on
    // discontinuities; the renderer explicitly clears them on replay seeks too.
    const continuous =
      this.hasState &&
      time > this.time &&
      time - this.time <= Math.max(0.5, this.dt * 2);
    for (let c = 0; c < this.entries.length; c++) {
      const entry = this.entries[c],
        b = entry.body,
        at = this.info[15] + 4 * c;
      const amount = clamp(state[at], this.capacity);
      const delivered = clamp(state[at + 2], Number.MAX_SAFE_INTEGER);
      const collected = amount + delivered;
      const gain = collected - entry.collected;
      if (
        (!continuous &&
          (time !== this.time ||
            amount !== entry.amount ||
            delivered !== entry.delivered)) ||
        gain < -1e-4 ||
        delivered < entry.delivered
      ) {
        entry.loadStarted = -Infinity;
        entry.loadingAmount = 0;
      } else if (
        continuous &&
        gain > 1e-4 &&
        Math.abs(gain - Math.round(gain)) < 1e-3
      ) {
        entry.loadStarted = time;
        entry.loadingAmount = Math.min(gain, amount);
        entry.loadingFrom = Math.max(0, amount - gain);
      }
      if (time - entry.loadStarted >= CARGO_LOAD_SECONDS)
        entry.loadingAmount = 0;
      entry.amount = amount;
      entry.delivered = delivered;
      entry.collected = collected;
      const x = state[8 + b],
        y = state[8 + this.info[1] + b];
      entry.zone =
        state[at + 1] > 0 && amount > 0
          ? (this.scene.refineries || []).find(
              (z) =>
                Math.hypot(x - z.position[0], y - z.position[1]) <= z.radius,
            ) || null
          : null;
      if (entry.zone) entry.loadingAmount = 0;
      entry.status = this.status(entry);
      this.bodyPose.position.set(x, y, this.bodies.models[b].position.z);
      this.bodyPose.rotation.set(0, 0, state[8 + 4 * this.info[1] + b]);
      this.bodyPose.scale.copy(this.bodies.models[b].scale);
      this.bodyPose.updateMatrix();
      entry.base.copy(this.bodyPose.matrix);
    }
    this.time = time;
    this.hasState = true;
    this.revision++;
    this.animate();
  }
  set(mesh, index, x, y, z, sx, sy, sz, angle = 0, local = true) {
    this.pose.position.set(x, y, z);
    this.pose.scale.set(sx, sy, sz);
    this.pose.rotation.set(0, 0, angle);
    this.pose.updateMatrix();
    this.matrix.copy(this.pose.matrix);
    if (local) this.matrix.premultiply(this.base);
    mesh.setMatrixAt(index, this.matrix);
  }
  animate() {
    if (!this.fill || !this.hasState) return;
    // BodyLayer versions already throttle crowds to 30 Hz and track culling.
    const version = this.bodies.presentationVersion;
    if (
      version != null &&
      this.lastVersion === version &&
      this.lastRevision === this.revision
    )
      return;
    this.lastVersion = version;
    this.lastRevision = this.revision;
    let fills = 0,
      meters = 0,
      transfers = 0;
    for (const entry of this.entries) {
      const b = entry.body,
        { center, size, intake } = entry.mount;
      this.base.copy(entry.base);
      const presentation = this.bodies.presentations?.[b];
      if (this.animationsEnabled && presentation) {
        presentation.updateMatrix();
        this.base.multiply(presentation.matrix);
      }
      entry.anchor
        .set(center[0], 0, Math.max(0.95, center[2] + size[2] + 0.2))
        .applyMatrix4(this.base);
      if (
        this.bodies.active?.[b] === false ||
        this.bodies.inView?.[b] === false
      )
        continue;
      const progress = clamp(
        (this.time - entry.loadStarted) / CARGO_LOAD_SECONDS,
      );
      const loading = this.animationsEnabled && entry.loadingAmount > 0;
      const shown = loading
        ? Math.min(
            entry.amount,
            entry.loadingFrom + entry.loadingAmount * smooth(progress),
          )
        : entry.amount;
      const slots = Math.min(CARGO_SLOTS, this.capacity);
      for (let j = 0; j < slots; j++) {
        const fraction = clamp((shown / this.capacity) * slots - j);
        if (!fraction) continue;
        const growth = Math.cbrt(fraction);
        const row = Math.floor(j / 2),
          side = j % 2 ? 1 : -1;
        this.set(
          this.fill,
          fills++,
          center[0] + (row - 1) * size[0] * 0.3,
          side * size[1] * 0.23,
          center[2] + size[2] * 0.04 * (j % 3),
          size[0] * 0.22 * growth,
          size[1] * 0.31 * growth,
          size[2] * 0.5 * growth,
          j * 1.7,
        );
      }
      for (let j = 0; j < 5; j++) {
        const fraction = clamp((entry.amount / this.capacity) * 5 - j);
        this.set(
          this.meter,
          meters,
          center[0] + (j - 2) * 0.09,
          -size[1] * 0.66,
          center[2] + size[2] + 0.015,
          0.072,
          0.045,
          1,
        );
        this.meter.setColorAt(
          meters++,
          this.segmentColor
            .copy(this.dimColor)
            .lerp(
              entry.amount >= this.capacity - 1e-5
                ? this.fullColor
                : this.color,
              fraction,
            ),
        );
      }
      if (!this.animationsEnabled || (!loading && !entry.zone)) continue;
      const points = loading
        ? Math.min(CARGO_TRANSFER_SLOTS, Math.ceil(entry.loadingAmount))
        : CARGO_TRANSFER_SLOTS;
      for (let j = 0; j < points; j++) {
        // Each observed pickup passes through the intake once. Unloading uses a
        // continuous conveyor flow whose endpoint is the refinery's rear bay.
        const t = loading
          ? clamp(progress * 1.45 - (j / Math.max(1, points)) * 0.45)
          : (this.time * 1.5 + j / points) % 1;
        if (loading && (t <= 0 || t >= 1)) continue;
        const q = smooth(t),
          lane = ((j % 3) - 1) * 0.07;
        const radius = 0.055 * Math.sin(Math.PI * t) ** 0.3;
        if (loading) {
          this.set(
            this.transfer,
            transfers++,
            intake[0] + (center[0] - intake[0]) * q,
            intake[1] + lane,
            intake[2] +
              (center[2] + size[2] * 0.5 - intake[2]) * q +
              Math.sin(Math.PI * t) * 0.32,
            radius,
            radius * 0.8,
            radius * 1.5,
            j + t * 3,
          );
        } else {
          this.point.set(...center).applyMatrix4(this.base);
          this.target.set(
            entry.zone.position[0] - (3.8 * entry.zone.radius) / 6,
            entry.zone.position[1] + (7.6 * entry.zone.radius) / 6,
            (2.9 * entry.zone.radius) / 6,
          );
          const scale = this.bodies.models[b].scale.x;
          this.set(
            this.transfer,
            transfers++,
            this.point.x + (this.target.x - this.point.x) * q,
            this.point.y + (this.target.y - this.point.y) * q + lane * scale,
            this.point.z +
              (this.target.z - this.point.z) * q +
              Math.sin(Math.PI * t) * 0.55,
            radius * scale,
            radius * scale,
            radius * scale * 1.5,
            j + t * 3,
            false,
          );
        }
      }
    }
    for (let i = fills; i < this.fill.count; i++)
      this.fill.setMatrixAt(i, this.zero);
    this.fill.count = fills;
    this.meter.count = meters;
    this.fill.visible = fills > 0;
    this.meter.visible = meters > 0;
    this.fill.instanceMatrix.needsUpdate = true;
    this.meter.instanceMatrix.needsUpdate = true;
    if (this.meter.instanceColor) this.meter.instanceColor.needsUpdate = true;
    this.transfer.visible = this.animationsEnabled && transfers > 0;
    if (this.animationsEnabled) {
      this.transfer.count = transfers;
      this.transfer.instanceMatrix.needsUpdate = true;
    }
  }
}
