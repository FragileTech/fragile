import * as T from "../vendor/three.module.js";

// Instanced cargo and transfer particles reconstruct directly from packed state.
export class CargoVisuals {
  constructor(scene, info, bodyLayer, parent, style) {
    this.animationsEnabled = true;
    this.scene = scene;
    this.info = info;
    this.bodies = bodyLayer;
    if (!info[15]) return;
    const count = bodyLayer.controlled.length;
    const color = style === "steampunk" ? 0xeab553 : 0xb78bff;
    this.group = new T.Group();
    parent.add(this.group);
    this.fill = new T.InstancedMesh(
      new T.BoxGeometry(1, 1, 1),
      new T.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.35,
        metalness: 0.5,
        roughness: 0.4,
      }),
      count,
    );
    this.meter = new T.InstancedMesh(
      new T.BoxGeometry(1, 1, 1),
      new T.MeshBasicMaterial({ color }),
      count * 5,
    );
    this.transfer = new T.InstancedMesh(
      new T.OctahedronGeometry(0.09),
      new T.MeshBasicMaterial({ color }),
      count * 6,
    );
    this.group.add(this.fill, this.meter, this.transfer);
    for (const mesh of [this.fill, this.meter, this.transfer]) {
      mesh.frustumCulled = false;
      mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
    }
    this.pose = new T.Object3D();
    this.fillPoses = bodyLayer.controlled.map(() => ({
      base: new T.Matrix4(),
      local: new T.Matrix4(),
    }));
    this.matrix = new T.Matrix4();
    this.bodyPose = new T.Object3D();
  }
  setAnimationsEnabled(enabled) {
    this.animationsEnabled = !!enabled;
    if (this.transfer) this.transfer.visible = !!enabled;
    this.animate();
  }
  animate() {
    if (!this.fill || !this.hasState) return;
    for (let c = 0; c < this.bodies.controlled.length; c++) {
      const b = this.bodies.controlled[c];
      const presentation = this.bodies.presentations?.[b];
      const { base, local } = this.fillPoses[c];
      this.matrix.copy(base);
      if (this.animationsEnabled && presentation) {
        presentation.updateMatrix();
        this.matrix.multiply(presentation.matrix);
      }
      this.matrix.multiply(local);
      this.fill.setMatrixAt(c, this.matrix);
    }
    this.fill.instanceMatrix.needsUpdate = true;
  }
  update(state) {
    if (!this.fill) return;
    const n = this.info[1],
      capacity = this.scene.cargo.capacity ?? 5;
    const time =
      new Uint32Array(state.buffer, state.byteOffset, 1)[0] *
      (this.scene.physics?.dt || 1 / 60);
    const set = (mesh, index, x, y, z, sx, sy, sz, angle = 0) => {
      this.pose.position.set(x, y, z);
      this.pose.scale.set(sx, sy, sz);
      this.pose.rotation.set(0, 0, angle);
      this.pose.updateMatrix();
      mesh.setMatrixAt(index, this.pose.matrix);
    };
    this.bodies.controlled.forEach((b, c) => {
      const at = this.info[15] + 4 * c,
        fraction = state[at] / capacity;
      const x = state[8 + b],
        y = state[8 + n + b],
        a = state[8 + 4 * n + b];
      const scale = this.bodies.models[b].scale.x;
      const drone = this.bodies.bodies[b].visual?.model === "drone";
      const back = drone ? 0 : -0.22 * scale;
      this.bodyPose.position.set(x, y, 0);
      this.bodyPose.rotation.set(0, 0, a);
      this.bodyPose.scale.setScalar(scale);
      this.bodyPose.updateMatrix();
      this.fillPoses[c].base.copy(this.bodyPose.matrix);
      this.pose.position.set(
        back / scale,
        0,
        (drone ? 0.85 : 0.48) + 0.12 * fraction,
      );
      this.pose.rotation.set(0, 0, 0);
      this.pose.scale.set(0.48, 0.36, Math.max(0, fraction) * 0.25);
      this.pose.updateMatrix();
      this.fillPoses[c].local.copy(this.pose.matrix);
      for (let j = 0; j < 5; j++) {
        const dx = (j - 2) * 0.14 * scale;
        set(
          this.meter,
          c * 5 + j,
          x + dx,
          y,
          (drone ? 1.2 : 1) * scale,
          0.1 * scale,
          0.09 * scale,
          j < fraction * 5 ? 0.06 : 0,
        );
      }
      if (!this.animationsEnabled) return;
      const zone = (this.scene.refineries || []).find(
        (z) => Math.hypot(x - z.position[0], y - z.position[1]) <= z.radius,
      );
      const unloading = zone && state[at + 1] > 0;
      for (let j = 0; j < 6; j++) {
        const t = (time * 1.5 + j / 6) % 1;
        set(
          this.transfer,
          c * 6 + j,
          unloading ? x + (zone.position[0] - x) * t : x,
          unloading ? y + (zone.position[1] + zone.radius - y) * t : y,
          0.6 + Math.sin(t * Math.PI) * 0.7,
          unloading ? 1 : 0,
          unloading ? 1 : 0,
          unloading ? 1 : 0,
        );
      }
    });
    this.hasState = true;
    this.animate();
    this.meter.instanceMatrix.needsUpdate = true;
    if (this.animationsEnabled) this.transfer.instanceMatrix.needsUpdate = true;
  }
}
