import * as T from "../vendor/three.module.js";
import { disposeGroup } from "./resources.js";

// Four crossed triangles per nozzle; all 32 native thrusters fit 128 triangles.
function plumeGeometry() {
  const g = new T.BufferGeometry();
  g.setAttribute(
    "position",
    new T.Float32BufferAttribute(
      [
        0, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, 0, -0.45, 0.002,
        0.72, 0, 0.002, 0, 0.45, 0.002, 0, 0.002, -0.45, 0, 0.002, 0.45, 0.72,
        0.002, 0,
      ],
      3,
    ),
  );
  const colors = [];
  for (let i = 0; i < 12; i++)
    colors.push(...(i < 6 ? [0.5, 0.65, 0.8] : [1, 1, 1]));
  g.setAttribute("color", new T.Float32BufferAttribute(colors, 3));
  return g;
}
const Z = new T.Vector3(0, 0, 1);
// Surface points measured from the authored, normalized drone export. Keep jets
// outside the rotor shrouds so small lateral commands remain visible overhead.
const droneMounts = {
  futuristic: {
    front: [0.57738, 0, 0.0496],
    rear: [-0.74032, 0, 0.15896],
    side: [0.00328, 0.74633, 0.15896],
    torqueFront: [0.38601, 0.74633, 0.15896],
    torqueRear: [-0.39586, 0.74633, 0.15896],
  },
  steampunk: {
    front: [0.48467, 0, 0.13468],
    rear: [-0.53928, 0, 0.22569],
    side: [0.00683, 0.75317, 0.22569],
    torqueFront: [0.32539, 0.75317, 0.22569],
    torqueRear: [-0.32539, 0.75317, 0.22569],
  },
};
export class ActionEffects {
  constructor(bodyLayer, parent, style = "futuristic") {
    this.layer = bodyLayer;
    this.droneMounts = droneMounts[style] || droneMounts.futuristic;
    this.sockets = [];
    this.jetHeights = [];
    this.rocketMount =
      style === "steampunk"
        ? [0.05466, 0.39773, 0.20329, -0.62716]
        : [0.03419, 0.38774, 0.19881, -0.6305];
    // Authored sockets are read once per style/LOD collection, never per frame.
    const inverse = new T.Matrix4(),
      relative = new T.Matrix4(),
      bounds = new T.Box3();
    for (const i of bodyLayer.controlled) {
      const root = bodyLayer.models[i];
      root.updateWorldMatrix(true, true);
      inverse.copy(root.matrixWorld).invert();
      const sockets = { brake: [], reverse: [], drive: [] };
      let height = 0;
      root.traverse((part) => {
        if (part.isMesh && part.userData.motion !== "thrust") {
          if (!part.geometry.boundingBox) part.geometry.computeBoundingBox();
          relative.multiplyMatrices(inverse, part.matrixWorld);
          bounds.copy(part.geometry.boundingBox).applyMatrix4(relative);
          height = Math.max(height, bounds.max.z);
        }
        const kind = part.userData.effectSocket;
        if (
          !sockets[kind] ||
          sockets[kind].length >= (kind === "brake" ? 2 : 1)
        )
          return;
        sockets[kind].push(
          new T.Vector3()
            .setFromMatrixPosition(part.matrixWorld)
            .applyMatrix4(inverse),
        );
      });
      this.sockets[i] = sockets;
      // Configured thruster locations are planar. Cosmetic height keeps even
      // a central jet visible without moving its native X/Y application point.
      this.jetHeights[i] = height + 0.04;
    }
    this.group = new T.Group();
    this.group.name = "Command effects";
    parent.add(this.group);
    this.local = new T.Matrix4();
    this.base = new T.Matrix4();
    this.matrix = new T.Matrix4();
    this.position = new T.Vector3();
    this.scale = new T.Vector3();
    this.quaternion = new T.Quaternion();
    this.jetColor = new T.Color(style === "steampunk" ? 0xffbb62 : 0x75e9ff);
    this.brakeColor = new T.Color(0xff394c);
    this.reverseColor = new T.Color(0xe6f7ff);
    this.driveColor = new T.Color(style === "steampunk" ? 0xffd06f : 0x7cffda);
    const count = Math.max(1, bodyLayer.controlled.length);
    this.jets = new T.InstancedMesh(
      plumeGeometry(),
      new T.MeshBasicMaterial({
        vertexColors: true,
        side: T.DoubleSide,
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
        toneMapped: false,
      }),
      count * 32,
    );
    this.lamps = new T.InstancedMesh(
      new T.BoxGeometry(1, 1, 1),
      new T.MeshBasicMaterial({ toneMapped: false }),
      count * 4,
    );
    for (const mesh of [this.jets, this.lamps]) {
      mesh.name = mesh === this.jets ? "Command jets" : "Command lamps";
      mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.count = 0;
      mesh.visible = false;
      this.group.add(mesh);
    }
    this.guideGeometry = new T.BufferGeometry();
    this.guidePositions = new Float32Array(256 * 9);
    this.guideColors = new Float32Array(256 * 9);
    this.guideGeometry.setAttribute(
      "position",
      new T.BufferAttribute(this.guidePositions, 3).setUsage(
        T.DynamicDrawUsage,
      ),
    );
    this.guideGeometry.setAttribute(
      "color",
      new T.BufferAttribute(this.guideColors, 3).setUsage(T.DynamicDrawUsage),
    );
    this.guideGeometry.setDrawRange(0, 0);
    this.guides = new T.Mesh(
      this.guideGeometry,
      new T.MeshBasicMaterial({
        vertexColors: true,
        side: T.DoubleSide,
        transparent: true,
        opacity: 0.9,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    this.guides.material.forceSinglePass = true;
    this.jets.material.forceSinglePass = true;
    this.guides.name = "Command guides";
    this.guides.frustumCulled = false;
    this.guides.matrixAutoUpdate = false;
    this.guides.visible = false;
    this.guides.renderOrder = 10;
    this.group.add(this.guides);
    this.label = "";
    this.guideBody = undefined;
  }
  active(i) {
    const l = this.layer;
    return (
      !!l.models[i] &&
      l.inView?.[i] !== false &&
      (l.active
        ? l.active[i]
        : l.instances?.length
          ? !!(l.instanceState?.[l.info[5] + i] & 1)
          : l.models[i].visible !== false)
    );
  }
  transform(i, cosmetic = true) {
    const root = this.layer.models[i];
    root.updateMatrix();
    this.base.copy(root.matrix);
    const pose = this.layer.presentations?.[i];
    if (cosmetic && pose) {
      pose.updateMatrix();
      this.base.multiply(pose.matrix);
    }
  }
  instance(mesh, index, x, y, z, angle, length, width, height, color) {
    this.position.set(x, y, z);
    this.scale.set(length, width, height);
    this.quaternion.setFromAxisAngle(Z, angle);
    this.local.compose(this.position, this.quaternion, this.scale);
    mesh.setMatrixAt(
      index,
      this.matrix.multiplyMatrices(this.base, this.local),
    );
    mesh.setColorAt(index, color);
  }
  jet(x, y, z, fx, fy, value) {
    if (Math.abs(value) < 1e-6) return;
    const m = Math.min(1, Math.abs(value)),
      sign = Math.sign(value);
    this.instance(
      this.jets,
      this.jetCount++,
      x,
      y,
      z,
      Math.atan2(-fy * sign, -fx * sign),
      0.1 + 0.75 * m,
      0.025 + 0.11 * m,
      0.025 + 0.11 * m,
      this.jetColor,
    );
  }
  lamp(x, y, value, color, z = 0.36) {
    if (value <= 0) return;
    this.instance(
      this.lamps,
      this.lampCount++,
      x,
      y,
      z,
      0,
      0.08,
      0.04 + 0.2 * value,
      0.04 + 0.08 * value,
      color,
    );
  }
  update() {
    if (
      this.layer.presentationVersion !== undefined &&
      this.version === this.layer.presentationVersion
    )
      return false;
    this.version = this.layer.presentationVersion;
    this.jetCount = 0;
    this.lampCount = 0;
    for (const i of this.layer.controlled) {
      if (!this.active(i)) continue;
      const c = this.layer.commands?.[i];
      if (!c) continue;
      this.transform(i);
      if (c.kind === "thrusters") {
        const s = Math.abs(this.layer.models[i].scale.x) || 1;
        for (const t of c.thrusters)
          this.jet(
            t.position[0] / s,
            t.position[1] / s,
            this.jetHeights[i],
            t.direction[0],
            t.direction[1],
            t.value,
          );
      } else if (c.kind === "kart") {
        const sockets = this.sockets[i];
        for (let j = 0; j < 2; j++) {
          const p = sockets.brake[j];
          this.lamp(
            p?.x ?? -0.82,
            p?.y ?? (j ? 0.38 : -0.38),
            c.brake,
            this.brakeColor,
            p?.z,
          );
        }
        const reverse = sockets.reverse[0],
          drive = sockets.drive[0];
        this.lamp(
          reverse?.x ?? -0.83,
          reverse?.y ?? 0,
          Math.max(0, -c.throttle),
          this.reverseColor,
          reverse?.z,
        );
        this.lamp(
          drive?.x ?? -0.3,
          drive?.y ?? 0,
          Math.abs(c.throttle),
          this.driveColor,
          drive?.z,
        );
      } else {
        const drone = this.layer.bodies?.[i]?.visual?.model === "drone";
        const mounts = drone ? this.droneMounts : null;
        if (c.kind === "holonomic") {
          const longitudinal = c.forceX >= 0 ? mounts?.rear : mounts?.front;
          this.jet(
            longitudinal?.[0] ?? -Math.sign(c.forceX) * 0.6,
            0,
            longitudinal?.[2] ?? 0.32,
            1,
            0,
            c.forceX,
          );
          if (mounts) {
            this.jet(
              mounts.torqueFront[0],
              -Math.sign(c.forceY) * mounts.side[1],
              mounts.side[2],
              0,
              1,
              c.forceY,
            );
            this.jet(
              mounts.torqueRear[0],
              -Math.sign(c.forceY) * mounts.side[1],
              mounts.side[2],
              0,
              1,
              c.forceY,
            );
          } else this.jet(0, -Math.sign(c.forceY) * 0.55, 0.32, 0, 1, c.forceY);
        }
        // Positive native torque is counterclockwise: +Y force at +X and
        // -Y force at -X. Exhaust points opposite to each applied force.
        this.jet(
          mounts?.torqueFront[0] ?? this.rocketMount[0],
          -Math.sign(c.torque) *
            (mounts?.torqueFront[1] ?? this.rocketMount[1]),
          mounts?.torqueFront[2] ?? this.rocketMount[2],
          0,
          1,
          c.torque,
        );
        this.jet(
          mounts?.torqueRear[0] ?? this.rocketMount[3],
          Math.sign(c.torque) * (mounts?.torqueRear[1] ?? this.rocketMount[1]),
          mounts?.torqueRear[2] ?? this.rocketMount[2],
          0,
          -1,
          c.torque,
        );
      }
    }
    this.upload(this.jets, this.jetCount);
    this.upload(this.lamps, this.lampCount);
    this.guides.visible = !!this.guideEnabled && this.active(this.guideBody);
    if (this.guides.visible) {
      this.transform(this.guideBody, false);
      this.guides.matrix.copy(this.base);
    }
    return true;
  }
  upload(mesh, count) {
    mesh.count = count;
    mesh.visible = count > 0;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
  triangle(ax, ay, bx, by, cx, cy, color) {
    if (this.guideVertices + 3 > 768) return;
    const n = this.guideVertices * 3;
    this.guidePositions.set([ax, ay, 0.95, bx, by, 0.95, cx, cy, 0.95], n);
    for (let k = 0; k < 3; k++) this.guideColors.set(color, n + k * 3);
    this.guideVertices += 3;
  }
  arrow(x, y, dx, dy, value, color) {
    if (Math.abs(value) < 1e-6) return;
    const sign = Math.sign(value),
      norm = Math.hypot(dx, dy) || 1,
      ux = (dx / norm) * sign,
      uy = (dy / norm) * sign;
    const length = 0.2 + Math.min(1, Math.abs(value)) * 0.9,
      w = 0.02;
    const ex = x + ux * length,
      ey = y + uy * length,
      bx = ex - ux * 0.2,
      by = ey - uy * 0.2;
    this.triangle(
      x - uy * w,
      y + ux * w,
      x + uy * w,
      y - ux * w,
      bx + uy * w,
      by - ux * w,
      color,
    );
    this.triangle(
      x - uy * w,
      y + ux * w,
      bx + uy * w,
      by - ux * w,
      bx - uy * w,
      by + ux * w,
      color,
    );
    this.triangle(
      bx - uy * 0.09,
      by + ux * 0.09,
      bx + uy * 0.09,
      by - ux * 0.09,
      ex,
      ey,
      color,
    );
  }
  updateGuides(bodyIndex, enabled) {
    const i = this.layer.controlled.includes(bodyIndex)
      ? bodyIndex
      : this.layer.controlled[0];
    this.guideBody = i;
    this.guideEnabled = !!enabled && i != null;
    this.guides.visible = this.guideEnabled && this.active(i);
    if (!this.guideEnabled) {
      this.label = "";
      return this.label;
    }
    const c = this.layer.commands[i];
    if (!c) {
      this.guides.visible = false;
      return "";
    }
    this.guideVertices = 0;
    const cyan = [0.3, 0.95, 1],
      green = [0.5, 1, 0.6],
      amber = [1, 0.7, 0.22];
    const pct = (v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%`;
    let detail;
    if (c.kind === "thrusters") {
      const s = Math.abs(this.layer.models[i].scale.x) || 1;
      for (const t of c.thrusters)
        this.arrow(
          t.position[0] / s,
          t.position[1] / s,
          ...t.direction,
          t.value,
          cyan,
        );
      detail = c.thrusters
        .map((t, j) => `Jet ${j + 1} ${pct(t.value)}`)
        .join(" · ");
    } else if (c.kind === "kart") {
      this.arrow(0, 0, 1, 0, c.throttle, cyan);
      this.arrow(0.5, 0, 0, 1, c.steering, green);
      detail = `Throttle ${pct(c.throttle)} · Steering ${pct(c.steering)} · Brake ${pct(c.brake)}`;
    } else {
      this.arrow(0, 0, 1, 0, c.kind === "vector" ? c.thrust : c.forceX, cyan);
      if (c.kind === "holonomic") this.arrow(0, 0, 0, 1, c.forceY, green);
      detail =
        c.kind === "vector"
          ? `Thrust ${pct(c.thrust)}`
          : `Force X ${pct(c.forceX)} · Force Y ${pct(c.forceY)}`;
      detail += ` · Torque ${pct(c.torque)}`;
    }
    if (Math.abs(c.torque) > 1e-6) {
      const sign = Math.sign(c.torque),
        end = sign * (0.25 + Math.abs(c.torque) * Math.PI * 1.3),
        r = 1.12;
      for (let k = 0; k < 24; k++) {
        const a = (end * k) / 24,
          b = (end * (k + 1)) / 24;
        this.triangle(
          Math.cos(a) * (r - 0.03),
          Math.sin(a) * (r - 0.03),
          Math.cos(a) * (r + 0.03),
          Math.sin(a) * (r + 0.03),
          Math.cos(b) * (r + 0.03),
          Math.sin(b) * (r + 0.03),
          amber,
        );
        this.triangle(
          Math.cos(a) * (r - 0.03),
          Math.sin(a) * (r - 0.03),
          Math.cos(b) * (r + 0.03),
          Math.sin(b) * (r + 0.03),
          Math.cos(b) * (r - 0.03),
          Math.sin(b) * (r - 0.03),
          amber,
        );
      }
      const x = Math.cos(end) * r,
        y = Math.sin(end) * r,
        tx = -Math.sin(end) * sign,
        ty = Math.cos(end) * sign;
      this.triangle(
        x - ty * 0.14 - tx * 0.18,
        y + tx * 0.14 - ty * 0.18,
        x + ty * 0.14 - tx * 0.18,
        y - tx * 0.14 - ty * 0.18,
        x + tx * 0.12,
        y + ty * 0.12,
        amber,
      );
    }
    this.guideGeometry.setDrawRange(0, this.guideVertices);
    this.guideGeometry.attributes.position.needsUpdate = true;
    this.guideGeometry.attributes.color.needsUpdate = true;
    this.transform(i, false);
    this.guides.matrix.copy(this.base);
    this.label = `Body ${i + 1} · Commands · ${detail}`;
    return this.label;
  }
  dispose() {
    this.group.removeFromParent();
    disposeGroup(this.group);
  }
}
