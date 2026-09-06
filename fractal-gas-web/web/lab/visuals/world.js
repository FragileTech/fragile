import * as T from "../vendor/three.module.js";
import { assetModel } from "./assets.js";
import { worldCatalog } from "./world-catalog.js";
import { visualInput } from "../actions.js";

export function worldModel(
  style,
  kind,
  size = worldCatalog[kind]?.size,
  lod = "low",
) {
  const model = assetModel(style, kind, lod);
  if (!model) return null;
  const dimensions = worldCatalog[kind].size;
  model.scale.set(...dimensions.map((v, i) => size[i] / v));
  model.userData.authoredWorld = true;
  return model;
}

// Batch repeated scenery and resources by authored geometry/material. Each batch
// owns only its instance buffers; shared GLB geometry is retained by the cache.
export class WorldInstances {
  constructor(style, parent) {
    this.style = style;
    this.parent = parent;
    this.entries = [];
    this.templates = new Map();
    this.batches = [];
  }
  add(kind, position, size = worldCatalog[kind]?.size, angle = 0) {
    if (!this.templates.has(kind)) {
      let model = worldModel(this.style, kind);
      if (!model && kind.startsWith("drop-")) {
        model = new T.Mesh(
          new T.OctahedronGeometry(0.5),
          new T.MeshStandardMaterial({
            color: this.style === "steampunk" ? 0xbc843d : 0x9a70ce,
          }),
        );
        model.position.z = 0.5;
      }
      if (!model) return null;
      const levels = {};
      for (const lod of ["low", "high"]) {
        const template =
          lod === "low"
            ? model
            : worldModel(this.style, kind, undefined, "high") || model;
        template.updateMatrixWorld(true);
        levels[lod] = [];
        template.traverse((part) => {
          if (part.isMesh)
            levels[lod].push({
              geometry: part.geometry,
              material: part.material,
              matrix: part.matrixWorld.clone(),
            });
        });
      }
      this.templates.set(kind, levels);
    }

    const object = new T.Object3D();
    object.position.set(...position);
    object.scale.set(...size.map((v, i) => v / worldCatalog[kind].size[i]));
    object.rotation.z = angle;
    object.userData.assetModel = kind;
    object.userData.size = size;
    object.userData.lod = "low";
    this.entries.push(object);
    return object;
  }
  build(dynamic = false) {
    for (const [kind, levels] of this.templates) {
      const entries = this.entries.filter(
        (e) => e.userData.assetModel === kind,
      );
      for (const [lod, parts] of Object.entries(levels))
        for (const part of parts) {
          const mesh = new T.InstancedMesh(
            part.geometry,
            part.material,
            entries.length,
          );
          mesh.name = `World instances / ${kind}`;
          mesh.userData.authoredWorld = true;
          mesh.frustumCulled = false;
          if (dynamic) mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
          this.parent.add(mesh);
          this.batches.push({ mesh, entries, matrix: part.matrix, lod });
        }
    }
    this.update();
    return this;
  }
  update() {
    const matrix = new T.Matrix4(),
      zero = new T.Matrix4().makeScale(0, 0, 0);
    for (const e of this.entries) e.updateMatrix();
    for (const { mesh, entries, matrix: local, lod } of this.batches) {
      entries.forEach((e, i) =>
        mesh.setMatrixAt(
          i,
          e.visible && e.userData.lod === lod
            ? matrix.multiplyMatrices(e.matrix, local)
            : zero,
        ),
      );
      mesh.instanceMatrix.needsUpdate = true;
      mesh.visible = entries.some((e) => e.visible && e.userData.lod === lod);
    }
  }
  updateLod(camera, height) {
    const pixels = height / (camera.top - camera.bottom);
    let changed = false;
    for (const entry of this.entries) {
      const size = Math.max(...entry.userData.size) * pixels;
      const old = entry.userData.lod;
      entry.userData.lod = size > 120 ? "high" : size < 90 ? "low" : old;
      changed ||= old !== entry.userData.lod;
    }
    if (changed) this.update();
  }
}

export function animateWorld(root, time) {
  root.traverse((part) => {
    const motion = part.userData.motion;
    if (motion === "world-spin") part.rotation.z = time * 0.16;
    if (motion === "world-gimbal") {
      const axis = ["x", "y", "z"][part.userData.axisIndex];
      part.rotation[axis] = time * part.userData.speed;
    }
  });
}

export function worldSurface(style, kind = "floor-tile") {
  const root = assetModel(style, kind, "low");
  let material;
  root?.traverse((part) => {
    if (part.material?.name.includes("road")) material = part.material;
  });
  if (!material) return null;
  const copy = material.clone();
  // UVs in the procedural boundary are world coordinates; repeat on a 2-unit grid.
  for (const key of ["map", "normalMap", "roughnessMap"])
    if (copy[key]) {
      copy[key] = copy[key].clone();
      copy[key].wrapS = copy[key].wrapT = T.RepeatWrapping;
      copy[key].repeat.set(0.5, 0.5);
      copy[key].needsUpdate = true;
    }
  return copy;
}

function tube() {
  const mesh = new T.Mesh(
    new T.CylinderGeometry(0.025, 0.025, 1, 6),
    new T.MeshStandardMaterial({
      color: 0x706052,
      metalness: 0.8,
      roughness: 0.6,
    }),
  );
  mesh.geometry.rotateX(Math.PI / 2);
  return mesh;
}
const Z = new T.Vector3(0, 0, 1);

// A pure pose function of native state/actions: scrubbing reconstructs effects
// without event queues, wall-clock timers, simulation mutations or GPU readback.
export class WorldDynamics {
  constructor(scene, info, style, bodyLayer, parent, tetherParent) {
    this.scene = scene;
    this.info = info;
    this.style = style;
    this.bodyLayer = bodyLayer;
    this.group = new T.Group();
    parent.add(this.group);
    this.pickupBatch = new WorldInstances(style, this.group);
    const variants = [
      "drop-crystal",
      "drop-nugget",
      "drop-salvage",
      "drop-capsule",
      "drop-core",
      "drop-pile",
    ];
    this.pickups = (scene.pickups || []).map((p, i) => {
      const radius = p.radius || 0.4;
      return this.pickupBatch.add(
        variants[i % variants.length],
        [...p.position, 0.04],
        [radius * 2, radius * 2, radius * 1.6],
      );
    });
    this.pickupBatch.build(true);
    this.bursts = (scene.pickups || []).map(() => {
      const model = worldModel(style, "pickup-burst");
      if (model) {
        this.group.add(model);
        model.visible = false;
      }
      return model;
    });
    this.tethers = (scene.tethers || []).map(() => {
      const group = new T.Group(),
        cable = tube();
      const clamp = worldModel(style, "capture-clamp"),
        latch = worldModel(style, "capture-latch");
      const fitting = worldModel(style, "tether-fitting");
      group.add(cable);
      if (clamp) group.add(clamp);
      if (latch) group.add(latch);
      if (fitting) group.add(fitting);
      group.visible = false;
      tetherParent.add(group);
      return { group, cable, clamp, latch, fitting };
    });
    this.effects = bodyLayer.controlled.map((i) => {
      const model = bodyLayer.bodies[i].visual?.model || "rocket";
      const kind =
        model === "drone"
          ? "rotor-airflow"
          : model === "harvester"
            ? "intake-swirl"
            : "path-trail";
      const effect = worldModel(style, kind);
      if (effect) this.group.add(effect);
      const thrust =
        model === "rocket" ? worldModel(style, "thrust-plume") : null;
      if (thrust) this.group.add(thrust);
      return { i, model, effect, thrust };
    });
  }
  update(state, action) {
    const { info, scene, bodyLayer } = this;
    const bits = new Uint32Array(state.buffer, state.byteOffset, state.length),
      n = bodyLayer.models.length;
    const time = bits[0] * (scene.physics?.dt || 1 / 60);
    this.pickups.forEach((model, i) => {
      const at = info[8] + i * 3;
      if (model) {
        model.position.set(state[at], state[at + 1], 0.04);
        model.visible = state[at + 2] <= 0;
      }
      const burst = this.bursts[i],
        elapsed = (scene.respawn_seconds ?? 4) - state[at + 2];
      if (burst) {
        burst.visible = state[at + 2] > 0 && elapsed >= 0 && elapsed < 0.45;
        burst.position.set(state[at], state[at + 1], 0.15);
        burst.scale.setScalar(0.2 + Math.max(0, Math.min(0.45, elapsed)) * 1.5);
        burst.rotation.z = time * 0.4;
      }
    });
    this.pickupBatch.update();
    this.tethers.forEach((entry, i) => {
      const b = bits[info[7] + 2 * i] - 1,
        a = scene.tethers[i].a;
      entry.group.visible = b >= 0;
      if (b < 0) return;
      const start = new T.Vector3(state[8 + a], state[8 + n + a], 0.4),
        end = new T.Vector3(state[8 + b], state[8 + n + b], 0.4);
      const delta = end.clone().sub(start),
        length = delta.length(),
        angle = Math.atan2(delta.y, delta.x);
      entry.cable.position.copy(start).addScaledVector(delta, 0.5);
      entry.cable.quaternion.setFromUnitVectors(Z, delta.clone().normalize());
      entry.cable.scale.set(1, 1, length);
      for (const [object, point] of [
        [entry.clamp, end],
        [entry.fitting, start],
        [entry.latch, entry.cable.position],
      ])
        if (object) {
          object.position.copy(point);
          object.rotation.z = angle;
        }
    });
    this.effects.forEach(({ i, model, effect, thrust }) => {
      const source = bodyLayer.models[i],
        speed = Math.hypot(state[8 + 2 * n + i], state[8 + 3 * n + i]);
      const input = visualInput(bodyLayer.channels, action, i);
      const active = !!(bits[info[5] + i] & 1);
      if (effect) {
        effect.position.copy(source.position);
        effect.rotation.z = source.rotation.z;
        if (model === "harvester") effect.rotation.z += time * 0.8;
        const scale = source.scale.x;
        effect.scale.setScalar(scale);
        if (model === "drone")
          effect.scale.z *= 0.8 + 0.15 * Math.sin(time * 8);
        effect.visible = active && (model === "drone" || speed > 0.05);
        if (model !== "drone" && model !== "harvester")
          effect.position.add(
            new T.Vector3(-0.6 * scale, 0, 0.02).applyAxisAngle(
              Z,
              source.rotation.z,
            ),
          );
        if (model === "harvester")
          effect.position.add(
            new T.Vector3(0.7 * scale, 0, 0.12).applyAxisAngle(
              Z,
              source.rotation.z,
            ),
          );
      }
      if (thrust) {
        const power = Math.abs(input.thrust || 0);
        thrust.position
          .copy(source.position)
          .add(
            new T.Vector3(-0.7 * source.scale.x, 0, 0.22).applyAxisAngle(
              Z,
              source.rotation.z,
            ),
          );
        thrust.rotation.z = source.rotation.z;
        thrust.scale.setScalar(source.scale.x);
        thrust.visible = active && power > 0.01;
        thrust.traverse((p) => {
          if (p.userData.motion === "effect-stage")
            p.visible = p.userData.stage === Math.min(2, Math.floor(power * 3));
        });
      }
    });
    return time;
  }
}
