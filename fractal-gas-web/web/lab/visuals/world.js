import { CargoVisuals } from "./cargo.js";
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
    this.frustum = new T.Frustum();
    this.viewProjection = new T.Matrix4();
    this.bounds = new T.Sphere();
    this.instanceTransform = new T.Matrix4();
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
    const matrix = this.instanceTransform;
    for (const e of this.entries) e.updateMatrix();
    for (const { mesh, entries, matrix: local, lod } of this.batches) {
      let count = 0;
      for (const e of entries)
        if (e.visible && e.userData.inView !== false && e.userData.lod === lod)
          mesh.setMatrixAt(count++, matrix.multiplyMatrices(e.matrix, local));
      // Zero-scale matrices still execute every vertex. Submit only visible entries.
      mesh.count = count;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.visible = count > 0;
    }
  }
  updateLod(camera, height) {
    const pixels = height / (camera.top - camera.bottom);
    camera.updateMatrixWorld();
    this.parent.updateWorldMatrix(true, false);
    this.frustum.setFromProjectionMatrix(
      this.viewProjection.multiplyMatrices(
        camera.projectionMatrix,
        camera.matrixWorldInverse,
      ),
    );
    let changed = false;
    for (const entry of this.entries) {
      const size = Math.max(...entry.userData.size) * pixels;
      const old = entry.userData.lod;
      entry.userData.lod = size > 120 ? "high" : size < 90 ? "low" : old;
      changed ||= old !== entry.userData.lod;
      this.bounds.center.copy(entry.position);
      this.bounds.center.z += entry.userData.size[2] / 2;
      this.bounds.radius = Math.hypot(...entry.userData.size) / 2;
      this.bounds.applyMatrix4(this.parent.matrixWorld);
      const inView = this.frustum.intersectsSphere(this.bounds);
      changed ||= entry.userData.inView !== inView;
      entry.userData.inView = inView;
    }
    if (changed) this.update();
  }
}

const worldMotion = new WeakMap();
export function animateWorld(root, time, { enabled = true } = {}) {
  let bindings = worldMotion.get(root);
  if (!bindings) {
    bindings = [];
    root.traverse((part) => {
      const motion = part.userData.motion;
      if (motion !== "world-spin" && motion !== "world-gimbal") return;
      const axis =
        motion === "world-spin"
          ? "z"
          : ["x", "y", "z"][part.userData.axisIndex];
      if (axis)
        bindings.push({
          part,
          axis,
          rest: part.rotation[axis],
          speed: motion === "world-spin" ? 0.16 : part.userData.speed,
        });
    });
    worldMotion.set(root, bindings);
  }
  for (const { part, axis, rest, speed } of bindings)
    part.rotation[axis] = enabled ? rest + time * speed : rest;
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
    this.animationsEnabled = true;
    this.offset = new T.Vector3();
    this.start = new T.Vector3();
    this.end = new T.Vector3();
    this.delta = new T.Vector3();
    this.direction = new T.Vector3();
    this.scene = scene;
    this.info = info;
    this.style = style;
    this.bodyLayer = bodyLayer;
    this.group = new T.Group();
    parent.add(this.group);
    this.cargo = new CargoVisuals(scene, info, bodyLayer, this.group, style);
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
      const stages = [];
      thrust?.traverse((part) => {
        if (part.userData.motion === "effect-stage") stages.push(part);
      });
      return {
        i,
        model,
        effect,
        thrust,
        stages,
        active: false,
        speed: 0,
        power: 0,
      };
    });
  }
  setAnimationsEnabled(enabled) {
    this.animationsEnabled = !!enabled;
    this.cargo.setAnimationsEnabled(enabled);
    if (!enabled) {
      for (const burst of this.bursts)
        if (burst) {
          burst.visible = false;
          burst.rotation.z = 0;
          burst.scale.setScalar(1);
        }
      for (const { effect, thrust } of this.effects) {
        if (effect) {
          effect.visible = false;
          effect.scale.setScalar(1);
        }
        if (thrust) {
          thrust.visible = false;
          thrust.scale.setScalar(1);
        }
      }
    }
  }
  // Simulation time drives events; idle time adds restrained engine motion.
  animate(idleTime, { playing = false } = {}) {
    if (!this.animationsEnabled) return;
    this.cargo.animate();
    for (const entry of this.effects) {
      const { i, model, effect, thrust, active, speed, power } = entry;
      const source = this.bodyLayer.models[i];
      if (
        !source ||
        this.bodyLayer.inView?.[i] === false ||
        (!this.bodyLayer.instances?.length && source.visible === false)
      ) {
        if (effect) effect.visible = false;
        if (thrust) thrust.visible = false;
        continue;
      }
      if (effect) {
        effect.visible =
          active && (model === "drone" || (playing && speed > 0.05));
        effect.position.copy(source.position);
        effect.rotation.z =
          source.rotation.z + (model === "harvester" ? this.time * 0.8 : 0);
        effect.scale.setScalar(source.scale.x);
        if (model === "drone")
          effect.scale.z *= 0.8 + 0.08 * Math.sin(idleTime * 6 + i);
        if (model !== "drone")
          effect.position.add(
            this.offset
              .set(
                (model === "harvester" ? 0.7 : -0.6) * source.scale.x,
                0,
                model === "harvester" ? 0.12 : 0.02,
              )
              .applyAxisAngle(Z, source.rotation.z),
          );
      }
      if (thrust) {
        thrust.visible = active && playing && power > 0.01;
        thrust.position
          .copy(source.position)
          .add(
            this.offset
              .set(-0.7 * source.scale.x, 0, 0.22)
              .applyAxisAngle(Z, source.rotation.z),
          );
        thrust.rotation.z = source.rotation.z;
        thrust.scale.set(
          source.scale.x * (1 + 0.04 * Math.sin(idleTime * 19 + i)),
          source.scale.x,
          source.scale.x,
        );
      }
    }
  }
  update(state, action) {
    const { info, scene, bodyLayer } = this;
    this.cargo.update(state);
    const bits = new Uint32Array(state.buffer, state.byteOffset, state.length),
      n = bodyLayer.models.length;
    const time = bits[0] * (scene.physics?.dt || 1 / 60);
    this.time = time;
    this.pickups.forEach((model, i) => {
      const at = info[8] + i * 3;
      if (model) {
        model.position.set(state[at], state[at + 1], 0.04);
        model.visible = state[at + 2] <= 0;
      }
      const burst = this.bursts[i],
        elapsed = (scene.respawn_seconds ?? 4) - state[at + 2];
      if (burst && this.animationsEnabled) {
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
      const start = this.start.set(state[8 + a], state[8 + n + a], 0.4),
        end = this.end.set(state[8 + b], state[8 + n + b], 0.4);
      const delta = this.delta.copy(end).sub(start),
        length = delta.length(),
        angle = Math.atan2(delta.y, delta.x);
      entry.cable.position.copy(start).addScaledVector(delta, 0.5);
      entry.cable.quaternion.setFromUnitVectors(
        Z,
        this.direction.copy(delta).normalize(),
      );
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
    if (this.animationsEnabled) {
      for (const entry of this.effects) {
        const { i, stages } = entry;
        entry.speed = Math.hypot(state[8 + 2 * n + i], state[8 + 3 * n + i]);
        entry.power = Math.abs(
          visualInput(bodyLayer.channels, action, i).thrust || 0,
        );
        entry.active = !!(bits[info[5] + i] & 1);
        for (const part of stages)
          part.visible =
            part.userData.stage === Math.min(2, Math.floor(entry.power * 3));
      }
      this.animate(time, { playing: true });
    }
    return time;
  }
}
