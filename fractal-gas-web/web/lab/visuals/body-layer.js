import { actionLayout, visualInput } from "../actions.js";
import { contactShadow } from "./lighting.js";
import * as T from "../vendor/three.module.js";
import { palette } from "./primitives.js";
import { rockModel } from "../models.js";
import { createAgentModel, animatedParts, animateAgent } from "./registry.js";
import { resolveBodies } from "../agent-types.js";
import { styleAssetsReady } from "./assets.js";
import { vehicleModels } from "./asset-manifest.js";
import { themeScenery } from "./style-palette.js";
import { worldModel } from "./world.js";

export function chooseLod(projectedPixels, current = "low", crowd = false) {
  if (crowd || projectedPixels < 90) return "low";
  return projectedPixels > 120 ? "high" : current;
}

function meshes(group) {
  const result = [];
  group.traverse((part) => {
    if (part.isMesh) result.push(part);
  });
  return result;
}
function visibleInModel(part, model) {
  for (let p = part; p && p !== model; p = p.parent)
    if (!p.visible) return false;
  return true;
}
export class BodyLayer {
  constructor(scene, info, parent, channels, { style } = {}) {
    this.info = info;
    this.parent = parent;
    this.parentInverse = new T.Matrix4();
    this.localInstanceMatrix = new T.Matrix4();
    this.channels = channels || actionLayout(scene);
    this.dt = scene.physics?.dt || 1 / 60;
    this.bodies = resolveBodies(scene);
    this.models = [];
    this.controlled = [];
    this.parts = [];
    this.animations = [];
    this.presentations = [];
    this.animationInputs = [];
    this.animationsEnabled = true;
    this.animationElapsed = 0;
    this.animationTime = 0;
    this.instances = [];
    this.lods = [];
    this.frustum = new T.Frustum();
    this.viewProjection = new T.Matrix4();
    this.sphere = new T.Sphere();
    this.inView = [];
    const templates = new Map(),
      groups = new Map();
    const crowd = this.bodies.filter((b) => b.controlled).length > 16;
    this.crowd = crowd;
    this.bodies.forEach((b, i) => {
      let model;
      if (b.controlled) {
        const color = this.controlled.length % 2 ? palette.rose : palette.cyan;
        // Untyped v1 scenes keep their original presentation.
        const visual = b.visual || {
          model: scene.task === "forage" ? "kart" : "rocket",
        };
        const key = JSON.stringify([visual, visual.color ?? color, style]);
        const authored =
          styleAssetsReady(style) &&
          vehicleModels.includes(visual.model ?? "rocket");
        if (!templates.has(key)) {
          if (authored && !crowd) {
            const levels = new T.Group();
            levels.add(
              createAgentModel(visual, color, { style, lod: "high" }),
              createAgentModel(visual, color, { style, lod: "low" }),
            );
            templates.set(key, levels);
          } else
            templates.set(
              key,
              createAgentModel(visual, color, {
                style,
                lod: crowd ? "low" : "high",
              }),
            );
        }
        model = templates.get(key).clone(true);
        if (authored && !crowd) {
          const [high, low] = model.children;
          high.visible = false;
          this.lods.push({ model, high, low, current: "low" });
        }
        model.add(contactShadow());
        model.scale.setScalar(((b.radius || 0.5) / 0.8) * (visual.scale ?? 1));
        this.controlled.push(i);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(i);
      } else {
        const vertices =
          b.vertices ||
          Array.from({ length: 7 }, (_, j) => [
            Math.cos((j * Math.PI * 2) / 7) * (b.radius || 0.5),
            Math.sin((j * Math.PI * 2) / 7) * (b.radius || 0.5),
          ]);
        const lo = [0, 1].map((axis) =>
          Math.min(...vertices.map((v) => v[axis])),
        );
        const hi = [0, 1].map((axis) =>
          Math.max(...vertices.map((v) => v[axis])),
        );
        const kind = ["ore-small", "ore-medium", "ore-large"][i % 3];
        const size = [
          hi[0] - lo[0],
          hi[1] - lo[1],
          Math.max(hi[0] - lo[0], hi[1] - lo[1]) * 0.8,
        ];
        const low = worldModel(style, kind, size);
        if (low) {
          const high = worldModel(style, kind, size, "high") || low.clone(true);
          model = new T.Group();
          for (const level of [high, low]) {
            // Keep the native local hull origin, including asymmetric cargo hulls.
            level.position.set((hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, 0);
            model.add(level);
          }
          high.visible = false;
          this.lods.push({
            model,
            high,
            low,
            current: "low",
            span: Math.max(size[0], size[1]),
          });
        } else model = themeScenery(rockModel(vertices), style);
      }
      model.position.set(...(b.position || [0, 0]), 0.08);
      model.rotation.z = b.angle || 0;
      parent.add(model);
      this.models.push(model);
      this.parts.push(meshes(model));
      const animation = animatedParts(model, {
        kind: b.controlled
          ? (b.visual?.model ?? (scene.task === "forage" ? "kart" : "rocket"))
          : undefined,
        style,
      });
      this.animations.push(animation);
      this.presentations.push(animation.pose);
      this.animationInputs.push({
        time: 0,
        speed: 0,
        signedSpeed: 0,
        thrust: 0,
        steer: 0,
        modelScale: model.scale.x,
        wheelTravel: 0,
        mechanicalTime: 0,
        throttleChannel: this.channels.findIndex(
          (c) => c.body === i && c.name === "throttle",
        ),
        brakeChannel: this.channels.findIndex(
          (c) => c.body === i && c.name === "brake",
        ),
        throttle: 0,
        brake: 0,
        displayThrottle: 0,
        displayBrake: 0,
        idleTime: 0,
        enabled: true,
        displayThrust: 0,
        displaySteer: 0,
      });
    });
    if (crowd)
      for (const bodies of groups.values()) {
        this.parts[bodies[0]].forEach((source, part) => {
          const mesh = new T.InstancedMesh(
            source.geometry,
            source.material,
            bodies.length,
          );
          mesh.frustumCulled = false;
          mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
          parent.add(mesh);
          this.instances.push({ mesh, bodies, part });
        });
      }
  }
  update(state, action) {
    const n = this.models.length,
      bits = new Uint32Array(state.buffer, state.byteOffset, state.length),
      time = bits[0] * this.dt;
    this.models.forEach((model, i) => {
      model.position.set(state[8 + i], state[8 + n + i], 0.1);
      model.rotation.z = state[8 + 4 * n + i];
      model.visible = !!(bits[this.info[5] + i] & 1);
    });
    for (const b of this.controlled) {
      const input = this.animationInputs[b];
      input.time = time;
      input.speed = Math.hypot(state[8 + 2 * n + b], state[8 + 3 * n + b]);
      input.signedSpeed =
        state[8 + 2 * n + b] * Math.cos(this.models[b].rotation.z) +
        state[8 + 3 * n + b] * Math.sin(this.models[b].rotation.z);
      Object.assign(input, visualInput(this.channels, action, b));
      input.throttle =
        input.throttleChannel >= 0
          ? action?.[input.throttleChannel] || 0
          : input.thrust;
      input.brake =
        input.brakeChannel >= 0 ? action?.[input.brakeChannel] || 0 : 0;
      input.enabled = this.animationsEnabled;
      // Once display animation starts, retain its last pose between state
      // deliveries. In particular, a throttled crowd must not snap back to
      // simulation-time wheel angles on the intervening display frames.
      if (!this.animationsEnabled || !this.animationStarted) {
        const travel = input.wheelTravel;
        input.wheelTravel = undefined;
        input.idleTime = undefined;
        animateAgent(this.animations[b], input);
        input.wheelTravel = travel;
      }
    }
    if (this.instances.length) {
      this.instanceState = bits;
      this.updateInstances();
      for (const b of this.controlled) this.models[b].visible = false;
    }
  }
  setAnimationsEnabled(enabled) {
    if (this.animationsEnabled === !!enabled) return;
    this.animationsEnabled = !!enabled;
    this.resetAnimation();
  }
  resetAnimation() {
    this.animationStarted = false;
    this.animationElapsed = 0;
    this.animationTime = 0;
    for (const b of this.controlled) {
      const input = this.animationInputs[b];
      input.wheelTravel = 0;
      input.mechanicalTime = 0;
      input.displayThrottle = 0;
      input.displayBrake = 0;
      input.idleTime = 0;
      input.enabled = false;
      input.displayThrust = 0;
      input.displaySteer = input.steer;
      animateAgent(this.animations[b], input);
    }
    if (this.instances.length) this.updateInstances();
  }
  animate(dt, idleTime, { playing = false, speed = 1 } = {}) {
    if (
      !this.animationsEnabled ||
      (typeof document !== "undefined" && document.hidden)
    )
      return;
    const elapsed = Math.min(0.1, Math.max(0, dt || 0));
    this.animationElapsed += elapsed;
    if (this.crowd && this.animationElapsed + 1e-9 < 1 / 30) return;
    const delta = this.animationElapsed;
    this.animationElapsed = 0;
    this.animationTime += delta;
    this.animationStarted = true;
    for (const b of this.controlled) {
      const active = this.instances.length
        ? this.instanceState && this.instanceState[this.info[5] + b] & 1
        : this.models[b].visible;
      if (!active || this.inView[b] === false) continue;
      const input = this.animationInputs[b];
      input.enabled = true;
      input.playing = playing;
      input.idleTime = this.animationTime + b * 0.73;
      if (playing) {
        input.wheelTravel += input.signedSpeed * delta * speed;
        input.mechanicalTime += delta * speed;
      }
      const thrust = input.thrust,
        steer = input.steer,
        throttle = input.throttle,
        brake = input.brake;
      const response = 1 - Math.exp(-delta * 10);
      input.displayThrust += (thrust - input.displayThrust) * response;
      input.displaySteer += (steer - input.displaySteer) * response;
      input.displayThrottle += (throttle - input.displayThrottle) * response;
      input.displayBrake += (brake - input.displayBrake) * response;
      input.throttle = input.displayThrottle;
      input.brake = input.displayBrake;
      input.thrust = input.displayThrust;
      input.steer = input.displaySteer;
      animateAgent(this.animations[b], input);
      input.thrust = thrust;
      input.steer = steer;
      input.throttle = throttle;
      input.brake = brake;
    }
    if (this.instances.length) this.updateInstances();
  }
  updateInstances() {
    if (!this.instanceState) return;
    this.parent.updateWorldMatrix(true, false);
    const parentInverse = this.parentInverse
      .copy(this.parent.matrixWorld)
      .invert();
    const localMatrix = this.localInstanceMatrix;
    for (const b of this.controlled) this.models[b].updateMatrixWorld(true);
    for (const { mesh, bodies, part } of this.instances) {
      let count = 0;
      for (const b of bodies) {
        const model = this.models[b],
          child = this.parts[b][part];
        if (
          this.instanceState[this.info[5] + b] & 1 &&
          this.inView[b] !== false &&
          visibleInModel(child, model)
        )
          mesh.setMatrixAt(
            count++,
            localMatrix.multiplyMatrices(parentInverse, child.matrixWorld),
          );
      }
      mesh.count = count;
      mesh.visible = count > 0;
      mesh.instanceMatrix.needsUpdate = true;
    }
  }
  updateLod(camera, viewportHeight) {
    if (camera.isCamera) {
      camera.updateMatrixWorld();
      this.frustum.setFromProjectionMatrix(
        this.viewProjection.multiplyMatrices(
          camera.projectionMatrix,
          camera.matrixWorldInverse,
        ),
      );
      let changed = false;
      for (const b of this.controlled) {
        const model = this.models[b];
        model.getWorldPosition(this.sphere.center);
        this.sphere.radius = 2 * model.scale.x;
        const visible = this.frustum.intersectsSphere(this.sphere);
        changed ||= this.inView[b] !== visible;
        this.inView[b] = visible;
      }
      if (changed) this.updateInstances();
    }
    const pixelsPerUnit = viewportHeight / (camera.top - camera.bottom);
    for (const entry of this.lods) {
      entry.current = chooseLod(
        (entry.span ?? 1.52) * entry.model.scale.x * pixelsPerUnit,
        entry.current,
      );
      entry.high.visible = entry.current === "high";
      entry.low.visible = entry.current === "low";
    }
  }
}
