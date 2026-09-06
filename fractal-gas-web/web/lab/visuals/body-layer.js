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
    this.channels = channels || actionLayout(scene);
    this.dt = scene.physics?.dt || 1 / 60;
    this.bodies = resolveBodies(scene);
    this.models = [];
    this.controlled = [];
    this.parts = [];
    this.animations = [];
    this.instances = [];
    this.lods = [];
    this.zero = new T.Matrix4().makeScale(0, 0, 0);
    const templates = new Map(),
      groups = new Map();
    const crowd = this.bodies.filter((b) => b.controlled).length > 16;
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
        model = themeScenery(rockModel(vertices), style);
      }
      model.position.set(...(b.position || [0, 0]), 0.08);
      model.rotation.z = b.angle || 0;
      parent.add(model);
      this.models.push(model);
      this.parts.push(meshes(model));
      this.animations.push(animatedParts(model));
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
    this.controlled.forEach((b, c) =>
      animateAgent(this.animations[b], {
        time,
        speed: Math.hypot(state[8 + 2 * n + b], state[8 + 3 * n + b]),
        modelScale: this.models[b].scale.x,
        ...visualInput(this.channels, action, b),
      }),
    );
    if (this.instances.length) {
      for (const b of this.controlled) this.models[b].updateMatrixWorld(true);
      for (const { mesh, bodies, part } of this.instances) {
        bodies.forEach((b, i) => {
          const model = this.models[b],
            child = this.parts[b][part];
          mesh.setMatrixAt(
            i,
            bits[this.info[5] + b] & 1 && visibleInModel(child, model)
              ? child.matrixWorld
              : this.zero,
          );
        });
        mesh.instanceMatrix.needsUpdate = true;
      }
      for (const b of this.controlled) this.models[b].visible = false;
    }
  }
  updateLod(camera, viewportHeight) {
    const pixelsPerUnit = viewportHeight / (camera.top - camera.bottom);
    for (const entry of this.lods) {
      entry.current = chooseLod(
        1.52 * entry.model.scale.x * pixelsPerUnit,
        entry.current,
      );
      entry.high.visible = entry.current === "high";
      entry.low.visible = entry.current === "low";
    }
  }
}
