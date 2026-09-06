import { stylePalette, themeScenery } from "../style-palette.js";
import { dressWorld } from "./world-dressing.js";
const registry = new Map();
export function registerEnvironment(id, create) {
  if (!id || registry.has(id) || typeof create !== "function")
    throw new Error(`Invalid or duplicate environment renderer: ${id}`);
  registry.set(id, create);
}
export function createEnvironment(scene, { style = "futuristic" } = {}) {
  const kind = scene.environment?.kind || "arena",
    create = registry.get(kind);
  if (!create) throw new Error(`Unknown environment renderer: ${kind}`);
  const environment = create(scene, { style });
  themeScenery(environment.group, style);
  dressWorld(environment, scene, style);
  // Apply after asset dressing so loaded road textures cannot darken the playable
  // area. A quiet matte surface separates agents and planning traces from the void.
  environment.group.traverse?.((object) => {
    if (
      object.name !== "Playable arena floor" &&
      object.name !== "Asphalt racing surface"
    )
      return;
    const material = object.material;
    material.map?.dispose();
    material.map = null;
    material.color.setHex(stylePalette[style].floor);
    material.metalness = 0;
    material.roughness = 0.95;
    material.normalScale?.set(0.2, 0.2);
    material.needsUpdate = true;
  });
  return environment;
}
