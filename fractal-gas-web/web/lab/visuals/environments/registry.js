const registry = new Map();
export function registerEnvironment(id, create) {
  if (!id || registry.has(id) || typeof create !== "function")
    throw new Error(`Invalid or duplicate environment renderer: ${id}`);
  registry.set(id, create);
}
export function createEnvironment(scene) {
  const kind = scene.environment?.kind || "arena",
    create = registry.get(kind);
  if (!create) throw new Error(`Unknown environment renderer: ${kind}`);
  return create(scene);
}
