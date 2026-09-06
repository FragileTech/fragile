import { NativeEngine } from "../native.js";
const registry = new Map();
export function registerController(id, definition) {
  if (!id || registry.has(id) || typeof definition.create !== "function")
    throw new Error(`Invalid or duplicate controller: ${id}`);
  registry.set(id, definition);
}
export function controllerDefinitions() {
  return Array.from(registry, ([id, d]) => ({
    id,
    label: d.label,
    parameters: d.parameters || {},
  }));
}
// Instantiate against any implementation of the descriptor/batch contract.
// Engine allocation and ownership belong to the host, not the algorithm.
export function instantiateController(id, engine, settings, scene) {
  const def = registry.get(id);
  if (!def) throw new Error(`Unknown controller: ${id}`);
  const controller = def.create({ engine, settings, scene });
  for (const method of ["begin", "advance", "result"])
    if (typeof controller[method] !== "function")
      throw new Error(`Controller missing ${method}`);
  return controller;
}
export function createController(module, scene, settings, threads = 1) {
  const id = settings.algorithm || "fmc",
    def = registry.get(id);
  if (!def) throw new Error(`Unknown controller: ${id}`);
  const engine = new NativeEngine(
    module,
    scene,
    def.worlds?.(settings) || 1,
    threads,
  );
  try {
    const controller = instantiateController(id, engine, settings, scene);
    return {
      id,
      engine,
      controller,
      dispose: () => {
        controller.dispose?.();
        engine.dispose();
      },
    };
  } catch (error) {
    engine.dispose();
    throw error;
  }
}
export class SeededRandom {
  constructor(seed) {
    this.state = seed >>> 0 || 0x9e3779b9;
  }
  uniform() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 4294967296;
  }
  normal() {
    return (
      Math.sqrt(-2 * Math.log(Math.max(1e-9, this.uniform()))) *
      Math.cos(2 * Math.PI * this.uniform())
    );
  }
}
export function emptyTree(engine, root) {
  return {
    dim: engine.dim,
    poseDim: engine.controlled * 2,
    meta: new Uint32Array(),
    values: new Float32Array(),
    root: root.slice(),
  };
}
