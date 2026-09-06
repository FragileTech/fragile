// Scene archetypes are portable data, shared by native physics and rendering.
// Physics overlays are shallow; shape arrays replace inherited arrays in full.
export function resolveAgentTypes(definitions = {}) {
  if (
    !definitions ||
    Array.isArray(definitions) ||
    typeof definitions !== "object"
  )
    throw new Error("agent_types must be an object");
  const resolved = new Map(),
    visiting = new Set();
  if (Object.keys(definitions).length > 256)
    throw new Error("At most 256 agent types are supported");
  const resolve = (name) => {
    if (resolved.has(name)) return resolved.get(name);
    if (!Object.hasOwn(definitions, name))
      throw new Error(`Unknown agent type: ${name}`);
    if (visiting.has(name))
      throw new Error(`Cyclic agent type inheritance: ${name}`);
    const def = definitions[name];
    if (!name || !def || Array.isArray(def) || typeof def !== "object")
      throw new Error("Agent types need a name and an object definition");
    visiting.add(name);
    const parent = def.extends == null ? {} : resolve(def.extends);
    for (const field of ["physics", "visual"])
      if (
        def[field] != null &&
        (typeof def[field] !== "object" || Array.isArray(def[field]))
      )
        throw new Error(`Agent ${field} defaults must be an object`);
    if (def.physics && Object.hasOwn(def.physics, "agent_type"))
      throw new Error("Use extends to inherit agent types");
    const type = {
      ...parent,
      ...def,
      physics: { ...parent.physics, ...def.physics },
      visual: { ...parent.visual, ...def.visual },
    };
    visiting.delete(name);
    resolved.set(name, type);
    return type;
  };
  Object.keys(definitions).forEach(resolve);
  return resolved;
}
export function resolveBodies(scene) {
  const types = resolveAgentTypes(scene.agent_types);
  return scene.bodies.map((body) => {
    if (body.agent_type == null) return { ...body };
    const type = types.get(body.agent_type);
    if (!type) throw new Error(`Unknown agent type: ${body.agent_type}`);
    return {
      ...type.physics,
      ...body,
      visual: { ...type.visual, ...body.visual },
    };
  });
}
