// Cached GLB resources belong to the asset store, not any one viewport.
const shared = new WeakSet();
export const retainResource = (resource) => shared.add(resource);
const textureKeys = [
  "map",
  "normalMap",
  "roughnessMap",
  "metalnessMap",
  "emissiveMap",
  "aoMap",
  "alphaMap",
];

export function retainAsset(root) {
  root.traverse((object) => {
    if (object.geometry) shared.add(object.geometry);
    for (const material of materials(object)) {
      shared.add(material);
      for (const key of textureKeys)
        if (material[key]) shared.add(material[key]);
    }
  });
}

function materials(object) {
  return !object.material
    ? []
    : Array.isArray(object.material)
      ? object.material
      : [object.material];
}

export function disposeGroup(group, releaseShared = false) {
  const disposed = new Set();
  const release = (resource) => {
    if (
      !resource ||
      disposed.has(resource) ||
      (!releaseShared && shared.has(resource))
    )
      return;
    disposed.add(resource);
    resource.dispose();
  };
  group.traverse((object) => {
    release(object.geometry);
    // InstancedMesh owns GPU instance buffers independently of its geometry.
    if (object.isInstancedMesh) release(object);
    for (const material of materials(object)) {
      for (const key of textureKeys) release(material[key]);
      release(material);
    }
  });
  group.clear();
}
