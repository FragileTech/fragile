import * as T from "../vendor/three.module.js";
export const palette = {
  cyan: 0x6ffff1,
  rose: 0xff538b,
  gold: 0xffce70,
  violet: 0x9f82ff,
  green: 0x7bffc1,
};
export const metal = (color) =>
  new T.MeshStandardMaterial({
    color,
    roughness: 0.48,
    metalness: 0.65,
    flatShading: true,
  });
export const glow = (color) =>
  new T.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 1.1,
    roughness: 0.3,
  });
export function prism(points, depth, material, bevel = 0.04) {
  const shape = new T.Shape(points.map((p) => new T.Vector2(...p)));
  return new T.Mesh(
    new T.ExtrudeGeometry(shape, {
      depth,
      bevelEnabled: bevel > 0,
      bevelSegments: 1,
      steps: 1,
      bevelSize: bevel,
      bevelThickness: bevel,
    }),
    material,
  );
}
export function box(size, position, material) {
  const mesh = new T.Mesh(new T.BoxGeometry(...size), material);
  mesh.position.set(...position);
  return mesh;
}
