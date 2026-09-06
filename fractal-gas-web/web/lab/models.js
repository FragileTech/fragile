// Original geometry assets, authored for this laboratory. All dimensions are
// visual only; the C++ scene remains the sole source of collision geometry.
import * as T from "./vendor/three.module.js";
import { palette, metal, glow, prism, box } from "./visuals/primitives.js";
export { palette, prism } from "./visuals/primitives.js";
export { shipModel, kartModel, droneModel } from "./visuals/vehicles.js";
export { harvesterModel } from "./visuals/harvester.js";
export function rockModel(vertices, color = palette.violet) {
  const g = new T.Group();
  g.name = "Veined ore";
  g.add(prism(vertices, 0.75, metal(0x555369), 0.08));
  const inner = vertices.map((p) => p.map((v) => v * 0.58));
  const ore = prism(inner, 0.22, glow(color), 0.025);
  ore.position.z = 0.78;
  g.add(ore);
  const edges = new T.LineSegments(
    new T.EdgesGeometry(g.children[0].geometry),
    new T.LineBasicMaterial({
      color: 0xb5a5cb,
      transparent: true,
      opacity: 0.35,
    }),
  );
  g.add(edges);
  return g;
}
export function zoneModel(radius, color, kind = "gate") {
  const g = new T.Group();
  g.name = kind === "base" ? "Orbital recovery dock" : "Flux checkpoint";
  for (const [r, tube] of [
    [radius, 0.07],
    [radius * 0.86, 0.025],
  ]) {
    const ring = new T.Mesh(new T.TorusGeometry(r, tube, 5, 72), glow(color));
    ring.position.z = 0.12;
    g.add(ring);
  }
  const disk = new T.Mesh(
    new T.CircleGeometry(radius, 64),
    new T.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.055,
      depthWrite: false,
    }),
  );
  disk.position.z = 0.05;
  g.add(disk);
  for (let i = 0; i < 8; i++) {
    const a = (i * Math.PI) / 4,
      mesh = box(
        [0.5, 0.12, kind === "base" ? 0.65 : 0.18],
        [Math.cos(a) * radius, Math.sin(a) * radius, 0.2],
        metal(0xb9a178),
      );
    mesh.rotation.z = a;
    g.add(mesh);
  }
  return g;
}
export function reactorModel() {
  const g = new T.Group();
  g.name = "Gravitational flux well";
  const core = new T.Mesh(new T.OctahedronGeometry(0.9, 0), glow(palette.rose));
  core.position.z = 1.5;
  g.add(core);
  for (let i = 0; i < 3; i++) {
    const ring = new T.Mesh(
      new T.TorusGeometry(1.3 + 0.22 * i, 0.035, 5, 64),
      glow(palette.rose),
    );
    ring.rotation.x = i * 0.7;
    ring.position.z = 1;
    g.add(ring);
  }
  return g;
}
