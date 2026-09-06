import * as T from "../../vendor/three.module.js";
import { WorldInstances, worldModel, worldSurface } from "../world.js";
import { disposeGroup } from "../resources.js";
import { worldCatalog } from "../world-catalog.js";

// Fit the kit to existing boundary segments. No collider, native scene field,
// track curve, hole or checkpoint position is generated or moved here.
export function dressWorld(environment, scene, style) {
  if (!worldModel(style, "rail")) return environment;
  const circuit = scene.environment?.kind === "circuit",
    group = environment.group;
  for (const object of [...group.children]) {
    if (
      object.geometry?.type === "BoxGeometry" ||
      (!circuit && object.isLine)
    ) {
      group.remove(object);
      const garbage = new T.Group();
      garbage.add(object);
      disposeGroup(garbage);
    }
    if (
      object.geometry?.type === "ShapeGeometry" &&
      (!circuit || object.name === "Asphalt racing surface")
    ) {
      const material = worldSurface(
        style,
        circuit ? "track-straight" : "floor-tile",
      );
      if (material) {
        object.material.map?.dispose();
        object.material.dispose();
        object.material = material;
      }
    }
  }
  const batch = new WorldInstances(style, group);
  const [sx, sy] = scene.size || [64, 44];
  const rings = [
    scene.boundary || [
      [0, 0],
      [sx, 0],
      [sx, sy],
      [0, sy],
    ],
    ...(scene.holes || []),
  ];
  for (const [index, ring] of rings.entries()) {
    const area = ring.reduce((sum, p, i) => {
      const q = ring[(i + 1) % ring.length];
      return sum + p[0] * q[1] - q[0] * p[1];
    }, 0);
    const inward = (index ? -1 : 1) * Math.sign(area || 1);
    let perimeter = 0,
      nextFixture = 4,
      fixture = 0;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i],
        b = ring[(i + 1) % ring.length],
        dx = b[0] - a[0],
        dy = b[1] - a[1],
        length = Math.hypot(dx, dy);
      if (length < 0.001) continue;
      const angle = Math.atan2(dy, dx),
        nx = (-dy / length) * inward,
        ny = (dx / length) * inward;
      const count = Math.ceil(length / 2.8);
      for (let j = 0; j < count; j++) {
        const t = (j + 0.5) / count,
          x = a[0] + dx * t,
          y = a[1] + dy * t;
        batch.add(
          circuit ? "guardrail" : "rail",
          [x, y, 0],
          [length / count + 0.015, circuit ? 0.28 : 0.65, circuit ? 0.5 : 0.85],
          angle,
        );
        if (circuit)
          batch.add(
            "kerb",
            [x + nx * 0.4, y + ny * 0.4, 0.02],
            [length / count, 0.6, 0.07],
            angle,
          );
      }
      // Put utility scenery beyond the physical playable boundary, preserving clearance.
      while (nextFixture < perimeter + length) {
        const t = (nextFixture - perimeter) / length,
          x = a[0] + dx * t - nx * 0.9,
          y = a[1] + dy * t - ny * 0.9;
        const kind = circuit
          ? ["lamp", "direction-board", "bollard"][fixture % 3]
          : ["lamp", "utility-column", "bollard", "deposit"][fixture % 4];
        const size = kind === "deposit" ? [1.1, 1.1, 0.85] : undefined;
        batch.add(kind, [x, y, 0], size, angle - Math.PI / 2);
        nextFixture += circuit ? 16 : 12;
        fixture++;
      }
      perimeter += length;
      if (!circuit)
        batch.add(
          index ? "corner-inner" : "corner-outer",
          [...a, 0],
          [0.65, 0.65, 0.85],
          angle,
        );
    }
  }
  if (circuit && scene.environment.start) {
    const { position: p, angle = 0 } = scene.environment.start,
      width = scene.environment.width;
    batch.add("finish-arch", [...p, 0], [0.6, width + 1, 3.2], angle);
    const offset = width / 2 + 1.2;
    const side = [
      p[0] - Math.sin(angle) * offset,
      p[1] + Math.cos(angle) * offset,
    ];
    batch.add("start-light", [...side, 0], undefined, angle);
    batch.add(
      "pit-station",
      [side[0] + Math.cos(angle) * 3, side[1] + Math.sin(angle) * 3, 0],
      undefined,
      angle,
    );
    for (const gate of scene.gates || []) {
      // Pylons flank each native checkpoint; native progress rings remain visible.
      const points = scene.environment.centerline;
      let nearest = 0,
        distance = Infinity;
      points.forEach((q, i) => {
        const d = Math.hypot(q[0] - gate.position[0], q[1] - gate.position[1]);
        if (d < distance) {
          distance = d;
          nearest = i;
        }
      });
      const a = points[nearest],
        b = points[(nearest + 1) % points.length],
        heading = Math.atan2(b[1] - a[1], b[0] - a[0]);
      batch.add(
        "gate-pylons",
        [...gate.position, 0],
        [0.45, width - 0.8, 1.2],
        heading,
      );
    }
  }
  // Portable optional placements make the entire kit available to custom scenes.
  for (const item of scene.environment?.assets || []) {
    if (
      !worldCatalog[item.model] ||
      !Array.isArray(item.position) ||
      item.position.length !== 2 ||
      !item.position.every(Number.isFinite)
    )
      throw new Error("Invalid world asset placement");
    const size = item.size || worldCatalog[item.model].size;
    if (
      !Array.isArray(size) ||
      size.length !== 3 ||
      !size.every((v) => Number.isFinite(v) && v > 0 && v <= 200) ||
      !Number.isFinite(item.angle ?? 0)
    )
      throw new Error("Invalid world asset dimensions");
    batch.add(item.model, [...item.position, 0], size, item.angle || 0);
  }
  batch.build();
  environment.updateLod = (camera, height) => batch.updateLod(camera, height);
  group.userData.worldStyle = style;
  return environment;
}
