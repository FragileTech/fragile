import * as T from "../../vendor/three.module.js";
import { palette } from "../primitives.js";
import { registerEnvironment } from "./registry.js";
function line(points, color) {
  return new T.Line(
    new T.BufferGeometry().setFromPoints(
      points.map((p) => new T.Vector3(...p)),
    ),
    new T.LineBasicMaterial({ color }),
  );
}
export function arenaEnvironment(scene) {
  const group = new T.Group();
  group.name = "Laboratory arena";
  const [sx, sy] = scene.size || [64, 44];
  const boundary = scene.boundary || [
    [0, 0],
    [sx, 0],
    [sx, sy],
    [0, sy],
  ];
  const floorShape = new T.Shape(boundary.map((p) => new T.Vector2(...p)));
  for (const hole of scene.holes || [])
    floorShape.holes.push(new T.Path(hole.map((p) => new T.Vector2(...p))));
  const floor = new T.Mesh(
    new T.ShapeGeometry(floorShape),
    new T.MeshStandardMaterial({
      color: 0x15212e,
      metalness: 0.5,
      roughness: 0.75,
    }),
  );
  floor.name = "Playable arena floor";
  floor.position.z = -0.05;
  group.add(floor);
  const grid = new T.GridHelper(
    Math.max(sx, sy),
    Math.round(Math.max(sx, sy) / 2),
    0x284253,
    0x1a2e3c,
  );
  grid.rotation.x = Math.PI / 2;
  grid.position.set(sx / 2, sy / 2, -0.01);
  group.add(grid);
  for (const ring of [boundary, ...(scene.holes || [])]) {
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i],
        b = ring[(i + 1) % ring.length],
        length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const wall = new T.Mesh(
        new T.BoxGeometry(length, 0.65, 1.4),
        new T.MeshStandardMaterial({
          color: 0x324151,
          metalness: 0.6,
          roughness: 0.55,
        }),
      );
      wall.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.6);
      wall.rotation.z = Math.atan2(b[1] - a[1], b[0] - a[0]);
      group.add(wall);
      group.add(
        line(
          [
            [...a, 1.35],
            [...b, 1.35],
          ],
          0x75a9bb,
        ),
      );
      for (let j = 1; j < length; j += 2.8) {
        const stripe = new T.Mesh(
          new T.BoxGeometry(0.42, 0.71, 0.025),
          new T.MeshBasicMaterial({ color: palette.gold }),
        );
        stripe.position.set(
          a[0] + ((b[0] - a[0]) * j) / length,
          a[1] + ((b[1] - a[1]) * j) / length,
          1.32,
        );
        stripe.rotation.z = wall.rotation.z;
        group.add(stripe);
      }
    }
  }
  return { group };
}
registerEnvironment("arena", arenaEnvironment);
