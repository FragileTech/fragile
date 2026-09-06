// Original model kits. +X is forward, +Z is up; nominal footprint radius 0.8 m.
// Motion tags are interpreted by the model registry, independent of agent type.
import * as T from "../vendor/three.module.js";
import { palette, metal, glow, prism, box } from "./primitives.js";

function cylinder(
  radius,
  length,
  material,
  position,
  axis = "x",
  end = radius,
) {
  const mesh = new T.Mesh(
    new T.CylinderGeometry(end, radius, length, 16),
    material,
  );
  mesh.rotation[axis === "x" ? "z" : "x"] = Math.PI / 2;
  mesh.position.set(...position);
  return mesh;
}
function plate(points, z, depth, material) {
  const mesh = prism(points, depth, material, 0.025);
  mesh.position.z = z;
  return mesh;
}
function canopy() {
  return new T.MeshPhysicalMaterial({
    color: 0x164e68,
    metalness: 0.65,
    roughness: 0.12,
    clearcoat: 1,
    emissive: 0x126979,
    emissiveIntensity: 0.45,
  });
}
function plume(color, position, radius = 0.12, length = 0.8) {
  const pivot = new T.Group();
  pivot.name = "thruster";
  pivot.position.set(...position);
  pivot.userData.motion = "thrust";
  const material = new T.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.65,
    depthWrite: false,
    blending: T.AdditiveBlending,
  });
  const flame = cylinder(
    0.005,
    length,
    material,
    [-length / 2, 0, 0],
    "x",
    radius,
  );
  pivot.add(
    flame,
    cylinder(
      0.005,
      length * 0.55,
      glow(0xcfffff),
      [-length * 0.275, 0, 0],
      "x",
      radius * 0.45,
    ),
  );
  return pivot;
}
export function shipModel(color = palette.cyan) {
  const g = new T.Group();
  g.name = "Kestrel / vector rocket";
  const armor = metal(0xd4dfdf),
    dark = metal(0x203144),
    alloy = metal(0x667c91),
    light = glow(color);
  g.add(
    plate(
      [
        [0.86, 0],
        [0.35, 0.23],
        [-0.64, 0.23],
        [-0.72, -0.23],
        [0.35, -0.23],
      ],
      0.14,
      0.24,
      armor,
    ),
  );
  // Faceted nose, raised cockpit, recessed service spine.
  g.add(
    plate(
      [
        [0.64, 0],
        [0.22, 0.18],
        [-0.18, 0.16],
        [-0.24, -0.16],
        [0.22, -0.18],
      ],
      0.39,
      0.13,
      canopy(),
    ),
  );
  g.add(box([0.4, 0.18, 0.1], [-0.46, 0, 0.45], dark));
  for (const side of [-1, 1]) {
    g.add(
      plate(
        [
          [-0.12, side * 0.19],
          [-0.46, side * 0.62],
          [-0.74, side * 0.59],
          [-0.6, side * 0.18],
        ],
        0.13,
        0.09,
        dark,
      ),
    );
    g.add(
      plate(
        [
          [-0.37, side * 0.41],
          [-0.55, side * 0.57],
          [-0.69, side * 0.55],
          [-0.62, side * 0.39],
        ],
        0.24,
        0.015,
        light,
      ),
    );
    g.add(cylinder(0.14, 0.66, alloy, [-0.39, side * 0.35, 0.27]));
    g.add(cylinder(0.18, 0.19, dark, [-0.79, side * 0.35, 0.27], "x", 0.11));
    g.add(cylinder(0.125, 0.025, light, [-0.895, side * 0.35, 0.27]));
    g.add(plume(color, [-0.92, side * 0.35, 0.27]));
    g.add(box([0.35, 0.024, 0.024], [0.19, side * 0.22, 0.405], light));
  }
  for (let i = 0; i < 4; i++)
    g.add(box([0.032, 0.15, 0.02], [-0.32 - i * 0.085, 0, 0.515], alloy));
  // Upright dorsal stabilizer and rescue beacon.
  g.add(box([0.33, 0.045, 0.3], [-0.52, 0, 0.61], armor));
  g.add(box([0.25, 0.05, 0.025], [-0.51, 0, 0.77], light));
  return g;
}
export function kartModel(color = palette.cyan) {
  const g = new T.Group();
  g.name = "Mite / electric kart";
  const dark = metal(0x1d2b3b),
    armor = metal(0x97b3bd),
    trim = metal(0x577286),
    light = glow(color);
  const rubber = new T.MeshStandardMaterial({
    color: 0x10151e,
    roughness: 0.95,
  });
  g.add(
    plate(
      [
        [0.64, 0.22],
        [0.42, 0.35],
        [-0.63, 0.34],
        [-0.69, -0.34],
        [0.42, -0.35],
        [0.64, -0.22],
      ],
      0.19,
      0.12,
      dark,
    ),
  );
  g.add(
    plate(
      [
        [0.63, 0.2],
        [0.2, 0.26],
        [0.12, -0.26],
        [0.63, -0.2],
      ],
      0.32,
      0.12,
      armor,
    ),
  );
  g.add(box([0.17, 0.49, 0.08], [0.44, 0, 0.47], light));
  // Open cockpit with roll cage and glossy helmet, separate from the rocket canopy.
  g.add(box([0.34, 0.31, 0.09], [-0.15, 0, 0.35], rubber));
  g.add(box([0.08, 0.32, 0.27], [-0.34, 0, 0.48], dark));
  const helmet = new T.Mesh(new T.SphereGeometry(0.14, 16, 10), canopy());
  helmet.position.set(-0.12, 0, 0.57);
  g.add(helmet);
  for (const side of [-1, 1]) {
    g.add(box([0.55, 0.045, 0.035], [-0.04, side * 0.26, 0.55], trim));
    g.add(box([0.045, 0.045, 0.24], [-0.31, side * 0.26, 0.45], trim));
    g.add(box([0.16, 0.12, 0.04], [0.67, side * 0.18, 0.31], glow(0xd3faff)));
    g.add(
      box([0.1, 0.12, 0.035], [-0.69, side * 0.2, 0.3], glow(palette.rose)),
    );
    g.add(box([0.4, 0.04, 0.035], [-0.17, side * 0.36, 0.23], light));
    for (const x of [-0.43, 0.42]) {
      const steering = new T.Group();
      steering.position.set(x, side * 0.43, 0.22);
      if (x > 0) steering.userData.motion = "steer";
      const wheel = new T.Group();
      wheel.userData.motion = "wheel";
      wheel.add(cylinder(0.215, 0.18, rubber, [0, 0, 0], "y"));
      wheel.add(cylinder(0.115, 0.185, trim, [0, 0, 0], "y"));
      for (let i = 0; i < 5; i++) {
        const spoke = box([0.027, 0.19, 0.19], [0, 0, 0], light);
        spoke.rotation.y = (i * Math.PI) / 5;
        wheel.add(spoke);
      }
      steering.add(wheel);
      g.add(steering);
    }
    g.add(box([0.045, 0.04, 0.28], [-0.58, side * 0.23, 0.5], trim));
  }
  g.add(box([0.2, 0.88, 0.065], [-0.6, 0, 0.67], dark));
  g.add(box([0.04, 0.85, 0.02], [-0.69, 0, 0.714], light));
  return g;
}
export function droneModel(color = palette.gold) {
  const g = new T.Group();
  g.name = "Wisp / survey drone";
  const armor = metal(0x39465c),
    light = glow(color);
  g.add(
    plate(
      [
        [0.48, 0],
        [0, 0.32],
        [-0.45, 0],
        [0, -0.32],
      ],
      0.3,
      0.17,
      armor,
    ),
  );
  const eye = new T.Mesh(new T.SphereGeometry(0.16, 12, 8), canopy());
  eye.position.set(0.24, 0, 0.49);
  g.add(eye);
  for (const x of [-0.34, 0.34])
    for (const y of [-0.43, 0.43]) {
      const ring = new T.Mesh(new T.TorusGeometry(0.23, 0.055, 6, 24), armor);
      ring.position.set(x, y, 0.28);
      g.add(ring);
      const rotor = new T.Group();
      rotor.position.copy(ring.position);
      rotor.userData.motion = "rotor";
      rotor.add(
        box([0.36, 0.04, 0.025], [0, 0, 0], light),
        box([0.04, 0.36, 0.025], [0, 0, 0], light),
      );
      g.add(rotor);
    }
  g.add(plume(color, [-0.46, 0, 0.38], 0.08, 0.4));
  return g;
}
