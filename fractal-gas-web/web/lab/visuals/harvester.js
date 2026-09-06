// Steampunk resource truck. Geometry is presentation only; the archetype owns
// its physical footprint and uses the existing three-channel ground actuator.
import * as T from "../vendor/three.module.js";
import { box, metal, prism } from "./primitives.js";

function cylinder(radius, length, material, position, axis = "z") {
  const mesh = new T.Mesh(
    new T.CylinderGeometry(radius, radius, length, 12),
    material,
  );
  if (axis === "x") mesh.rotation.z = Math.PI / 2;
  else if (axis === "z") mesh.rotation.x = Math.PI / 2;
  mesh.position.set(...position);
  return mesh;
}

function pipe(points, material, radius = 0.022) {
  const curve = new T.CatmullRomCurve3(points.map((p) => new T.Vector3(...p)));
  return new T.Mesh(new T.TubeGeometry(curve, 12, radius, 6, false), material);
}

export function harvesterModel(color = 0xb88a43) {
  const truck = new T.Group();
  truck.name = "Harvester / steam ore collector";
  const iron = metal(0x343434),
    brass = metal(color),
    copper = metal(0xa65d37),
    enamel = metal(0x542b28),
    rubber = new T.MeshStandardMaterial({ color: 0x22201e, roughness: 0.95 }),
    glass = new T.MeshStandardMaterial({
      color: 0xeab965,
      emissive: 0xb97223,
      emissiveIntensity: 0.3,
      metalness: 0.3,
      roughness: 0.25,
    });

  truck.add(box([1.7, 0.88, 0.17], [-0.04, 0, 0.27], iron));
  truck.add(box([1.4, 0.94, 0.06], [-0.13, 0, 0.38], brass));

  // Six road wheels, front steering pivots, tread blocks and brass hubs.
  for (const side of [-1, 1]) {
    for (const x of [-0.65, -0.06, 0.53]) {
      const pivot = new T.Group();
      pivot.position.set(x, side * 0.52, 0.225);
      if (x > 0) pivot.userData.motion = "steer";
      const wheel = new T.Group();
      wheel.userData.motion = "wheel";
      wheel.add(cylinder(0.215, 0.2, rubber, [0, 0, 0], "y"));
      wheel.add(cylinder(0.12, 0.215, brass, [0, 0, 0], "y"));
      wheel.add(cylinder(0.055, 0.23, iron, [0, 0, 0], "y"));
      for (let i = 0; i < 10; i++) {
        const angle = (i * Math.PI * 2) / 10;
        const tread = box(
          [0.075, 0.205, 0.032],
          [Math.sin(angle) * 0.215, 0, Math.cos(angle) * 0.215],
          rubber,
        );
        tread.rotation.y = angle;
        wheel.add(tread);
      }
      pivot.add(wheel);
      truck.add(pivot);
      truck.add(box([0.4, 0.26, 0.055], [x, side * 0.49, 0.49], enamel));
    }
    truck.add(
      pipe(
        [
          [0.38, side * 0.42, 0.49],
          [0.15, side * 0.46, 0.58],
          [-0.55, side * 0.46, 0.58],
          [-0.75, side * 0.4, 0.67],
        ],
        copper,
      ),
    );
    truck.add(box([0.08, 0.08, 0.12], [0.8, side * 0.38, 0.48], glass));
    truck.add(box([0.15, 0.14, 0.045], [-0.84, side * 0.4, 0.29], brass));
  }

  // Open hopper with sloped walls and raised rim; no decorative ore implies
  // inventory that the current collection task does not actually simulate.
  truck.add(box([0.8, 0.54, 0.05], [-0.28, 0, 0.46], iron));
  for (const side of [-1, 1]) {
    const wall = box([0.92, 0.045, 0.31], [-0.28, side * 0.34, 0.64], enamel);
    wall.rotation.x = -side * 0.24;
    truck.add(wall);
    truck.add(box([1.0, 0.065, 0.045], [-0.28, side * 0.39, 0.8], brass));
    for (const x of [-0.63, -0.28, 0.07]) {
      truck.add(box([0.035, 0.045, 0.28], [x, side * 0.38, 0.63], brass));
      truck.add(cylinder(0.026, 0.06, iron, [x, side * 0.405, 0.73], "y"));
    }
  }
  for (const x of [-0.73, 0.17]) {
    truck.add(box([0.045, 0.72, 0.31], [x, 0, 0.64], enamel));
    truck.add(box([0.065, 0.82, 0.045], [x, 0, 0.8], brass));
  }

  // Cab and amber windows face +X. A brass pressure dial replaces neon trim.
  truck.add(box([0.41, 0.57, 0.36], [0.46, 0, 0.61], iron));
  truck.add(box([0.025, 0.44, 0.19], [0.674, 0, 0.67], glass));
  for (const side of [-1, 1])
    truck.add(box([0.24, 0.025, 0.18], [0.48, side * 0.295, 0.67], glass));
  truck.add(box([0.48, 0.65, 0.055], [0.46, 0, 0.82], brass));
  truck.add(cylinder(0.08, 0.07, brass, [0.47, 0, 0.87]));
  truck.add(cylinder(0.06, 0.075, glass, [0.47, 0, 0.875]));

  // Rear copper pressure vessel, steel bands, chimney and relief valve.
  truck.add(cylinder(0.2, 0.56, copper, [-0.78, 0, 0.63], "y"));
  for (const y of [-0.19, 0.19])
    truck.add(cylinder(0.211, 0.035, brass, [-0.78, y, 0.63], "y"));
  truck.add(cylinder(0.055, 0.42, iron, [-0.76, -0.22, 0.96]));
  truck.add(cylinder(0.08, 0.045, brass, [-0.76, -0.22, 1.18]));
  truck.add(cylinder(0.025, 0.16, brass, [-0.78, 0.19, 0.88]));
  truck.add(cylinder(0.055, 0.025, copper, [-0.78, 0.19, 0.97]));
  truck.add(
    pipe(
      [
        [-0.73, 0.28, 0.64],
        [-0.87, 0.36, 0.53],
        [-0.4, 0.39, 0.43],
        [0.38, 0.39, 0.48],
      ],
      copper,
      0.03,
    ),
  );

  // Broad toothed collecting drum, animated through the shared wheel contract.
  const intake = new T.Group();
  intake.name = "Mineral intake";
  intake.position.set(0.94, 0, 0.23);
  intake.userData.motion = "wheel";
  intake.add(cylinder(0.15, 0.77, iron, [0, 0, 0], "y"));
  for (let i = 0; i < 6; i++) {
    const angle = (i * Math.PI * 2) / 6;
    const tooth = box(
      [0.065, 0.74, 0.075],
      [Math.sin(angle) * 0.15, 0, Math.cos(angle) * 0.15],
      brass,
    );
    tooth.rotation.y = angle;
    intake.add(tooth);
  }
  truck.add(intake);
  for (const side of [-1, 1]) {
    const guard = prism(
      [
        [0.63, side * 0.32],
        [1.07, side * 0.42],
        [1.03, side * 0.49],
        [0.55, side * 0.44],
      ],
      0.12,
      iron,
      0.01,
    );
    guard.position.z = 0.27;
    truck.add(guard);
  }
  // Stay within the shared nominal 0.8 m footprint used by BodyLayer.
  truck.scale.setScalar(0.68);
  const model = new T.Group();
  model.name = truck.name;
  model.add(truck);
  return model;
}
