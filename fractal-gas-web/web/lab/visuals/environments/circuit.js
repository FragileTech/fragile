import * as T from "../../vendor/three.module.js";
import { registerEnvironment } from "./registry.js";

function surface(boundary, holes, color, z) {
  const shape = new T.Shape(boundary.map((p) => new T.Vector2(...p)));
  for (const hole of holes)
    shape.holes.push(new T.Path(hole.map((p) => new T.Vector2(...p))));
  const mesh = new T.Mesh(
    new T.ShapeGeometry(shape),
    new T.MeshStandardMaterial({ color, roughness: 0.96, metalness: 0.05 }),
  );
  mesh.position.z = z;
  return mesh;
}
function segments(group, parts, color, emissive = false) {
  if (!parts.length) return;
  const material = new T.MeshStandardMaterial({
    color,
    roughness: 0.7,
    emissive: emissive ? color : 0,
    emissiveIntensity: emissive ? 0.5 : 0,
  });
  const mesh = new T.InstancedMesh(
    new T.BoxGeometry(1, 1, 1),
    material,
    parts.length,
  );
  const object = new T.Object3D();
  parts.forEach((p, i) => {
    object.position.set(...p.position);
    object.scale.set(...p.size);
    object.rotation.z = p.angle || 0;
    object.updateMatrix();
    mesh.setMatrixAt(i, object.matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  group.add(mesh);
}
function asphaltTexture() {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const context = canvas.getContext("2d"),
    pixels = context.createImageData(128, 128);
  let rng = 73;
  for (let i = 0; i < pixels.data.length; i += 4) {
    rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
    const value = 175 + (rng >>> 27);
    pixels.data.set([value, value, value, 255], i);
  }
  context.putImageData(pixels, 0, 0);
  const texture = new T.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = T.RepeatWrapping;
  texture.repeat.set(0.7, 0.7);
  texture.colorSpace = T.SRGBColorSpace;
  return texture;
}
function checkedPoints(points) {
  return (
    Array.isArray(points) &&
    points.length >= 3 &&
    points.length <= 4096 &&
    points.every(
      (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
    )
  );
}

// Physical limits always come from scene.boundary / holes. Environment metadata
// only adds track dressing, never modifies the native simulation or action space.
export function circuitEnvironment(scene) {
  const spec = scene.environment,
    group = new T.Group();
  group.name = "Violet Circuit";
  if (
    !checkedPoints(scene.boundary) ||
    !(scene.holes || []).every(checkedPoints) ||
    !checkedPoints(spec.centerline) ||
    !Number.isFinite(spec.width) ||
    spec.width < 1 ||
    spec.width > 100
  )
    throw new Error("Invalid circuit rendering geometry");
  const road = surface(scene.boundary, scene.holes || [], 0x34303d, 0);
  road.name = "Asphalt racing surface";
  road.material.map = asphaltTexture();
  group.add(road);
  for (const hole of scene.holes || []) {
    const infield = surface(hole, [], 0x211b2b, -0.03);
    infield.name = "Circuit infield";
    group.add(infield);
  }
  const purple = [],
    white = [],
    walls = [],
    paint = [];
  for (const [index, ring] of [
    scene.boundary,
    ...(scene.holes || []),
  ].entries()) {
    const area = ring.reduce((s, a, i) => {
      const b = ring[(i + 1) % ring.length];
      return s + a[0] * b[1] - b[0] * a[1];
    }, 0);
    const side = (index ? -1 : 1) * Math.sign(area);
    let stripe = 0;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i],
        b = ring[(i + 1) % ring.length],
        dx = b[0] - a[0],
        dy = b[1] - a[1];
      const length = Math.hypot(dx, dy),
        angle = Math.atan2(dy, dx);
      const nx = (-dy / length) * side,
        ny = (dx / length) * side;
      walls.push({
        position: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.25],
        size: [length + 0.05, 0.28, 0.5],
        angle,
      });
      paint.push({
        position: [
          (a[0] + b[0]) / 2 + nx * 0.9,
          (a[1] + b[1]) / 2 + ny * 0.9,
          0.015,
        ],
        size: [length, 0.07, 0.02],
        angle,
      });
      const count = Math.ceil(length / 1.2);
      for (let j = 0; j < count; j++) {
        const t = (j + 0.5) / count;
        (stripe++ % 2 ? purple : white).push({
          position: [a[0] + dx * t + nx * 0.4, a[1] + dy * t + ny * 0.4, 0.035],
          size: [length / count + 0.02, 0.6, 0.07],
          angle,
        });
      }
    }
  }
  segments(group, walls, 0x756784);
  segments(group, white, 0xece5ef);
  segments(group, purple, 0x9059a4);
  segments(group, paint, 0xb9a9c5);

  // Dashed centre guide is decorative; checkpoint rewards use the native zones.
  const guide = [];
  for (let i = 0; i < spec.centerline.length; i++) {
    const a = spec.centerline[i],
      b = spec.centerline[(i + 1) % spec.centerline.length];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let distance = 0.8; distance < length; distance += 2.5) {
      const t = distance / length;
      guide.push({
        position: [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.02],
        size: [0.7, 0.065, 0.02],
        angle: Math.atan2(b[1] - a[1], b[0] - a[0]),
      });
    }
  }
  segments(group, guide, 0x6d617a);
  if (spec.start) {
    const { position: p, angle = 0 } = spec.start;
    if (
      !Array.isArray(p) ||
      p.length !== 2 ||
      !p.every(Number.isFinite) ||
      !Number.isFinite(angle)
    )
      throw new Error("Invalid circuit start line");
    const light = [],
      dark = [],
      posts = [];
    const across = spec.width - 1.8,
      count = Math.max(2, Math.round(across / 0.65));
    const point = (x, y, z) => [
      p[0] + x * Math.cos(angle) - y * Math.sin(angle),
      p[1] + x * Math.sin(angle) + y * Math.cos(angle),
      z,
    ];
    for (let x = 0; x < 2; x++)
      for (let y = 0; y < count; y++)
        ((x + y) % 2 ? light : dark).push({
          position: point(
            (x - 0.5) * 0.65,
            ((y + 0.5) / count - 0.5) * across,
            0.025,
          ),
          size: [0.65, across / count, 0.025],
          angle,
        });
    for (const sign of [-1, 1])
      posts.push({
        position: point(0, sign * (spec.width / 2 + 0.3), 1.5),
        size: [0.28, 0.35, 3],
        angle,
      });
    posts.push({
      position: point(0, 0, 3),
      size: [0.35, spec.width + 1, 0.22],
      angle,
    });
    segments(group, light, 0xf7f0fb);
    segments(group, dark, 0x13101a);
    segments(group, posts, 0xc393d6, true);
  }
  if (spec.sponsor && typeof document !== "undefined") {
    const { position, size = 11 } = spec.sponsor;
    if (
      !Array.isArray(position) ||
      position.length !== 2 ||
      !position.every(Number.isFinite) ||
      !Number.isFinite(size) ||
      size <= 0 ||
      size > 100
    )
      throw new Error("Invalid circuit sponsor placement");
    const texture = new T.TextureLoader().load(
      new URL("../../branding/logo.png", import.meta.url).href,
    );
    texture.colorSpace = T.SRGBColorSpace;
    const sponsor = new T.Mesh(
      new T.PlaneGeometry(size, size),
      new T.MeshBasicMaterial({ map: texture }),
    );
    sponsor.name = "Fragile documentation logo";
    sponsor.position.set(...position, 0.01);
    group.add(sponsor);
  }
  const markers = (scene.gates || []).map((gate, i) => {
    const circle = new T.Mesh(
      new T.RingGeometry(gate.radius - 0.05, gate.radius, 48),
      new T.MeshBasicMaterial({
        color: 0xc6a0d6,
        transparent: true,
        opacity: 0.2,
        depthWrite: false,
      }),
    );
    circle.position.set(...gate.position, 0.04);
    circle.name = `Checkpoint ${i + 1}`;
    group.add(circle);
    return circle;
  });
  return {
    group,
    replacesGates: true,
    update(state, info) {
      const bits = new Uint32Array(
        state.buffer,
        state.byteOffset,
        state.length,
      );
      const next = bits[info[6]] % markers.length;
      markers.forEach((m, i) => {
        m.material.opacity = i === next ? 0.95 : 0.12;
        m.material.color.setHex(i === next ? 0xf1cd84 : 0x9a6fac);
      });
    },
  };
}
registerEnvironment("circuit", circuitEnvironment);
