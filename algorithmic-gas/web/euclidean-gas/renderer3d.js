import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/addons/controls/OrbitControls.js";
import { alive } from "./config.js";
import {
  bestIndex,
  buildSurface,
  companionPairs,
  contourLevels,
  contourSegments,
  coord,
  heightOf,
  metricRange,
  pixelStats,
  robustScale,
  writeInstance,
} from "./geometry3d.js";

const BACKGROUND = "#090d16";
const rgb = (hex) => {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
};
// Linear working-space endpoints, written straight into colour buffers.
const LOW = rgb("#7ef5df"),
  HIGH = rgb("#ff729b"),
  DEAD = rgb("#555267"),
  BEST = rgb("#efc87b"),
  SELECTED = rgb("#ffffff");
const LINK_CAP = 32768;
function dispose(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    if (Array.isArray(child.material))
      child.material.forEach((m) => m.dispose());
    else child.material?.dispose();
  });
  object.clear();
}
function sprite() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(16, 16, 14, 0, Math.PI * 2);
  ctx.fill();
  return new THREE.CanvasTexture(canvas);
}
function lines(color, opacity, floats) {
  const geometry = new THREE.BufferGeometry();
  const attribute = new THREE.BufferAttribute(new Float32Array(floats), 3);
  attribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("position", attribute);
  geometry.setDrawRange(0, 0);
  const object = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
  object.frustumCulled = false;
  return object;
}

export class SwarmRenderer3D {
  constructor(container, onSelect) {
    this.container = container;
    this.onSelect = onSelect;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const canvas = this.renderer.domElement;
    canvas.setAttribute(
      "aria-label",
      "Walker positions in three dimensions over the objective landscape",
    );
    canvas.style.touchAction = "none";
    container.prepend(canvas);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 400);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.addEventListener("change", () => this.draw());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x352141, 1.2));
    const light = new THREE.DirectionalLight(0xffffff, 1.2);
    light.position.set(5, -10, 20);
    this.scene.add(light);
    this.surface = new THREE.Group();
    this.overlays = new THREE.Group();
    this.frameGroup = new THREE.Group();
    this.scene.add(this.surface, this.overlays, this.frameGroup);
    this.bounds = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(20, 20, 20)),
      new THREE.LineBasicMaterial({
        color: 0x665477,
        transparent: true,
        opacity: 0.4,
      }),
    );
    this.frameGroup.add(this.bounds);
    this.links = lines(0xd0a9e2, 0.22, 1536);
    this.trailLines = lines(0x7ef5df, 0.3, 6);
    this.overlays.add(this.links, this.trailLines);
    this.texture = sprite();
    this.highlight = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 12, 8),
      new THREE.MeshStandardMaterial({ roughness: 0.5 }),
      2,
    );
    this.highlight.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(6),
      3,
    );
    this.highlight.frustumCulled = false;
    this.highlight.count = 0;
    this.scene.add(this.highlight);
    this.walkers = null;
    this.mode = null;
    this.capacity = 0;
    this.n = 0;
    this.xyz = new Float32Array(0);
    this.eligible = new Uint8Array(0);
    this.shown = new Uint8Array(0);
    this.pairs = null;
    this.domain = { low: -1, high: 1, dimensions: 2, minimum: null };
    this.settings = null;
    this.frame = null;
    this.selected = 0;
    this.trails = null;
    this.scale = null;
    this.surfaceData = null;
    this.surfaceKey = "";
    this.linkInfo = { drawn: 0, skippedHistorical: 0, capped: false };
    this.renderMs = 0;
    this.vector = new THREE.Vector3();
    this.trailPoint = (positions, offset, raw, target) =>
      this.place(positions, offset, raw, target);
    let down = null;
    canvas.addEventListener("pointerdown", (e) => {
      down = [e.clientX, e.clientY];
    });
    canvas.addEventListener("pointerup", (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5)
        this.selectAt(e.clientX, e.clientY);
      down = null;
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.resetCamera();
  }
  resize() {
    const width = this.container.clientWidth,
      height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.draw();
  }
  resetCamera() {
    this.camera.position.set(28, -34, 25);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.draw();
  }
  // Frames the finite eligible walkers, keeping the current view direction.
  fit() {
    let count = 0,
      cx = 0,
      cy = 0,
      cz = 0;
    for (let i = 0; i < this.n; i++)
      if (this.shown[i] && this.eligible[i]) {
        cx += this.xyz[i * 3];
        cy += this.xyz[i * 3 + 1];
        cz += this.xyz[i * 3 + 2];
        count++;
      }
    if (!count) return this.resetCamera();
    cx /= count;
    cy /= count;
    cz /= count;
    let radius = 0.5;
    for (let i = 0; i < this.n; i++)
      if (this.shown[i] && this.eligible[i])
        radius = Math.max(
          radius,
          Math.hypot(
            this.xyz[i * 3] - cx,
            this.xyz[i * 3 + 1] - cy,
            this.xyz[i * 3 + 2] - cz,
          ),
        );
    const direction = this.vector
      .copy(this.camera.position)
      .sub(this.controls.target)
      .normalize();
    const distance =
      (radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.2;
    this.controls.target.set(cx, cy, cz);
    this.camera.position.set(
      cx + direction.x * distance,
      cy + direction.y * distance,
      cz + direction.z * distance,
    );
    this.controls.update();
    this.draw();
  }
  setDomain(domain) {
    this.domain = domain;
  }
  // Display position of a d-dimensional point with raw reward `raw`.
  place(values, offset, raw, target) {
    const s = this.settings,
      { low, high } = this.domain,
      [ax, ay, az] = s.axes;
    target[0] = coord(values[offset + ax], low, high);
    target[1] = coord(values[offset + ay], low, high);
    target[2] =
      s.view === "landscape"
        ? this.scale
          ? heightOf(raw, this.scale, s.heightScale)
          : 0
        : az < this.d
          ? coord(values[offset + az], low, high)
          : 0;
    return (
      Number.isFinite(target[0]) &&
      Number.isFinite(target[1]) &&
      Number.isFinite(target[2])
    );
  }
  ensureWalkers(n) {
    const mode = n <= 2048 ? "sphere" : n <= 8192 ? "ico" : "points";
    if (this.walkers && mode === this.mode && n <= this.capacity) return;
    if (this.walkers) {
      this.scene.remove(this.walkers);
      this.walkers.geometry.dispose();
      this.walkers.material.dispose();
    }
    this.mode = mode;
    this.capacity = Math.max(n, this.capacity * 2, 256);
    if (mode === "points") {
      const geometry = new THREE.BufferGeometry();
      for (const name of ["position", "color"]) {
        const attribute = new THREE.BufferAttribute(
          new Float32Array(this.capacity * 3),
          3,
        );
        attribute.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute(name, attribute);
      }
      this.walkers = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({
          size: 0.4,
          vertexColors: true,
          map: this.texture,
          alphaTest: 0.5,
          sizeAttenuation: true,
        }),
      );
    } else {
      this.walkers = new THREE.InstancedMesh(
        mode === "sphere"
          ? new THREE.SphereGeometry(1, 10, 7)
          : new THREE.IcosahedronGeometry(1, 1),
        new THREE.MeshStandardMaterial({ roughness: 0.5 }),
        this.capacity,
      );
      this.walkers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.walkers.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(this.capacity * 3),
        3,
      );
      this.walkers.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    this.walkers.frustumCulled = false;
    this.scene.add(this.walkers);
    this.xyz = new Float32Array(this.capacity * 3);
    this.eligible = new Uint8Array(this.capacity);
    this.shown = new Uint8Array(this.capacity);
  }
  update(frame, settings, selected, trails = null) {
    this.frame = frame;
    this.settings = settings;
    this.selected = selected;
    this.trails = trails;
    const population = frame.population,
      field = population.observations.fields.positions;
    if (!field) return;
    const n = field.rows,
      d = field.item_shape.reduce((a, b) => a * b, 1),
      raw = population.rewards.raw;
    this.n = n;
    this.d = d;
    this.ensureSurface();
    this.ensureWalkers(n);
    for (let i = 0; i < n; i++)
      this.eligible[i] = alive(
        population.validity[i],
        settings.includeTruncated,
      )
        ? 1
        : 0;
    const fitness = frame.report?.pre_clone_fitness?.fitness,
      metric =
        settings.color === "uniform"
          ? null
          : settings.color === "fitness" && fitness
            ? fitness
            : raw,
      [min, max] = metric ? metricRange(metric, this.eligible) : [0, 0],
      best = bestIndex(raw, this.eligible, settings.direction),
      radius = settings.pointSize / 22,
      points = this.mode === "points",
      matrices = points ? null : this.walkers.instanceMatrix.array,
      colors = points
        ? this.walkers.geometry.attributes.color.array
        : this.walkers.instanceColor.array,
      positions = points
        ? this.walkers.geometry.attributes.position.array
        : null,
      target = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const visible = this.place(field.values, i * d, raw[i], target);
      this.shown[i] = visible ? 1 : 0;
      this.xyz[i * 3] = visible ? target[0] : 0;
      this.xyz[i * 3 + 1] = visible ? target[1] : 0;
      this.xyz[i * 3 + 2] = visible ? target[2] : 0;
      let color = LOW,
        t = -1;
      if (!this.eligible[i]) color = DEAD;
      else if (i === selected) color = SELECTED;
      else if (i === best) color = BEST;
      else if (metric && Number.isFinite(metric[i]))
        t = max > min ? (metric[i] - min) / (max - min) : 0.5;
      for (let k = 0; k < 3; k++)
        colors[i * 3 + k] = t < 0 ? color[k] : LOW[k] + (HIGH[k] - LOW[k]) * t;
      if (points) {
        positions[i * 3] = target[0];
        positions[i * 3 + 1] = target[1];
        positions[i * 3 + 2] = visible ? target[2] : 1e9;
        if (!visible) positions[i * 3] = positions[i * 3 + 1] = 0;
      } else
        writeInstance(
          matrices,
          i,
          this.xyz[i * 3],
          this.xyz[i * 3 + 1],
          this.xyz[i * 3 + 2],
          !visible
            ? 0
            : !this.eligible[i]
              ? radius * 0.6
              : i === selected
                ? radius * 1.65
                : radius,
        );
    }
    if (points) {
      this.walkers.geometry.setDrawRange(0, n);
      this.walkers.geometry.attributes.position.needsUpdate = true;
      this.walkers.geometry.attributes.color.needsUpdate = true;
      this.walkers.material.size = radius * 2;
      let count = 0;
      for (const [index, color, scale] of [
        [selected, SELECTED, 1.65],
        [best, BEST, 1.2],
      ]) {
        if (index < 0 || index >= n || !this.shown[index]) continue;
        if (index === selected && color === BEST) continue;
        writeInstance(
          this.highlight.instanceMatrix.array,
          count,
          this.xyz[index * 3],
          this.xyz[index * 3 + 1],
          this.xyz[index * 3 + 2],
          radius * scale,
        );
        this.highlight.instanceColor.array.set(color, count * 3);
        count++;
      }
      this.highlight.count = count;
      this.highlight.instanceMatrix.needsUpdate = true;
      this.highlight.instanceColor.needsUpdate = true;
    } else {
      this.walkers.count = n;
      this.walkers.instanceMatrix.needsUpdate = true;
      this.walkers.instanceColor.needsUpdate = true;
      this.highlight.count = 0;
    }
    this.updateLinks(frame, settings, selected, n);
    this.updateTrails(settings, trails);
    this.draw();
  }
  updateLinks(frame, settings, selected, n) {
    this.linkInfo = { drawn: 0, skippedHistorical: 0, capped: false };
    if (settings.links === "none" || !frame.report) {
      this.links.geometry.setDrawRange(0, 0);
      return;
    }
    const per =
        settings.links === "distance"
          ? frame.report.distance_companions.count
          : 1,
      capped = n * per > LINK_CAP,
      result = companionPairs(
        frame.report,
        settings.links,
        n,
        this.pairs,
        capped ? selected : -1,
      );
    this.pairs = result.out;
    let attribute = this.links.geometry.attributes.position;
    if (attribute.array.length < result.count * 6) {
      attribute = new THREE.BufferAttribute(
        new Float32Array(
          Math.max(result.count * 6, attribute.array.length * 2),
        ),
        3,
      );
      attribute.setUsage(THREE.DynamicDrawUsage);
      this.links.geometry.dispose();
      this.links.geometry = new THREE.BufferGeometry();
      this.links.geometry.setAttribute("position", attribute);
    }
    let vertices = 0;
    for (let p = 0; p < result.count; p++) {
      const a = this.pairs[p * 2],
        b = this.pairs[p * 2 + 1];
      if (
        !this.shown[a] ||
        !this.shown[b] ||
        !this.eligible[a] ||
        !this.eligible[b]
      )
        continue;
      attribute.array.set(this.xyz.subarray(a * 3, a * 3 + 3), vertices * 3);
      attribute.array.set(
        this.xyz.subarray(b * 3, b * 3 + 3),
        (vertices + 1) * 3,
      );
      vertices += 2;
    }
    attribute.needsUpdate = true;
    this.links.geometry.setDrawRange(0, vertices);
    this.linkInfo = {
      drawn: vertices / 2,
      skippedHistorical: result.skippedHistorical,
      capped,
    };
  }
  updateTrails(settings, trails) {
    if (!settings.trails || !trails) {
      this.trailLines.geometry.setDrawRange(0, 0);
      return;
    }
    let attribute = this.trailLines.geometry.attributes.position;
    if (attribute.array.length < trails.capacity) {
      attribute = new THREE.BufferAttribute(
        new Float32Array(trails.capacity),
        3,
      );
      attribute.setUsage(THREE.DynamicDrawUsage);
      this.trailLines.geometry.dispose();
      this.trailLines.geometry = new THREE.BufferGeometry();
      this.trailLines.geometry.setAttribute("position", attribute);
    }
    const floats =
      trails.d === this.d
        ? trails.segments(this.trailPoint, attribute.array)
        : 0;
    attribute.needsUpdate = true;
    this.trailLines.geometry.setDrawRange(0, floats / 3);
  }
  setSurface(data, settings) {
    this.surfaceData = data;
    this.settings = settings;
    this.surfaceKey = "";
    this.ensureSurface();
    if (this.frame)
      this.update(this.frame, settings, this.selected, this.trails);
    else this.draw();
  }
  // Rebuilds from the cached samples when a display setting changes; no
  // engine round trip.
  ensureSurface() {
    const s = this.settings,
      data = this.surfaceData;
    this.bounds.scale.z = s.view === "landscape" ? 0.005 : 1;
    this.surface.visible = s.showSurface;
    if (!data) {
      if (this.surface.children.length) dispose(this.surface);
      this.scale = null;
      this.surfaceKey = "";
      return;
    }
    const plane =
        s.view === "landscape"
          ? 0
          : s.axes[2] < this.domain.dimensions
            ? coord(s.slice[s.axes[2]], this.domain.low, this.domain.high)
            : 0,
      key = [s.view, s.heightScale, s.direction, plane].join("|");
    const material = this.surface.children[0]?.material;
    if (material) {
      material.opacity =
        s.view === "landscape" ? s.surfaceOpacity : 0.4 * s.surfaceOpacity;
      material.depthWrite = s.view === "landscape";
    }
    if (key === this.surfaceKey) return;
    this.surfaceKey = key;
    dispose(this.surface);
    this.scale = robustScale(data.values, s.direction);
    if (!this.scale) return;
    const built = buildSurface(
      data.values,
      data.resolution,
      s.view === "landscape"
        ? (v) => heightOf(v, this.scale, s.heightScale)
        : () => plane,
      this.scale,
    );
    const colors = new Float32Array(built.t.length * 3);
    for (let k = 0; k < built.t.length; k++) {
      const t = built.t[k];
      colors[k * 3] = 0.24 + 0.69 * t;
      colors[k * 3 + 1] = 0.12 + 0.68 * t;
      colors[k * 3 + 2] = 0.48 + 0.12 * (1 - t);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(built.positions, 3),
    );
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(built.indices, 1));
    geometry.computeVertexNormals();
    this.surface.add(
      new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          transparent: true,
          opacity:
            s.view === "landscape" ? s.surfaceOpacity : 0.4 * s.surfaceOpacity,
          roughness: 1,
          depthWrite: s.view === "landscape",
        }),
      ),
    );
    const contours = new THREE.BufferGeometry();
    contours.setAttribute(
      "position",
      new THREE.BufferAttribute(
        contourSegments(
          data.values,
          built.positions,
          built.indices,
          contourLevels(this.scale, 9),
        ),
        3,
      ),
    );
    this.surface.add(
      new THREE.LineSegments(
        contours,
        new THREE.LineBasicMaterial({
          color: 0xe7d5f1,
          transparent: true,
          opacity: 0.2,
        }),
      ),
    );
  }
  // Screen-space nearest walker: identical for spheres and sprites.
  selectAt(x, y) {
    if (!this.frame) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    let best = 14,
      depth = Infinity,
      selected = null;
    for (let i = 0; i < this.n; i++) {
      if (!this.shown[i]) continue;
      const p = this.vector
        .set(this.xyz[i * 3], this.xyz[i * 3 + 1], this.xyz[i * 3 + 2])
        .project(this.camera);
      if (p.z < -1 || p.z > 1) continue;
      const distance = Math.hypot(
        rect.left + ((p.x + 1) * rect.width) / 2 - x,
        rect.top + ((1 - p.y) * rect.height) / 2 - y,
      );
      if (distance >= 14) continue;
      if (distance < best - 0.5 || (distance < best + 0.5 && p.z < depth)) {
        best = Math.min(best, distance);
        depth = p.z;
        selected = i;
      }
    }
    if (selected !== null) this.onSelect(selected);
  }
  // Display position of one walker, for tests and overlays.
  screenPoint(index) {
    const rect = this.renderer.domElement.getBoundingClientRect(),
      p = this.vector
        .set(
          this.xyz[index * 3],
          this.xyz[index * 3 + 1],
          this.xyz[index * 3 + 2],
        )
        .project(this.camera);
    return [
      rect.left + ((p.x + 1) * rect.width) / 2,
      rect.top + ((1 - p.y) * rect.height) / 2,
    ];
  }
  pixelStats() {
    this.draw();
    return pixelStats(this.renderer.getContext());
  }
  draw() {
    const start = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.renderMs = performance.now() - start;
  }
  dispose() {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    dispose(this.scene);
    this.texture.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
}

export class MoleculeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
    this.camera.position.set(12, -18, 14);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.addEventListener("change", () => this.draw());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x463253, 3));
    this.mesh = null;
    this.capacity = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }
  resize() {
    const width = this.canvas.clientWidth,
      height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.draw();
  }
  // x holds 3·atoms coordinates; atoms are centred and scaled to the view.
  update(x) {
    const atoms = Math.floor(x.length / 3);
    if (!this.mesh || atoms > this.capacity) {
      if (this.mesh) {
        this.scene.remove(this.mesh);
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
      }
      this.capacity = Math.max(atoms, this.capacity * 2, 16);
      this.mesh = new THREE.InstancedMesh(
        new THREE.SphereGeometry(1, 16, 12),
        new THREE.MeshStandardMaterial({ color: 0xd0a9e2 }),
        this.capacity,
      );
      this.mesh.frustumCulled = false;
      this.scene.add(this.mesh);
    }
    const center = [0, 0, 0];
    let finite = 0;
    for (let i = 0; i < atoms; i++) {
      if (![0, 1, 2].every((k) => Number.isFinite(x[i * 3 + k]))) continue;
      for (let k = 0; k < 3; k++) center[k] += x[i * 3 + k];
      finite++;
    }
    for (let k = 0; k < 3; k++) center[k] /= Math.max(1, finite);
    let radius = 0;
    for (let i = 0; i < atoms; i++) {
      const r = Math.hypot(
        x[i * 3] - center[0],
        x[i * 3 + 1] - center[1],
        x[i * 3 + 2] - center[2],
      );
      if (Number.isFinite(r)) radius = Math.max(radius, r);
    }
    const scale = 6 / Math.max(1, radius),
      size = atoms > 100 ? 0.25 : 0.5;
    for (let i = 0; i < atoms; i++) {
      const ok = [0, 1, 2].every((k) => Number.isFinite(x[i * 3 + k]));
      writeInstance(
        this.mesh.instanceMatrix.array,
        i,
        ok ? (x[i * 3] - center[0]) * scale : 0,
        ok ? (x[i * 3 + 1] - center[1]) * scale : 0,
        ok ? (x[i * 3 + 2] - center[2]) * scale : 0,
        ok ? size : 0,
      );
    }
    this.mesh.count = atoms;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.draw();
  }
  draw() {
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    dispose(this.scene);
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    // A lost context stays bound to its canvas; leave a fresh one behind.
    this.canvas.replaceWith(this.canvas.cloneNode(false));
  }
}
