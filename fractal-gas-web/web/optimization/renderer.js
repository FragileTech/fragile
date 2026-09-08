import * as T from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/addons/controls/OrbitControls.js";
import { frameInfo, row } from "./native.js";
const color = (t) =>
  new T.Color().setRGB(0.24 + 0.69 * t, 0.12 + 0.68 * t, 0.48 + 0.12 * (1 - t));
function dispose(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    if (Array.isArray(child.material))
      child.material.forEach((m) => m.dispose());
    else child.material?.dispose();
  });
  object.clear();
}
export class SwarmRenderer {
  constructor(canvas, select) {
    this.renderer = new T.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x15111b);
    this.scene = new T.Scene();
    this.camera = new T.PerspectiveCamera(42, 1, 0.01, 300);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.addEventListener("change", () => this.draw());
    this.scene.add(new T.HemisphereLight(0xffffff, 0x352141, 1.2));
    const light = new T.DirectionalLight(0xffffff, 1.2);
    light.position.set(5, -10, 20);
    this.scene.add(light);
    this.surface = new T.Group();
    this.overlays = new T.Group();
    this.frameGroup = new T.Group();
    this.scene.add(this.surface, this.overlays, this.frameGroup);
    const box = new T.EdgesGeometry(new T.BoxGeometry(20, 20, 20));
    this.bounds = new T.LineSegments(
      box,
      new T.LineBasicMaterial({
        color: 0x665477,
        transparent: true,
        opacity: 0.4,
      }),
    );
    this.frameGroup.add(this.bounds);
    this.mesh = null;
    this.capacity = 0;
    this.config = null;
    this.frame = null;
    this.settings = {};
    this.selected = -1;
    this.surfaceScale = 1;
    this.surfaceBase = 0;
    this.ray = new T.Raycaster();
    let down;
    canvas.addEventListener("pointerdown", (e) => {
      down = [e.clientX, e.clientY];
    });
    canvas.addEventListener("pointerup", (e) => {
      if (
        !down ||
        Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 ||
        !this.mesh
      )
        return;
      const rect = canvas.getBoundingClientRect();
      this.ray.setFromCamera(
        new T.Vector2(
          (2 * (e.clientX - rect.left)) / rect.width - 1,
          1 - (2 * (e.clientY - rect.top)) / rect.height,
        ),
        this.camera,
      );
      const hit = this.ray.intersectObject(this.mesh)[0];
      if (hit) select(hit.instanceId);
    });
    this.resizeObserver = new ResizeObserver(() => {
      const w = canvas.clientWidth,
        h = canvas.clientHeight;
      if (!w || !h) return;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.draw();
    });
    this.resizeObserver.observe(canvas);
    this.resetCamera();
  }
  resetCamera() {
    this.camera.position.set(28, -34, 25);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.draw();
  }
  setConfig(config) {
    this.config = config;
    this.surfaceBase = 0;
    this.surfaceScale = 1;
    dispose(this.surface);
    dispose(this.overlays);
  }
  coord(x) {
    return (
      ((x - this.config.low) / (this.config.high - this.config.low)) * 20 - 10
    );
  }
  height(value) {
    return (
      Math.asinh((value - this.surfaceBase) / this.surfaceScale) *
      Number(this.settings.height || 3)
    );
  }
  point(walker) {
    const s = this.settings,
      axes = s.axes || [0, 1, 2];
    return new T.Vector3(
      this.coord(walker.x[axes[0]]),
      walker.x.length > 1 ? this.coord(walker.x[axes[1] ?? axes[0]]) : 0,
      s.view === "landscape"
        ? this.height(walker.value)
        : axes[2] < walker.x.length
          ? this.coord(walker.x[axes[2]])
          : 0,
    );
  }
  setSurface(values, resolution, settings) {
    this.settings = settings;
    this.surfaceValues = values;
    this.resolution = resolution;
    dispose(this.surface);
    const finite = Array.from(values)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    if (!finite.length) {
      this.draw();
      return;
    }
    this.surfaceBase = finite[0];
    this.surfaceScale = Math.max(
      1e-9,
      (finite[Math.floor((finite.length - 1) * 0.95)] -
        finite[Math.floor((finite.length - 1) * 0.05)]) /
        3,
      Math.abs(finite[0]) * 1e-8,
    );
    const positions = new Float32Array(values.length * 3),
      colors = new Float32Array(values.length * 3),
      indices = [];
    const position = (i, j) => {
      const k = j * resolution + i;
      return [
        (i / (resolution - 1)) * 20 - 10,
        (j / (resolution - 1)) * 20 - 10,
        settings.view === "landscape"
          ? this.height(values[k])
          : settings.axes[2] < this.config.dimensions
            ? this.coord(settings.slice[settings.axes[2]])
            : 0,
      ];
    };
    for (let j = 0; j < resolution; j++)
      for (let i = 0; i < resolution; i++) {
        const k = j * resolution + i;
        positions.set(position(i, j), k * 3);
        const t = Math.max(
          0,
          Math.min(
            1,
            Math.asinh((values[k] - this.surfaceBase) / this.surfaceScale) / 3,
          ),
        );
        color(Number.isFinite(t) ? t : 0).toArray(colors, k * 3);
        if (i < resolution - 1 && j < resolution - 1) {
          const a = k,
            b = k + 1,
            c = k + resolution,
            d = c + 1;
          if ([a, b, c].every((n) => Number.isFinite(values[n])))
            indices.push(a, b, c);
          if ([b, c, d].every((n) => Number.isFinite(values[n])))
            indices.push(b, d, c);
        }
      }
    // Replace nonfinite unused coordinates before uploading buffers.
    for (let i = 0; i < positions.length; i++)
      if (!Number.isFinite(positions[i])) positions[i] = 0;
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new T.BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const material = new T.MeshStandardMaterial({
      vertexColors: true,
      side: T.DoubleSide,
      transparent: true,
      opacity: settings.view === "landscape" ? 0.77 : 0.3,
      roughness: 1,
      depthWrite: settings.view === "landscape",
    });
    this.surface.add(new T.Mesh(geometry, material));
    // Marching triangle contours use the exact sampled objective values.
    const lines = [];
    for (let level = 1; level < 10; level++) {
      const target =
        this.surfaceBase + this.surfaceScale * Math.sinh(level / 3);
      for (let t = 0; t < indices.length; t += 3) {
        const hits = [];
        for (let e = 0; e < 3; e++) {
          const a = indices[t + e],
            b = indices[t + ((e + 1) % 3)],
            va = values[a],
            vb = values[b];
          if ((va < target && vb >= target) || (vb < target && va >= target)) {
            const f = (target - va) / (vb - va);
            hits.push(
              [0, 1, 2].map(
                (k) =>
                  positions[a * 3 + k] +
                  f * (positions[b * 3 + k] - positions[a * 3 + k]) +
                  (k === 2 ? 0.012 : 0),
              ),
            );
          }
        }
        if (hits.length === 2) lines.push(...hits[0], ...hits[1]);
      }
    }
    const contours = new T.BufferGeometry();
    contours.setAttribute("position", new T.Float32BufferAttribute(lines, 3));
    this.surface.add(
      new T.LineSegments(
        contours,
        new T.LineBasicMaterial({
          color: 0xe7d5f1,
          transparent: true,
          opacity: 0.2,
        }),
      ),
    );
    this.surface.visible = settings.view === "landscape" || settings.showSlice;
    this.bounds.scale.z = settings.view === "landscape" ? 0.005 : 1;
    if (this.frame)
      this.update(this.frame, settings, this.history || [], this.selected);
    else this.draw();
  }
  update(frame, settings, history, selected) {
    this.frame = frame;
    this.settings = settings;
    this.history = history;
    this.selected = selected;
    const info = frameInfo(frame);
    if (!this.mesh || info.n > this.capacity) {
      if (this.mesh) {
        this.scene.remove(this.mesh);
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
      }
      this.capacity = Math.max(info.n, this.capacity * 2);
      this.mesh = new T.InstancedMesh(
        new T.SphereGeometry(1, 10, 7),
        new T.MeshStandardMaterial({
          roughness: 0.5,
          transparent: true,
          opacity: Number(settings.opacity),
          depthWrite: true,
        }),
        this.capacity,
      );
      this.mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
      this.mesh.frustumCulled = false;
      this.scene.add(this.mesh);
    }
    this.mesh.count = info.n;
    this.mesh.material.opacity = Number(settings.opacity);
    let min = INFINITY,
      max = -INFINITY;
    const data = [];
    for (let i = 0; i < info.n; i++) {
      const w = row(frame, i);
      data.push(w);
      if (w.alive) {
        const v = settings.color === "fitness" ? w.fitness : w.value;
        if (Number.isFinite(v)) {
          min = Math.min(min, v);
          max = Math.max(max, v);
        }
      }
    }
    const transform = new T.Object3D(),
      pointSize = Number(settings.pointSize) / 22;
    for (let i = 0; i < info.n; i++) {
      const w = data[i];
      transform.position.copy(w.alive ? this.point(w) : new T.Vector3());
      const radius = w.alive
        ? pointSize *
          (i === selected || (settings.planning && i === info.n - 1) ? 1.65 : 1)
        : 0;
      transform.scale.setScalar(radius);
      transform.updateMatrix();
      this.mesh.setMatrixAt(i, transform.matrix);
      const metric = settings.color === "fitness" ? w.fitness : w.value;
      let c =
        settings.color === "constant" || !Number.isFinite(metric)
          ? new T.Color(0x7ef5df)
          : color(max > min ? (metric - min) / (max - min) : 0.5);
      if (settings.planning && i === info.n - 1) c = new T.Color(0x7ef5df);
      if (i === selected) c = new T.Color(0xffffff);
      else if (i === info.bestIndex) c = new T.Color(0xefc87b);
      this.mesh.setColorAt(i, c);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    dispose(this.overlays);
    const line = (points, c, opacity) => {
      if (!points.length) return;
      const g = new T.BufferGeometry();
      g.setAttribute("position", new T.Float32BufferAttribute(points, 3));
      this.overlays.add(
        new T.LineSegments(
          g,
          new T.LineBasicMaterial({ color: c, transparent: true, opacity }),
        ),
      );
    };
    if (settings.edges !== "none") {
      const points = [];
      for (let i = 0; i < info.n; i++) {
        const w = data[i],
          j =
            settings.edges === "companions"
              ? w.companion
              : settings.edges === "cloning"
                ? w.cloneCompanion
                : w.parent;
        if (j === i || !w.alive || !data[j]?.alive) continue;
        points.push(
          ...this.point(w).toArray(),
          ...this.point(data[j]).toArray(),
        );
      }
      line(points, 0xd0a9e2, 0.22);
    }
    if (settings.trails && history.length > 1) {
      const points = [],
        start = Math.max(1, history.length - 30);
      const selectedIndices =
        selected >= 0
          ? [selected]
          : Array.from({ length: Math.min(info.n, 96) }, (_, i) => i);
      for (let t = start; t < history.length; t++) {
        const prev = history[t - 1],
          next = history[t];
        for (const i of selectedIndices) {
          if (i >= prev[1] || i >= next[1]) continue;
          const a = row(prev, i),
            b = row(next, i);
          if (
            !a.alive ||
            !b.alive ||
            b.cloned ||
            (settings.planning && i !== info.n - 1 && b.parent === info.n - 1)
          )
            continue;
          points.push(...this.point(a).toArray(), ...this.point(b).toArray());
        }
      }
      line(points, 0x7ef5df, 0.3);
    }
    this.surface.visible = settings.view === "landscape" || settings.showSlice;
    this.draw();
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
    this.renderer.dispose();
  }
}
const INFINITY = Number.POSITIVE_INFINITY;
export class MoleculeRenderer {
  constructor(canvas) {
    this.renderer = new T.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new T.Scene();
    this.camera = new T.PerspectiveCamera(40, 1, 0.01, 200);
    this.camera.position.set(12, -18, 14);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.addEventListener("change", () => this.draw());
    this.scene.add(new T.HemisphereLight(0xffffff, 0x463253, 3));
    this.group = new T.Group();
    this.scene.add(this.group);
    this.resize = new ResizeObserver(() => {
      const w = canvas.clientWidth,
        h = canvas.clientHeight;
      if (w && h) {
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.draw();
      }
    });
    this.resize.observe(canvas);
  }
  update(x) {
    dispose(this.group);
    const atoms = x.length / 3,
      center = [0, 0, 0];
    for (let i = 0; i < atoms; i++)
      for (let k = 0; k < 3; k++) center[k] += x[i * 3 + k] / atoms;
    let radius = 0;
    for (let i = 0; i < atoms; i++)
      radius = Math.max(
        radius,
        Math.hypot(...center.map((v, k) => x[i * 3 + k] - v)),
      );
    const scale = 6 / Math.max(1, radius);
    for (let i = 0; i < atoms; i++) {
      const sphere = new T.Mesh(
        new T.SphereGeometry(atoms > 100 ? 0.25 : 0.5, 16, 12),
        new T.MeshStandardMaterial({ color: 0xd0a9e2 }),
      );
      sphere.position.set(...center.map((v, k) => (x[i * 3 + k] - v) * scale));
      this.group.add(sphere);
    }
    this.draw();
  }
  draw() {
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this.resize.disconnect();
    this.controls.dispose();
    dispose(this.scene);
    this.renderer.dispose();
  }
}
