import { treePoseDim, treeWidth } from "./actions.js";
import { laboratoryEnvironment } from "./visuals/lighting.js";
import { createEnvironment } from "./visuals/environments/index.js";
import { BodyLayer } from "./visuals/body-layer.js";
import * as T from "./vendor/three.module.js";
import { palette, prism, zoneModel, reactorModel } from "./models.js";
import { assetModel, preloadStyle } from "./visuals/assets.js";
import { disposeGroup as dispose } from "./visuals/resources.js";
import { stylePalette } from "./visuals/style-palette.js";
import { labStyle } from "./visual-style.js";

function line(points, color, dashed = false) {
  const geometry = new T.BufferGeometry().setFromPoints(
    points.map((p) => new T.Vector3(...p)),
  );
  const material = dashed
    ? new T.LineDashedMaterial({
        color,
        dashSize: 0.3,
        gapSize: 0.2,
        transparent: true,
        opacity: 0.8,
      })
    : new T.LineBasicMaterial({ color });
  const mesh = new T.Line(geometry, material);
  if (dashed) mesh.computeLineDistances();
  return mesh;
}
function prop(kind, style, radius = 1, color = palette.gold) {
  const model = assetModel(style, kind, "high");
  if (!model)
    return kind === "reactor"
      ? reactorModel()
      : zoneModel(radius, color, kind === "dock" ? "base" : "gate");
  const root = new T.Group();
  model.scale.set(radius, radius, 1);
  root.add(model);
  return root;
}
export class LabRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.style = labStyle.current;
    this.unsubscribeStyle = labStyle.subscribe((style) =>
      this.prepareStyle(style),
    );
    this.renderer = new T.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x090d16);
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.world = new T.Scene();
    this.world.background = new T.Color(0x0c1422);
    this.world.fog = new T.FogExp2(0x0c1422, 0.002);
    this.environment = laboratoryEnvironment(this.renderer, this.style);
    this.world.environment = this.environment.texture;
    this.world.environmentIntensity = 0.65;
    this.restoreContext = () => {
      this.environment.dispose();
      this.environment = laboratoryEnvironment(this.renderer, this.style);
      this.world.environment = this.environment.texture;
    };
    canvas.addEventListener("webglcontextrestored", this.restoreContext);
    this.camera = new T.OrthographicCamera(-40, 40, 30, -30, 0.1, 300);
    this.camera.up.set(0, 0, 1);
    this.hemisphere = new T.HemisphereLight(0xbcecff, 0x3e204f, 1.6);
    this.world.add(this.hemisphere);
    const key = new T.DirectionalLight(0xe0e4ff, 2.2);
    key.position.set(20, -30, 70);
    this.world.add(key);
    const rim = new T.DirectionalLight(0xad70ff, 2);
    rim.position.set(-40, 50, 30);
    this.world.add(rim);
    this.keyLight = key;
    this.rimLight = rim;
    this.applyLighting(this.style);
    this.static = new T.Group();
    this.dynamic = new T.Group();
    this.overlays = new T.Group();
    this.world.add(this.static, this.dynamic, this.overlays);
    this.layers = {
      tree: true,
      cloud: true,
      geometry: false,
      tethers: true,
    };
    this.zoom = 1;
    this.top = false;
    this.viewCenter = [32, 22];
    this.ray = new T.Raycaster();
    this.plane = new T.Plane(new T.Vector3(0, 0, 1), 0);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.zoom = T.MathUtils.clamp(
          this.zoom * Math.exp(-e.deltaY * 0.001),
          0.6,
          12,
        );
        this.resize();
      },
      { passive: false },
    );
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
      if (e.button === 1 || e.button === 2 || e.altKey) {
        this.panAnchor = this.worldPoint(e);
        this.followBody = null;
        canvas.setPointerCapture(e.pointerId);
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    });
    canvas.addEventListener("pointermove", (e) => {
      if (this.panAnchor) {
        const point = this.worldPoint(e);
        if (point) {
          this.viewCenter = this.viewCenter.map(
            (v, i) => v + this.panAnchor[i] - point[i],
          );
          this.resize();
        }
        e.stopImmediatePropagation();
      }
    });
    canvas.addEventListener("pointerup", () => {
      this.panAnchor = null;
    });
    canvas.addEventListener("lostpointercapture", () => {
      this.panAnchor = null;
    });
    this.animate = this.animate.bind(this);
    this.frame = requestAnimationFrame(this.animate);
  }
  resize() {
    const { width, height } = this.canvas.parentElement.getBoundingClientRect();
    if (!height) return;
    this.renderer.setSize(width, height, false);
    const span =
      (Math.max(
        this.size?.[1] || 44,
        (this.size?.[0] || 64) / (width / height),
      ) *
        0.66) /
      this.zoom;
    this.camera.left = (-span * width) / height;
    this.camera.right = (span * width) / height;
    this.camera.top = span;
    this.camera.bottom = -span;
    this.camera.position.set(
      this.viewCenter[0],
      this.viewCenter[1] - (this.top ? 0.001 : 45),
      this.top ? 90 : 60,
    );
    this.camera.lookAt(this.viewCenter[0], this.viewCenter[1], 0);
    this.camera.updateProjectionMatrix();
  }
  worldPoint(event) {
    const r = this.canvas.getBoundingClientRect();
    this.ray.setFromCamera(
      new T.Vector2(
        ((event.clientX - r.left) / r.width) * 2 - 1,
        1 - ((event.clientY - r.top) / r.height) * 2,
      ),
      this.camera,
    );
    const p = this.ray.ray.intersectPlane(this.plane, new T.Vector3());
    return p ? [p.x, p.y] : null;
  }
  load(scene, info, channels) {
    dispose(this.static);
    dispose(this.dynamic);
    dispose(this.overlays);
    this.config = scene;
    this.info = info;
    this.channels = channels;
    this.state = null;
    this.action = null;
    this.size = scene.size || [64, 44];
    this.viewCenter = this.size.map((v) => v / 2);
    this.zoom = 1;
    this.followBody = null;
    const presentation = this.makeStatic(scene, this.style);
    this.static.add(presentation.group);
    Object.assign(this, {
      scenery: presentation.scenery,
      bases: presentation.bases,
      reactors: presentation.reactors,
    });
    this.bodyGroup = new T.Group();
    this.dynamic.add(this.bodyGroup);
    this.bodyLayer = new BodyLayer(scene, info, this.bodyGroup, channels, {
      style: this.style,
    });
    this.models = this.bodyLayer.models;
    this.controlled = this.bodyLayer.controlled;
    this.food = (scene.pickups || []).map((def) => {
      const food = new T.Mesh(
        new T.OctahedronGeometry(def.radius || 0.4),
        new T.MeshStandardMaterial({
          color: stylePalette[this.style].ore,
          emissive: stylePalette[this.style].ore,
          emissiveIntensity: this.style === "steampunk" ? 0.12 : 0.65,
        }),
      );
      food.position.set(...def.position, 0.6);
      this.dynamic.add(food);
      return food;
    });
    this.tethers = new T.Group();
    this.overlays.add(this.tethers);
    this.treeGroup = new T.Group();
    this.cloudGroup = new T.Group();
    this.hulls = new T.Group();
    this.overlays.add(this.treeGroup, this.cloudGroup, this.hulls);
    for (const b of this.bodyLayer.bodies) {
      const vertices =
        b.vertices ||
        Array.from({ length: 32 }, (_, i) => [
          (b.radius || 0.5) * Math.cos((i * Math.PI) / 16),
          (b.radius || 0.5) * Math.sin((i * Math.PI) / 16),
        ]);
      this.hulls.add(
        line(
          [...vertices, vertices[0]].map((p) => [...p, 0.95]),
          palette.gold,
        ),
      );
    }
    this.inspection = new T.Group();
    this.multiSelection = new T.Group();
    this.overlays.add(this.inspection, this.multiSelection);
    this.selection = zoneModel(0.95, 0xffffff);
    this.selection.visible = false;
    this.overlays.add(this.selection);
    this.resize();
    this.setLayers(this.layers);
  }
  makeStatic(scene, style) {
    const group = new T.Group(),
      bases = [],
      reactors = [];
    const scenery = createEnvironment(scene, { style });
    group.add(scenery.group);
    for (const [kind, color] of [
      ["bases", palette.green],
      ["gates", palette.gold],
    ])
      for (const [i, def] of (kind === "gates" && scenery.replacesGates
        ? []
        : scene[kind] || []
      ).entries()) {
        const mesh = prop(
          kind === "bases" ? "dock" : "gate",
          style,
          def.radius || 2,
          color,
        );
        mesh.position.set(...def.position, 0);
        group.add(mesh);
        if (kind === "bases") bases.push(mesh);
        this.label(
          kind === "bases"
            ? "RECOVERY / 01"
            : `GATE / ${String(i + 1).padStart(2, "0")}`,
          [...def.position, 0.3],
          style === "steampunk" ? stylePalette[style].accent : color,
          4.2,
          group,
        );
      }
    for (const def of scene.gravity || []) {
      const model = prop("reactor", style);
      model.position.set(...def.position, 0);
      group.add(model);
      reactors.push(model);
      const ring = zoneModel(
        (def.softening || 2) * 1.8,
        stylePalette[style].energy,
      );
      ring.position.set(...def.position, 0.03);
      group.add(ring);
      this.label(
        "GRAVITY WELL",
        [def.position[0], def.position[1] - 4, 0.1],
        stylePalette[style].energy,
        4,
        group,
      );
    }
    return { group, scenery, bases, reactors };
  }
  applyLighting(style) {
    const steam = style === "steampunk";
    this.world.background.setHex(stylePalette[style].background);
    this.world.fog.color.copy(this.world.background);
    this.hemisphere.color.setHex(steam ? 0xffdfac : 0xbcecff);
    this.hemisphere.groundColor.setHex(steam ? 0x3b281b : 0x3e204f);
    this.keyLight.color.setHex(steam ? 0xffdfb7 : 0xe0e4ff);
    this.rimLight.color.setHex(steam ? 0xd79250 : 0xad70ff);
  }
  prepareStyle(style) {
    const environment = laboratoryEnvironment(this.renderer, style);
    const bodyGroup = new T.Group();
    let presentation, bodyLayer;
    try {
      if (this.config) {
        presentation = this.makeStatic(this.config, style);
        bodyLayer = new BodyLayer(
          this.config,
          this.info,
          bodyGroup,
          this.channels,
          { style },
        );
        if (this.state) bodyLayer.update(this.state, this.action);
      }
    } catch (error) {
      environment.dispose();
      if (presentation) dispose(presentation.group);
      dispose(bodyGroup);
      throw error;
    }
    return {
      cancel: () => {
        environment.dispose();
        if (presentation) dispose(presentation.group);
        dispose(bodyGroup);
      },
      commit: () => {
        this.style = style;
        this.environment.dispose();
        this.environment = environment;
        this.world.environment = environment.texture;
        this.applyLighting(style);
        if (!presentation) return;
        dispose(this.static);
        this.static.add(presentation.group);
        Object.assign(this, {
          scenery: presentation.scenery,
          bases: presentation.bases,
          reactors: presentation.reactors,
        });
        this.dynamic.remove(this.bodyGroup);
        dispose(this.bodyGroup);
        this.bodyGroup = bodyGroup;
        this.dynamic.add(bodyGroup);
        this.bodyLayer = bodyLayer;
        this.models = bodyLayer.models;
        this.controlled = bodyLayer.controlled;
        for (const food of this.food) {
          food.material.color.setHex(stylePalette[style].ore);
          food.material.emissive.setHex(stylePalette[style].ore);
          food.material.emissiveIntensity = style === "steampunk" ? 0.12 : 0.65;
        }
        if (this.state) this.update(this.state, this.action);
      },
    };
  }
  async setStyle(style) {
    const request = (this.styleRequest || 0) + 1;
    this.styleRequest = request;
    await preloadStyle(style);
    if (this.disposed || request !== this.styleRequest) return false;
    this.prepareStyle(style).commit();
    return true;
  }
  label(text, pos, color, size, parent = this.static) {
    const c = document.createElement("canvas");
    c.width = 512;
    c.height = 80;
    const ctx = c.getContext("2d");
    ctx.font = "32px monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = "#" + color.toString(16).padStart(6, "0");
    ctx.fillText(text, 256, 49);
    const texture = new T.CanvasTexture(c),
      sprite = new T.Sprite(
        new T.SpriteMaterial({
          map: texture,
          transparent: true,
          depthTest: false,
        }),
      );
    sprite.position.set(pos[0], pos[1] - 1, pos[2]);
    sprite.scale.set(size, (size * 80) / 512, 1);
    parent.add(sprite);
  }
  update(state, action) {
    if (!this.info) return;
    this.state = state;
    this.action = action;
    const n = this.models.length,
      bits = new Uint32Array(state.buffer, state.byteOffset, state.length);
    this.bodyLayer.update(state, action);
    this.scenery.update?.(state, this.info);
    this.models.forEach((mesh, i) => {
      this.hulls.children[i].position.copy(mesh.position);
      this.hulls.children[i].rotation.z = mesh.rotation.z;
    });
    this.simulationTime = bits[0] * (this.config.physics?.dt || 1 / 60);
    if (this.followBody != null) {
      const model = this.models[this.followBody];
      this.viewCenter = [model.position.x, model.position.y];
      this.resize();
    }
    this.food.forEach((food, i) => {
      const offset = this.info[8] + 3 * i;
      food.position.set(state[offset], state[offset + 1], 0.55);
      food.visible = state[offset + 2] <= 0;
    });
    dispose(this.tethers);
    (this.config.tethers || []).forEach((def, i) => {
      const b = bits[this.info[7] + 2 * i] - 1;
      if (b >= 0)
        this.tethers.add(
          line(
            [
              [state[8 + def.a], state[8 + n + def.a], 0.5],
              [state[8 + b], state[8 + n + b], 0.5],
            ],
            stylePalette[this.style].energy,
            true,
          ),
        );
    });
    if (this.config.task === "tandem" && this.controlled.length >= 2) {
      const points = this.controlled.map((b) => [
        state[8 + b],
        state[8 + n + b],
        0.3,
      ]);
      const center = points.reduce(
        (s, p) => s.map((v, k) => v + p[k] / points.length),
        [0, 0, 0],
      );
      const gate = this.config.gates?.[
        bits[this.info[6]] % this.config.gates.length
      ]?.position || [center[0] + 3, center[1]];
      const d = Math.hypot(gate[0] - center[0], gate[1] - center[1]) || 1;
      const anchor = [
        center[0] + ((gate[0] - center[0]) * 3) / d,
        center[1] + ((gate[1] - center[1]) * 3) / d,
        0.3,
      ];
      this.tethers.add(
        line([points[0], anchor, points[1], points[0]], palette.gold, true),
      );
    }
    if (bits[4] !== this.delivered) {
      this.deliveryFlash = performance.now();
      this.delivered = bits[4];
    }
  }
  diagnostics(tree, cloud) {
    dispose(this.treeGroup);
    dispose(this.cloudGroup);
    if (!tree) return;
    const width = treeWidth(tree),
      count = tree.meta.length / 5,
      index = new Map(),
      points = [],
      colors = [];
    for (let i = 0; i < count; i++) index.set(tree.meta[i * 5], i);
    let avg = 0;
    for (let i = 0; i < count; i++)
      avg += tree.values[i * width] / Math.max(1, count);
    const skip = Math.max(
      1,
      Math.ceil((count * treePoseDim(tree)) / 2 / 50000),
    );
    for (let i = 0; i < count; i += skip) {
      const parent = index.get(tree.meta[i * 5 + 1]);
      if (parent === undefined) continue;
      const flags = tree.meta[i * 5 + 4];
      const color = new T.Color(
        flags & 1
          ? palette.rose
          : flags & 2
            ? palette.violet
            : tree.values[i * width] >= avg
              ? palette.green
              : 0x4883a4,
      );
      for (let c = 0; c < treePoseDim(tree); c += 2) {
        points.push(
          tree.values[parent * width + 3 + c],
          tree.values[parent * width + 4 + c],
          0.34,
          tree.values[i * width + 3 + c],
          tree.values[i * width + 4 + c],
          0.34,
        );
        colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    }
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.Float32BufferAttribute(points, 3));
    geometry.setAttribute("color", new T.Float32BufferAttribute(colors, 3));
    this.treeGroup.add(
      new T.LineSegments(
        geometry,
        new T.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity: 0.33,
          depthWrite: false,
        }),
      ),
    );
    if (cloud) {
      const particles = [],
        stride = this.info[3],
        n = this.info[1];
      for (let w = 0; w < cloud.length / stride; w++)
        for (const b of this.controlled)
          particles.push(
            cloud[w * stride + 8 + b],
            cloud[w * stride + 8 + n + b],
            0.45,
          );
      const g = new T.BufferGeometry();
      g.setAttribute("position", new T.Float32BufferAttribute(particles, 3));
      this.cloudGroup.add(
        new T.Points(
          g,
          new T.PointsMaterial({
            color: palette.cyan,
            size: 0.14,
            transparent: true,
            opacity: 0.5,
            depthWrite: false,
          }),
        ),
      );
    }
  }
  clearDiagnostics() {
    if (this.treeGroup) dispose(this.treeGroup);
    if (this.cloudGroup) dispose(this.cloudGroup);
  }
  setLayers(layers) {
    Object.assign(this.layers, layers);
    if (this.treeGroup) {
      this.treeGroup.visible = this.layers.tree;
      this.cloudGroup.visible = this.layers.cloud;
      this.hulls.visible = this.layers.geometry;
      this.tethers.visible = this.layers.tethers;
    }
  }
  select(position) {
    if (this.selection) {
      this.selection.visible = !!position;
      if (position) this.selection.position.set(...position, 0.1);
    }
  }
  selectMany(positions) {
    this.select();
    if (!this.multiSelection) return;
    dispose(this.multiSelection);
    for (const pos of positions) {
      const marker = zoneModel(0.95, 0xffffff);
      marker.position.set(...pos, 0.1);
      this.multiSelection.add(marker);
    }
  }
  inspectVectors(vectors, state) {
    if (!this.inspection) return;
    dispose(this.inspection);
    const arrow = (x, y, dx, dy, color) => {
      const length = Math.hypot(dx, dy);
      if (length < 1e-5) return;
      const scale = Math.min(8 / length, 1);
      dx *= scale;
      dy *= scale;
      const tip = [x + dx, y + dy, 0.8],
        side = 0.22,
        ux = dx / Math.hypot(dx, dy),
        uy = dy / Math.hypot(dx, dy);
      this.inspection.add(
        line([[x, y, 0.8], tip], color),
        line(
          [
            [
              tip[0] - side * ux + side * uy,
              tip[1] - side * uy - side * ux,
              0.8,
            ],
            tip,
            [
              tip[0] - side * ux - side * uy,
              tip[1] - side * uy + side * ux,
              0.8,
            ],
          ],
          color,
        ),
      );
    };
    if (state) {
      const B = this.info[1];
      for (let i = 0; i < B; i++)
        arrow(
          state[8 + i],
          state[8 + B + i],
          state[8 + 2 * B + i] * 0.25,
          state[8 + 3 * B + i] * 0.25,
          0x45e9e5,
        );
    }
    for (let i = 0; i < vectors.length; i += 8) {
      const kind = vectors[i],
        scale = kind === 1 ? 1 : 0.05;
      arrow(
        vectors[i + 3],
        vectors[i + 4],
        vectors[i + 5] * scale,
        vectors[i + 6] * scale,
        [0xffbb45, 0xff526e, 0xd670ff][kind] || 0xffffff,
      );
    }
  }
  focus(body = null) {
    this.followBody = body;
    this.viewCenter =
      body == null
        ? this.size.map((v) => v / 2)
        : [this.models[body].position.x, this.models[body].position.y];
    this.zoom = body == null ? 1 : 7;
    this.resize();
  }
  dispose() {
    this.disposed = true;
    this.unsubscribeStyle();
    this.canvas.removeEventListener(
      "webglcontextrestored",
      this.restoreContext,
    );
    cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    dispose(this.static);
    dispose(this.dynamic);
    dispose(this.overlays);
    this.environment?.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }
  animate(time) {
    const start = performance.now();
    this.frame = requestAnimationFrame(this.animate);
    this.bodyLayer?.updateLod(this.camera, this.canvas.clientHeight);
    for (const model of this.reactors || [])
      if (!model.children[0].userData.assetModel)
        model.children[0].rotation.z = (this.simulationTime || 0) * 0.3;
    for (const food of this.food || [])
      food.rotation.z = this.simulationTime || 0;
    for (const base of this.bases || [])
      base.scale.setScalar(
        1 +
          Math.max(0, 1 - (time - (this.deliveryFlash || -10000)) / 500) * 0.15,
      );
    this.renderer.render(this.world, this.camera);
    const previous = this.performance || {};
    this.performance = {
      fps: this.lastFrame
        ? 0.9 * (previous.fps || 60) +
          (0.1 * 1000) / Math.max(1, time - this.lastFrame)
        : 0,
      cpuMs: performance.now() - start,
      calls: this.renderer.info.render.calls,
      triangles: this.renderer.info.render.triangles,
    };
    this.lastFrame = time;
  }
}
