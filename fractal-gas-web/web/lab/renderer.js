import { harvestPresentation } from "./harvest-hooks.js";
import { treePoseDim, treeWidth } from "./actions.js";
import { laboratoryEnvironment } from "./visuals/lighting.js";
import { createEnvironment } from "./visuals/environments/index.js";
import { BodyLayer } from "./visuals/body-layer.js";
import * as T from "./vendor/three.module.js";
import { palette, prism, zoneModel, reactorModel } from "./models.js";
import { assetModel, preloadStyle } from "./visuals/assets.js";
import { disposeGroup as dispose } from "./visuals/resources.js";
import { stylePalette } from "./visuals/style-palette.js";
import { WorldDynamics, animateWorld } from "./visuals/world.js";
import { labStyle } from "./visual-style.js";
import { flightMode } from "./agent-types.js";
import { labAnimations } from "./animations.js";
import { AnimationClock } from "./animation-clock.js";
import { labActionGuides } from "./action-guides.js";
import { ActionEffects } from "./visuals/action-effects.js";
import { CargoReadout } from "./visuals/cargo-readout.js";
import { CheckpointPresentation } from "./visuals/checkpoints.js";
import { arenaBounds, arenaHalfSpan } from "./camera-fit.js";
import {
  FormationOverlay,
  FORMATION_GRADIENT,
} from "./visuals/formation-overlay.js";
import { dragOrbit, orbitPosition, presetOrbit } from "./camera-orbit.js";

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
  constructor(canvas, { isEditing = () => false } = {}) {
    this.canvas = canvas;
    this.checkpointMotion = canvas.ownerDocument.defaultView.matchMedia(
      "(prefers-reduced-motion: reduce)",
    );
    this.checkpointMotionChanged = () =>
      this.checkpoints?.setEnabled(
        this.animationsEnabled && !this.checkpointMotion.matches,
      );
    this.checkpointMotion.addEventListener(
      "change",
      this.checkpointMotionChanged,
    );
    this.formationLegend =
      canvas.ownerDocument.getElementById("formation-legend");
    if (this.formationLegend)
      this.formationLegend.querySelector(
        ".formation-quality-ramp",
      ).style.background = FORMATION_GRADIENT;
    this.actionGuidesEnabled = labActionGuides.enabled;
    this.unsubscribeActionGuides = labActionGuides.subscribe((enabled) => {
      this.actionGuidesEnabled = enabled;
      this.refreshActionGuides();
    });
    this.actionReadout = document.createElement("output");
    this.actionReadout.className = "action-guide-readout";
    this.actionReadout.hidden = true;
    canvas.parentElement.append(this.actionReadout);
    this.cargoReadout = new CargoReadout(canvas);
    this.animationClock = new AnimationClock();
    this.animationStep = { playing: false, speed: 1 };
    this.animationsEnabled = labAnimations.enabled;
    this.unsubscribeAnimations = labAnimations.subscribe((enabled) =>
      this.setAnimationsEnabled(enabled),
    );
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
    this.coordinateFrame = new T.Group();
    this.coordinateFrame.add(this.static, this.dynamic, this.overlays);
    this.world.add(this.coordinateFrame);
    this.layers = {
      tree: true,
      cloud: true,
      geometry: false,
      tethers: true,
    };
    this.zoom = 1;
    this.top = false;
    this.flightMode = false;
    this.orbit = presetOrbit(false, false);
    this.arena = arenaBounds({ size: [64, 44] });
    this.viewCenter = [32, 22];
    this.ray = new T.Raycaster();
    this.plane = new T.Plane(new T.Vector3(0, 0, 1), 0);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.inputController = new AbortController();
    const listen = (type, handler, options = {}) =>
      canvas.addEventListener(type, handler, {
        ...options,
        signal: this.inputController.signal,
      });
    listen(
      "wheel",
      (e) => {
        e.preventDefault();
        this.clearCameraGesture();
        this.zoom = T.MathUtils.clamp(
          this.zoom * Math.exp(-e.deltaY * 0.001),
          0.6,
          12,
        );
        this.resize();
      },
      { passive: false },
    );
    listen("contextmenu", (e) => e.preventDefault());
    listen("pointerdown", (e) => {
      if (this.cameraGesture) {
        e.stopImmediatePropagation();
        return;
      }
      const rotating = e.button === 2 && e.pointerType !== "touch";
      const shortcut = e.button === 1 || (e.button === 0 && e.altKey);
      if (!rotating && !shortcut && (e.button !== 0 || isEditing())) return;
      const anchor = rotating ? null : this.worldPoint(e);
      if (!rotating && !anchor) return;
      this.cameraGesture = {
        id: e.pointerId,
        kind: rotating ? "rotate" : "pan",
        anchor,
        x: e.clientX,
        y: e.clientY,
        dragging: false,
        click: !shortcut && !rotating,
      };
      canvas.setPointerCapture(e.pointerId);
      if (shortcut || rotating) e.preventDefault();
      e.stopImmediatePropagation();
    });
    const move = (e) => {
      const gesture = this.cameraGesture;
      if (!gesture || gesture.id !== e.pointerId) return;
      if (
        !gesture.dragging &&
        Math.hypot(e.clientX - gesture.x, e.clientY - gesture.y) >= 4
      ) {
        gesture.dragging = true;
        if (gesture.kind === "pan") this.followBody = null;
        canvas.classList.add(gesture.kind === "pan" ? "panning" : "rotating");
        canvas.dispatchEvent(new Event("camerachange"));
      }
      if (gesture.dragging && gesture.kind === "rotate") {
        this.orbit = dragOrbit(
          this.orbit,
          e.clientX - gesture.x,
          e.clientY - gesture.y,
          this.flightMode && !this.top,
        );
        gesture.x = e.clientX;
        gesture.y = e.clientY;
        this.updateCamera();
      } else if (gesture.dragging) {
        const point = this.worldPoint(e);
        if (point) {
          this.viewCenter = this.viewCenter.map(
            (v, i) => v + gesture.anchor[i] - point[i],
          );
          this.updateCamera();
        }
      }
      e.stopImmediatePropagation();
    };
    listen("pointermove", move);
    listen("pointerup", (e) => {
      const gesture = this.cameraGesture;
      if (!gesture || gesture.id !== e.pointerId) return;
      move(e);
      const point =
        !gesture.dragging && gesture.click && !isEditing()
          ? this.worldPoint(e)
          : null;
      this.clearCameraGesture();
      if (point)
        canvas.dispatchEvent(new CustomEvent("worldclick", { detail: point }));
    });
    for (const type of ["pointercancel", "lostpointercapture"])
      listen(type, (e) => {
        if (this.cameraGesture?.id === e.pointerId) this.clearCameraGesture();
      });
    this.animate = this.animate.bind(this);
    this.frame = requestAnimationFrame(this.animate);
  }
  clearCameraGesture() {
    const gesture = this.cameraGesture;
    this.cameraGesture = null;
    this.canvas.classList.remove("panning", "rotating");
    if (gesture && this.canvas.hasPointerCapture(gesture.id))
      this.canvas.releasePointerCapture(gesture.id);
  }
  resize() {
    const { width, height } = this.canvas.parentElement.getBoundingClientRect();
    if (!height) return;
    this.renderer.setSize(width, height, false);
    const span = arenaHalfSpan(this.arena, width / height) / this.zoom;
    this.camera.left = (-span * width) / height;
    this.camera.right = (span * width) / height;
    this.camera.top = span;
    this.camera.bottom = -span;
    this.cargoReadout.resize(width, height);
    this.updateCamera();
  }
  setViewPreset(top) {
    this.clearCameraGesture();
    this.top = top;
    this.orbit = presetOrbit(top, this.flightMode && !top);
    this.updateCamera();
  }
  updateCamera() {
    const side = this.flightMode && !this.top;
    this.coordinateFrame.rotation.x = side ? Math.PI / 2 : 0;
    this.plane.normal.set(0, side ? 1 : 0, side ? 0 : 1);
    this.plane.constant = 0;
    const pose = orbitPosition(this.orbit, this.arena, this.viewCenter, side);
    this.camera.position.set(...pose.position);
    this.camera.lookAt(...pose.target);
    this.camera.far = pose.far;
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);
    this.refreshCargoReadout();
  }
  worldPoint(event) {
    this.camera.updateMatrixWorld(true);
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    this.ray.setFromCamera(
      new T.Vector2(
        ((event.clientX - r.left) / r.width) * 2 - 1,
        1 - ((event.clientY - r.top) / r.height) * 2,
      ),
      this.camera,
    );
    if (
      Math.abs(this.ray.ray.direction.dot(this.plane.normal)) <
      Math.sin((5 * Math.PI) / 180) - 1e-12
    )
      return null;
    const p = this.ray.ray.intersectPlane(this.plane, new T.Vector3());
    return p ? [p.x, this.flightMode && !this.top ? p.z : p.y] : null;
  }
  setAnimationsEnabled(enabled) {
    this.animationsEnabled = !!enabled;
    this.checkpointMotionChanged();
    this.animationClock.reset(this.simulationTime || 0);
    this.bodyLayer?.setAnimationsEnabled(this.animationsEnabled);
    this.worldDynamics?.setAnimationsEnabled(this.animationsEnabled);
    if (this.state) this.update(this.state, this.action);
    if (!this.animationsEnabled) {
      for (const model of this.reactors || []) {
        animateWorld(model, 0, { enabled: false });
        if (!model.children[0].userData.assetModel)
          model.children[0].rotation.z = 0;
      }
    }
  }
  setAnimationPlayback({ seek = false, ...playback } = {}) {
    this.animationClock.setPlayback(playback);
    if (seek) {
      this.animationClock.reset(this.simulationTime || 0);
      this.bodyLayer?.resetAnimation();
      this.worldDynamics?.cargo.resetTransitions();
      this.refreshCargoReadout();
      this.animationPulseUntil = 0;
    }
  }
  setActionGuideBody(index) {
    this.actionGuideBody = index;
    this.refreshActionGuides();
    this.refreshCargoReadout();
  }
  refreshCargoReadout() {
    this.cargoReadout.update(
      this.worldDynamics?.cargo.entries,
      this.bodyLayer,
      this.camera,
      this.coordinateFrame,
      this.actionGuideBody,
      this.followBody,
      this.style,
    );
  }
  refreshActionGuides() {
    if (!this.actionEffects) return;
    const text = this.actionEffects.updateGuides(
      this.actionGuideBody,
      this.actionGuidesEnabled,
    );
    if (text !== this.actionReadout.textContent)
      this.actionReadout.textContent = text;
    this.actionReadout.hidden = !text;
  }
  pulseAnimation() {
    this.animationPulseUntil = performance.now() + 120;
  }
  load(scene, info, channels) {
    scene = harvestPresentation(scene, info);
    this.clearCameraGesture();
    this.simulationTime = 0;
    this.animationPulseUntil = 0;
    this.animationClock.reset();
    this.formationOverlay?.dispose();
    this.checkpoints?.dispose();
    dispose(this.static);
    dispose(this.dynamic);
    dispose(this.overlays);
    this.config = scene;
    this.info = info;
    this.channels = channels;
    this.state = null;
    this.action = null;
    this.actionGuideBody = undefined;
    this.actionReadout.hidden = true;
    this.cargoReadout.clear();
    this.flightMode = flightMode(scene);
    this.top = false;
    this.orbit = presetOrbit(false, this.flightMode);
    this.size = scene.size || [64, 44];
    this.arena = arenaBounds(scene);
    this.viewCenter = [...this.arena.center];
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
    this.checkpoints = new CheckpointPresentation(scene, info, this.canvas);
    this.coordinateFrame.add(this.checkpoints.group);
    this.checkpointMotionChanged();
    this.tethers = new T.Group();
    this.formationOverlay = new FormationOverlay(
      scene,
      this.controlled,
      info[1],
    );
    this.overlays.add(this.tethers, this.formationOverlay.group);
    this.worldDynamics = new WorldDynamics(
      scene,
      info,
      this.style,
      this.bodyLayer,
      this.bodyGroup,
      this.tethers,
    );
    this.actionEffects = new ActionEffects(
      this.bodyLayer,
      this.bodyGroup,
      this.style,
    );
    this.food = this.worldDynamics.pickups;
    this.setAnimationsEnabled(this.animationsEnabled);
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
      for (const [i, def] of (kind === "gates" &&
      (scenery.replacesGates || (scene.task === "tandem" && this.info?.[2]))
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
        const retained =
          kind === "bases" &&
          scene.task === "harvest" &&
          scene.keep_delivered_rocks;
        if (retained) {
          for (const [name, radius, ringColor, dashed] of [
            ["Delivery", (def.radius ?? 1) * 0.5, palette.green, false],
            ["Release", def.radius ?? 1, palette.gold, true],
          ]) {
            const ring = line(
              Array.from({ length: 97 }, (_, k) => {
                const angle = (k * Math.PI * 2) / 96;
                return [
                  Math.cos(angle) * radius,
                  Math.sin(angle) * radius,
                  0.25,
                ];
              }),
              ringColor,
              dashed,
            );
            ring.name = `${name} boundary`;
            ring.userData.radius = radius;
            ring.material.depthTest = false;
            ring.renderOrder = 10;
            ring.position.set(...def.position, 0);
            group.add(ring);
            this.label(
              name.toUpperCase(),
              [def.position[0], def.position[1] + radius + 0.25, 0.3],
              ringColor,
              2.2,
              group,
            );
          }
        }
        this.label(
          kind === "bases"
            ? retained
              ? "DROP / RELEASE"
              : "RECOVERY / 01"
            : `GATE / ${String(i + 1).padStart(2, "0")}`,
          [...def.position, 0.3],
          style === "steampunk" ? stylePalette[style].accent : color,
          4.2,
          group,
        );
      }
    for (const def of scene.refineries || []) {
      const refinery = new T.Group();
      const scale = (def.radius || 6) / 6;
      for (const lod of ["high", "low"]) {
        const model = assetModel(style, "refinery", lod);
        if (model) {
          model.name = `refinery-${lod}`;
          model.visible = lod === "low";
          refinery.add(model);
        }
      }
      const pad = zoneModel(6, stylePalette[style].energy);
      pad.position.z = 0.1;
      refinery.add(pad);
      refinery.scale.setScalar(scale);
      refinery.position.set(...def.position, 0);
      refinery.userData.refinery = true;
      group.add(refinery);
      this.label(
        "REFINERY / UNLOAD",
        [def.position[0], def.position[1] - def.radius - 0.8, 0.2],
        stylePalette[style].energy,
        5,
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
    const tetherGroup = new T.Group();
    let presentation, bodyLayer, worldDynamics, actionEffects;
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
        worldDynamics = new WorldDynamics(
          this.config,
          this.info,
          style,
          bodyLayer,
          bodyGroup,
          tetherGroup,
        );
        actionEffects = new ActionEffects(bodyLayer, bodyGroup, style);
        bodyLayer.setAnimationsEnabled(this.animationsEnabled);
        worldDynamics.setAnimationsEnabled(this.animationsEnabled);
        if (this.state) {
          bodyLayer.update(this.state, this.action);
          worldDynamics.update(this.state, this.action);
        }
      }
    } catch (error) {
      environment.dispose();
      if (presentation) dispose(presentation.group);
      dispose(bodyGroup);
      dispose(tetherGroup);
      throw error;
    }
    return {
      cancel: () => {
        environment.dispose();
        if (presentation) dispose(presentation.group);
        dispose(bodyGroup);
        dispose(tetherGroup);
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
        dispose(this.tethers);
        this.tethers.add(tetherGroup);
        this.worldDynamics = worldDynamics;
        this.actionEffects = actionEffects;
        this.food = worldDynamics.pickups;
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
  update(state, action, { discontinuity = false } = {}) {
    if (!this.info) return;
    this.state = state;
    this.checkpoints?.update(state, { discontinuity });
    this.checkpoints?.project(this.camera);
    this.action = action;
    const bits = new Uint32Array(state.buffer, state.byteOffset, state.length);
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
      this.updateCamera();
    }
    this.worldDynamics.update(state, action);
    this.worldDynamics.pickupBatch.updateLod(
      this.camera,
      this.canvas.clientHeight,
    );
    this.bodyLayer.updateLod(this.camera, this.canvas.clientHeight);
    this.actionEffects?.update();
    this.refreshActionGuides();
    this.refreshCargoReadout();
    this.static.traverse((object) => {
      if (!object.userData.refinery) return;
      const pixels =
        ((12 * object.scale.x * this.canvas.clientHeight) /
          (this.camera.top - this.camera.bottom)) *
        this.camera.zoom;
      const high = object.getObjectByName("refinery-high"),
        low = object.getObjectByName("refinery-low");
      if (high && low) {
        if (pixels > 120) high.visible = true;
        else if (pixels < 90) high.visible = false;
        low.visible = !high.visible;
      }
    });
    this.scenery.updateLod?.(this.camera, this.canvas.clientHeight);
    for (const reactor of this.reactors)
      animateWorld(reactor, this.simulationTime, {
        enabled: this.animationsEnabled,
      });
    this.formationOverlay.update(state);
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
    this.formationOverlay?.setVisible(this.layers.tethers);
    if (this.formationLegend)
      this.formationLegend.hidden = !this.formationOverlay?.group.visible;
  }
  clearDraft() {
    if (this.draftGroup) {
      dispose(this.draftGroup);
      this.overlays.remove(this.draftGroup);
      this.draftGroup = null;
    }
  }
  showDraft(scene) {
    this.clearDraft();
    this.draftGroup = new T.Group();
    for (const body of scene.bodies || []) {
      if (!body.position) continue;
      const marker = zoneModel(body.radius || 0.8, 0xd0a9e2);
      marker.position.set(...body.position, 0.2);
      this.draftGroup.add(marker);
    }
    this.overlays.add(this.draftGroup);
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
    if (body == null) return this.resetView();
    this.clearCameraGesture();
    this.followBody = body;
    this.viewCenter = [
      this.models[body].position.x,
      this.models[body].position.y,
    ];
    this.zoom = 7;
    this.resize();
    this.canvas.dispatchEvent(new Event("camerachange"));
  }
  resetView() {
    this.clearCameraGesture();
    this.followBody = null;
    this.viewCenter = [...this.arena.center];
    this.zoom = 1;
    this.top = false;
    this.orbit = presetOrbit(false, this.flightMode);
    this.resize();
    this.canvas.dispatchEvent(new Event("camerachange"));
  }
  dispose() {
    this.disposed = true;
    this.clearCameraGesture();
    this.inputController.abort();
    this.unsubscribeStyle();
    this.unsubscribeAnimations();
    this.unsubscribeActionGuides();
    this.actionReadout.remove();
    this.cargoReadout.dispose();
    this.checkpoints?.dispose();
    this.checkpointMotion.removeEventListener(
      "change",
      this.checkpointMotionChanged,
    );
    this.formationOverlay?.dispose();
    if (this.formationLegend) this.formationLegend.hidden = true;
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
    const visible =
      !document.hidden && !this.renderer.getContext().isContextLost();
    const motion = this.animationClock.tick(
      time,
      visible && this.animationsEnabled,
    );
    if (!visible) return;
    this.checkpoints?.paint(time);
    this.checkpoints?.project(this.camera);
    this.bodyLayer?.updateLod(this.camera, this.canvas.clientHeight);
    this.scenery?.updateLod?.(this.camera, this.canvas.clientHeight);
    this.worldDynamics?.pickupBatch.updateLod(
      this.camera,
      this.canvas.clientHeight,
    );
    let animationCpuMs = 0;
    if (this.animationsEnabled) {
      const animationStart = performance.now();
      this.animationStep.playing =
        motion.playing || time < (this.animationPulseUntil || 0);
      this.animationStep.speed = motion.speed;
      this.bodyLayer?.animate(motion.dt, motion.idleTime, this.animationStep);
      this.worldDynamics?.animate(motion.idleTime, this.animationStep);
      for (const model of this.reactors || []) {
        if (!model.children[0].userData.assetModel)
          model.children[0].rotation.z = motion.idleTime * 0.3;
        else animateWorld(model, motion.idleTime);
      }
      animationCpuMs = performance.now() - animationStart;
    }
    // Static action cues also follow authoritative state with animation off.
    // Versioning skips uploads when neither pose nor visibility changed.
    this.actionEffects?.update();
    if (this.actionReadout && this.actionEffects)
      this.actionReadout.hidden = !this.actionEffects.guides.visible;
    this.worldDynamics?.cargo.animate();
    this.refreshCargoReadout();
    this.renderer.render(this.world, this.camera);
    const previous = this.performance || {};
    this.performance = {
      fps: this.lastFrame
        ? 0.9 * (previous.fps || 60) +
          (0.1 * 1000) / Math.max(1, time - this.lastFrame)
        : 0,
      cpuMs: performance.now() - start,
      animationCpuMs,
      calls: this.renderer.info.render.calls,
      triangles: this.renderer.info.render.triangles,
    };
    this.lastFrame = time;
  }
}
