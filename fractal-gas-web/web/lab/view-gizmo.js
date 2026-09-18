import { gizmoAxes, lerpOrbit, orbitBasis, snapOrbit } from "./camera-orbit.js";

const SVG = "http://www.w3.org/2000/svg";
const REACH = 33;
const SNAP_MS = 220;
const YAW_STEP = (15 * Math.PI) / 180 / 0.008;
const PITCH_STEP = (15 * Math.PI) / 180 / 0.006;
const ARROWS = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};
const ICONS = {
  zoom: '<circle cx="10.5" cy="10.5" r="5.5"/><path d="M14.6 14.6 20 20M10.5 8v5M8 10.5h5"/>',
  pan: '<path d="M12 3v18M3 12h18M12 3 9.5 5.5M12 3l2.5 2.5M12 21l-2.5-2.5M12 21l2.5-2.5M3 12l2.5-2.5M3 12l2.5 2.5M21 12l-2.5-2.5M21 12l-2.5 2.5"/>',
};

const svgNode = (doc, name, attributes = {}) => {
  const node = doc.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, value);
  return node;
};

// Blender-style navigation overlay. It is plain DOM beside the canvas, so it
// adds no WebGL draw calls and never competes with the canvas camera gestures.
export class ViewGizmo {
  constructor(container, renderer) {
    const doc = container.ownerDocument;
    this.renderer = renderer;
    this.root = doc.createElement("div");
    this.root.className = "view-gizmo";
    this.root.setAttribute("role", "group");
    this.root.setAttribute("aria-label", "Camera navigation");

    this.orbit = svgNode(doc, "svg", {
      class: "view-gizmo-orbit",
      viewBox: "-50 -50 100 100",
      tabindex: "0",
      role: "img",
      "aria-label":
        "View orientation. Drag or use the arrow keys to rotate; click an axis to look along it.",
    });
    this.orbit.append(
      svgNode(doc, "circle", { class: "view-gizmo-disc", r: 48 }),
    );
    this.handles = new Map();
    for (const axis of "xyz")
      for (const sign of [1, -1]) {
        const group = svgNode(doc, "g", {
          class: `view-gizmo-axis axis-${axis} ${sign > 0 ? "positive" : "negative"}`,
          "data-axis": axis,
          "data-sign": sign,
        });
        const title = svgNode(doc, "title");
        const name = `${sign > 0 ? "" : "−"}${axis.toUpperCase()}`;
        title.textContent = `Look along ${name}`;
        const ball = svgNode(doc, "circle", { r: sign > 0 ? 9 : 6.5 });
        const label = svgNode(doc, "text", { dy: "0.35em" });
        label.textContent = name;
        const stem = sign > 0 ? svgNode(doc, "line", { x1: 0, y1: 0 }) : null;
        group.append(title, ...(stem ? [stem] : []), ball, label);
        this.orbit.append(group);
        this.handles.set(`${axis}${sign}`, { group, ball, label, stem });
      }

    const button = (kind, text) => {
      const node = doc.createElement("button");
      node.type = "button";
      node.className = `view-gizmo-button view-gizmo-${kind}`;
      node.title = text;
      node.setAttribute("aria-label", text);
      node.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[kind]}</svg>`;
      return node;
    };
    this.zoomButton = button(
      "zoom",
      "Zoom: drag up or down, or use the arrow keys",
    );
    this.panButton = button("pan", "Pan: drag, or use the arrow keys");
    this.root.append(this.orbit, this.zoomButton, this.panButton);
    container.append(this.root);

    this.input = new AbortController();
    const listen = (target, type, handler, options = {}) =>
      target.addEventListener(type, handler, {
        ...options,
        signal: this.input.signal,
      });
    listen(this.root, "contextmenu", (e) => e.preventDefault());
    listen(
      this.root,
      "wheel",
      (e) => {
        e.preventDefault();
        renderer.zoomBy(Math.exp(-e.deltaY * 0.001));
      },
      { passive: false },
    );
    this.drag(listen, this.orbit, {
      move: (dx, dy) => renderer.orbitBy(dx, dy),
      click: (target) => {
        const handle = target.closest?.("[data-axis]");
        if (handle && !handle.classList.contains("disabled"))
          this.snap(handle.dataset.axis, Number(handle.dataset.sign));
      },
    });
    this.drag(listen, this.zoomButton, {
      move: (dx, dy) => renderer.zoomBy(Math.exp(-dy * 0.01)),
    });
    this.drag(listen, this.panButton, {
      move: (dx, dy) => renderer.panByPixels(dx, dy),
    });
    const keys = (target, act) =>
      listen(target, "keydown", (e) => {
        const arrow = ARROWS[e.key];
        if (!arrow || e.altKey || e.ctrlKey || e.metaKey) return;
        e.preventDefault();
        act(...arrow);
      });
    keys(this.orbit, (x, y) => renderer.orbitBy(x * YAW_STEP, y * PITCH_STEP));
    keys(this.zoomButton, (x, y) => renderer.zoomBy(1.1 ** (x - y)));
    keys(this.panButton, (x, y) => renderer.panByPixels(-x * 40, -y * 40));
  }
  // Pointer drags report pixel deltas; a press released in place is a click.
  drag(listen, target, { move, click }) {
    let gesture = null;
    const end = (e) => {
      if (gesture?.id !== e.pointerId) return null;
      const ended = gesture;
      gesture = null;
      this.root.classList.remove("dragging");
      if (target.hasPointerCapture(e.pointerId))
        target.releasePointerCapture(e.pointerId);
      return ended;
    };
    listen(target, "pointerdown", (e) => {
      if (gesture || (e.button !== 0 && e.pointerType === "mouse")) return;
      cancelAnimationFrame(this.frame);
      gesture = {
        id: e.pointerId,
        x: e.clientX,
        y: e.clientY,
        dragging: false,
        target: e.target,
      };
      target.setPointerCapture(e.pointerId);
    });
    listen(target, "pointermove", (e) => {
      if (gesture?.id !== e.pointerId) return;
      const dx = e.clientX - gesture.x,
        dy = e.clientY - gesture.y;
      if (!gesture.dragging && Math.hypot(dx, dy) < 4) return;
      gesture.dragging = true;
      this.root.classList.add("dragging");
      gesture.x = e.clientX;
      gesture.y = e.clientY;
      move(dx, dy);
    });
    listen(target, "pointerup", (e) => {
      const ended = end(e);
      if (ended && !ended.dragging) click?.(ended.target);
    });
    listen(target, "pointercancel", end);
    listen(target, "lostpointercapture", end);
  }
  // Clicking the axis already in view turns to the opposite one, as Blender does.
  snap(axis, sign) {
    const r = this.renderer;
    const side = r.flightMode && !r.top;
    let target = snapOrbit(r.orbit, axis, sign, side);
    const facing = (orbit) => {
      const a = orbitBasis(orbit).toCamera,
        b = orbitBasis(r.orbit).toCamera;
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > 1 - 1e-6;
    };
    if (
      facing(target) &&
      gizmoAxes(r.orbit, side).find((h) => h.axis === axis && h.sign === -sign)
        .enabled
    )
      target = snapOrbit(r.orbit, axis, -sign, side);
    cancelAnimationFrame(this.frame);
    if (
      !r.animationsEnabled ||
      matchMedia("(prefers-reduced-motion: reduce)").matches
    )
      return r.setOrbit(target);
    const from = r.orbit,
      start = performance.now();
    let written = from;
    // Every other camera control replaces renderer.orbit, which ends the turn.
    const step = (now) => {
      if (r.disposed || r.cameraGesture || r.orbit !== written) return;
      const t = Math.min(1, Math.max(0, (now - start) / SNAP_MS));
      written = t < 1 ? lerpOrbit(from, target, t * (2 - t)) : target;
      r.setOrbit(written);
      if (t < 1) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }
  sync(orbit, side) {
    if (
      this.last &&
      this.last.yaw === orbit.yaw &&
      this.last.pitch === orbit.pitch &&
      this.last.side === side
    )
      return;
    this.last = { yaw: orbit.yaw, pitch: orbit.pitch, side };
    const axes = gizmoAxes(orbit, side).sort((a, b) => a.depth - b.depth);
    for (const { axis, sign, x, y, depth, enabled } of axes) {
      const { group, ball, label, stem } = this.handles.get(`${axis}${sign}`);
      const px = (x * REACH).toFixed(2),
        py = (-y * REACH).toFixed(2);
      ball.setAttribute("cx", px);
      ball.setAttribute("cy", py);
      label.setAttribute("x", px);
      label.setAttribute("y", py);
      stem?.setAttribute("x2", px);
      stem?.setAttribute("y2", py);
      group.style.opacity = enabled ? (0.8 + 0.2 * depth).toFixed(3) : "0.25";
      group.classList.toggle("disabled", !enabled);
    }
    const order = axes.map((a) => `${a.axis}${a.sign}`).join();
    if (order !== this.order) {
      this.order = order;
      for (const a of axes)
        this.orbit.append(this.handles.get(`${a.axis}${a.sign}`).group);
    }
  }
  dispose() {
    cancelAnimationFrame(this.frame);
    this.input.abort();
    this.root.remove();
  }
}
