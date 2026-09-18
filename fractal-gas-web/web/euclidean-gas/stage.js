import { PopulationRenderer } from "./renderer.js";
import { SwarmRenderer3D } from "./renderer3d.js";

/**
 * Shared stage renderer interface (PopulationRenderer, SwarmRenderer3D):
 * @typedef {object} StageRenderer
 * @property {(domain: {low:number, high:number, dimensions:number, minimum:number|null}) => void} setDomain
 * @property {(frame: object, settings: object, selected: number, trails?: object) => void} update
 * @property {(data: object|null, settings: object) => void} setSurface
 * @property {() => void} fit
 * @property {() => void} resetCamera
 * @property {() => void} resize
 * @property {() => {distinctColors:number, nonBackgroundFraction:number}} pixelStats
 * @property {() => void} dispose
 */

// One live WebGL stage at a time; cached state is replayed after a switch.
export class StageController {
  constructor(container, onSelect) {
    this.container = container;
    this.onSelect = onSelect;
    this.renderer = null;
    this.kind = null;
    this.domain = null;
    this.frame = null;
    this.settings = null;
    this.selected = 0;
    this.trails = null;
    this.surface = null;
  }
  ensure(view) {
    const kind = view === "2d" ? "2d" : "3d";
    if (kind === this.kind) return false;
    this.renderer?.dispose();
    this.kind = kind;
    this.renderer =
      kind === "2d"
        ? new PopulationRenderer(this.container, this.onSelect)
        : new SwarmRenderer3D(this.container, this.onSelect);
    if (this.domain) this.renderer.setDomain(this.domain);
    return true;
  }
  setDomain(domain) {
    this.domain = domain;
    this.renderer?.setDomain(domain);
  }
  update(frame, settings, selected, trails = null) {
    this.frame = frame;
    this.settings = settings;
    this.selected = selected;
    this.trails = trails;
    const switched = this.ensure(settings.view);
    if (switched && this.surface)
      this.renderer.setSurface(this.surface, settings);
    this.renderer.update(frame, settings, selected, trails);
    if (switched) this.home();
  }
  setSurface(data, settings) {
    this.surface = data;
    this.settings = settings;
    this.ensure(settings.view);
    this.renderer.setSurface(data, settings);
  }
  // Initial framing: population fit in 2D, whole domain in 3D.
  home() {
    if (this.kind === "2d") this.renderer.fit();
    else this.renderer?.resetCamera();
  }
  fit() {
    this.renderer?.fit();
  }
  resetCamera() {
    this.renderer?.resetCamera();
  }
  resize() {
    this.renderer?.resize();
  }
  pixelStats() {
    return this.renderer?.pixelStats() ?? null;
  }
  get renderMs() {
    return this.renderer?.renderMs ?? null;
  }
  get linkInfo() {
    return this.renderer?.linkInfo ?? null;
  }
  dispose() {
    this.renderer?.dispose();
    this.renderer = null;
    this.kind = null;
  }
}
