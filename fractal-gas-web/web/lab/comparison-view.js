import { LabRenderer } from "./renderer.js";
import { importRecording } from "./archive.js";
export class ComparisonView {
  constructor(container) {
    this.container = container;
    this.renderers = [];
  }
  load(branches) {
    this.dispose();
    this.branches = branches.map((b) => ({
      ...b,
      ...importRecording(b.archive),
    }));
    for (const branch of this.branches) {
      const pane = document.createElement("section"),
        title = document.createElement("h3"),
        view = document.createElement("div"),
        canvas = document.createElement("canvas");
      title.textContent = `${branch.settings.label || branch.settings.algorithm} · ${branch.stats.success ? "success" : branch.stats.dead ? "collision death" : "time limit"}`;
      view.className = "comparison-world";
      view.append(canvas);
      pane.append(title, view);
      this.container.append(pane);
      const renderer = new LabRenderer(canvas);
      renderer.load(branch.scene, branch.motion.info, branch.motion.channels);
      renderer.setLayers({
        tree: false,
        cloud: false,
        geometry: false,
        tethers: true,
      });
      this.renderers.push(renderer);
    }
    this.seek(0);
    return Math.max(...this.branches.map((b) => b.motion.length)) - 1;
  }
  seek(index) {
    this.branches?.forEach((b, i) => {
      const frame = b.motion.frame(Math.min(index, b.motion.length - 1));
      this.renderers[i].update(frame.state, frame.action);
      this.renderers[i].setAnimationPlayback({ playing: false, seek: true });
    });
  }
  dispose() {
    for (const renderer of this.renderers) renderer.dispose();
    this.renderers = [];
    this.container.replaceChildren();
  }
}
