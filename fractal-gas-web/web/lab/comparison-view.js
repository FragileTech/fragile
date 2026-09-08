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
      branch.title = title;
      title.textContent = `${branch.settings.label || branch.settings.algorithm} · ${branch.stats.success ? "success" : branch.stats.dead ? "terminal state" : "time limit"}`;
      branch.titleText = title.textContent;
      const summary = document.createElement("p");
      summary.textContent = `Planning ${(branch.stats.planningMs || 0).toFixed(1)} ms · ${branch.stats.simulatorFrames ?? "—"} world frames`;
      pane.append(summary);
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
    this.durations = this.branches.map((b) => {
      const tick = (i) =>
        new Uint32Array(
          b.motion.frame(i).state.buffer,
          b.motion.frame(i).state.byteOffset,
        )[0];
      b.startTick = tick(0);
      return (tick(b.motion.length - 1) - b.startTick) * b.motion.dt;
    });
    return Math.max(...this.durations);
  }
  seek(seconds) {
    this.branches?.forEach((b, i) => {
      let low = 0,
        high = b.motion.length - 1;
      const first = b.motion.frame(0);
      const start = new Uint32Array(
        first.state.buffer,
        first.state.byteOffset,
      )[0];
      const target = start + seconds / b.motion.dt;
      while (low < high) {
        const mid = Math.ceil((low + high) / 2),
          frame = b.motion.frame(mid);
        if (
          new Uint32Array(frame.state.buffer, frame.state.byteOffset)[0] <=
          target
        )
          low = mid;
        else high = mid - 1;
      }
      const frame = b.motion.frame(low);
      this.renderers[i].update(frame.state, frame.action);
      this.renderers[i].setAnimationPlayback({ playing: false, seek: true });
      b.title.textContent =
        b.titleText +
        (low === b.motion.length - 1
          ? " · Finished"
          : ` · ${seconds.toFixed(2)} s`);
    });
  }
  dispose() {
    for (const renderer of this.renderers) renderer.dispose();
    this.renderers = [];
    this.container.replaceChildren();
  }
}
