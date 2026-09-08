import { VariantSettings, configurationDiff } from "./variant-settings.js";
import "./controllers/index.js";
import { controllerDefinitions } from "./controllers/registry.js";
import { ComparisonView } from "./comparison-view.js";
export class ExperimentPanel {
  constructor({ getScene, getSettings, getRoot, stop, download, error }) {
    const $ = (id) => document.getElementById(id);
    this.$ = $;
    this.download = download;
    this.report = null;
    this.comparison = new ComparisonView($("comparison-worlds"));
    for (const id of ["algorithm", "variant-a", "variant-b"])
      $(id).replaceChildren(
        ...controllerDefinitions().map((d) => new Option(d.label, d.id)),
      );
    $("variant-a").value = "fmc";
    $("variant-b").value = "cem";
    const differences = document.createElement("p");
    differences.id = "variant-differences";
    const duplicate = document.createElement("button");
    duplicate.textContent = "Duplicate A into B";
    $("experiment-dialog")
      .querySelector(".experiment-fields")
      .after(duplicate, differences);
    const changed = () => {
      differences.textContent =
        configurationDiff(this.variants[0].model, this.variants[1].model).join(
          " · ",
        ) || "Identical configurations. Change B to compare one parameter.";
    };
    this.variants = ["a", "b"].map(
      (key) =>
        new VariantSettings(
          $("variant-" + key),
          $("variant-settings-" + key),
          changed,
        ),
    );
    duplicate.onclick = () => {
      this.variants[1].set(this.variants[0].values());
      changed();
    };
    for (const id of [
      "benchmark-walkers",
      "benchmark-horizon",
      "benchmark-action-frames",
    ])
      $(id).closest("label").hidden = true;
    $("experiments").onclick = () => {
      stop();
      const scene = getScene();
      this.variants.forEach((v) => v.set(getSettings()));
      changed();
      if (scene !== this.lastScene) {
        const goal = scene.evaluation;
        if (
          goal &&
          Array.from($("benchmark-goal").options).some(
            (o) => o.value === goal.metric,
          )
        ) {
          $("benchmark-goal").value = goal.metric;
          $("benchmark-target").value = goal.target;
        }
        this.lastScene = scene;
      }
      $("experiment-dialog").showModal();
    };
    $("close-experiments").onclick = () => {
      $("experiment-dialog").close();
      this.comparison.dispose();
    };
    $("experiment-dialog").addEventListener("close", () => {
      this.worker?.terminate();
      this.busy(false);
      this.comparison.dispose();
      $("comparison-section").hidden = true;
    });
    const spec = () => ({
      seeds: $("benchmark-seeds")
        .value.split(",")
        .map((s) => Number(s.trim())),
      maxFrames: +$("benchmark-frames").value,
      goal: {
        metric: $("benchmark-goal").value,
        target: +$("benchmark-target").value,
      },
      variants: this.variants.map((v) => ({ ...v.values(), recording: 0 })),
    });
    const start = (type) => {
      try {
        const submitted = structuredClone({
          scene: getScene(),
          spec: spec(),
          root: type === "compare" ? getRoot() : undefined,
        });
        this.worker?.terminate();
        this.worker = new Worker(
          new URL("./experiment-worker.js", import.meta.url),
          { type: "module" },
        );
        this.worker.onmessage = ({ data }) => {
          if (data.type === "error") {
            error(data.message);
            $("experiment-status").textContent = data.message;
            this.busy(false);
          }
          if (data.type === "progress")
            $("experiment-status").textContent =
              `Completed ${data.index} / ${data.total}`;
          if (data.type === "benchmark") {
            this.report = data.report;
            this.table(data.report.summary);
            $("experiment-status").textContent = "Benchmark complete";
            this.busy(false);
          }
          if (data.type === "comparison") {
            this.report = {
              version: 1,
              scene: submitted.scene,
              spec: submitted.spec,
              branches: data.branches,
            };
            $("comparison-timeline").max = this.comparison.load(data.branches);
            $("comparison-timeline").step = "0.001";
            $("comparison-timeline").value = 0;
            $("comparison-section").hidden = false;
            $("experiment-status").textContent =
              "Same world, independent controller outcomes";
            this.busy(false);
          }
          if (data.type === "profile") {
            this.showProfile(data);
            $("experiment-status").textContent = "Batch probe complete";
            this.busy(false);
          }
        };
        this.worker.onerror = (e) => {
          error(e.message);
          this.busy(false);
        };
        this.busy(true);
        this.worker.postMessage({
          type,
          scenarios:
            type === "benchmark" && $("benchmark-all-scenes").checked
              ? Array.from($("scenario").options, (o) => o.value)
              : undefined,
          scene: submitted.scene,
          spec: submitted.spec,
          root: submitted.root,
          worlds: 256,
        });
      } catch (e) {
        error(e);
        this.busy(false);
      }
    };
    $("run-benchmark").onclick = () => start("benchmark");
    $("compare-branches").onclick = () => start("compare");
    $("profile-batch").onclick = () => start("profile");
    $("cancel-experiment").onclick = () => {
      this.worker?.postMessage({ type: "cancel" });
      this.worker?.terminate();
      this.busy(false);
      $("experiment-status").textContent = "Cancelled";
    };
    $("export-experiment").onclick = () => {
      if (this.report)
        download(JSON.stringify(this.report), `control-experiment.json`);
    };
    $("comparison-timeline").oninput = () =>
      this.comparison.seek(+$("comparison-timeline").value);
    $("comparison-play").onclick = () => {
      if (this.playTimer) {
        clearInterval(this.playTimer);
        this.playTimer = null;
        $("comparison-play").textContent = "Play both";
        return;
      }
      if (+$("comparison-timeline").value >= +$("comparison-timeline").max)
        $("comparison-timeline").value = 0;
      $("comparison-play").textContent = "Pause both";
      let previous = performance.now();
      this.playTimer = setInterval(() => {
        const now = performance.now();
        const value = +$("comparison-timeline").value + (now - previous) / 1000;
        previous = now;
        $("comparison-timeline").value = value;
        this.comparison.seek(Math.min(value, +$("comparison-timeline").max));
        if (value >= +$("comparison-timeline").max)
          $("comparison-play").click();
      }, 1000 / 60);
    };
    $("experiment-dialog").addEventListener("close", () => {
      clearInterval(this.playTimer);
      this.playTimer = null;
    });
  }
  busy(value) {
    for (const id of ["run-benchmark", "compare-branches", "profile-batch"])
      this.$(id).disabled = value;
    this.$("cancel-experiment").disabled = !value;
    for (const input of document.querySelectorAll(
      "#experiment-dialog input,#experiment-dialog select,#variant-settings-a,#variant-settings-b",
    ))
      input.disabled = value;
    this.$("variant-differences").previousElementSibling.disabled = value;
    if (value)
      this.$("experiment-status").textContent = "Running in a separate worker…";
  }
  table(rows) {
    const body = this.$("benchmark-results");
    body.replaceChildren();
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const value of [
        row.settings.algorithm,
        row.episodes,
        `${(row.successRate * 100).toFixed(1)}%`,
        row.collisions.toFixed(1),
        row.completionSeconds?.toFixed(2) ?? "—",
        row.controlEffort.toFixed(2),
        row.planningMs.toFixed(1),
        row.simulatorFrames?.toFixed(0) ?? "—",
      ]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      body.append(tr);
    }
  }
  showProfile(data) {
    const p = data.profile;
    this.$("batch-profile").textContent =
      `${data.worlds} worlds · ${((p[1] / Math.max(p[0], 0.001)) * 1000).toFixed(0)} world frames/s · get ${(p[3] / Math.max(p[2], 0.001) / 1e6).toFixed(2)} GB/s · set ${(p[5] / Math.max(p[4], 0.001) / 1e6).toFixed(2)} GB/s · gather ${(p[7] / Math.max(p[6], 0.001) / 1e6).toFixed(2)} GB/s · ${(data.wasmBytes / 1048576).toFixed(1)} MiB WASM memory`;
  }
}
