import { alive } from "./config.js";
import { bestIndex } from "./geometry3d.js";

const VIEWS = ["2d", "spatial", "landscape"];
// View controls, slices and landscape sampling. `state` exposes the live
// config, frame, selected walker and benchmarkInfo(); display changes never
// touch the run.
export function createView({ $, client, stage, state, redraw, fail }) {
  let slice = new Float64Array(0),
    timer = null,
    settle = null,
    revision = 0;
  const view = {
    applied: 0,
    get slice() {
      return slice;
    },
    settings() {
      const config = state.config,
        boundary = config?.gas.boundary.kind;
      return {
        view: VIEWS.includes($("view").value) ? $("view").value : "2d",
        axes: ["x-axis", "y-axis", "z-axis"].map((id) => Number($(id).value)),
        color: $("color-mode").value,
        pointSize: Number($("point-size").value),
        surfaceOpacity: Number($("surface-opacity").value),
        heightScale: Number($("height-scale").value),
        links: $("links").value,
        trails: $("trails").checked,
        showSurface: $("landscape-toggle").checked,
        slice,
        resolution: Number($("resolution").value),
        direction: config?.gas.fitness.direction ?? "minimize",
        includeTruncated: !!config?.gas.include_truncated,
        periodic: boundary === "periodic_box",
      };
    },
    // Rebuilds the axis selects and the default slice for a new run.
    configureAxes() {
      const d = state.config.dimensions;
      for (const [id, axis] of [
        ["x-axis", 0],
        ["y-axis", 1],
        ["z-axis", 2],
      ]) {
        const select = $(id);
        select.replaceChildren();
        for (let k = 0; k < d; k++) select.add(new Option(`x${k + 1}`, k));
        if (axis === 2 && d < 3) select.add(new Option("Plane", d));
        select.value = Math.min(axis, axis === 2 ? d : d - 1);
      }
      view.resetSlice();
    },
    resetSlice() {
      const { config, info, frame } = state,
        d = config.dimensions;
      slice = new Float64Array(d);
      // Slice through the known minimiser (shift included) when the engine reports one.
      for (let k = 0; k < d; k++)
        slice[k] = Math.max(
          info.low,
          Math.min(
            info.high,
            info.minimizer?.[k] ?? config.reward_shift?.[k] ?? 0,
          ),
        );
      // Every atom coincides at the origin: slice through the best walker.
      if (info.molecule && frame) view.sliceFrom(view.best(), false);
      view.sliceControls();
    },
    best() {
      const p = state.frame.population,
        include = !!state.config.gas.include_truncated;
      return bestIndex(
        p.rewards.raw,
        p.validity.map((v) => alive(v, include)),
        state.config.gas.fitness.direction,
      );
    },
    sliceFrom(index, refresh = true) {
      const field = state.frame?.population.observations.fields.positions;
      if (!field || index < 0 || index >= field.rows) return;
      const d = slice.length;
      for (let k = 0; k < d; k++) {
        const x = field.values[index * d + k];
        if (Number.isFinite(x))
          slice[k] = Math.max(state.info.low, Math.min(state.info.high, x));
      }
      if (refresh) {
        view.sliceControls();
        view.refreshSurface();
      }
    },
    sliceControls() {
      const [x, y] = view.settings().axes,
        { low, high } = state.info,
        panel = $("slice-controls");
      panel.replaceChildren();
      for (let k = 0; k < slice.length; k++) {
        if (k === x || k === y) continue;
        const label = document.createElement("label"),
          input = document.createElement("input"),
          output = document.createElement("output");
        input.type = "range";
        input.id = `slice-${k}`;
        input.min = low;
        input.max = high;
        input.step = "any";
        input.value = slice[k];
        output.htmlFor = input.id;
        output.textContent = Number(slice[k]).toPrecision(4);
        label.append(`Slice x${k + 1} `, output, input);
        input.addEventListener("input", () => {
          slice[k] = Number(input.value);
          output.textContent = slice[k].toPrecision(4);
          view.refreshSurface();
        });
        panel.append(label);
      }
      $("slice-empty").hidden = panel.childElementCount > 0;
    },
    // Keeps x, y and z distinct after one of them changed.
    distinctAxes(changed) {
      const d = state.config.dimensions,
        value = (id) => Number($(id).value);
      if (value("x-axis") === value("y-axis"))
        $(changed === "x-axis" ? "y-axis" : "x-axis").value =
          (value(changed) + 1) % d;
      const taken = [value("x-axis"), value("y-axis")];
      if (taken.includes(value("z-axis"))) {
        let free = d;
        for (let k = 0; k < d; k++)
          if (!taken.includes(k)) {
            free = k;
            break;
          }
        $("z-axis").value = free;
      }
    },
    // Debounced landscape query; stale answers are dropped by revision.
    refreshSurface(immediate = false) {
      clearTimeout(timer);
      settle?.();
      const current = ++revision;
      return new Promise((resolve) => {
        settle = resolve;
        timer = setTimeout(
          async () => {
            try {
              if (!state.config || !state.frame) return;
              const settings = view.settings();
              const data = await client.request("landscape", {
                x: settings.axes[0],
                y: settings.axes[1],
                resolution: settings.view === "2d" ? 72 : settings.resolution,
                center: Array.from(slice),
              });
              if (current !== revision) return;
              stage.setSurface(data, view.settings());
              view.applied++;
              redraw();
            } catch (error) {
              fail(error);
            } finally {
              resolve();
            }
          },
          immediate ? 0 : 80,
        );
      });
    },
  };
  return view;
}
