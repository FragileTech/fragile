import { LabRenderer } from "../renderer.js";
import { loadNative, NativeEngine } from "../native.js";
import { configureAntsScene } from "../ants-scene.js";
import { labStyle } from "../visual-style.js";
const output = document.querySelector("#result"),
  button = document.querySelector("#run");
const renderer = new LabRenderer(document.querySelector("#world"));
let engine;
button.onclick = async () => {
  button.disabled = true;
  const rows = [];
  try {
    const module = await loadNative(false),
      template = await (await fetch("../scenarios/ants.json")).json();
    for (const style of ["steampunk", "futuristic"]) {
      await labStyle.change(style);
      for (const type of ["harvester", "drone"])
        for (const count of [1, 48, 128]) {
          engine?.dispose();
          const scene = configureAntsScene(template, {
            agentType: type,
            count,
          });
          engine = new NativeEngine(module, scene);
          renderer.load(scene, engine.info, engine.channels);
          const state = engine.states();
          for (let c = 0; c < count; c++) {
            state[engine.info[15] + c * 4] = 5;
            state[engine.info[15] + c * 4 + 1] = 1;
            state[engine.info[15] + c * 4 + 3] = 1;
          }
          state[8] = 12;
          state[8 + count] = 36;
          engine.restoreRows(state);
          engine.step(engine.neutralAction(), 60);
          const middle = engine.states(),
            before = engine.snapshot();
          renderer.update(middle, engine.neutralAction());
          await new Promise(requestAnimationFrame);
          await new Promise(requestAnimationFrame);
          const pose =
            renderer.worldDynamics.cargo.fill.instanceMatrix.array.slice();
          engine.step(engine.neutralAction(), 30);
          renderer.update(engine.states(), engine.neutralAction());
          engine.restore(before);
          renderer.update(engine.states(), engine.neutralAction());
          if (
            !pose.every(
              (x, i) =>
                x === renderer.worldDynamics.cargo.fill.instanceMatrix.array[i],
            )
          )
            throw new Error("Cargo replay pose mismatch");
          if (!renderer.static.getObjectByName("refinery-high"))
            throw new Error("Refinery missing");
          rows.push(
            `${style} ${count} ${type}: cargo pose replay passed; ${renderer.performance.calls} draws`,
          );
          output.textContent = rows.join("\n");
        }
    }
    output.textContent =
      "PASS: 12 fleet/style combinations; full and unloading poses reconstruct exactly.\n" +
      rows.join("\n");
  } catch (e) {
    output.textContent = "FAIL: " + e.stack;
  } finally {
    button.disabled = false;
  }
};
