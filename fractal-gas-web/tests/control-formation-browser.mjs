// Exercise the actual reward controls and scene editor with deterministic pointer positions.
import assert from "node:assert/strict";
import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({ serviceWorkers: "block" });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // Keep this editor test independent of the planner and animation loop.
  await page.route("**/lab/main.js", (route) =>
    route.fulfill({ contentType: "text/javascript", body: "" }),
  );
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8099/lab/");
  const result = await page.evaluate(async () => {
    const { RewardSettings, rewardDefaults } = await import(
      "./reward-settings.js"
    );
    const { createSceneEditor } = await import("./scene-editor.js");
    const container = document.getElementById("reward-terms");
    let applied;
    const settings = new RewardSettings(container, (values) => {
      applied = values;
    });
    const values = () =>
      Object.fromEntries(
        [...settings.inputs].map(([key, { number }]) => [
          key,
          Number(number.value),
        ]),
      );
    const reset = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Reset defaults",
    );
    const resetCases = [];
    for (const task of ["tandem", "harvest", "navigation", "tandem"]) {
      settings.render({
        task,
        rewards: { formation: 7, progress: 8, gate: 9 },
      });
      reset.click();
      settings.apply.click();
      resetCases.push({
        task,
        progressLabel: settings.inputs
          .get("progress")
          .number.getAttribute("aria-label"),
        actual: applied,
        expected: {
          ...rewardDefaults({ task }),
          distance_coef: 1,
          reward_coef: 1,
        },
      });
    }
    const formationInputs = settings.inputs.get("formation");
    const label = formationInputs.number.getAttribute("aria-label");
    const formationRange = {
      numberMax: formationInputs.number.max,
      sliderMax: formationInputs.slider.max,
    };
    let scene = {
      task: "tandem",
      formation_distance: 4,
      bodies: [
        { position: [5, 5] },
        { controlled: true, position: [20, 20] },
        { controlled: true, position: [25, 20] },
        { controlled: true, position: [20, 27] },
      ],
      formation_pairs: [
        { a: 1, b: 2, distance: 5 },
        { a: 1, b: 3, distance: 7 },
      ],
    };
    let point;
    const editor = createSceneEditor({
      renderer: { worldPoint: () => point, selectMany() {} },
      getState: () => null,
      getInfo: () => [],
      getChannels: () => [],
      isReady: () => true,
      loadScene: (next) => {
        scene = structuredClone(next);
      },
      stop() {},
      status() {},
      error(error) {
        throw error;
      },
    });
    editor.setScene(scene);
    document.getElementById("editor").hidden = false;
    document.getElementById("tool").value = "select";
    const select = (position, shiftKey = false) => {
      point = position;
      document
        .getElementById("world")
        .dispatchEvent(
          new PointerEvent("pointerdown", { button: 0, shiftKey }),
        );
      window.dispatchEvent(new PointerEvent("pointerup"));
    };
    select([20, 20]);
    select([25, 20], true);
    document.getElementById("duplicate-entities").click();
    const duplicated = structuredClone(scene);
    select([5, 5]);
    document.getElementById("delete-entity").click();
    const remapped = structuredClone(scene);
    select([20, 20]);
    document.getElementById("delete-entity").click();
    const removed = structuredClone(scene);
    document.getElementById("undo").click();
    const undone = structuredClone(scene);
    document.getElementById("redo").click();
    return {
      resetCases,
      label,
      formationRange,
      duplicated,
      remapped,
      removed,
      undone,
      redone: scene,
      values: values(),
    };
  });
  for (const { task, actual, expected, progressLabel } of result.resetCases) {
    assert.deepEqual(actual, expected, `${task} reset defaults`);
    assert.equal(
      progressLabel,
      task === "tandem" ? "Checkpoint proximity" : "Target progress",
    );
  }
  assert.equal(result.label, "Formation reward");
  assert.deepEqual(result.formationRange, {
    numberMax: "100",
    sliderMax: "100",
  });
  assert.equal(result.duplicated.bodies.length, 6);
  assert.deepEqual(result.duplicated.formation_pairs, [
    { a: 1, b: 2, distance: 5 },
    { a: 1, b: 3, distance: 7 },
    { a: 4, b: 5, distance: 5 },
  ]);
  assert.deepEqual(result.remapped.formation_pairs, [
    { a: 0, b: 1, distance: 5 },
    { a: 0, b: 2, distance: 7 },
    { a: 3, b: 4, distance: 5 },
  ]);
  assert.deepEqual(result.removed.formation_pairs, [
    { a: 2, b: 3, distance: 5 },
  ]);
  assert.deepEqual(result.undone, result.remapped);
  assert.deepEqual(result.redone, result.removed);
  assert.deepEqual(errors, []);
  console.log(
    "Formation defaults, editor duplication/deletion, and undo/redo passed",
  );
} finally {
  await browser.close();
}
