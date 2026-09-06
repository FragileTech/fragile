import { manualAction } from "./actions.js";
// Keyboard input adapter; the simulation worker remains the only world writer.
export function installManualControl({
  isReady,
  channels,
  selectedBody,
  apply,
}) {
  const $ = (id) => document.getElementById(id);
  const keys = new Set();
  window.addEventListener("keydown", (e) => {
    if (
      $("manual").checked &&
      !["INPUT", "TEXTAREA", "SELECT"].includes(
        document.activeElement.tagName,
      ) &&
      ["w", "a", "s", "d", "q", "e", " "].includes(e.key.toLowerCase())
    ) {
      keys.add(e.key.toLowerCase());
      e.preventDefault();
    }
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => keys.clear());
  setInterval(() => {
    if (isReady() && $("manual").checked && keys.size) {
      const action = manualAction(channels(), keys, selectedBody());
      apply(action);
    }
  }, 1000 / 30);
}
