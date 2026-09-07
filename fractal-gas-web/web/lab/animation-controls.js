import { labAnimations } from "./animations.js";

export function installAnimationControls() {
  const inputs = [...document.querySelectorAll('input[name="animations"]')];
  const check = (enabled) => {
    for (const input of inputs) input.checked = enabled;
  };
  const change = (event) => labAnimations.setEnabled(event.target.checked);
  check(labAnimations.enabled);
  const unsubscribe = labAnimations.subscribe(check);
  for (const input of inputs) input.addEventListener("change", change);
  return () => {
    unsubscribe();
    for (const input of inputs) input.removeEventListener("change", change);
  };
}
