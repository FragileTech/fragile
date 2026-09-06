import { labStyle } from "./visual-style.js";

export function installStyleControls() {
  const inputs = [...document.querySelectorAll('input[name="visual-style"]')];
  const status = document.getElementById("style-status");
  const retry = document.getElementById("style-retry");
  let request = 0,
    requested = labStyle.preferred();
  const check = (style) =>
    inputs.forEach((input) => {
      input.checked = input.value === style;
    });
  async function change(style) {
    const revision = ++request;
    requested = style;
    check(style);
    retry.hidden = true;
    status.textContent = `Loading ${style} models…`;
    status.parentElement.setAttribute("aria-busy", "true");
    try {
      const changed = await labStyle.change(style);
      if (!changed || revision !== request) return;
      document.documentElement.dataset.visualStyle = style;
      status.textContent = "";
    } catch (error) {
      if (revision !== request) return;
      check(labStyle.current);
      status.textContent = `Could not load ${style} models. Current view kept.`;
      retry.hidden = false;
      console.error("Visual style loading failed", error);
    } finally {
      if (revision === request)
        status.parentElement.setAttribute("aria-busy", "false");
    }
  }
  for (const input of inputs)
    input.addEventListener("change", () => {
      if (input.checked) void change(input.value);
    });
  retry.addEventListener("click", () => void change(requested));
  void change(requested);
}
