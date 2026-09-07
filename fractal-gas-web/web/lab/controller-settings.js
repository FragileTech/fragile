import { controllerDefinitions } from "./controllers/index.js";
import { initHelp } from "../help.js";

// Controller plugins own their parameter schema. The UI renders numeric
// and boolean descriptors; algorithms need no branching in the application host.
export class ControllerSettings {
  constructor(container, select, changed) {
    this.container = container;
    this.select = select;
    this.saved = new Map();
    this.changed = changed;
    this.render();
  }
  values() {
    return Object.fromEntries(
      Array.from(this.container.querySelectorAll("input"), (input) => {
        if (!input.checkValidity())
          throw new Error(`Invalid planner setting: ${input.name}`);
        return [
          input.name,
          input.type === "checkbox" ? input.checked : Number(input.value),
        ];
      }),
    );
  }
  render(values) {
    if (this.id) {
      // An unfinished/invalid edit must not prevent switching controllers or
      // loading a valid checkpoint. Keep the last valid values instead.
      try {
        this.saved.set(this.id, this.values());
      } catch {
        /* incomplete input */
      }
    }
    this.id = this.select.value;
    const schema =
      controllerDefinitions().find((d) => d.id === this.id)?.parameters || {};
    const stored = values || this.saved.get(this.id) || {};
    this.container.replaceChildren();
    for (const [key, d] of Object.entries(schema)) {
      const label = document.createElement("label"),
        input = document.createElement("input");
      label.className = d.type === "boolean" ? "check" : "field";
      label.textContent = d.label;
      label.dataset.help =
        d.help || `${d.label} controls this controller's planning behavior.`;
      input.type = d.type === "boolean" ? "checkbox" : "number";
      input.name = key;
      input.id = `planner-${key}`;
      input.min = d.min;
      input.max = d.max;
      // Schema step is a display increment, not a quantization constraint.
      input.step = d.step === 1 ? "1" : "any";
      input.required = input.type !== "checkbox";
      input.value = stored[key] ?? d.default;
      if (input.type === "checkbox") input.checked = stored[key] ?? d.default;
      input.onchange = () => {
        if (!input.checkValidity()) {
          input.reportValidity();
          return;
        }
        this.changed();
      };
      if (input.type === "checkbox") label.prepend(input);
      else label.append(input);
      this.container.append(label);
    }
    initHelp(this.container);
  }
}
