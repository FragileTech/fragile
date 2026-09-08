import { controllerDefinitions } from "./controllers/registry.js";

const common = {
  walkers: {
    label: "Candidate worlds",
    min: 1,
    max: 8192,
    default: 128,
    step: 1,
  },
  horizon: {
    label: "Lookahead actions",
    min: 1,
    max: 4096,
    default: 32,
    step: 1,
  },
  frames: {
    label: "Action duration (frames)",
    min: 1,
    max: 60,
    default: 12,
    step: 1,
  },
};
export class VariantSettings {
  constructor(select, json, changed) {
    Object.assign(this, { select, json, changed });
    this.fields = document.createElement("div");
    this.fields.className = "variant-parameters fields-two";
    select.closest("label").after(this.fields);
    const advanced = document.createElement("details"),
      summary = document.createElement("summary");
    summary.textContent = "Advanced variant JSON";
    const jsonLabel = json.closest("label");
    jsonLabel.firstChild.textContent = "Variant settings";
    jsonLabel.before(advanced);
    advanced.append(summary, jsonLabel);
    json.addEventListener("change", () => {
      try {
        const data = JSON.parse(json.value);
        if (!data || Array.isArray(data) || typeof data !== "object")
          throw new Error("Enter a settings object.");
        this.set(data);
        this.changed();
      } catch (e) {
        json.setCustomValidity(e.message);
      }
    });
    select.addEventListener("change", () => {
      this.set({ ...this.model, algorithm: select.value });
      this.changed();
    });
  }
  set(settings) {
    this.model = structuredClone(settings);
    this.select.value = settings.algorithm;
    const def = controllerDefinitions().find((d) => d.id === this.select.value);
    if (!def) throw new Error("Unknown comparison controller");
    this.inputs = new Map();
    this.fields.replaceChildren();
    for (const [key, spec] of Object.entries({
      ...common,
      ...def.parameters,
    })) {
      const label = document.createElement("label");
      label.className = "field";
      label.textContent = spec.label;
      const input = document.createElement("input");
      input.type = spec.type === "boolean" ? "checkbox" : "number";
      input.setAttribute(
        "aria-label",
        `${this.select.id === "variant-a" ? "A" : "B"} ${spec.label}`,
      );
      input.min = spec.min;
      input.max = spec.max;
      input.step = spec.step === 1 ? 1 : "any";
      input.required = input.type === "number";
      if (input.type === "checkbox")
        input.checked = settings[key] ?? spec.default;
      else input.value = settings[key] ?? spec.default;
      input.oninput = () => {
        this.model[key] =
          input.type === "checkbox" ? input.checked : Number(input.value);
        this.sync();
        this.changed();
      };
      label.append(input);
      this.fields.append(label);
      this.inputs.set(key, input);
      this.model[key] =
        input.type === "checkbox" ? input.checked : Number(input.value);
    }
    this.model.algorithm = this.select.value;
    this.sync();
  }
  sync() {
    this.json.value = JSON.stringify(this.model, null, 2);
    this.json.setCustomValidity("");
  }
  values() {
    if (!this.json.reportValidity())
      throw new Error("Correct the variant JSON.");
    for (const input of this.inputs.values())
      if (!input.reportValidity())
        throw new Error("Correct the variant parameters.");
    return structuredClone(this.model);
  }
}
export function configurationDiff(a, b) {
  return [...new Set([...Object.keys(a), ...Object.keys(b)])]
    .filter((key) => JSON.stringify(a[key]) !== JSON.stringify(b[key]))
    .map(
      (key) =>
        `${key.replaceAll("_", " ")}: ${JSON.stringify(a[key] ?? "default")} → ${JSON.stringify(b[key] ?? "default")}`,
    );
}
