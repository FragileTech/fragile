// The numeric editor recursively exposes extension parameters as well as common
// physical quantities. Unknown parameters remain editable through scene JSON.
const units = {
  mass: "kg",
  radius: "m",
  thrust: "N",
  torque: "N·m",
  drag: "1/s",
  strength: "m³/s²",
  softening: "m",
  wheelbase: "m",
  steering_limit: "rad",
  lateral_grip: "1/s",
  yaw_response: "1/s",
  brake_deceleration: "m/s²",
  stiffness: "N/m",
  damping: "N·s/m",
  rest_length: "m",
  angle: "rad",
};
const pretty = (key) => key.replaceAll("_", " ");
export class EntityProperties {
  constructor(container) {
    this.container = container;
    this.fields = [];
  }
  show(entity) {
    this.container.replaceChildren();
    this.fields = [];
    if (!entity) return;
    const walk = (value, path = []) => {
      if (typeof value === "number" && Number.isFinite(value)) {
        const label = document.createElement("label");
        label.className = "field";
        const name = path.join(" · "),
          unit =
            path[0] === "position"
              ? "m"
              : path[0] === "velocity"
                ? "m/s"
                : units[path.at(-1)];
        label.textContent = `${pretty(name)}${unit ? ` (${unit})` : ""}`;
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.value = value;
        input.dataset.path = path.join(".");
        label.append(input);
        this.container.append(label);
        this.fields.push({ path, input });
      } else if (value && typeof value === "object" && path[0] !== "vertices")
        for (const [key, child] of Object.entries(value))
          walk(child, [...path, key]);
    };
    walk(entity);
  }
  apply(entity) {
    const out = structuredClone(entity);
    for (const { path, input } of this.fields) {
      if (
        !input.checkValidity() ||
        input.value === "" ||
        !Number.isFinite(+input.value)
      )
        throw new Error("Properties must contain finite numbers");
      let target = out;
      for (const key of path.slice(0, -1)) target = target[key];
      target[path.at(-1)] = +input.value;
    }
    return out;
  }
}
export function actionSliders(container, channels) {
  container.replaceChildren();
  const action = Float32Array.from(channels || [], (c) =>
    Math.max(c.low, Math.min(c.high, 0)),
  );
  (channels || []).forEach((channel, index) => {
    const label = document.createElement("label");
    label.className = "channel-control";
    label.textContent = `Body ${channel.body} · ${channel.name}`;
    const output = document.createElement("output");
    output.textContent = action[index].toFixed(2);
    const input = document.createElement("input");
    input.type = "range";
    input.min = channel.low;
    input.max = channel.high;
    input.step = (channel.high - channel.low) / 200;
    input.value = action[index];
    input.oninput = () => {
      action[index] = +input.value;
      output.textContent = (+input.value).toFixed(2);
    };
    label.append(output, input);
    container.append(label);
  });
  return action;
}
