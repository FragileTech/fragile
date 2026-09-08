import { manualAction } from "./actions.js";

// Input events update a command. The worker owns the continuous physics clock.
export function installManualControl({
  isReady,
  channels,
  selectedBody,
  apply,
  step,
  pause,
}) {
  const $ = (id) => document.getElementById(id),
    keys = new Set();
  let sliders = [],
    signature;
  const bindings = {
    w: ["thrust", "throttle", "force_x"],
    s: ["thrust", "throttle", "force_x"],
    a: ["torque", "steering"],
    d: ["torque", "steering"],
    q: ["force_y"],
    e: ["force_y"],
    " ": ["brake"],
  };
  const labels = {
    w: "W · Forward",
    s: "S · Reverse",
    a: "A · Left",
    d: "D · Right",
    q: "Q · Strafe",
    e: "E · Strafe",
    " ": "Space · Brake",
  };
  const command = () => {
    const action = manualAction(channels(), keys, selectedBody());
    if (!keys.size)
      for (const { input, index } of sliders) action[index] = +input.value;
    return action;
  };
  const send = () => {
    if (isReady()) apply(command());
  };
  function clear() {
    keys.clear();
    for (const { input, channel } of sliders) {
      input.value = Math.max(channel.low, Math.min(channel.high, 0));
      input.nextElementSibling.textContent = input.value;
    }
    $("drive-keys")
      ?.querySelectorAll("button")
      .forEach((b) => b.classList.remove("held"));
    send();
  }
  function refresh() {
    const all = channels(),
      body = selectedBody(),
      next = JSON.stringify([body, all]);
    if (next === signature) return;
    clear();
    signature = next;
    sliders = [];
    const selected = all.filter((c) => c.body === body);
    $("drive-channels").replaceChildren();
    $("drive-keys").replaceChildren();
    for (const [key, names] of Object.entries(bindings)) {
      if (!selected.some((c) => names.includes(c.name))) continue;
      const button = document.createElement("button");
      button.textContent = labels[key];
      button.dataset.key = key;
      const release = () => {
        keys.delete(key);
        button.classList.remove("held");
        clearSliders();
        send();
      };
      button.onpointerdown = (event) => {
        event.preventDefault();
        button.setPointerCapture(event.pointerId);
        if (!isReady()) return;
        keys.add(key);
        button.classList.add("held");
        send();
      };
      button.onpointerup =
        button.onpointercancel =
        button.onlostpointercapture =
          release;
      $("drive-keys").append(button);
    }
    all.forEach((channel, index) => {
      if (channel.body !== body) return;
      const label = document.createElement("label");
      label.className = "field";
      label.textContent = channel.name;
      const input = document.createElement("input");
      input.type = "range";
      input.min = channel.low;
      input.max = channel.high;
      input.step = (channel.high - channel.low) / 200 || 1;
      input.value = Math.max(channel.low, Math.min(channel.high, 0));
      input.setAttribute("aria-label", `Drive ${channel.name}`);
      const output = document.createElement("output");
      output.textContent = input.value;
      input.oninput = () => {
        output.textContent = Number(input.value).toFixed(2);
        send();
      };
      label.append(input, output);
      $("drive-channels").append(label);
      sliders.push({ input, index, channel });
    });
    $("drive-hint").textContent =
      `Vehicle ${(body ?? 0) + 1} · ${selected.length ? "Use the supported keys or actuator sliders. Start driving to advance physics." : "No controllable channels."}`;
  }
  function clearSliders() {
    for (const { input, channel } of sliders)
      input.value = Math.max(channel.low, Math.min(channel.high, 0));
  }
  window.addEventListener("keydown", (e) => {
    if (
      !isReady() ||
      e.target.closest?.("input,textarea,select,[contenteditable=true]") ||
      document.querySelector("dialog[open]")
    )
      return;
    const key = e.key.toLowerCase();
    if (
      !bindings[key] ||
      !channels().some(
        (c) => c.body === selectedBody() && bindings[key].includes(c.name),
      )
    )
      return;
    e.preventDefault();
    keys.add(key);
    clearSliders();
    send();
  });
  window.addEventListener("keyup", (e) => {
    if (keys.delete(e.key.toLowerCase())) {
      clearSliders();
      send();
    }
  });
  window.addEventListener("blur", () => {
    clear();
    if (isReady()) pause();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clear();
      if (isReady()) pause();
    }
  });
  $("drive-step").onclick = () => {
    if (isReady()) step(command());
  };
  return {
    clear,
    refresh,
    step: () => {
      if (isReady()) step(command());
    },
  };
}
