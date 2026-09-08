// Defaults and limits mirror Scene::compile. Weights travel with scene JSON.
export const REWARD_TERMS = Object.freeze([
  {
    key: "distance_squared",
    label: "Distance travelled²",
    def: 1,
    max: 1000,
    slider: 10,
    step: 0.01,
    help: "Weight × mean vehicle displacement squared (Δx² + Δy²), in metres squared per physics frame. Rewards movement in any direction. Summed over frames, not squared over an entire action or journey. Zero disables it.",
  },
  {
    key: "hooked_rock_distance",
    label: "Hooked rock travel",
    def: 1,
    max: 1000,
    slider: 10,
    step: 0.1,
    help: "Reward per metre travelled by rocks hooked to an active vehicle at the start of each physics frame. Each rock counts once even with several tow hooks. Linear distance in any direction; excludes vehicle-only movement and delivery respawns. Zero disables it. Click Apply settings to use edits.",
  },
  {
    key: "progress",
    label: "Target progress",
    def: 1,
    max: 1000,
    slider: 10,
    step: 0.1,
    help: "Reward for getting closer to the next gate, refinery, pickup or cargo target. Moving away gives a penalty. Also scales formation shaping.",
  },
  {
    key: "collision",
    label: "Collision penalty",
    def: 2,
    max: 10000,
    slider: 100,
    step: 0.1,
    help: "Penalty per qualifying vehicle contact. Repeated contacts can cost more than one penalty.",
  },
  {
    key: "pickup",
    label: "Pickup bonus",
    def: 10,
    max: 10000,
    slider: 100,
    step: 1,
    help: "Reward for each collected drop. Has no effect without pickups.",
  },
  {
    key: "delivery",
    label: "Delivery bonus",
    def: 100,
    max: 10000,
    slider: 1000,
    step: 1,
    help: "Reward per delivered rock, or distributed over unloading one full vehicle load.",
  },
  {
    key: "gate",
    label: "Checkpoint bonus",
    def: 30,
    max: 10000,
    slider: 300,
    step: 1,
    help: "Reward for each vehicle reaching its next ordered gate. Has no effect without gates.",
  },
  {
    key: "formation",
    label: "Formation shaping",
    def: 0.15,
    max: 1000,
    slider: 5,
    step: 0.01,
    help: "Weight of formation improvement for tandem tasks. Multiplied by Target progress; zero progress disables formation shaping.",
  },
  {
    key: "full_reward",
    label: "Full-load bonus",
    def: 10,
    max: 10000,
    slider: 100,
    step: 1,
    help: "Additional reward when cargo storage first becomes full. Has no effect without cargo storage.",
  },
]);
export const COEFFICIENTS = Object.freeze([
  {
    key: "distance_coef",
    label: "Diversity coefficient",
    def: 1,
    max: 10,
    slider: 10,
    step: 0.1,
    help: "FMC exponent on rescaled observation distance. Higher values favor diverse futures; zero ignores diversity. Apply settings to replan from the current world.",
  },
  {
    key: "reward_coef",
    label: "Reward coefficient",
    def: 1,
    max: 10,
    slider: 10,
    step: 0.1,
    help: "FMC exponent on rescaled accumulated reward. Higher values favor exploitation; zero ignores reward in cloning fitness. Apply settings to replan from the current world.",
  },
]);
const SETTINGS_TERMS = [...COEFFICIENTS, ...REWARD_TERMS];
export function coefficientValues(values = {}) {
  return Object.fromEntries(
    COEFFICIENTS.map((term) => {
      const value = values[term.key] ?? term.def;
      if (!Number.isFinite(value) || value < 0 || value > term.max)
        throw new RangeError(`${term.label} must be between 0 and ${term.max}`);
      return [term.key, value];
    }),
  );
}
export function rewardValues(scene) {
  return Object.fromEntries(
    REWARD_TERMS.map((t) => [
      t.key,
      t.key === "full_reward"
        ? (scene.cargo?.full_reward ?? scene.rewards?.pickup ?? t.def)
        : (scene.rewards?.[t.key] ?? t.def),
    ]),
  );
}
export function withRewards(scene, values) {
  const next = structuredClone(scene);
  next.rewards = { ...next.rewards };
  for (const t of REWARD_TERMS) {
    const v = values[t.key];
    if (!Number.isFinite(v) || v < 0 || v > t.max)
      throw new RangeError(`${t.label} must be between 0 and ${t.max}`);
    if (t.key === "full_reward") {
      if (next.cargo != null) next.cargo.full_reward = v;
    } else next.rewards[t.key] = v;
  }
  return next;
}
export class RewardSettings {
  constructor(container, apply) {
    this.inputs = new Map();
    for (const term of SETTINGS_TERMS) {
      const label = document.createElement("label");
      label.className = "field reward-field";
      label.dataset.help = term.help;
      label.textContent = term.label;
      const row = document.createElement("span");
      row.className = "reward-inputs";
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = term.slider;
      slider.step = term.step;
      slider.setAttribute("aria-label", `${term.label} slider`);
      const number = document.createElement("input");
      number.id = `lab-reward-${term.key}`;
      label.htmlFor = number.id;
      number.type = "number";
      number.min = 0;
      number.max = term.max;
      number.step = "any";
      number.required = true;
      number.setAttribute("aria-label", term.label);
      slider.oninput = () => {
        number.value = slider.value;
        this.updatePending();
      };
      number.oninput = () => {
        slider.max = Math.max(term.slider, Number(number.value) || 0);
        slider.value = number.value;
        this.updatePending();
      };
      row.append(slider, number);
      label.append(row);
      container.append(label);
      this.inputs.set(term.key, { number, slider, term });
    }
    this.pending = document.createElement("p");
    this.pending.id = "reward-settings-status";
    this.pending.setAttribute("role", "status");
    this.pending.textContent =
      "Active reward settings. Edit a value to prepare a change.";
    this.apply = document.createElement("button");
    this.apply.type = "button";
    this.apply.textContent = "Apply to current run";
    this.apply.onclick = () => {
      for (const { number } of this.inputs.values())
        if (!number.reportValidity()) return;
      apply(
        Object.fromEntries(
          [...this.inputs].map(([key, { number }]) => [
            key,
            Number(number.value),
          ]),
        ),
      );
    };
    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset defaults";
    reset.onclick = () =>
      this.setValues(
        Object.fromEntries(SETTINGS_TERMS.map((t) => [t.key, t.def])),
      );
    container.append(this.pending, this.apply, reset);
  }
  updatePending() {
    const dirty =
      this.appliedValues &&
      [...this.inputs].some(
        ([key, { number }]) =>
          number.value === "" ||
          Number(number.value) !== this.appliedValues[key],
      );
    this.pending.textContent = dirty
      ? "Reward changes are pending. Apply to current run preserves world state."
      : "Active reward settings. Edit a value to prepare a change.";
  }
  setValues(values) {
    for (const [key, { number, slider, term }] of this.inputs) {
      number.value = values[key];
      slider.max = Math.max(term.slider, values[key]);
      slider.value = values[key];
    }
    this.updatePending();
  }
  render(scene, coefficients = {}) {
    this.appliedValues = {
      ...rewardValues(scene),
      ...coefficientValues(coefficients),
    };
    this.setValues(this.appliedValues);
  }
  setEnabled(enabled) {
    this.apply.disabled = !enabled;
    for (const { number, slider } of this.inputs.values()) {
      number.disabled = slider.disabled = !enabled;
    }
  }
}
