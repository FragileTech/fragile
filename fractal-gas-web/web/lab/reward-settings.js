// Defaults and limits mirror Scene::compile. Weights travel with scene JSON.
const HARVEST_REWARDS = new Set([
  "progress",
  "distance_squared",
  "catch",
  "wall_collision",
]);
const TANDEM_DEFAULTS = {
  progress: 1,
  gate: 30,
  distance_squared: 1,
  wall_collision: 100,
  collision: 2,
  formation: 50,
};
export const REWARD_TERMS = Object.freeze([
  {
    key: "catch",
    label: "Asteroid catch bonus",
    def: 10,
    max: 10000,
    slider: 100,
    step: 1,
    help: "Bonus on every asteroid catch, including re-catching after a break. Harvesting only.",
  },
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
    help: "In formation flight, rewards proximity each physics frame: checkpoint radius / (checkpoint radius + mean distance of all agents to the shared active checkpoint), including agents that already crossed. Staying nearby earns more; moving away lowers this positive reward without a negative penalty. Default weight 1. Other tasks reward progress toward their target, with a penalty for moving away; harvesting uses hook-to-asteroid or attached asteroid-to-discharge distance.",
  },
  {
    key: "wall_collision",
    label: "Wall collision penalty",
    def: 100,
    max: 10000,
    slider: 100,
    step: 0.1,
    help: "Penalty once per controlled vehicle per physics frame touching an arena wall or obstacle boundary. Sustained contact keeps costing reward; corners and solver substeps do not multiply it. Cargo and hooks do not count. Zero disables the penalty, independently of wall death.",
  },
  {
    key: "collision",
    label: "Vehicle/body collision penalty",
    def: 2,
    max: 10000,
    slider: 100,
    step: 0.1,
    help: "Penalty per qualifying vehicle contact with another body, not walls. Repeated contacts can cost more than one penalty. Disabled in harvesting tasks.",
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
    help: "Reward for each vehicle reaching its next ordered gate. In formation flight, defaults to 30 divided by total agent count per crossing: the whole team earns 30 per shared checkpoint. Later checkpoints stay locked until everyone clears the current one. Has no effect without gates.",
  },
  {
    key: "formation",
    label: "Formation reward",
    def: 0.15,
    max: 100,
    slider: 100,
    step: 0.01,
    help: "Reward each physics frame in formation flight: multiply target / (target + absolute separation error) over every vehicle pair. Perfect formation scores 1 before weighting, even while stationary. Independent of Checkpoint proximity. Defaults to weight 50 in formation flight; accepts 0 to 100, and zero disables it. Set pair distances in scene JSON.",
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
export function rewardDefaults(scene = {}) {
  return Object.fromEntries(
    REWARD_TERMS.map((term) => [
      term.key,
      scene.task === "harvest" && !HARVEST_REWARDS.has(term.key)
        ? 0
        : scene.task === "tandem"
          ? (TANDEM_DEFAULTS[term.key] ?? 0)
          : term.def,
    ]),
  );
}
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
  const defaults = rewardDefaults(scene);
  return Object.fromEntries(
    REWARD_TERMS.map((t) => [
      t.key,
      scene.task === "harvest" && !HARVEST_REWARDS.has(t.key)
        ? 0
        : t.key === "full_reward"
          ? (scene.cargo?.full_reward ??
            (scene.task === "tandem" ? 0 : scene.rewards?.pickup) ??
            defaults[t.key])
          : (scene.rewards?.[t.key] ?? defaults[t.key]),
    ]),
  );
}
export function withRewards(scene, values) {
  const next = structuredClone(scene);
  next.rewards = { ...next.rewards };
  for (const t of REWARD_TERMS) {
    const v =
      scene.task === "harvest" && !HARVEST_REWARDS.has(t.key)
        ? 0
        : values[t.key];
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
      this.setValues({
        ...coefficientValues(),
        ...rewardDefaults({ task: this.task }),
      });
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
    this.task = scene.task;
    const progress = this.inputs.get("progress");
    const progressLabel =
      scene.task === "tandem" ? "Checkpoint proximity" : progress.term.label;
    progress.number.closest("label").firstChild.textContent = progressLabel;
    progress.number.setAttribute("aria-label", progressLabel);
    progress.slider.setAttribute("aria-label", `${progressLabel} slider`);
    for (const [key, { number }] of this.inputs)
      number.closest("label").hidden =
        scene.task === "harvest"
          ? !HARVEST_REWARDS.has(key) &&
            !COEFFICIENTS.some((t) => t.key === key)
          : key === "catch";
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
