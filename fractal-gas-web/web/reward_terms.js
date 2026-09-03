// "Reward terms" sidebar panel: one live slider per term of the shaped
// Mario / Sonic / Montezuma rewards (src/mario_reward.cpp,
// src/retro_game_logic.hpp, src/montezuma_logic.hpp). The weights are sent
// to C++ as a plain float array in the order below, which must match the
// MarioRewardWeights / SonicRewardWeights / MontezumaRewardWeights field
// order. Keys are UI console ids (main.js).

const LIVE = " Applies live to rewards earned from now on.";

export const REWARD_TERMS = {
  // console 0: Super Mario Bros
  0: [
    { key: "x", label: "X progress", def: 1, min: 0, max: 5, step: 0.1,
      help: "Reward per pixel Mario moves right each frame (negative when moving left). Also scales the shortcut payout: coming out of a pipe further along the level pays the distance skipped. The main driving force of the swarm." + LIVE },
    { key: "time", label: "Time penalty", def: 1, min: 0, max: 5, step: 0.1,
      help: "Penalty per unit the in-game clock loses. Pushes walkers to hurry; 0 removes any time pressure." + LIVE },
    { key: "death", label: "Death penalty", def: 25, min: 0, max: 100, step: 1,
      help: "One-off penalty on the frame Mario dies or falls. It is added inside the per-frame clip, so values above the clip are cut to the clip." + LIVE },
    { key: "clip", label: "Per-frame clip", def: 15, min: 1, max: 100, step: 1,
      help: "The sum of x progress, time penalty and death penalty is clipped to plus/minus this value every frame, so no single frame dominates. The flag and area bonuses are added after the clip." + LIVE },
    { key: "flag", label: "Flag bonus", def: 500, min: 0, max: 2000, step: 10,
      help: "One-off bonus for grabbing the end-of-level flagpole. Makes finishing the level the most valuable event." + LIVE },
    { key: "area", label: "Area bonus", def: 100, min: 0, max: 500, step: 5,
      help: "One-off bonus for entering an area (pipe bonus room, warp, intro sub-area) not visited before in this stage. Pays for the detour of stopping to enter a pipe." + LIVE },
  ],
  // console 2: Sonic the Hedgehog (Genesis)
  2: [
    { key: "dx", label: "X progress", def: 1, min: 0, max: 5, step: 0.1,
      help: "Reward per pixel Sonic moves right each frame (negative when moving left). Disabled while a boss is on screen." + LIVE },
    { key: "rings", label: "Rings", def: 3, min: 0, max: 20, step: 0.5,
      help: "Reward per ring gained, and the same penalty per ring lost (a hit drops all rings, so this is also the damage penalty)." + LIVE },
    { key: "score", label: "Score", def: 0.5, min: 0, max: 5, step: 0.1,
      help: "Reward per point of in-game score (badniks, monitors, boss hits, end-of-act tally)." + LIVE },
    { key: "cell", label: "Exploration bonus", def: 500, min: 0, max: 2000, step: 10,
      help: "One-off bonus for entering a 64x64 px map cell this walker's lineage has never visited. Makes backtracking and exploring vertically worthwhile." + LIVE },
    { key: "life", label: "Extra-life bonus", def: 1000, min: 0, max: 5000, step: 50,
      help: "Bonus per life gained (1-ups, 100-ring bonus)." + LIVE },
    { key: "boss", label: "Boss hit bonus", def: 2000, min: 0, max: 10000, step: 100,
      help: "Bonus per hit point removed from a boss (bosses take 8 hits)." + LIVE },
    { key: "act", label: "Act clear bonus", def: 5000, min: 0, max: 20000, step: 100,
      help: "One-off bonus for reaching the next act or zone. Keeps finishing the level the dominant goal." + LIVE },
  ],
  // console 3: Montezuma's Revenge (Atari, src/montezuma_logic.hpp)
  3: [
    { key: "score", label: "Game score", def: 1, min: 0, max: 5, step: 0.1,
      help: "Multiplier on the in-game score (keys, doors, jewels, enemies). Sparse: the first points only come with the first key." + LIVE },
    { key: "room", label: "New-room bonus", def: 500, min: 0, max: 2000, step: 10,
      help: "One-off bonus the first time a walker's lineage enters one of the 24 rooms of the temple level (the bitmask resets on the next level). Not paid while dying. 0 leaves exploration entirely to the distance term." + LIVE },
  ],
};

const fmt = (v, step) => (step < 1 ? v.toFixed(1) : String(Math.round(v)));
const sliderId = (consoleId, key) => `term-${consoleId}-${key}`;

/**
 * Build the per-console slider groups inside `container` (the section).
 * `onChange(consoleId)` is called on every slider input.
 */
export function buildRewardPanel(container, onChange) {
  for (const [consoleId, terms] of Object.entries(REWARD_TERMS)) {
    const group = document.createElement("div");
    group.id = `reward-terms-${consoleId}`;
    group.className = "reward-terms";
    group.hidden = true;
    for (const t of terms) {
      const label = document.createElement("label");
      label.dataset.help = t.help;
      label.append(document.createTextNode(t.label + " "));
      const value = document.createElement("span");
      value.className = "term-value";
      value.textContent = fmt(t.def, t.step);
      label.append(value);
      const input = document.createElement("input");
      input.type = "range";
      input.id = sliderId(consoleId, t.key);
      input.min = t.min;
      input.max = t.max;
      input.step = t.step;
      input.value = t.def;
      input.addEventListener("input", () => {
        value.textContent = fmt(parseFloat(input.value), t.step);
        onChange(parseInt(consoleId, 10));
      });
      label.append(input);
      group.append(label);
    }
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "term-defaults";
    reset.textContent = "Reset to defaults";
    reset.addEventListener("click", () => {
      for (const t of terms) {
        const input = document.getElementById(sliderId(consoleId, t.key));
        input.value = t.def;
        input.dispatchEvent(new Event("input"));
      }
    });
    group.append(reset);
    // Keep the section's hint paragraph below the sliders.
    container.insertBefore(group, container.querySelector(".hint"));
  }
}

/** Float array in C++ field order, or null for consoles without terms. */
export function readRewardWeights(consoleId) {
  const terms = REWARD_TERMS[consoleId];
  if (!terms) return null;
  return terms.map((t) => parseFloat(document.getElementById(sliderId(consoleId, t.key)).value));
}

/** Show the group for `consoleId`; hide the whole section when none. */
export function showRewardPanel(section, consoleId) {
  let any = false;
  for (const id of Object.keys(REWARD_TERMS)) {
    const show = parseInt(id, 10) === consoleId;
    document.getElementById(`reward-terms-${id}`).hidden = !show;
    any = any || show;
  }
  section.hidden = !any;
}
