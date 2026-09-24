// Preset recommendations apply when starting a task or replacing unchanged
// defaults. Explicit controller tuning survives a scenario switch.
export const DEFAULT_PLANNING_BUDGET = Object.freeze({
  walkers: 128,
  horizon: 64,
  frames: 12,
  elites: 0,
});

const MAXIMUM = { walkers: 8192, horizon: 4096, frames: 60, elites: 128 };

export function presetControllerSettings(
  scene,
  settings,
  previousScene,
  reset = false,
) {
  const next = { ...settings };
  for (const [key, fallback] of Object.entries(DEFAULT_PLANNING_BUDGET)) {
    const previous = previousScene?.controller_defaults?.[key] ?? fallback;
    if (!reset && settings[key] != null && settings[key] !== previous) continue;
    const value = scene.controller_defaults?.[key] ?? fallback;
    if (
      !Number.isInteger(value) ||
      value < (key === "elites" ? 0 : 1) ||
      value > MAXIMUM[key]
    )
      throw new RangeError(`Invalid preset controller default: ${key}`);
    next[key] = key === "elites" ? Math.min(value, next.walkers ?? 128) : value;
  }
  if (next.max_walkers != null)
    next.max_walkers =
      settings.max_walkers === settings.walkers
        ? next.walkers
        : Math.max(next.max_walkers, next.walkers);
  return next;
}
