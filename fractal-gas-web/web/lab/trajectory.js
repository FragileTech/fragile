// Shared validation and execution cursor for live control and experiments.
export function validateTrajectory(trajectory, channels) {
  if (!Array.isArray(trajectory) || !trajectory.length)
    throw new Error("Controller returned an empty trajectory");
  return trajectory.map(({ action, frames }) => {
    if (!Number.isSafeInteger(frames) || frames <= 0)
      throw new Error("Invalid trajectory duration");
    const values = Float32Array.from(action);
    if (
      values.length !== channels.length ||
      !values.every(
        (v, i) =>
          Number.isFinite(v) &&
          v >= channels[i].low - 1e-6 &&
          v <= channels[i].high + 1e-6,
      )
    )
      throw new Error("Invalid trajectory action");
    return { action: values, frames };
  });
}

export class TrajectoryCursor {
  constructor(trajectory, channels, saved) {
    this.trajectory = validateTrajectory(trajectory, channels);
    this.index = saved?.index ?? 0;
    this.remaining = saved?.remaining ?? this.trajectory[0].frames;
    if (
      !Number.isInteger(this.index) ||
      this.index < 0 ||
      this.index >= this.trajectory.length ||
      !Number.isInteger(this.remaining) ||
      this.remaining < 1 ||
      this.remaining > this.trajectory[this.index].frames
    )
      throw new Error("Invalid trajectory checkpoint progress");
  }
  get done() {
    return this.index === this.trajectory.length;
  }
  get action() {
    return this.trajectory[this.index].action;
  }
  advance() {
    if (--this.remaining === 0) {
      this.index++;
      this.remaining = this.done ? 0 : this.trajectory[this.index].frames;
    }
  }
  checkpoint() {
    return {
      trajectory: this.trajectory,
      index: this.index,
      remaining: this.remaining,
    };
  }
}
