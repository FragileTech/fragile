// Presentation time never advances the engine or a recording. Reuse the frame
// object so the render loop does not allocate on each tick.
export class AnimationClock {
  constructor() {
    this.frame = { dt: 0, idleTime: 0, playing: false, speed: 1 };
  }
  setPlayback({ playing = this.frame.playing, speed = this.frame.speed } = {}) {
    this.frame.playing = !!playing;
    this.frame.speed = Number.isFinite(speed) && speed > 0 ? speed : 1;
  }
  reset(time = 0) {
    this.frame.idleTime = Number.isFinite(time) ? time : 0;
    this.frame.dt = 0;
    this.last = undefined;
  }
  tick(now, enabled = true) {
    this.frame.dt =
      enabled && this.last != null
        ? Math.max(0, Math.min(0.05, (now - this.last) / 1000))
        : 0;
    this.last = enabled ? now : undefined;
    this.frame.idleTime += this.frame.dt;
    return this.frame;
  }
}
