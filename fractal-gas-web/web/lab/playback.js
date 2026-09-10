// Playback is a presentation clock. Seeking never advances the native world.
export class WorldPlayback {
  constructor({ show, changed = () => {} }) {
    this.show = show;
    this.changed = changed;
    this.speed = 1;
    this.cursor = 0;
    this.active = false;
    this.playing = false;
    this.animate = this.animate.bind(this);
  }
  attach(recording) {
    this.live();
    this.recording = recording;
    this.active = false;
    this.cursor = 0;
  }
  async seek(index, { discontinuity = true } = {}) {
    if (!this.recording?.length) return;
    this.active = true;
    this.cursor = Math.max(
      0,
      Math.min(this.recording.length - 1, Math.round(index)),
    );
    const cursor = this.cursor,
      generation = (this.seekGeneration || 0) + 1;
    this.seekGeneration = generation;
    try {
      const frame = this.recording.getFrame
        ? await this.recording.getFrame(cursor)
        : this.recording.frame(cursor);
      if (this.seekGeneration === generation && this.active)
        this.show(frame, cursor, { discontinuity });
    } catch (error) {
      this.pause();
      this.onError?.(error);
    }
    this.changed();
  }
  play() {
    if (!this.recording?.length) return;
    if (this.cursor === this.recording.length - 1) this.cursor = 0;
    this.seek(this.cursor);
    this.playing = true;
    this.last = undefined;
    this.accumulator = 0;
    this.changed();
    cancelAnimationFrame(this.request);
    this.request = requestAnimationFrame(this.animate);
  }
  pause() {
    this.playing = false;
    cancelAnimationFrame(this.request);
    this.changed();
  }
  live() {
    this.pause();
    this.active = false;
    this.seekGeneration = (this.seekGeneration || 0) + 1;
    this.changed();
  }
  animate(now) {
    if (!this.playing) return;
    if (this.last !== undefined)
      this.accumulator += Math.min((now - this.last) / 1000, 0.25) * this.speed;
    this.last = now;
    const frames = Math.floor(this.accumulator / this.recording.dt);
    if (frames) {
      this.accumulator -= frames * this.recording.dt;
      this.seek(this.cursor + frames, { discontinuity: false });
    }
    if (this.cursor >= this.recording.length - 1) {
      this.pause();
      return;
    }
    this.request = requestAnimationFrame(this.animate);
  }
}
