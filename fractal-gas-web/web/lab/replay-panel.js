import { WorldPlayback } from "./playback.js";

// DOM adapter for the independent playback clock. No physics or archive code.
export class ReplayPanel {
  constructor({ stop, show, live, resume, decision, error }) {
    this.$ = (id) => document.getElementById(id);
    this.playback = new WorldPlayback({
      show: (frame, index) => {
        show(frame);
        decision(frame.decision);
        this.$("motion-timeline").value = index;
      },
      changed: () => this.update(),
    });
    this.playback.onError = error;
    this.$("motion-timeline").oninput = () => {
      const index = +this.$("motion-timeline").value;
      stop();
      this.playback.pause();
      this.playback.seek(index);
    };
    this.$("motion-play").onclick = () => {
      stop();
      if (this.playback.playing) this.playback.pause();
      else this.playback.play();
    };
    this.$("motion-speed").onchange = () =>
      (this.playback.speed = +this.$("motion-speed").value);
    this.$("motion-live").onclick = () => {
      this.playback.live();
      live();
    };
    this.$("motion-resume").onclick = async () => {
      try {
        stop();
        const r = this.recording,
          index = this.playback.cursor;
        const rows = r.getRows ? await r.getRows(index) : r.rows(index);
        if (this.recording !== r) return;
        this.playback.live();
        resume(rows);
      } catch (e) {
        error(e);
      }
    };
    this.$("motion-events").onchange = () => {
      stop();
      this.playback.pause();
      this.playback.seek(+this.$("motion-events").value);
    };
    this.$("add-event").onclick = () => {
      const r = this.recording;
      if (!r?.length) return;
      r.addEvent(
        this.active ? this.playback.cursor : r.length - 1,
        this.$("event-note").value || "Marker",
      );
      this.$("event-note").value = "";
      this.update();
    };
  }
  get active() {
    return this.playback.active;
  }
  attach(recording) {
    this.recording = recording;
    this.playback.attach(recording);
    this.update();
  }
  append(data) {
    if (data.initial && this.recording.length) return;
    this.recording.append(data.packet, data.label);
    this.update();
  }
  update() {
    const p = this.playback,
      r = this.recording,
      length = r?.length || 0;
    this.$("motion-timeline").max = Math.max(0, length - 1);
    if (!p.active) this.$("motion-timeline").value = Math.max(0, length - 1);
    this.$("motion-play").textContent = p.playing
      ? "Ⅱ Pause replay"
      : "▶ Play world";
    this.$("motion-play").disabled = length < 2;
    this.$("motion-timeline").disabled = !length;
    this.$("motion-resume").disabled = !p.active;
    this.$("motion-live").disabled = !p.active;
    const events = r?.events || [];
    if (this.eventsRecording !== r || this.eventCount !== events.length) {
      this.eventsRecording = r;
      this.eventCount = events.length;
      this.$("motion-events").replaceChildren(
        new Option("Jump to event…", ""),
        ...events
          .slice(-500)
          .map((e) => new Option(`${e.frame + 1} · ${e.label}`, e.frame)),
      );
    }
    this.$("motion-position").textContent = length
      ? `${p.active ? p.cursor + 1 : length} / ${length} frames · ${((r.residentBytes ?? r.bytes) / 1024).toFixed(0)} KiB resident${r.id ? ` · ${r.durableFrames} stored` : ""}`
      : "No world frames";
    this.$("motion-segment").textContent = length
      ? p.active
        ? r.segment(p.cursor)
        : "Recording executed world"
      : "WORLD REPLAY";
  }
}
