import { StoredMotionRecording } from "./storage/recording-store.js";
import { CHUNK_FRAMES } from "./motion.js";

export class RunSession {
  constructor({ getRecording, attach, getEntries, download, status }) {
    Object.assign(this, { getRecording, attach, getEntries, download, status });
  }
  async save() {
    let recording = this.getRecording();
    if (!recording?.length || recording.readOnly) return;
    this.status("Saving…");
    try {
      if (!recording.flush) {
        const stored = new StoredMotionRecording(
          recording.info,
          recording.root,
          recording.dt,
          {
            scene: recording.scene,
            settings: recording.settings,
          },
        );
        stored.channels = recording.channels;
        // Copy one bounded chunk at a time, applying backpressure to the store.
        for (let start = 0; start < recording.length; start += CHUNK_FRAMES) {
          const count = Math.min(CHUNK_FRAMES, recording.length - start);
          stored.append(
            recording.chunks[start / CHUNK_FRAMES].subarray(
              0,
              count * recording.frameBytes,
            ),
          );
          await stored.flush();
        }
        stored.events = structuredClone(recording.events);
        stored.segments = structuredClone(recording.segments);
        stored.restoreRewardChanges(recording.rewardChanges);
        recording = stored;
        this.attach(stored);
      }
      for (const entry of this.getEntries())
        recording.saveObject("tree", entry.decision, entry);
      await recording.flush();
      this.status("Saved on this device");
    } catch (error) {
      this.status("Save failed");
      throw error;
    }
  }
  async preserve() {
    for (;;) {
      try {
        await this.save();
        return;
      } catch (error) {
        const choice = await this.failure(error);
        if (choice === "retry") {
          await this.getRecording()?.retry?.();
          continue;
        }
        if (choice === "discard") return;
        throw new Error(
          "Change cancelled. Your current run and edits are still available.",
        );
      }
    }
  }
  failure(error) {
    const $ = (id) => document.getElementById(id),
      dialog = $("save-failure");
    $("save-error-message").textContent = error.message;
    return new Promise((resolve) => {
      let choice = "cancel";
      dialog.onclose = () => resolve(choice);
      for (const key of ["retry", "discard", "cancel"])
        $(`save-${key}`).onclick = () => {
          choice = key;
          dialog.close();
        };
      $("save-export").onclick = async () => {
        try {
          const r = this.getRecording();
          const file = await r.exportFile({ recovery: true });
          this.download(
            file,
            "recovered-run.fgcrec",
            "application/octet-stream",
          );
          $("save-error-message").textContent =
            "Recovery file exported. Retry device storage or continue without saving.";
        } catch (e) {
          $("save-error-message").textContent =
            `Export failed: ${e.message}. The current run is retained.`;
        }
      };
      dialog.showModal();
    });
  }
}
