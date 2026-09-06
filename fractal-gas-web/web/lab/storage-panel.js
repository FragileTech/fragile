import {
  StoredMotionRecording,
  listRuns,
  deleteRun,
  importStoredFile,
} from "./storage/recording-store.js";
import { encodeObject, decodeObject } from "./storage/codec.js";
import { MotionRecording } from "./motion.js";

// Persistence and file IO are adapters around the same recording interface.
export class StoragePanel {
  constructor({
    getRecording,
    getScene,
    getSettings,
    loadRun,
    saveCheckpoint,
    loadCheckpoint,
    upload,
    download,
    status,
    error,
  }) {
    this.options = { getRecording, getScene, getSettings, error };
    const $ = (id) => document.getElementById(id);
    $("flush-run").onclick = async () => {
      try {
        const r = getRecording();
        if (!r?.flush) {
          status("Enable device storage and reset to start a persistent run.");
          return;
        }
        await r.flush();
        status("Recording saved on this device.");
      } catch (e) {
        error(e);
      }
    };
    const autoSave = () => {
      const record = getRecording();
      if (record?.flush && record.length > record.durableFrames)
        record.flush().catch(error);
    };
    setInterval(autoSave, 5000);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) autoSave();
    });
    const refresh = async () => {
      const list = $("stored-runs");
      list.replaceChildren();
      for (const run of await listRuns()) {
        const row = document.createElement("div");
        row.className = "stored-run";
        const title = document.createElement("span");
        title.textContent = `${run.name} · ${run.length} frames · ${new Date(run.updated).toLocaleString()}`;
        const open = document.createElement("button");
        open.textContent = "Open";
        open.onclick = async () => {
          try {
            await loadRun(await StoredMotionRecording.open(run.id, error));
            $("storage-dialog").close();
          } catch (e) {
            error(e);
          }
        };
        const remove = document.createElement("button");
        remove.textContent = "Delete";
        remove.disabled = run.id === getRecording()?.id;
        remove.onclick = async () => {
          try {
            await deleteRun(run.id);
            await refresh();
          } catch (e) {
            error(e);
          }
        };
        row.append(title, open, remove);
        list.append(row);
      }
      if (!list.children.length)
        list.textContent = "No recordings saved on this device.";
    };
    $("open-library").onclick = async () => {
      try {
        await getRecording()?.flush?.();
        await refresh();
        $("storage-dialog").showModal();
      } catch (e) {
        error(e);
      }
    };
    $("close-storage").onclick = () => $("storage-dialog").close();
    $("save-checkpoint").onclick = saveCheckpoint;
    $("load-checkpoint").onclick = () =>
      upload(".fgcp", async (file) => {
        const data = decodeObject(new Uint8Array(await file.arrayBuffer()));
        if (
          data.format !== "fractal-controller-checkpoint" ||
          data.version !== 1 ||
          !data.scene ||
          !data.checkpoint?.algorithm ||
          !data.root
        )
          throw new Error("Unsupported planner checkpoint");
        await loadCheckpoint(data);
      });
    this.downloadCheckpoint = (data) =>
      download(
        encodeObject({
          ...data,
          format: "fractal-controller-checkpoint",
          version: 1,
          scene: getScene(),
          settings: getSettings(),
        }),
        "fractal-control.fgcp",
        "application/octet-stream",
      );
  }
  create(info, root, dt, channels) {
    const { getScene, getSettings, error } = this.options;
    const r = document.getElementById("persistent-recording").checked
      ? new StoredMotionRecording(info, root, dt, {
          scene: structuredClone(getScene()),
          settings: getSettings(),
          onError: error,
        })
      : new MotionRecording(info, root, dt);
    r.channels = channels;
    return r;
  }
  async import(file) {
    return importStoredFile(file, this.options.error);
  }
}
