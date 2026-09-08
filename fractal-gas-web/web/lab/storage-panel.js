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
      if (
        record?.flush &&
        !record.readOnly &&
        record.length > record.durableFrames
      ) {
        const status = document.getElementById("save-status");
        if (status) status.textContent = "Saving…";
        record
          .flush()
          .then(() => {
            if (status) status.textContent = "Saved on this device";
          })
          .catch((e) => {
            if (status) status.textContent = "Save failed";
            error(e);
          });
      }
    };
    setInterval(autoSave, 5000);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) autoSave();
    });
    const refresh = async () => {
      const list = $("stored-runs");
      list.replaceChildren();
      const runs = await listRuns();
      const ordered = [],
        seen = new Set();
      const visit = (run) => {
        if (seen.has(run.id)) return;
        seen.add(run.id);
        ordered.push(run);
        runs.filter((child) => child.parent?.run === run.id).forEach(visit);
      };
      runs
        .filter(
          (run) => !run.parent || !runs.some((p) => p.id === run.parent.run),
        )
        .forEach(visit);
      runs.forEach(visit);
      for (const run of ordered) {
        const row = document.createElement("div");
        row.className = "stored-run";
        if (run.parent) row.dataset.parent = run.parent.run || "imported";
        const title = document.createElement("span");
        title.textContent = `${run.parent ? "↳ " : ""}${run.name} · ${run.length} frames · ${new Date(run.updated).toLocaleString()}`;
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
        if (!getRecording()?.readOnly) await getRecording()?.flush?.();
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
    r.scene = structuredClone(getScene());
    r.settings = structuredClone(getSettings());
    return r;
  }
  async import(file) {
    return importStoredFile(file, this.options.error);
  }
}
