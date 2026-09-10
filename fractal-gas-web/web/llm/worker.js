import { TogetherScorer } from "./scoring.js";
import { configuration } from "./config.js";
import { OpenRouter } from "./openrouter.js";
import { Recording } from "./recording.js";
import { TokenEnvironment } from "./environment.js";
import { NativeLlm } from "./native.js";
import { STOP_LABELS } from "./run-control.js";
let record,
  engine,
  env,
  abort,
  busy = false,
  continuous = false,
  failed = false;
const send = (type, value) => postMessage({ type, ...value });
function update(status) {
  send("update", { status, data: record?.data });
}
async function drive(single) {
  if (busy || failed || !engine) return;
  if (record.data.run.stop_reason) {
    update(STOP_LABELS[record.data.run.stop_reason]);
    send("idle", { failed });
    return;
  }
  busy = true;
  continuous = !single;
  try {
    do {
      const snapshot = await engine.advance();
      abort.signal.throwIfAborted();
      env.commit({
        ...snapshot,
        step: record.data.snapshots.length + 1,
        time: Date.now(),
      });
      if (record.data.run.stop_reason) continuous = false;
      update(
        STOP_LABELS[record.data.run.stop_reason] ??
          (continuous ? "Running" : "Paused"),
      );
    } while (continuous && !abort.signal.aborted);
    update(STOP_LABELS[record.data.run.stop_reason] ?? "Paused");
  } catch (error) {
    continuous = false;
    failed = true;
    try {
      record.append("errors", { message: error.message, time: Date.now() });
    } catch {}
    update(abort.signal.aborted ? "Stopped" : error.message);
  } finally {
    busy = false;
    send("idle", { failed });
  }
}
onmessage = async ({ data }) => {
  if (data.type === "pause") {
    continuous = false;
    return;
  }
  if (data.type === "stop") {
    continuous = false;
    failed = true;
    abort?.abort(new Error("Stopped"));
    if (!busy) update("Stopped");
    return;
  }
  if (data.type === "export") {
    send("export", { text: record?.export() });
    return;
  }
  if (data.type === "catalogs") {
    if (busy) return;
    try {
      send("catalogs", await new OpenRouter(data.key).catalogs());
    } catch (e) {
      send("error", { message: e.message });
    }
    return;
  }
  if (data.type === "start") {
    if (busy) return;
    busy = true;
    failed = false;
    continuous = !data.single;
    try {
      engine?.close();
      engine = null;
      abort = new AbortController();
      const config = configuration(data.config);
      record = new Recording(config);
      update("Checking model support…");
      const api = new OpenRouter(data.key, {
        signal: abort.signal,
        onRequestStart: (r) =>
          record.append("requests", { ...r, status: "started" }),
        onRequest: (r) => record.append("requests", r),
      });
      let scorer = null;
      if (config.objective === "xed") {
        scorer = new TogetherScorer(data.togetherKey, {
          signal: abort.signal,
          concurrency: config.concurrency,
          onRequestStart: (r) => record.append("requests", r),
          onRequest: (r) => record.append("requests", r),
        });
      }
      const scoring = scorer ? await scorer.prepare(config) : null;
      const metadata = await api.prepare(config);
      if (scoring) metadata.scoring = scoring;
      record.metadata(metadata);
      abort.signal.throwIfAborted();
      env = new TokenEnvironment(config, api, record, scorer);
      engine = await NativeLlm.create(
        { ...config, dimensions: metadata.dimensions },
        (r) => env.transition(r),
      );
      busy = false;
      await drive(!continuous);
    } catch (error) {
      busy = false;
      failed = true;
      try {
        record?.append("errors", { message: error.message });
      } catch {}
      update(abort?.signal.aborted ? "Stopped" : error.message);
      send("idle", { failed });
    }
    return;
  }
  if (data.type === "run" || data.type === "step")
    await drive(data.type === "step");
};
