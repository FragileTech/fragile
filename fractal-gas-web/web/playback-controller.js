// Owns one disposable player. Search owns only capture; playback owns all seeks.
export function playbackController({ sendSearch, receive, memory, config }) {
  let worker = null, runtime = 0, request = null, timeout = null;
  function dispose() {
    clearTimeout(timeout); timeout = null;
    runtime++; request = null;
    worker?.terminate(); worker = null; memory(0);
  }
  function fail(error, selectedRequest) {
    dispose();
    receive({ type: "trajectoryFrame", request: selectedRequest, error });
  }
  function send(msg) {
    if (msg.type === "trajectorySelect") {
      dispose(); request = msg.request; sendSearch(msg);
    } else if (worker && msg.request === request) {
      const token = runtime, selectedRequest = request;
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        if (token === runtime) fail("Playback frame timed out; load the path again.", selectedRequest);
      }, 30000);
      worker.postMessage({ ...msg, runtime });
    }
  }
  function captured(msg) {
    if (msg.request !== request) return;
    if (msg.error) { receive(msg); return; }
    const token = runtime, selectedRequest = request;
    try {
      worker = new Worker(new URL("./playback-worker.js", import.meta.url), { type: "module" });
      worker.onerror = event => { if (token === runtime) fail(`Playback: ${event.message}`, selectedRequest); };
      timeout = setTimeout(() => fail("Playback initialization timed out; load the path again.", selectedRequest), 120000);
      worker.onmessage = ({ data }) => {
        if (token !== runtime || data.runtime !== runtime || data.request !== request) return;
        if (data.error) { fail(data.error, selectedRequest); return; }
        clearTimeout(timeout); memory(data.playbackBytes);
        receive(data);
      };
      const setup = config();
      const rom = setup.rom.slice(0);
      worker.postMessage({ ...msg, type: "init", runtime, rom, params: setup.params },
        [rom, msg.root, msg.actions]);
    } catch (error) { fail(`Playback: ${error.message || error}`, selectedRequest); }
  }
  return { send, captured, dispose };
}
