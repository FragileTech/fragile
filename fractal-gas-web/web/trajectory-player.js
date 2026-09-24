// Requests carry a generation so resets, new selections and rapid scrubbing
// cannot paint stale frames. Only one reconstruction request is in flight.
export function trajectoryPlayer(send, release = () => {}) {
  const $ = id => document.getElementById(`trajectory-${id}`);
  const canvas = $("screen"), ctx = canvas.getContext("2d");
  let generation = 0, length = 0, index = 0, desired = 0, busy = false;
  let playing = false, timer = null, walker = 0, iteration = 0;
  const controls = ["seek", "home", "prev", "play", "next"];
  function stop() { playing = false; clearTimeout(timer); $("play").textContent = "Play"; }
  function clear(available = false) {
    release();
    stop(); generation++; length = 0; busy = false; index = desired = 0;
    for (const id of controls) $(id).disabled = true;
    $("load").disabled = $("best").disabled = !available;
    $("seek").value = $("seek").max = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    $("status").textContent = "Load a fixed path to replay while the search continues.";
  }
  function request() {
    if (busy || !length) return;
    busy = true;
    send({ type: "trajectoryFrame", index: desired, request: generation });
  }
  function seek(target) {
    desired = Math.max(0, Math.min(length - 1, target));
    $("seek").value = desired;
    request();
  }
  function load(best = false) {
    const value = best ? "" : $("walker").value;
    if (value !== "" && (!Number.isInteger(Number(value)) || Number(value) < 0)) {
      $("status").textContent = "Choose a nonnegative walker index.";
      return;
    }
    clear(true);
    if (best) $("walker").value = "";
    $("status").textContent = "Loading ancestry…";
    send({ type: "trajectorySelect", walker: value === "" ? -1 : Number(value), request: generation });
  }
  $("load").onclick = () => load();
  $("best").onclick = () => load(true);
  $("walker").onchange = () => load();
  $("seek").oninput = () => { stop(); seek(Number($("seek").value)); };
  $("home").onclick = () => { stop(); seek(0); };
  $("prev").onclick = () => { stop(); seek(desired - 1); };
  $("next").onclick = () => { stop(); seek(desired + 1); };
  $("play").onclick = () => {
    if (playing) { stop(); return; }
    playing = true; $("play").textContent = "Pause";
    seek(index >= length - 1 ? 0 : index);
  };
  $("fullscreen").onclick = () => {
    const result = document.fullscreenElement ? document.exitFullscreen() :
      document.getElementById("trajectory-panel").requestFullscreen();
    result?.catch(() => { $("status").textContent = "Full screen is unavailable in this browser."; });
  };
  function receive(msg) {
    if (msg.request !== generation) return;
    if (msg.error) {
      stop(); busy = false;
      $("status").textContent = msg.error;
      return;
    }
    if (msg.type === "trajectorySelected") {
      length = msg.length; walker = msg.walker; iteration = msg.iteration;
      $("walker").max = Math.max(0, msg.walkerCount - 1);
      $("seek").max = Math.max(0, length - 1);
      for (const id of controls) $(id).disabled = !length;
      seek(0);
      return;
    }
    busy = false;
    if (msg.index !== desired || !msg.ready) {
      $("status").textContent = `Reconstructing state ${desired + 1} / ${length}…`;
      request();
      return;
    }
    index = msg.index;
    canvas.width = msg.frameWidth; canvas.height = msg.frameHeight;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(msg.frame), canvas.width, canvas.height), 0, 0);
    $("seek").value = index;
    $("status").textContent = `Walker ${walker} · state ${index + 1} / ${length} · captured at update ${iteration}`;
    if (playing && index < length - 1)
      timer = setTimeout(() => seek(index + 1), 1000 / Number($("speed").value));
    else if (playing) stop();
  }
  clear();
  return { clear, receive };
}
