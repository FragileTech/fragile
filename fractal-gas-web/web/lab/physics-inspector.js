export class PhysicsInspector {
  constructor({ renderer, getState, getInfo, getSelection, request }) {
    this.options = { renderer, getState, getInfo, getSelection };
    this.lastProfile = null;
    const $ = (id) => document.getElementById(id);
    this.$ = $;
    $("inspect-physics").onchange = () => {
      const enabled = $("inspect-physics").checked;
      $("physics-readout").hidden = !enabled;
      if (!enabled) renderer.inspectVectors([]);
    };
    this.timer = setInterval(() => {
      const perf = renderer.performance || {},
        p = this.lastProfile;
      $("performance-readout").textContent =
        `${(perf.fps || 0).toFixed(0)} FPS · ${(perf.cpuMs || 0).toFixed(2)} ms render submission · ${perf.calls || 0} draws · ${((perf.triangles || 0) / 1000).toFixed(1)}k triangles${p ? ` · ${((p[1] / Math.max(p[0], 0.001)) * 1000).toFixed(0)} world frames/s · ${(p[10] / 1048576).toFixed(1)} MiB tracked native buffers` : ""}`;
      if ($("inspect-physics").checked && getState()) request();
    }, 250);
  }
  profile(p) {
    this.lastProfile = p;
  }
  update(vectors) {
    if (!this.$("inspect-physics").checked) return;
    const { renderer, getState, getInfo, getSelection } = this.options,
      state = getState(),
      info = getInfo();
    if (!state || !info) return;
    renderer.inspectVectors(vectors, state);
    const selected = getSelection(),
      body = selected?.key === "bodies" ? selected.i : renderer.controlled[0];
    if (body == null) return;
    const B = info[1];
    let force = [0, 0],
      contacts = 0,
      tension = 0;
    for (let i = 0; i < vectors.length; i += 8) {
      const [kind, a, b] = vectors.subarray(i, i + 3);
      if (a !== body && b !== body) continue;
      if (kind === 0) force = [vectors[i + 5], vectors[i + 6]];
      if (kind === 1) contacts++;
      if (kind === 2) tension = Math.max(tension, Math.abs(vectors[i + 7]));
    }
    this.$("physics-readout").textContent =
      `Body ${body} · velocity (${state[8 + 2 * B + body].toFixed(2)}, ${state[8 + 3 * B + body].toFixed(2)}) m/s · angular velocity ${state[8 + 5 * B + body].toFixed(2)} rad/s · external force (${force[0].toFixed(2)}, ${force[1].toFixed(2)}) N · nearby contacts ${contacts} · peak tether force ${tension.toFixed(2)} N\nCyan: velocity × 0.25 s · amber: external force × 0.05 m/N · red: nearby contact normals · magenta: tether force × 0.05 m/N`;
  }
}
