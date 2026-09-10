import { objectiveLabel } from "./config.js";
import {
  AnalysisIndex,
  METRICS,
  number,
  probability,
  metricFormat,
  color,
  stateName,
  decisionReason,
  tokenSpans,
} from "./analysis-data.js";
import { AnalysisCanvas } from "./analysis-canvas.js";
const $ = (id) => document.getElementById(id);
const el = (tag, text, className) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (className) n.className = className;
  return n;
};
const button = (text, fn) => {
  const b = el("button", text);
  b.type = "button";
  b.onclick = fn;
  return b;
};
export function renderTokens(target, tokens, onToken) {
  target.replaceChildren(
    ...tokenSpans(tokens).map((s) => {
      const span = el(onToken ? "button" : "span", s.text);
      const description = s.tokens
        .map(
          (t) =>
            `Token ${t.index + 1}: log p ${number(t.logprob)}, p ${probability(t.logprob)}`,
        )
        .join("\n");
      span.title = description;
      if (onToken) {
        span.type = "button";
        span.setAttribute("aria-label", description);
        span.onclick = () => onToken(s);
      }
      span.style.background = `rgba(144,89,164,${Math.min(0.65, 0.08 + s.tokens.reduce((v, t) => v - t.logprob, 0) / 8)})`;
      return span;
    }),
  );
}
export class AnalysisView {
  constructor(host, { onSelect, onStep, onFollow, onStatus }) {
    this.onSelect = onSelect;
    this.onStep = onStep;
    this.onFollow = onFollow;
    this.onStatus = onStatus;
    this.collapsed = new Set();
    this.pins = [];
    this.selectedKey = null;
    this.active = false;
    host.innerHTML = `
      <div class="analysis-heading"><div><h2>Generation atlas</h2><p id="analysis-description" class="hint">Every continuation, from prompt to answer.</p></div><button id="analysis-best">Show best</button></div>
      <div class="analysis-controls">
        <label>View<select id="analysis-mode"><option value="tree">Generation tree</option><option value="graph">Stored Graph</option></select></label>
        <label>Horizontal position<select id="analysis-axis"><option value="tokens">Generated tokens</option><option value="iteration">Generation iteration</option></select></label>
        <label>Color by<select id="analysis-metric"></select></label>
        <label class="analysis-search">Find a branch<input id="analysis-search" type="search" placeholder="Node ID or generated text" /></label>
        <label>Show<select id="analysis-filter"><option value="all">All branches</option><option value="population">Current population</option><option value="finished">Finished / capped</option><option value="discarded">Discarded branches</option><option value="subtree">Selected subtree</option></select></label>
      </div>
      <div class="analysis-playback" aria-label="Analysis playback">
        <button id="analysis-prev" aria-label="Previous iteration">←</button><button id="analysis-play">Play history</button><button id="analysis-next" aria-label="Next iteration">→</button>
        <input id="analysis-iteration" aria-label="Analysis iteration" type="range" min="1" value="1" max="1" /><output id="analysis-step">0 / 0</output>
        <label class="check"><input id="analysis-follow" type="checkbox" checked />Follow latest</label>
      </div>
      <div class="analysis-workspace">
        <section class="analysis-graph" aria-label="Branch visualization">
          <div class="analysis-navigation"><button id="analysis-fit">Fit</button><button id="analysis-zoom-in" aria-label="Zoom in">+</button><button id="analysis-zoom-out" aria-label="Zoom out">−</button><button id="analysis-focus">Focus selected</button><button id="analysis-collapse">Collapse branch</button><button id="analysis-expand">Expand all</button><button id="analysis-png">Save PNG</button></div>
          <div class="analysis-stage"><canvas id="analysis-canvas" tabindex="0" role="img" aria-label="Generation tree. Use arrow keys to navigate nodes, plus and minus to zoom, F to focus, Enter to collapse a branch." aria-describedby="analysis-selection"></canvas><canvas id="analysis-minimap" width="190" height="110" aria-label="Tree overview; click to move the main view"></canvas><p id="analysis-empty">Run a generation or import a recording to explore its branches.</p></div>
          <div id="analysis-legend" class="analysis-legend"></div>
          <div class="analysis-options"><label class="check"><input id="analysis-overlays" type="checkbox" />Decision connections</label><label class="check"><input id="analysis-unused" type="checkbox" />Show unused Graph slots</label><span class="hint">Solid: ancestry · dotted mint: companion · dashed gold: donor</span></div>
          <p id="analysis-selection" class="hint" role="status"></p>
          <details class="analysis-node-list"><summary id="analysis-node-count">Visible nodes</summary><div id="analysis-node-links" class="ancestry"></div></details>
        </section>
        <section class="analysis-inspector" aria-label="Analysis node inspector">
          <div class="section-title"><h2 id="analysis-node-title">Node inspector</h2><button id="analysis-pin">Pin to compare</button></div>
          <p id="analysis-node-summary" class="hint">Select a node to inspect the answer and its decisions.</p>
          <p class="hint">Recorded sequence ancestry</p><div id="analysis-ancestors" class="ancestry"></div><div id="analysis-slot-links" class="ancestry"></div>
          <div id="analysis-node-metrics" class="analysis-metrics"></div>
          <details><summary>Original prompt</summary><pre id="analysis-prompt"></pre></details>
          <div class="analysis-answer-heading"><h3>Answer up to this node</h3><button id="analysis-copy">Copy answer</button><button id="analysis-copy-chunk">Copy chunk</button></div>
          <p class="hint">The highlighted text is the new chunk at this node.</p><pre id="analysis-answer"></pre>
          <details><summary>Token probabilities and chunk boundaries</summary><div id="analysis-tokens" class="tokens"></div><pre id="analysis-token-detail" class="hint">Select a token to see its probability and bytes.</pre></details>
          <div id="analysis-children" class="ancestry"></div><h3>Generation decision</h3><p class="hint">The decision that produced this chunk evaluated the previous population.</p><div id="analysis-origins" class="ancestry"></div>
          <details open><summary>Decision history</summary><p id="analysis-measured" class="hint"></p><div id="analysis-decision-pages" class="ancestry"></div><div id="analysis-decisions"></div><div id="analysis-decision-detail"></div></details>
        </section>
      </div>
      <section id="analysis-comparison" class="analysis-comparison" hidden><div class="section-title"><h2>Branch comparison</h2><button id="analysis-clear-pins">Clear pins</button></div><p id="analysis-compare-summary" class="hint"></p><div id="analysis-compare-columns"></div></section>`;
    this.canvas = new AnalysisCanvas(
      $("analysis-canvas"),
      $("analysis-minimap"),
      (r) => this.selectRow(r),
      (key) => {
        this.collapsed.has(key)
          ? this.collapsed.delete(key)
          : this.collapsed.add(key);
        this.render();
      },
    );
    $("analysis-metric").replaceChildren(
      ...Object.entries(METRICS).map(([key, m]) => {
        const o = el("option", m.label);
        o.value = key;
        return o;
      }),
    );
    for (const id of ["mode", "axis", "metric", "filter", "overlays", "unused"])
      $(`analysis-${id}`).onchange = () => this.render();
    let searchTimer;
    $("analysis-search").oninput = () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => this.render(), 120);
    };
    $("analysis-fit").onclick = () => this.canvas.fit();
    $("analysis-zoom-in").onclick = () => this.canvas.zoom(1.35);
    $("analysis-zoom-out").onclick = () => this.canvas.zoom(1 / 1.35);
    $("analysis-focus").onclick = () => this.reveal(this.selected);
    $("analysis-collapse").onclick = () => {
      const key = this.canvas.selectedKey;
      if (key) {
        this.collapsed.has(key)
          ? this.collapsed.delete(key)
          : this.collapsed.add(key);
        this.render();
      }
    };
    $("analysis-expand").onclick = () => {
      this.collapsed.clear();
      this.render();
    };
    $("analysis-best").onclick = () => {
      const n = this.index.best(this.step);
      if (n) this.reveal(n.id);
    };
    $("analysis-iteration").oninput = () =>
      this.seek(Number($("analysis-iteration").value));
    $("analysis-prev").onclick = () => this.seek(this.step - 1);
    $("analysis-next").onclick = () => this.seek(this.step + 1);
    $("analysis-follow").onchange = () => {
      this.pause();
      this.onFollow($("analysis-follow").checked);
    };
    $("analysis-play").onclick = () => {
      if (this.timer) {
        this.pause();
        return;
      }
      this.onFollow(false);
      if (this.step >= this.index.snapshots.length) this.onStep(1);
      $("analysis-play").textContent = "Pause history";
      this.timer = setInterval(() => {
        if (this.step >= this.index.snapshots.length) {
          this.pause();
          return;
        }
        this.onStep(this.step + 1);
      }, 900);
    };
    $("analysis-pin").onclick = () => {
      if (this.selected !== null && !this.pins.includes(this.selected)) {
        if (this.pins.length === 2) this.pins.shift();
        this.pins.push(this.selected);
      }
      this.render();
    };
    $("analysis-clear-pins").onclick = () => {
      this.pins = [];
      this.render();
    };
    $("analysis-copy").onclick = () =>
      this.copy(this.index.nodes[this.selected]?.text ?? "");
    $("analysis-copy-chunk").onclick = () =>
      this.copy(this.index.chunk(this.selected));
    $("analysis-png").onclick = async () => {
      try {
        const blob = await this.canvas.export();
        downloadBlob(blob, `llm-analysis-step-${this.step}.png`);
      } catch (e) {
        this.onStatus(`Could not export visualization: ${e.message}`);
      }
    };
  }
  pause() {
    clearInterval(this.timer);
    this.timer = null;
    $("analysis-play").textContent = "Play history";
  }
  seek(step) {
    this.pause();
    this.onFollow(false);
    this.onStep(Math.max(1, Math.min(this.index.snapshots.length, step)));
  }
  selectRow(row) {
    this.selectedKey = row.key;
    this.decision = null;
    this.unusedSlot = row.unused ? row.slot : null;
    this.onSelect(row.id);
  }
  reveal(id) {
    if (id === null || !this.index.nodes[id]) return;
    this.collapsed.clear();
    $("analysis-filter").value = "all";
    $("analysis-search").value = "";
    if (
      $("analysis-mode").value === "graph" &&
      !this.index.snapshots[this.step - 1]?.walkers.some((w) => w.node === id)
    )
      $("analysis-mode").value = "tree";
    this.unusedSlot = null;
    this.onSelect(id);
    this.canvas.focus();
  }
  setActive(active) {
    this.active = active;
    if (!active) this.pause();
    else {
      this.render();
      if (this.needsFit) {
        this.canvas.fit();
        this.needsFit = false;
      }
    }
  }
  update(record, step, selected, follow) {
    // New worker messages replace the object. Updating a cursor reuses the index.
    if (record !== this.record) {
      const fresh =
        !this.record ||
        !record ||
        (record.config !== this.record.config &&
          (record.snapshots.length < this.record.snapshots.length ||
            record.nodes[1]?.request_id !== this.record.nodes[1]?.request_id));
      this.record = record;
      this.index = new AnalysisIndex(record);
      if (fresh) {
        this.pause();
        this.pins = [];
        this.collapsed.clear();
        this.decision = null;
        this.selectedKey = null;
        this.decisionNode = undefined;
        this.unusedSlot = null;
        $("analysis-mode").value =
          record?.config.algorithm === "graph" ? "graph" : "tree";
        this.needsFit = true;
      }
    }
    if (!this.index) this.index = new AnalysisIndex(record);
    this.step = step;
    this.selected = selected;
    this.follow = follow;
    if (this.decision && this.decision.step > step) this.decision = null;
    if (this.active) this.render();
  }
  render() {
    if (!this.active || !this.index) return;
    const count = this.index.snapshots.length;
    const metric = $("analysis-metric").value,
      mode = $("analysis-mode").value;
    $("analysis-mode").querySelector('[value="graph"]').disabled =
      this.record?.config.algorithm !== "graph";
    $("analysis-unused").disabled = mode !== "graph";
    $("analysis-description").textContent =
      mode === "graph"
        ? "Stored population relationships at this step. Repeated prefixes remain separate slots."
        : "Immutable parent links, including branches no longer in the population.";
    $("analysis-iteration").max = Math.max(1, count);
    $("analysis-iteration").value = Math.max(1, this.step);
    $("analysis-iteration").disabled = !count;
    $("analysis-step").value = `${this.step} / ${count}`;
    $("analysis-follow").checked = this.follow;
    $("analysis-prev").disabled = this.step <= 1;
    $("analysis-next").disabled = this.step >= count;
    $("analysis-play").disabled = count < 2;
    $("analysis-empty").hidden = this.index.nodes.length > 1;
    $("analysis-best").disabled = !this.index.best(this.step);
    const model = this.index.model({
      step: this.step,
      mode,
      unused: $("analysis-unused").checked,
      filter: $("analysis-filter").value,
      search: $("analysis-search").value,
      selected: this.selected,
      collapsed: this.collapsed,
    });
    const shared =
      this.pins.length === 2
        ? new Set(
            this.index
              .chain(this.index.compare(...this.pins).shared?.id)
              .map((n) => n.id),
          )
        : new Set();
    this.canvas.setData(model, this.index, {
      axis: $("analysis-axis").value,
      metric,
      step: this.step,
      selected: this.selected,
      selectedKey: this.selectedKey,
      decision: this.decision,
      overlays: $("analysis-overlays").checked,
      shared,
    });
    if (this.needsFit && this.index.nodes.length > 1) {
      this.canvas.fit();
      this.needsFit = false;
    }
    $("analysis-node-count").textContent =
      `${model.matches} matching nodes · ${model.nodes.length} including ancestors`;
    $("analysis-node-links").replaceChildren(
      ...model.nodes
        .filter((n) => n.match || n.key === "root")
        .slice(0, 100)
        .map((n) =>
          button(
            n.unused
              ? `Unused ${n.slot}`
              : n.key === "root"
                ? "Prompt"
                : `#${n.id}${n.slot !== null ? ` / slot ${n.slot}` : ""}`,
            () => {
              this.selectRow(n);
              this.canvas.focus(n.key);
            },
          ),
        ),
    );
    if (model.nodes.length > 100)
      $("analysis-node-links").append(
        el("span", "First 100 listed. Search to narrow the list.", "hint"),
      );
    const domain = this.index.domains[metric],
      legend = $("analysis-legend");
    legend.replaceChildren(
      el(
        "strong",
        metric === "objective"
          ? objectiveLabel(this.record?.config)
          : METRICS[metric].label,
      ),
    );
    if (metric === "status") {
      ["Partial", "Finished", "Capped"].forEach((label, i) => {
        const s = el("span", label);
        s.style.color = color(metric, i, domain);
        legend.append(s);
      });
    } else {
      const scale = el("span", undefined, "analysis-scale");
      scale.style.background = `linear-gradient(to right,${color(metric, domain?.[0], domain)},${color(metric, domain?.[1], domain)})`;
      legend.append(
        el("span", metricFormat(metric, domain?.[0])),
        scale,
        el("span", metricFormat(metric, domain?.[1])),
      );
    }
    legend.append(
      el("span", METRICS[metric].unit, "hint"),
      el("span", "● Not recorded", "analysis-missing"),
    );
    this.inspect(metric);
    this.compare();
    $("analysis-collapse").textContent = this.collapsed.has(
      this.canvas.selectedKey,
    )
      ? "Expand branch"
      : "Collapse branch";
  }
  inspect(metric) {
    const n = this.index.nodes[this.selected],
      inspector = $("analysis-node-title");
    inspector.textContent =
      this.unusedSlot !== null && this.unusedSlot !== undefined
        ? `Unused slot ${this.unusedSlot}`
        : n
          ? `${n.id === 0 ? "Prompt root" : `Node #${n.id}`}${this.canvas.selectedKey?.startsWith("g:") ? ` · slot ${this.canvas.selectedKey.slice(2)}` : ""}`
          : "Node inspector";
    const valid = !!n;
    for (const id of ["pin", "copy", "copy-chunk", "focus", "collapse"])
      $(`analysis-${id}`).disabled = !valid;
    $("analysis-node-summary").textContent = n
      ? `${n.tokens} tokens · ${stateName(n)} · born at step ${this.index.birth[n.id]}${n.finish_reason ? ` · finish: ${n.finish_reason}` : ""}`
      : "Select a generated node to inspect its text. Unused slots have no sequence.";
    $("analysis-selection").textContent = n
      ? `${inspector.textContent} selected. ${this.canvas.ancestry.size} visible ancestors. Arrow keys navigate; double-click or Enter collapses branches.`
      : "Drag to pan, scroll to zoom, or choose a node from the list.";
    const chain = this.index.chain(this.selected);
    $("analysis-ancestors").replaceChildren(
      ...(chain.length > 60 ? [chain[0], ...chain.slice(-59)] : chain).map(
        (p) => button(p.id ? `#${p.id}` : "Prompt", () => this.reveal(p.id)),
      ),
    );
    const graphLinks = $("analysis-slot-links");
    graphLinks.replaceChildren();
    if ($("analysis-mode").value === "graph") {
      const row = this.canvas.positions.get(this.canvas.selectedKey);
      const relatives = [
        ["Graph parent", this.canvas.positions.get(row?.parent)],
        ...this.canvas.rows
          .filter((p) => p.parent === row?.key && p.key !== row?.key)
          .map((p) => ["Graph child", p]),
      ];
      for (const [label, p] of relatives)
        if (p)
          graphLinks.append(
            button(`${label}: slot ${p.slot}`, () => {
              this.selectRow(p);
              this.canvas.focus(p.key);
            }),
          );
    }
    const metrics = $("analysis-node-metrics");
    metrics.replaceChildren();
    for (const key of [
      "objective",
      "utility",
      "logp",
      "probability",
      "mean",
      "chunk_logp",
      "reward",
      "fitness",
      "distance",
      "clone_probability",
    ]) {
      const v = this.index.value(this.selected, key, this.step);
      const item = el("div");
      item.append(
        el("span", METRICS[key].label),
        el("strong", metricFormat(key, v.value)),
      );
      if (METRICS[key].dynamic && v.count)
        item.append(
          el(
            "small",
            `${v.count} evaluation${v.count === 1 ? "" : "s"} · step ${v.step}${v.count > 1 ? ` · range ${number(v.min)} to ${number(v.max)}` : ""}`,
          ),
        );
      metrics.append(item);
    }
    $("analysis-prompt").textContent = this.record?.config.prompt ?? "";
    const answer = $("analysis-answer");
    answer.replaceChildren();
    if (n) {
      answer.append(
        document.createTextNode(this.index.nodes[n.parent]?.text ?? ""),
        el("mark", this.index.chunk(n.id)),
      );
    }
    const tokenHost = $("analysis-tokens");
    tokenHost.replaceChildren();
    $("analysis-token-detail").textContent =
      "Select a token to see its probability and bytes.";
    for (const part of chain.filter((p) => p.id)) {
      const section = el("div", undefined, "analysis-token-chunk");
      section.append(
        el(
          "div",
          `Chunk #${part.id} · ${part.token_data.length} tokens · ${number(part.logp - this.index.nodes[part.parent].logp)} ln p`,
          "hint",
        ),
      );
      const text = el("div");
      renderTokens(text, part.token_data, (s) => {
        $("analysis-token-detail").textContent =
          `Chunk #${part.id}\n` +
          s.tokens
            .map(
              (t) =>
                `Token ${t.index + 1}\nlog p = ${number(t.logprob)}\np = ${probability(t.logprob)}\nbytes = ${JSON.stringify(t.bytes ?? Array.from(new TextEncoder().encode(t.text)))}`,
            )
            .join("\n\n");
      });
      section.append(text);
      tokenHost.append(section);
    }
    const children = this.index.children[this.selected] ?? [];
    $("analysis-children").replaceChildren(
      ...children
        .filter((id) => this.index.birth[id] <= this.step)
        .slice(0, 100)
        .map((id) => button(`Child #${id}`, () => this.reveal(id))),
    );
    const origins = this.index.origins[this.selected] ?? [];
    $("analysis-origins").replaceChildren(
      ...origins.map((d) =>
        button(
          `Step ${d.step} · slot ${d.slot} · evaluated ${d.evaluated === null ? "unused slot" : `#${d.evaluated}`}`,
          () => this.selectDecision(d),
        ),
      ),
    );
    if (!origins.length)
      $("analysis-origins").append(
        el(
          "span",
          n?.id &&
            this.record?.config.algorithm === "graph" &&
            this.index.birth[n.id] === 1
            ? "Initial generation; no cloning decision."
            : "No generating decision recorded.",
          "hint",
        ),
      );
    const events = this.index.events[this.selected] ?? [],
      past = events.filter((d) => d.step <= this.step);
    $("analysis-measured").textContent = !past.length
      ? "Unmeasured at this step. This prefix has not yet been evaluated for cloning."
      : `${past.length} recorded evaluations up to this step. Values belong to the sequence before cloning.${this.record?.version === 1 ? " Legacy recording: thresholds, distances and protection are Not recorded." : ""}`;
    const pageCount = Math.max(1, Math.ceil(events.length / 100));
    if (this.decisionNode !== this.selected) {
      this.decisionNode = this.selected;
      this.decisionPage = Math.max(0, Math.ceil(past.length / 100) - 1);
    }
    this.decisionPage = Math.min(this.decisionPage ?? 0, pageCount - 1);
    const pages = $("analysis-decision-pages");
    pages.replaceChildren();
    if (pageCount > 1) {
      const previous = button("Earlier decisions", () => {
        this.decisionPage--;
        this.render();
      });
      const next = button("Later decisions", () => {
        this.decisionPage++;
        this.render();
      });
      previous.disabled = this.decisionPage === 0;
      next.disabled = this.decisionPage === pageCount - 1;
      pages.append(
        previous,
        el("span", `${this.decisionPage + 1} / ${pageCount}`, "hint"),
        next,
      );
    }
    $("analysis-decisions").replaceChildren(
      ...events
        .slice(this.decisionPage * 100, (this.decisionPage + 1) * 100)
        .map((d) => {
          const b = button(
            `Step ${d.step} · slot ${d.slot} · ${d.cloned ? "cloned" : "retained"}${d.step > this.step ? " · later" : ""}`,
            () => this.selectDecision(d),
          );
          if (this.decision?.key === d.key) b.className = "selected";
          return b;
        }),
    );
    const detail = $("analysis-decision-detail");
    detail.replaceChildren();
    const d = this.decision;
    if (d) {
      detail.append(
        el("h3", `Decision at step ${d.step}, slot ${d.slot}`),
        el("p", decisionReason(d), "hint"),
      );
      const links = el("div", undefined, "ancestry");
      for (const [label, id] of [
        ["Evaluated", d.evaluated],
        ["Companion", d.companion],
        ["Donor", d.donor],
        ["Result", d.result],
      ])
        if (id !== null)
          links.append(button(`${label} #${id}`, () => this.reveal(id)));
        else links.append(el("span", `${label}: unused slot`, "hint"));
      detail.append(links);
      const list = el("dl", undefined, "analysis-decision-values");
      for (const [label, value] of [
        ["Raw distance", d.distance],
        ["Normalized distance", d.distance_norm],
        ["Normalized reward", d.reward_norm],
        ["Other factor", d.other],
        ["Fitness", d.fitness],
        ["Donor fitness", d.donor_fitness],
        ["Clone score (unclamped)", d.clone_score],
        ["Uniform draw", d.draw],
      ])
        list.append(el("dt", label), el("dd", number(value)));
      detail.append(
        list,
        el(
          "p",
          `Fitness = normalized distance ^ ${this.record.config.distance_coef} × normalized reward ^ ${this.record.config.reward_coef} × other factor. ${d.legacy ? "" : `Included in leaf reward normalization: ${d.normalization_leaf ? "yes" : "no"}.`}`,
          "hint",
        ),
      );
      if (this.record.config.algorithm === "graph")
        detail.append(
          el(
            "p",
            "Graph keeps its recorded observation geometry. The prompt slot’s reset observation can differ from its empty sequence embedding.",
            "hint",
          ),
        );
    }
  }
  selectDecision(d) {
    this.pause();
    this.onFollow(false);
    this.decision = d;
    $("analysis-overlays").checked = true;
    // The completed Graph may have replaced the original slot. The immutable
    // tree keeps every sequence involved in this recorded decision available.
    $("analysis-mode").value = "tree";
    this.collapsed.clear();
    $("analysis-filter").value = "all";
    $("analysis-search").value = "";
    this.onStep(d.step);
    this.canvas.focus();
  }
  compare() {
    const box = $("analysis-comparison");
    box.hidden = !this.pins.length;
    $("analysis-pin").textContent = this.pins.includes(this.selected)
      ? "Pinned"
      : "Pin to compare";
    const columns = $("analysis-compare-columns");
    columns.replaceChildren();
    const result =
      this.pins.length === 2 ? this.index.compare(...this.pins) : null;
    $("analysis-compare-summary").textContent = result
      ? `Shared ancestry through #${result.shared.id} (${result.shared.tokens} tokens). Right − left: ${number(result.logp)} total log likelihood; ${result.tokens >= 0 ? "+" : ""}${result.tokens} tokens. Gold edges mark shared recorded ancestry.`
      : "Pin a second node to compare its continuation with this one.";
    for (const id of this.pins) {
      const n = this.index.nodes[id],
        column = el("article");
      column.append(
        button(`Inspect #${id}`, () => {
          if (this.index.birth[id] > this.step) this.seek(this.index.birth[id]);
          this.reveal(id);
        }),
        el(
          "p",
          `${n.tokens} tokens · total ln p ${number(n.logp)} · mean ln p ${number(n.logp / Math.max(1, n.tokens))} · ${stateName(n)}`,
          "hint",
        ),
      );
      const text = el("pre");
      if (result) {
        text.append(
          el("span", result.shared.text, "analysis-shared"),
          el("mark", n.text.slice(result.shared.text.length)),
        );
      } else text.textContent = n.text;
      column.append(text);
      columns.append(column);
    }
  }
  async copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      this.onStatus("Copied to clipboard.");
    } catch {
      this.onStatus("Clipboard unavailable. Select and copy the answer text.");
    }
  }
}
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob),
    a = el("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
