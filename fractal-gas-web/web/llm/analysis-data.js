import { objective, selectedScore, bestNode } from "./config.js";

export const METRICS = {
  objective: {
    label: "Selected objective",
    unit: "raw score · follows the recorded optimization direction",
  },
  utility: {
    label: "Internal utility",
    unit: "signed objective · higher is better",
  },
  logp: { label: "Total log likelihood", unit: "ln p · higher is more likely" },
  probability: {
    label: "Accumulated probability",
    unit: "p(sequence | prompt) · logarithmic scale · higher is more likely",
  },
  mean: {
    label: "Mean token log likelihood",
    unit: "ln p / token · higher is more likely",
  },
  chunk_logp: {
    label: "Chunk log probability",
    unit: "ln p · new tokens only",
  },
  reward: {
    label: "Chunk reward",
    unit: "change in selected objective · higher is better",
  },
  fitness: {
    label: "Fitness",
    unit: "normalized fitness · higher favors survival",
    dynamic: true,
  },
  distance: {
    label: "Distance to companion",
    unit: "observation distance · higher is more diverse",
    dynamic: true,
  },
  clone_probability: {
    label: "Clone probability",
    unit: "clamped stochastic score · before protection / dead forcing",
    dynamic: true,
  },
  tokens: { label: "Token depth", unit: "generated tokens" },
  iteration: { label: "Generation iteration", unit: "recorded step" },
  status: { label: "Termination status", unit: "Partial / finished / capped" },
};
export const stateName = (n) =>
  n?.status === 1 ? "Finished" : n?.status === 2 ? "Capped" : "Partial";
export const number = (value) =>
  Number.isFinite(value)
    ? Number(value).toLocaleString("en", { maximumFractionDigits: 5 })
    : "Not recorded";
// Work in log space all the way through the palette and formatting. exp(logp)
// alone would turn perfectly valid long-sequence probabilities into zero.
export function probability(logp) {
  if (!Number.isFinite(logp)) return "Not recorded";
  const power = Math.floor(logp / Math.LN10);
  return `${Math.exp(logp - power * Math.LN10).toFixed(3)}e${power >= 0 ? "+" : ""}${power}`;
}
export const metricFormat = (key, value) =>
  key === "probability"
    ? probability(value)
    : key === "status"
      ? (["Partial", "Finished", "Capped"][value] ?? "Not recorded")
      : number(value);
export function decisionReason(d) {
  if (d.legacy)
    return `${d.cloned ? "Cloned" : "Retained"}. Fitness recovered from the prior slot; thresholds and protection were not recorded.`;
  const reasons = [];
  if (!d.alive)
    reasons.push("Dead walker: cloning requested regardless of the draw");
  else
    reasons.push(
      `Clone score ${number(d.clone_score)} ${d.clone_score > d.draw ? ">" : "≤"} draw ${number(d.draw)}`,
    );
  if (!d.leaf) reasons.push("Graph parent protected");
  if (d.donor_protected) reasons.push("Selected donor protected");
  if (d.best_protected) reasons.push("Best candidate protected");
  if (d.elite_protected) reasons.push("Elite protected");
  if (d.invalid_donor) reasons.push("Donor slot has no state");
  reasons.push(d.cloned ? "Cloned from donor, then continued" : "Retained");
  return reasons.join(". ") + ".";
}
export class AnalysisIndex {
  constructor(record) {
    this.record = record;
    this.nodes = record?.nodes ?? [];
    this.snapshots = record?.snapshots ?? [];
    this.children = this.nodes.map(() => []);
    this.birth = new Int32Array(this.nodes.length);
    this.events = this.nodes.map(() => []);
    this.origins = this.nodes.map(() => []);
    this.decisions = [];
    this.nodes.forEach((n) => {
      if (n.parent !== null) this.children[n.parent]?.push(n.id);
    });
    let count = 1;
    this.snapshots.forEach((s, ix) => {
      const step = ix + 1;
      const firstNew = count;
      for (; count < s.node_count; count++) this.birth[count] = step;
      let decisions = s.decisions;
      if (record.version === 1) {
        // Fitness is evaluated BEFORE the transition. Only the previous slot
        // population (or Wave's known reset roots) can identify that sequence.
        const prior =
          this.snapshots[ix - 1]?.walkers ??
          (record.config.algorithm === "wave" && ix === 0
            ? s.walkers.map((w) => ({ ...w, node: 0 }))
            : []);
        decisions = prior.map((w, slot) => ({
          slot,
          evaluated: w.node,
          result: s.walkers[slot]?.node ?? null,
          companion: prior[s.walkers[slot]?.fitnessCompanion]?.node ?? null,
          donor: prior[s.walkers[slot]?.cloneCompanion]?.node ?? null,
          companion_slot: s.walkers[slot]?.fitnessCompanion,
          donor_slot: s.walkers[slot]?.cloneCompanion,
          fitness: s.walkers[slot]?.fitness,
          cloned: s.walkers[slot]?.cloned,
          legacy: true,
        }));
      }
      for (const [ordinal, d] of (decisions ?? []).entries()) {
        const event = { ...d, step, key: `${step}:${ordinal}` };
        this.decisions.push(event);
        if (
          d.result !== null &&
          d.result >= firstNew &&
          d.result < s.node_count
        )
          this.origins[d.result]?.push(event);
        if (d.evaluated !== null) this.events[d.evaluated]?.push(event);
      }
    });
    this.domains = {};
    for (const key of Object.keys(METRICS)) {
      let low = Infinity,
        high = -Infinity;
      const include = (v) => {
        if (Number.isFinite(v)) {
          low = Math.min(low, v);
          high = Math.max(high, v);
        }
      };
      if (METRICS[key].dynamic)
        this.decisions.forEach((d) => include(this.decisionValue(d, key)));
      else
        this.nodes.forEach((n) => {
          if (n.id) include(this.value(n.id, key, this.snapshots.length).value);
        });
      this.domains[key] = low === Infinity ? null : [low, high];
    }
  }
  chain(id) {
    const out = [];
    for (let n = this.nodes[id]; n; n = this.nodes[n.parent]) out.push(n);
    return out.reverse();
  }
  chunk(id) {
    const n = this.nodes[id];
    return n ? n.text.slice(this.nodes[n.parent]?.text.length ?? 0) : "";
  }
  decisionValue(d, key) {
    return key === "clone_probability"
      ? Number.isFinite(d.clone_score)
        ? Math.max(0, Math.min(1, d.clone_score))
        : null
      : (d[key] ?? null);
  }
  value(id, key, step, slot = null) {
    const n = this.nodes[id];
    if (!n) return { value: null, count: 0 };
    if (METRICS[key].dynamic) {
      const events = this.events[id].filter(
        (e) => e.step <= step && (slot === null || e.slot === slot),
      );
      const last = events.at(-1)?.step;
      const values = events
        .filter((e) => e.step === last)
        .map((e) => this.decisionValue(e, key))
        .filter(Number.isFinite);
      return values.length
        ? {
            value: values.reduce((a, b) => a + b, 0) / values.length,
            count: values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            step: last,
          }
        : { value: null, count: 0 };
    }
    const parent = this.nodes[n.parent];
    const value = {
      objective: n.tokens ? selectedScore(n, this.record.config) : null,
      utility: n.tokens ? objective(n, this.record.config) : null,
      logp: n.logp,
      probability: n.logp,
      mean: n.tokens ? n.logp / n.tokens : null,
      chunk_logp: parent ? n.logp - parent.logp : null,
      reward: parent
        ? objective(n, this.record.config) -
          objective(parent, this.record.config)
        : null,
      tokens: n.tokens,
      iteration: this.birth[id],
      status: n.status,
    }[key];
    return { value, count: 1 };
  }
  best(step) {
    return bestNode(
      this.nodes.slice(
        0,
        this.snapshots[step - 1]?.node_count ?? this.nodes.length,
      ),
      this.record?.config,
    );
  }
  compare(a, b) {
    const left = this.chain(a),
      right = this.chain(b);
    let shared = null;
    for (
      let i = 0;
      i < Math.min(left.length, right.length) && left[i].id === right[i].id;
      i++
    )
      shared = left[i];
    return {
      shared,
      left: this.nodes[a],
      right: this.nodes[b],
      logp: this.nodes[b].logp - this.nodes[a].logp,
      tokens: this.nodes[b].tokens - this.nodes[a].tokens,
    };
  }
  model({
    step,
    mode = "tree",
    unused = false,
    filter = "all",
    search = "",
    selected = null,
    collapsed = new Set(),
  }) {
    const snapshot = this.snapshots[step - 1];
    const count = snapshot?.node_count ?? 1;
    const population = new Set(
      snapshot?.walkers.map((w) => w.node).filter((n) => n !== null),
    );
    const kept = new Set(population);
    // Descending IDs visit each ancestor once, even for a large shared prefix.
    for (let i = count - 1; i > 0; i--)
      if (kept.has(i)) kept.add(this.nodes[i].parent);
    const rows = [];
    if (mode === "graph" && snapshot) {
      const key = (slot) =>
        slot === 0 && snapshot.walkers[0]?.node === 0 ? "root" : `g:${slot}`;
      rows.push({
        key: "root",
        id: 0,
        parent: null,
        slot: snapshot.walkers[0]?.node === 0 ? 0 : null,
      });
      for (const w of snapshot.walkers) {
        if (key(w.slot) === "root" || (w.node === null && !unused)) continue;
        rows.push({
          key: key(w.slot),
          id: w.node,
          slot: w.slot,
          parent: key(w.parentSlot),
          unused: w.node === null,
        });
      }
    } else {
      for (let id = 0; id < count; id++)
        rows.push({
          key: id ? `n:${id}` : "root",
          id,
          slot: null,
          parent: id
            ? this.nodes[id].parent
              ? `n:${this.nodes[id].parent}`
              : "root"
            : null,
        });
    }
    const byKey = new Map(rows.map((r) => [r.key, r]));
    const query = search.trim().toLowerCase().replace(/^#/, "");
    const subtree = new Set();
    if (selected !== null) {
      subtree.add(selected);
      for (let i = selected + 1; i < count; i++)
        if (subtree.has(this.nodes[i].parent)) subtree.add(i);
    }
    const visible = new Set(["root"]);
    for (const row of rows) {
      const n = this.nodes[row.id];
      row.current = population.has(row.id);
      row.discarded = row.id !== null && !kept.has(row.id);
      row.match =
        (!query ||
          String(row.id) === query ||
          n?.text.toLowerCase().includes(query) ||
          `slot ${row.slot}` === query) &&
        (filter === "all" ||
          (filter === "population" && row.current) ||
          (filter === "finished" && n?.status > 0) ||
          (filter === "discarded" && row.discarded) ||
          (filter === "subtree" && subtree.has(row.id)));
      if (row.match) {
        let p = row;
        while (p && !visible.has(p.key)) {
          visible.add(p.key);
          p = byKey.get(p.parent);
        }
      }
    }
    const hidden = new Set();
    // Graph slot order need not be topological. Walk only the collapsed roots'
    // descendants using an adjacency list, with cycle protection.
    const children = new Map();
    for (const row of rows) {
      if (!children.has(row.parent)) children.set(row.parent, []);
      children.get(row.parent).push(row.key);
    }
    const stack = [...collapsed].flatMap((k) => children.get(k) ?? []);
    while (stack.length) {
      const k = stack.pop();
      if (k === "root" || collapsed.has(k) || hidden.has(k)) continue;
      hidden.add(k);
      stack.push(...(children.get(k) ?? []));
    }
    const nodes = rows.filter((r) => visible.has(r.key) && !hidden.has(r.key));
    const keys = new Set(nodes.map((n) => n.key));
    for (const row of nodes) {
      row.collapsed = collapsed.has(row.key);
      row.children = (children.get(row.key) ?? []).length;
    }
    const edges = nodes
      .filter((n) => n.parent && keys.has(n.parent) && n.key !== "root")
      .map((n) => ({ source: n.parent, target: n.key }));
    return {
      nodes,
      edges,
      population,
      count,
      matches: nodes.filter((n) => n.match && n.key !== "root").length,
    };
  }
}
export function color(key, value, domain) {
  if (!Number.isFinite(value)) return "#796e83";
  if (key === "status") return ["#ae96d1", "#7ef5df", "#efc87b"][value];
  const t =
    !domain || domain[0] === domain[1]
      ? 0.5
      : Math.max(0, Math.min(1, (value - domain[0]) / (domain[1] - domain[0])));
  const stops = [
    [97, 78, 141],
    [195, 156, 221],
    [126, 245, 223],
  ];
  const i = t < 0.5 ? 0 : 1,
    u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return `rgb(${stops[i].map((c, j) => Math.round(c + (stops[i + 1][j] - c) * u)).join(",")})`;
}
// Decode bytes as a stream: one glyph can span several provider tokens. Each
// display span carries all contributing tokens, while the exact answer remains
// the canonical provider text. Never render individual partial UTF-8 decodes.
export function tokenSpans(tokens) {
  const encoder = new TextEncoder();
  const intervals = [];
  let length = 0;
  const buffers = tokens.map((t, index) => {
    const bytes = t.bytes ? Uint8Array.from(t.bytes) : encoder.encode(t.text);
    intervals.push({
      start: length,
      end: length + bytes.length,
      token: { ...t, index },
    });
    length += bytes.length;
    return bytes;
  });
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const buffer of buffers) {
    bytes.set(buffer, offset);
    offset += buffer.length;
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    spans = [];
  let pos = 0,
    first = 0;
  for (const glyph of text) {
    const end = pos + encoder.encode(glyph).length;
    while (first < intervals.length && intervals[first].end <= pos) first++;
    const contributors = [];
    for (let i = first; i < intervals.length && intervals[i].start < end; i++)
      if (intervals[i].end > pos) contributors.push(intervals[i].token);
    const key = contributors.map((t) => t.index).join(",");
    if (spans.at(-1)?.key === key) spans.at(-1).text += glyph;
    else spans.push({ text: glyph, tokens: contributors, key });
    pos = end;
  }
  return spans;
}
