import { element as el, fmt, chart, download } from "./comparison-charts.js";
import { METHOD_STYLE } from "./comparison-metrics.js";
import { usageTotals } from "./benchmark-data.js";
import { prepareJudge, DEFAULT_RUBRIC } from "./grading.js";
import {
  DEFAULT_RANKING,
  rankingConfig,
  rankingCandidates,
  samplingPlan,
  createRankingSession,
  resultKey,
  pairComplete,
  validateVerdict,
} from "./ranking-data.js";
import { RANKING_CRITERIA } from "./ranking-math.js";
import { BrowserRankingStore } from "./ranking-store.js";
import {
  RankingController,
  createIndependentAudit,
  runIndependentAudit,
} from "./ranking-runner.js";
import { auditAgreement } from "./ranking-statistics.js";

const button = (text, fn, id) => {
  const b = el("button", text, { type: "button", ...(id ? { id } : {}) });
  b.onclick = fn;
  return b;
};
const label = (text, input) => {
  const e = el("label", text);
  e.append(input);
  return e;
};
const select = (id, options) => {
  const e = el("select", null, { id });
  for (const [value, text] of options) e.append(new Option(text, value));
  return e;
};
const interval = (v) => (v ? `${fmt(v[0])}–${fmt(v[1])}` : "Unavailable");
function table(headers, rows) {
  const wrap = el("div", null, { class: "compare-table-wrap" }),
    t = el("table"),
    head = el("thead"),
    tr = el("tr");
  headers.forEach((h) => tr.append(el("th", h, { scope: "col" })));
  head.append(tr);
  t.append(head);
  const body = el("tbody");
  for (const cells of rows) {
    const row = el("tr");
    for (const c of cells) {
      const td = el("td");
      if (c instanceof Node) td.append(c);
      else td.textContent = c;
      row.append(td);
    }
    body.append(row);
  }
  t.append(body);
  wrap.append(t);
  return wrap;
}
function pagedTable(headers, items, cells, pageSize = 50) {
  const host = el("div");
  let page = 0;
  const render = () => {
    host.replaceChildren(
      table(
        headers,
        items.slice(page * pageSize, (page + 1) * pageSize).map(cells),
      ),
    );
    if (items.length <= pageSize) return;
    const previous = button("Previous rows", () => {
      page--;
      render();
    });
    const next = button("Next rows", () => {
      page++;
      render();
    });
    previous.disabled = page === 0;
    next.disabled = (page + 1) * pageSize >= items.length;
    host.append(
      previous,
      el(
        "span",
        ` Rows ${page * pageSize + 1}–${Math.min(items.length, (page + 1) * pageSize)} of ${items.length} `,
      ),
      next,
    );
  };
  render();
  return host;
}
export class RankingView {
  constructor(parent, host) {
    this.parent = parent;
    this.host = host;
    this.pending = new Map();
    this.counter = 0;
    this.worker = new Worker(new URL("./ranking-worker.js", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      data.error ? p.reject(Error(data.error)) : p.resolve(data.value);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(Error(e.message));
      this.pending.clear();
      this.status(`Ranking worker failed: ${e.message}`);
    };
    this.build();
  }
  $(id) {
    return this.host.querySelector(`#ranking-${id}`);
  }
  rpc(type, session, extra = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.counter;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, session, ...extra });
    });
  }
  status(text) {
    this.$("status").textContent = text;
  }
  build() {
    this.host.append(
      el("h3", "Pairwise ranking"),
      el(
        "p",
        "Compare answers with a blinded judge. Ratings predict preferences; they do not establish factual correctness.",
        { class: "hint" },
      ),
    );
    const controls = el("div", null, { class: "compare-filters" });
    controls.append(
      label(
        "Saved ranking",
        select("ranking-sessions", [["", "New ranking session"]]),
      ),
      label(
        "Maximum provider requests",
        el("input", null, {
          id: "ranking-budget",
          type: "number",
          min: 2,
          max: 10000,
          value: 600,
        }),
      ),
      label(
        "Sampling seed",
        el("input", null, {
          id: "ranking-seed",
          type: "number",
          min: 0,
          max: 2147483647,
          value: 7,
        }),
      ),
    );
    this.host.append(controls);
    const advanced = el("details");
    advanced.append(
      el("summary", "Judge and ranking rubric"),
      label(
        "Judge model",
        el("input", null, {
          id: "ranking-model",
          value: DEFAULT_RANKING.model,
          autocomplete: "off",
        }),
      ),
    );
    for (const k of RANKING_CRITERIA)
      advanced.append(
        label(
          k,
          el(
            "textarea",
            k === "overall" ? DEFAULT_RANKING.overall : DEFAULT_RUBRIC[k],
            { id: `ranking-rubric-${k}`, rows: 2 },
          ),
        ),
      );
    advanced.append(
      label(
        "Reference answer (optional)",
        el("textarea", "", { id: "ranking-reference", rows: 3 }),
      ),
    );
    this.host.append(advanced);
    const actions = el("div", null, { class: "toolbar" });
    actions.append(
      button("Estimate plan", () => this.preview(), "ranking-preview"),
      button("Run pairwise evaluation", () => this.start(), "ranking-start"),
      button("Resume / retry missing", () => this.resume(), "ranking-resume"),
      button("Pause", () => this.controller?.pause(), "ranking-pause"),
      button("Continue", () => this.controller?.continue(), "ranking-continue"),
      button("Stop", () => this.stop(), "ranking-stop"),
      button("New extension", () => this.extension(), "ranking-extension"),
    );
    this.host.append(
      actions,
      el("p", "Set a fixed request cap, then run the evaluation.", {
        id: "ranking-estimate",
        class: "hint",
      }),
      el("p", "No pairwise requests have been made.", {
        id: "ranking-status",
        role: "status",
        "aria-live": "polite",
      }),
      el("div", null, { id: "ranking-results" }),
    );
    this.$("sessions").onchange = () => {
      if (!this.parent.report) return;
      this.parent.report.selected_ranking = this.$("sessions").value || null;
      this.session = this.parent.report.rankings?.find(
        (s) => s.id === this.$("sessions").value,
      );
      this.prepared = null;
      this.refresh();
      this.saveReport();
    };
    this.$("results").append(
      el("p", "Generate or select completed answers to compare."),
    );
    this.controls(false);
    for (const input of this.host.querySelectorAll("input,textarea"))
      input.addEventListener("input", () => {
        this.prepared = null;
        this.schedulePreview();
      });
  }
  config() {
    return rankingConfig({
      budget: Number(this.$("budget").value),
      seed: Number(this.$("seed").value),
      model: this.$("model").value,
      trial: this.parent.filters.trial,
      rubric: Object.fromEntries(
        RANKING_CRITERIA.filter((k) => k !== "overall").map((k) => [
          k,
          this.$(`rubric-${k}`).value,
        ]),
      ),
      overall: this.$("rubric-overall").value,
      reference: this.$("reference").value,
    });
  }
  schedulePreview() {
    if (this.host.hidden || this.busy) return;
    clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => this.preview(), 400);
  }
  async preview() {
    try {
      const source = this.parent.source;
      if (!source) throw Error("Generate or select a dataset first");
      const config = this.config(),
        { candidates, groups } = await rankingCandidates(source, config.trial),
        plan = samplingPlan(candidates, groups, config);
      this.$("estimate").textContent =
        `${candidates.length} distinct completed answers · ${plan.cohort.length} ranking candidates · ${plan.allocation.training} ranking / ${plan.allocation.audit} method-audit / ${plan.allocation.validation} validation pair slots. Each pair uses both orders. Cap ${config.budget} POST attempts, including retries. ${plan.notes.join(" ")}`;
      if (!this.parent.getKey()) {
        this.$("estimate").append(
          " Enter the OpenRouter session key to obtain provider pricing.",
        );
        return;
      }
      const prepared = await prepareJudge(this.parent.getKey(), config);
      if (source !== this.parent.source) return;
      const unit = prepared.endpoint.pricing,
        price = (key) =>
          unit?.[key] != null && Number.isFinite(Number(unit[key]))
            ? Number(unit[key])
            : null;
      const maxBytes =
        Math.max(
          0,
          ...candidates.map((c) => new TextEncoder().encode(c.answer).length),
        ) *
          2 +
        new TextEncoder().encode(
          JSON.stringify({ ...config, prompt: candidates[0]?.prompt }),
        ).length +
        2000;
      const cost =
        price("prompt") !== null && price("completion") !== null
          ? config.budget *
            (maxBytes * price("prompt") + 2048 * price("completion"))
          : null;
      this.$("estimate").append(
        ` Route: ${prepared.profile.model} · ${prepared.profile.provider}. ${cost === null ? "Cost estimate unavailable." : `Conservative token-cost estimate up to $${fmt(cost)} using UTF-8 input bytes; billed usage may differ.`}`,
      );
      this.prepared = { source, config, prepared };
    } catch (e) {
      this.$("estimate").textContent = e.message;
    }
  }
  async saveReport() {
    if (!this.parent.report) return;
    try {
      await this.parent.persist();
      await this.parent.refreshReports();
    } catch (e) {
      this.status(
        `Report storage failed: ${e.message}. Export the in-memory report to retain evidence.`,
      );
      throw e;
    }
  }
  sync() {
    const report = this.parent.report,
      id = report?.id ?? null;
    if (this.reportId !== id) {
      this.stop();
      this.reportId = id;
      this.session = null;
      this.data = null;
      this.extensionParent = null;
    }
    this.$("sessions").replaceChildren(
      new Option("New ranking session", ""),
      ...(report?.rankings ?? []).map(
        (s) =>
          new Option(
            `${new Date(s.created_at).toLocaleString()} · ${s.status} · ${s.id.slice(0, 8)}`,
            s.id,
          ),
      ),
    );
    this.$("sessions").value = report?.selected_ranking ?? "";
    const selected = report?.rankings?.find(
      (s) => s.id === report.selected_ranking,
    );
    if (selected !== this.session) {
      this.predictionText = "";
      this.session = selected;
      this.refresh();
    }
    if (report && !this.busy && this.loadedReport !== id) {
      this.loadedReport = id;
      this.recover(report);
    }
    this.controls(!!this.busy);
  }
  async recover(report) {
    try {
      const rows = await BrowserRankingStore.list(report.id);
      if (this.parent.report !== report) return;
      for (const row of rows) {
        const existing = report.rankings?.find((s) => s.id === row.id);
        if (!existing || row.lastSeq > existing.seq) {
          const saved = await (await BrowserRankingStore.open(row.id)).read();
          if (this.parent.report !== report) return;
          report.rankings ??= [];
          const i = report.rankings.findIndex((s) => s.id === row.id);
          if (i < 0) report.rankings.push(saved);
          else report.rankings[i] = saved;
        }
      }
      this.sync();
    } catch (e) {
      this.status(
        `Ranking journal unavailable: ${e.message}. Imported report evidence remains available.`,
      );
    }
  }
  controls(busy) {
    for (const id of [
      "preview",
      "start",
      "sessions",
      "budget",
      "seed",
      "model",
      "reference",
      "extension",
    ])
      this.$(id).disabled = busy;
    for (const k of RANKING_CRITERIA) this.$(`rubric-${k}`).disabled = busy;
    this.$("resume").disabled =
      busy ||
      !this.session ||
      ["completed", "budget_exhausted"].includes(this.session.status);
    this.$("extension").disabled = busy || !this.session;
    for (const id of ["pause", "continue"])
      this.$(id).disabled = !busy || !this.controller;
    this.$("stop").disabled = !busy;
  }
  stop() {
    this.preparation?.abort();
    this.controller?.stop();
    this.auditAbort?.abort();
  }
  extension() {
    this.extensionParent = this.session?.id;
    if (this.session)
      this.$("seed").value = (this.session.config.seed + 1) % 2147483648;
    this.parent.report.selected_ranking = null;
    this.session = null;
    this.sync();
    this.status(
      "New extension selected. Earlier judgments and conclusions remain unchanged; this extension uses a fresh fixed-budget plan.",
    );
  }
  async start() {
    if (this.busy || this.parent.grader || this.parent.gradingStarting) return;
    this.busy = true;
    this.preparation = new AbortController();
    this.controls(true);
    this.parent.gradingControls(true);
    const epoch = this.parent.epoch;
    try {
      this.status("Preparing a frozen ranking session…");
      const config = this.config(),
        prepared = await prepareJudge(this.parent.getKey(), config, {
          signal: this.preparation.signal,
        });
      if (epoch !== this.parent.epoch)
        throw Error("Source changed while preparing ranking");
      const session = await createRankingSession(
        this.parent.source,
        config,
        prepared,
        { parent_id: this.extensionParent ?? null },
      );
      if (epoch !== this.parent.epoch)
        throw Error("Source changed while preparing ranking");
      const report = await this.parent.ensureReport();
      report.version = 2;
      (report.rankings ??= []).push(session);
      report.selected_ranking = session.id;
      report.evaluation_mode = "pairwise";
      this.reportId = report.id;
      this.loadedReport = report.id;
      this.session = session;
      await this.saveReport();
      const store = await BrowserRankingStore.create(session, report.id);
      await this.execute(session, store, false);
    } catch (e) {
      this.status(`Pairwise evaluation failed: ${e.message}`);
    } finally {
      this.busy = false;
      this.preparation = null;
      this.controller = null;
      this.parent.gradingControls(false);
      this.controls(false);
      this.sync();
    }
  }
  async resume() {
    if (this.busy || !this.session) return;
    this.busy = true;
    this.preparation = new AbortController();
    this.controls(true);
    this.parent.gradingControls(true);
    try {
      let store;
      try {
        store = await BrowserRankingStore.open(this.session.id);
      } catch {
        store = await BrowserRankingStore.create(
          this.session,
          this.parent.report.id,
        );
      }
      if (store.lastSeq !== this.session.seq)
        throw Error(
          "A newer journal exists. Reselect this report before resuming",
        );
      await this.execute(this.session, store, true);
    } catch (e) {
      this.status(`Pairwise evaluation failed: ${e.message}`);
    } finally {
      this.busy = false;
      this.preparation = null;
      this.controller = null;
      this.parent.gradingControls(false);
      this.controls(false);
      this.sync();
    }
  }
  async execute(session, store, retry) {
    this.preparation?.signal.throwIfAborted();
    this.controller = new RankingController(
      this.parent.getKey(),
      session,
      store,
      {
        fit: (s) => this.rpc("fit", s),
        onStatus: (s, message) => {
          this.status(
            message ??
              `${s.status} · ${s.attempts.length}/${s.config.budget} provider attempts · ${Object.values(s.results).filter((r) => r.status === "valid").length} valid presentations. Evidence is provisional until this fixed-budget execution ends.`,
          );
          if (s.events.at(-1)?.type === "freeze") this.refresh();
        },
      },
    );
    this.controls(true);
    await this.controller.run({ retry });
    await this.saveReport();
    await this.refresh();
    this.status(
      `${session.status} · ${session.attempts.length}/${session.config.budget} requests. ${session.status === "needs_retry" ? "Retry missing or failed judgments; completed judgments are retained." : "Results distinguish observed audits from model predictions."}`,
    );
  }
  async refresh() {
    const s = this.session;
    if (!s) {
      this.$("results").replaceChildren(el("p", "No ranking selected."));
      return;
    }
    if (this.dataSession !== s) {
      this.dataSession = s;
      this.data = null;
      this.$("results").replaceChildren(
        el("p", "Processing saved ranking evidence…"),
      );
    }
    const job = (this.analysisJob = (this.analysisJob ?? 0) + 1);
    try {
      const data = await this.rpc("analyze", s);
      if (this.session === s && job === this.analysisJob) {
        this.data = data;
        this.render();
      }
    } catch (e) {
      this.status(`Ranking analysis failed: ${e.message}`);
    }
  }
  pin(candidate, slot = 0) {
    const s = this.session;
    // Pin against the immutable ranking snapshot, including after a live source has advanced.
    this.parent.pins[slot] = candidate.occurrences[0].key;
    this.parent.renderPins(s.source);
    this.parent.$("pins").scrollIntoView({ block: "start" });
  }
  render() {
    const s = this.session,
      d = this.data,
      out = this.$("results");
    out.replaceChildren();
    if (!s || !d) return;
    const byId = new Map(s.candidates.map((c) => [c.id, c])),
      usage = usageTotals(s.requests),
      criterion = this.criterion ?? "overall";
    if (!["completed", "budget_exhausted"].includes(s.status))
      out.append(
        el(
          "p",
          "Unfinished execution: audit results are provisional. Fixed-budget conclusions require the planned execution to end.",
          { class: "compare-notice" },
        ),
      );
    const controls = el("div", null, { class: "compare-filters" }),
      metric = select(
        "ranking-criterion",
        RANKING_CRITERIA.map((k) => [k, k]),
      );
    metric.value = criterion;
    metric.onchange = () => {
      this.criterion = metric.value;
      this.predictionText = "";
      this.render();
    };
    controls.append(
      label("Ranking criterion", metric),
      button("Export ranking data", () =>
        download(
          JSON.stringify(
            {
              session_id: s.id,
              ratings: d.ratings,
              methods: d.methods,
              validation: d.validation,
              audits: d.audits,
            },
            null,
            2,
          ),
          `ranking-${s.id}.json`,
        ),
      ),
    );
    out.append(
      controls,
      el(
        "p",
        `${s.plan.cohort.length}/${s.candidates.length} distinct answers in the ranking cohort · ${s.attempts.length} primary provider attempts · Primary ranking cost $${fmt(usage.cost.total)}. ${s.plan.notes.join(" ")}`,
        { class: "compare-notice" },
      ),
    );
    out.append(
      el("h3", "Observed evidence · Fractal preference share"),
      el(
        "p",
        "Win = 1, tie = ½, loss = 0. Trials have equal weight. Bounds describe the saved populations under this judge protocol; they do not establish factual correctness.",
        { class: "hint" },
      ),
      table(
        [
          "Baseline / population",
          "Trials",
          "Samples",
          "Preference share",
          "Win / tie / loss",
          "Coverage",
          "95% simultaneous bounds",
          "Missing-assessment bounds",
          "Across-trial interval (approx.)",
        ],
        d.methods
          .filter((r) => r.criterion === criterion)
          .map((r) => [
            `${METHOD_STYLE[r.baseline].label} · ${r.pool}`,
            r.trials,
            r.samples,
            fmt(r.preference_share),
            `${fmt(r.win)} / ${fmt(r.tie)} / ${fmt(r.loss)}`,
            fmt(r.coverage),
            interval(r.interval),
            interval(r.missing_bounds),
            interval(r.generalization_interval),
          ]),
      ),
    );
    if (!d.methods.length)
      out.append(
        el(
          "p",
          "No baseline is available. Only Fractal answer rankings are shown.",
        ),
      );
    if (d.methods.length) {
      const trials = el("details");
      trials.append(el("summary", "Inspect trial-level method variation"));
      trials.ontoggle = () => {
        if (trials.open && trials.children.length === 1)
          trials.append(
            pagedTable(
              [
                "Baseline / population",
                "Trial",
                "Sampled pairs",
                "Preference share",
                "Coverage",
              ],
              d.methods
                .filter((r) => r.criterion === criterion)
                .flatMap((r) => r.trial_values.map((t) => ({ r, t }))),
              ({ r, t }) => [
                `${METHOD_STYLE[r.baseline].label} · ${r.pool}`,
                t.trial,
                t.n,
                fmt(t.preference_share),
                fmt(t.coverage),
              ],
            ),
          );
      };
      out.append(trials);
    }
    const plot = el("div", null, { class: "compare-chart-grid" });
    out.append(plot);
    chart(plot, {
      title: "Observed method preference share",
      xLabel: "Preference share with simultaneous interval",
      yLabel: "Population / baseline",
      series: d.methods
        .filter((r) => r.criterion === criterion && r.interval)
        .map((r, i) => ({
          method: r.baseline,
          label: `${r.pool} · ${METHOD_STYLE[r.baseline].label}`,
          points: [
            { x: r.interval[0], y: i },
            {
              x: r.preference_share ?? (r.interval[0] + r.interval[1]) / 2,
              y: i,
            },
            { x: r.interval[1], y: i },
          ],
        })),
    });
    const diagnostics = d.validation.find((r) => r.criterion === criterion);
    chart(plot, {
      title: "Held-out calibration",
      xLabel: "Predicted outcome probability",
      yLabel: "Observed outcome frequency",
      series: [
        {
          method: "fractal",
          label: "Held-out outcomes",
          points: diagnostics.calibration
            .filter((b) => b.count)
            .map((b) => ({ x: b.predicted, y: b.observed })),
        },
      ],
    });
    out.append(
      el("h3", "Model predictions · answer ratings"),
      el(
        "p",
        "Intervals are approximate Laplace posterior intervals. Rank intervals and top-five probabilities apply within a connected component. Unmeasured answers remain unranked.",
        { class: "hint" },
      ),
    );
    const sort = select("ranking-sort", [
      ["elo", "Rating, highest first"],
      ["opponents", "Fewest opponents"],
      ["width", "Widest uncertainty first"],
    ]);
    sort.value = this.sort ?? "elo";
    sort.onchange = () => {
      this.sort = sort.value;
      this.render();
    };
    out.append(label("Sort answers", sort));
    const ratings = d.ratings
      .filter((r) => r.criterion === criterion)
      .sort((a, b) =>
        a.component !== b.component
          ? (a.component ?? Infinity) - (b.component ?? Infinity)
          : this.sort === "opponents"
            ? a.opponents - b.opponents
            : this.sort === "width"
              ? (b.interval ? b.interval[1] - b.interval[0] : Infinity) -
                (a.interval ? a.interval[1] - a.interval[0] : Infinity)
              : (b.elo ?? -Infinity) - (a.elo ?? -Infinity),
      );
    out.append(
      table(
        [
          "Answer / methods",
          "Elo",
          "95% model interval",
          "Rank interval",
          "Top five",
          "Opponents",
          "Component / diagnostics",
          "Trace",
        ],
        ratings.map((r) => {
          const c = byId.get(r.id);
          return [
            c.answer.slice(0, 110) +
              " · " +
              [
                ...new Set(
                  c.occurrences.map((o) => METHOD_STYLE[o.method].label),
                ),
              ].join(", "),
            fmt(r.elo),
            interval(r.interval),
            interval(r.rank_interval),
            fmt(r.top_five),
            r.opponents,
            `${r.component === null ? "Insufficient evidence" : r.component + 1}${r.prior_sensitive ? " · Prior-sensitive" : ""}${r.unreliable ? " · Fit unreliable" : ""}`,
            button("Inspect answer", () => this.pin(c)),
          ];
        }),
      ),
    );
    const unranked = s.candidates.filter((c) => !s.plan.cohort.includes(c.id));
    if (unranked.length) {
      const details = el("details");
      details.append(
        el(
          "summary",
          `${unranked.length} answers outside the cohort · unranked`,
        ),
      );
      details.ontoggle = () => {
        if (!details.open || details.children.length > 1) return;
        details.append(
          pagedTable(["Answer", "Trace"], unranked, (c) => [
            c.answer.slice(0, 120),
            button("Inspect", () => this.pin(c)),
          ]),
        );
      };
      out.append(details);
    }
    out.append(el("h3", "Predict a comparison"));
    const pairControls = el("div", null, { class: "compare-filters" }),
      options = s.plan.cohort.map((id) => [
        id,
        byId.get(id).answer.slice(0, 90),
      ]),
      a = select("ranking-answer-a", options),
      b = select("ranking-answer-b", options);
    if (options.length > 1) b.selectedIndex = 1;
    if (options.some(([id]) => id === this.answerA)) a.value = this.answerA;
    if (options.some(([id]) => id === this.answerB)) b.value = this.answerB;
    const prediction = el("p", this.predictionText ?? "", {
      id: "ranking-prediction",
      role: "status",
    });
    a.onchange = b.onchange = () => {
      this.answerA = a.value;
      this.answerB = b.value;
      this.predictionText = "";
      prediction.textContent = "";
    };
    pairControls.append(
      label("Answer A", a),
      label("Answer B", b),
      button("Compare pair", async () => {
        try {
          const p = await this.rpc("predict", null, {
            criterion,
            a: a.value,
            b: b.value,
            seed: s.config.seed,
          });
          this.predictionText =
            a.value === b.value
              ? "Identical answer: equal content."
              : p
                ? `Model prediction: win ${fmt(p.win)}, tie ${fmt(p.tie)}, loss ${fmt(p.loss)} · P(A stronger) ${fmt(p.stronger_probability)} · ${s.pairs.some((x) => [x.a, x.b].includes(a.value) && [x.a, x.b].includes(b.value)) ? "Pair has recorded evidence." : "Unjudged pair: inferred from shared opponents."}`
                : "Insufficient evidence: the answers are unobserved, disconnected, or the fit is unreliable.";
          this.$("prediction").textContent = this.predictionText;
          this.pin(byId.get(a.value), 0);
          this.pin(byId.get(b.value), 1);
        } catch (e) {
          this.predictionText = e.message;
          this.$("prediction").textContent = e.message;
        }
      }),
    );
    out.append(pairControls, prediction);
    const matrix = el("details");
    matrix.append(
      el("summary", "Win / tie / loss prediction matrix (up to 30 answers)"),
    );
    matrix.ontoggle = async () => {
      if (!matrix.open || matrix.children.length > 1) return;
      const ids = s.plan.cohort.slice(0, 30),
        rows = [];
      const { predictPair } = await import("./ranking-math.js");
      for (const id of ids)
        rows.push([
          byId.get(id).answer.slice(0, 35),
          ...ids.map((other) => {
            if (id === other) return "—";
            const p = predictPair(d.fits[criterion], id, other);
            return p
              ? `${fmt(p.win)} / ${fmt(p.tie)} / ${fmt(p.loss)}`
              : "Unavailable";
          }),
        ]);
      matrix.append(
        table(["A against B", ...ids.map((_, i) => String(i + 1))], rows),
      );
    };
    out.append(matrix);
    out.append(
      el("h3", "Validation and limitations"),
      table(
        [
          "Criterion",
          "Held-out pair weight",
          "Log loss / constant",
          "Brier / constant",
          "Order disagreements",
          "Presentation-position effect",
          "Cycles / measured triangles",
          "Connectivity",
          "Convergence",
        ],
        d.validation.map((r) => [
          r.criterion,
          r.assessed_weight,
          `${fmt(r.log_loss)} / ${fmt(r.constant_log_loss)}`,
          `${fmt(r.brier)} / ${fmt(r.constant_brier)}`,
          `${r.order_disagreements}/${r.swapped_pairs}`,
          fmt(r.position_effect),
          `${r.cycles}/${r.directed_triangles}`,
          `${r.components} components · ${r.unobserved} unobserved`,
          r.converged ? "Converged" : "Unreliable",
        ]),
      ),
      el(
        "p",
        "Random validation pairs were excluded from fitting. A narrow model interval can still be wrong if judge preferences violate the ranking model. Position disagreement is reported separately from explicit ties.",
        { class: "hint" },
      ),
    );
    const evidence = el("details");
    evidence.append(
      el(
        "summary",
        "Inspect pair judgments, both orders, requests and failures",
      ),
    );
    evidence.ontoggle = () => {
      if (evidence.open && evidence.children.length === 1)
        evidence.append(
          pagedTable(
            [
              "Pair / partition",
              "Presented A",
              "Presented B",
              "Verdict / explanation",
              "Trace and request evidence",
            ],
            s.pairs.flatMap((p) => [0, 1].map((o) => ({ p, o }))),
            ({ p, o }) => {
              const r = s.results[resultKey(p, o)],
                ca = byId.get(o ? p.b : p.a),
                cb = byId.get(o ? p.a : p.b);
              const actions = el("div");
              actions.append(
                button("Pin both", () => {
                  this.pin(ca, 0);
                  this.pin(cb, 1);
                }),
                button("Inspect request", () => {
                  const dialog = el("dialog", null, {
                    class: "ranking-human-dialog",
                    "aria-label": "Saved pairwise request evidence",
                  });
                  dialog.append(
                    button("Close evidence", () => dialog.close()),
                    el("h3", "Saved request evidence"),
                    el(
                      "pre",
                      JSON.stringify(
                        {
                          pair: p,
                          orientation: o,
                          result: r ?? null,
                          attempts: s.attempts.filter(
                            (e) => e.pair_id === p.id && e.orientation === o,
                          ),
                          requests: s.requests.filter(
                            (e) => e.pair_id === p.id && e.orientation === o,
                          ),
                        },
                        null,
                        2,
                      ),
                    ),
                  );
                  dialog.onclose = () => dialog.remove();
                  this.host.append(dialog);
                  dialog.showModal();
                }),
              );
              return [
                p.partition + " · " + (o ? "reversed" : "forward"),
                ca.answer.slice(0, 300),
                cb.answer.slice(0, 300),
                r?.status === "valid"
                  ? `${r.verdict[criterion].verdict}: ${r.verdict[criterion].explanation}`
                  : (r?.error ?? r?.status ?? "Not requested"),
                actions,
              ];
            },
          ),
        );
    };
    out.append(evidence);
    const failures = Object.values(s.results).filter((r) =>
      ["failed", "interrupted"].includes(r.status),
    );
    if (failures.length)
      out.append(
        el(
          "p",
          `${failures.length} presentations failed or were interrupted. ${failures[0].error ?? ""}`,
          { class: "compare-notice" },
        ),
      );
    this.renderAudits(out, s, byId);
  }
  renderAudits(out, s, byId) {
    out.append(
      el("h3", "Independent audit · optional"),
      el(
        "p",
        "Audit judgments remain separate from the primary ratings. The random sample contains up to 20 completed pairs.",
        { class: "hint" },
      ),
    );
    const fields = el("div", null, { class: "compare-filters" }),
      model = el("input", null, {
        placeholder: "Choose an independent judge model",
        id: "ranking-audit-model",
        value: this.auditModelInput ?? "",
      }),
      limit = el("input", null, {
        id: "ranking-audit-budget",
        type: "number",
        min: 2,
        max: 10000,
        value: this.auditBudgetInput ?? 80,
      });
    model.oninput = () => {
      this.auditModelInput = model.value;
    };
    limit.oninput = () => {
      this.auditBudgetInput = limit.value;
    };
    fields.append(
      label("Independent model", model),
      label("Audit request cap", limit),
      button("Run model audit", () =>
        this.auditModel(model.value, Number(limit.value)),
      ),
      button("Start blinded human audit", () => this.auditHuman()),
    );
    out.append(fields);
    for (const audit of s.audits) {
      const section = el("section"),
        usage = usageTotals(audit.requests);
      section.append(
        el(
          "h4",
          `${audit.kind === "human" ? "Human audit" : audit.profile.model} · ${audit.status}`,
        ),
        el(
          "p",
          `${audit.attempts.length}/${audit.budget} independent-audit requests · cost $${fmt(usage.cost.total)}`,
        ),
        table(
          ["Criterion", "Compared pair weight", "Agreement"],
          auditAgreement(s, audit).map((r) => [
            r.criterion,
            r.compared_pairs,
            fmt(r.agreement),
          ]),
        ),
      );
      if (
        audit.kind === "model" &&
        audit.status !== "completed" &&
        audit.attempts.length < audit.budget
      )
        section.append(
          button("Retry missing audit judgments", () =>
            this.auditModel(null, null, audit),
          ),
        );
      if (audit.kind === "human") {
        const task = audit.pairs
          .flatMap((p) => [0, 1].map((o) => ({ p, o, key: resultKey(p, o) })))
          .find((t) => audit.results[t.key]?.status !== "valid");
        if (task) {
          const a = byId.get(task.o ? task.p.b : task.p.a),
            b = byId.get(task.o ? task.p.a : task.p.b);
          const dialog = el("dialog", null, {
            class: "ranking-human-dialog",
            "aria-label": "Blinded human assessment",
          });
          section.append(
            dialog,
            button("Assess next pair", () => dialog.showModal()),
          );
          dialog.append(
            button("Close assessment", () => dialog.close()),
            el("h3", "Blinded assessment"),
            el("p", a.prompt),
            el("pre", `Candidate A\n${a.answer}\n\nCandidate B\n${b.answer}`),
          );
          if (s.profile.reference)
            dialog.append(
              el("h4", "Reference answer"),
              el("pre", s.profile.reference),
            );
          const inputs = {};
          for (const k of RANKING_CRITERIA) {
            const input = select(`human-${audit.id}-${k}`, [
              ["cannot_assess", "Cannot assess"],
              ["A", "A better"],
              ["B", "B better"],
              ["tie", "Tie"],
            ]);
            inputs[k] = input;
            dialog.append(
              label(
                `${k}: ${k === "overall" ? s.profile.overall : s.profile.rubric[k]}`,
                input,
              ),
            );
          }
          dialog.append(
            button("Save blinded assessment", () =>
              this.saveHuman(
                audit,
                task,
                Object.fromEntries(
                  RANKING_CRITERIA.map((k) => [
                    k,
                    {
                      verdict: inputs[k].value,
                      explanation: "Human assessment",
                    },
                  ]),
                ),
              ),
            ),
          );
        }
      }
      out.append(section);
    }
  }
  async primaryWriter() {
    let store;
    try {
      store = await BrowserRankingStore.open(this.session.id);
    } catch {
      store = await BrowserRankingStore.create(
        this.session,
        this.parent.report.id,
      );
    }
    if (store.lastSeq !== this.session.seq)
      throw Error("Newer ranking journal exists; reload before auditing");
    return {
      store,
      writer: new RankingController(
        this.parent.getKey() || "human-audit",
        this.session,
        store,
      ),
    };
  }
  async auditModel(model, budget, existing) {
    if (this.busy) return;
    this.busy = true;
    this.controls(true);
    this.parent.gradingControls(true);
    this.auditAbort = new AbortController();
    let stage = "preparing the judge";
    try {
      const { store, writer } = await this.primaryWriter();
      await store.lock(async () => {
        let audit = existing;
        if (!audit) {
          const prepared = await prepareJudge(
            this.parent.getKey(),
            {
              ...this.session.profile,
              model,
              provider: null,
            },
            { signal: this.auditAbort.signal },
          );
          audit = createIndependentAudit(this.session, {
            kind: "model",
            profile: {
              ...prepared.profile,
              overall: this.session.profile.overall,
            },
            endpoint: prepared.endpoint,
            budget,
          });
          stage = "saving the audit plan";
          await writer.emit("audit", audit);
        }
        stage = "running the independent judge";
        await runIndependentAudit(writer, audit, this.parent.getKey(), {
          signal: this.auditAbort.signal,
          onController: (controller) => {
            this.controller = controller;
            this.controls(true);
          },
          onStatus: (a) =>
            this.status(
              `Independent audit ${a.status} · ${a.attempts.length}/${a.budget} requests`,
            ),
        });
      });
      stage = "saving the report";
      await this.saveReport();
      stage = "processing audit results";
      await this.refresh();
    } catch (e) {
      this.status(`Independent audit failed while ${stage}: ${e.message}`);
    } finally {
      this.busy = false;
      this.auditAbort = null;
      this.controller = null;
      this.parent.gradingControls(false);
      this.controls(false);
    }
  }
  async auditHuman() {
    if (this.busy) return;
    try {
      const { store, writer } = await this.primaryWriter();
      await store.lock(() =>
        writer.emit(
          "audit",
          createIndependentAudit(this.session, { kind: "human" }),
        ),
      );
      await this.saveReport();
      this.render();
    } catch (e) {
      this.status(e.message);
    }
  }
  async saveHuman(audit, task, verdict) {
    try {
      validateVerdict(verdict);
      const { store, writer } = await this.primaryWriter();
      await store.lock(async () => {
        const next = structuredClone(audit);
        if (next.results[task.key]?.status === "valid")
          throw Error("Completed human assessment is immutable");
        next.results[task.key] = {
          status: "valid",
          verdict,
          ended_at: Date.now(),
        };
        next.status = next.pairs.every((p) =>
          [0, 1].every(
            (o) => next.results[resultKey(p, o)]?.status === "valid",
          ),
        )
          ? "completed"
          : "running";
        await writer.emit("audit", next);
      });
      await this.saveReport();
      this.render();
    } catch (e) {
      this.status(e.message);
    }
  }
}
