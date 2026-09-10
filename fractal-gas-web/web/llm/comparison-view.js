import { objectiveLabel } from "./config.js";
import {
  BrowserBenchmarkStore,
  BrowserComparisonStore,
} from "./benchmark-store.js";
import {
  sourceRuns,
  traceComparison,
  distance,
  DEFAULT_FILTERS,
  METHOD_STYLE,
  METRICS,
  finite,
  describe,
} from "./comparison-metrics.js";
import {
  createReport,
  parseReport,
  exportReport,
  verifyReportIdentities,
} from "./comparison-report.js";
import {
  CRITERIA,
  DEFAULT_GRADING_MODEL,
  DEFAULT_RUBRIC,
  gradingCandidates,
  prepareJudge,
  estimateGrading,
  gradesForSession,
  GradingController,
} from "./grading.js";
import { usageTotals } from "./benchmark-data.js";
import {
  element as el,
  fmt,
  chart,
  download,
  legend,
} from "./comparison-charts.js";
import { renderTokens } from "./analysis.js";
import { RankingView } from "./ranking-view.js";

const poolName = (p) =>
  p === "archive" ? "Archived answers" : "Retained population";
const termination = (n) =>
  !n.tokens
    ? "Empty"
    : n.status === 1
      ? "EOS"
      : n.status === 2
        ? "Sequence cap"
        : "Partial";
function select(label, id, options, value) {
  const wrap = el("label", label),
    input = el("select", null, { id });
  for (const [v, t] of options) input.append(new Option(t, v));
  input.value = value;
  wrap.append(input);
  return wrap;
}
function button(label, fn, id) {
  const b = el("button", label, { type: "button", ...(id ? { id } : {}) });
  b.onclick = fn;
  return b;
}
function table(headers, rows) {
  const wrap = el("div", null, { class: "compare-table-wrap" }),
    t = el("table"),
    thead = el("thead"),
    tr = el("tr");
  for (const h of headers) tr.append(el("th", h, { scope: "col" }));
  thead.append(tr);
  const tbody = el("tbody");
  for (const cells of rows) {
    const row = el("tr");
    for (const c of cells) {
      const td = el("td");
      if (c instanceof Node) td.append(c);
      else td.textContent = c;
      row.append(td);
    }
    tbody.append(row);
  }
  t.append(thead, tbody);
  wrap.append(t);
  return wrap;
}
export class ComparisonView {
  constructor(
    host,
    {
      evaluationHost,
      getKey,
      setKey,
      getConfig,
      onGeneration,
      onEvaluation,
      onBenchmark,
      onBenchmarkSource,
      onCurrentSource = () => {},
    },
  ) {
    this.host = host;
    this.evaluationHost = evaluationHost;
    this.setKey = setKey;
    this.onEvaluation = onEvaluation;
    this.onBenchmark = onBenchmark;
    this.onBenchmarkSource = onBenchmarkSource;
    this.getKey = getKey;
    this.getConfig = getConfig;
    this.onGeneration = onGeneration;
    this.onCurrentSource = onCurrentSource;
    this.filters = { ...DEFAULT_FILTERS };
    this.revision = 0;
    this.job = 0;
    this.epoch = 0;
    this.pins = [];
    this.selection = null;
    this.sort = "mean";
    this.desc = true;
    this.page = 0;
    this.report = null;
    this.worker = new Worker(
      new URL("./comparison-worker.js", import.meta.url),
      { type: "module" },
    );
    this.worker.onmessage = ({ data }) => {
      if (data.id !== this.job || data.revision !== this.revision) return;
      if (data.error) {
        this.status(data.error);
        return;
      }
      this.data = data.data;
      this.status(
        this.data?.partialPreview
          ? "Partial preview · No eligible full answers are available yet."
          : this.source?.live
            ? "Live benchmark · Saved generation boundaries"
            : this.source
              ? "Saved-data comparisons · No grading requests needed"
              : "Generate or import a Fractal recording, or start a benchmark.",
      );
      this.render();
    };
    this.worker.onerror = (e) =>
      this.status(`Comparison worker failed: ${e.message}`);
    this.build();
    this.refreshReports();
  }
  $(id) {
    return (
      this.host.querySelector(`#compare-${id}`) ??
      this.evaluationHost.querySelector(`#compare-${id}`)
    );
  }
  setPanel(tab) {
    this.panel = tab;
    if (tab === "evaluation") {
      this.evaluationHost.insertBefore(
        this.sharedControls,
        this.gradingSection,
      );
      this.$("evaluation-key").value = this.getKey();
      this.refreshEvaluationSources();
      this.schedulePreview();
      this.ranking?.schedulePreview();
    } else if (tab === "benchmark")
      this.host.insertBefore(this.sharedControls, this.$("results"));
  }
  async refreshEvaluationSources() {
    try {
      const entries = await BrowserBenchmarkStore.list();
      this.$("evaluation-source").replaceChildren(
        new Option("Current Fractal run", ""),
        ...entries.map(
          (e) =>
            new Option(
              `${new Date(e.manifest.created_at).toLocaleString()} · ${e.manifest.settings.config.algorithm} · ${e.id.slice(0, 8)}`,
              e.id,
            ),
        ),
      );
      if (this.importedReport)
        this.$("evaluation-source").append(
          new Option("Imported comparison report", "report"),
        );
      this.$("evaluation-source").value = this.importedReport
        ? "report"
        : this.benchmarkId || "";
    } catch (e) {
      this.status(`Source storage unavailable: ${e.message}`);
    }
  }
  status(text) {
    this.$("status").textContent = text;
  }
  build() {
    const title = el("div", null, { class: "section-title" });
    title.append(
      el("h2", "Sampling comparisons"),
      button("Generation settings", () => this.onGeneration()),
    );
    this.host.append(
      title,
      el("p", "", { id: "compare-config", class: "hint" }),
      el("div", null, { id: "compare-method-legend" }),
    );
    const reportbar = el("div", null, { class: "compare-filters" });
    reportbar.append(
      select(
        "Saved comparison reports",
        "compare-reports",
        [["", "Select a report"]],
        "",
      ),
      button("Export report", () => this.export(), "compare-export"),
      button("Import report", () => this.$("file").click(), "compare-import"),
    );
    const file = el("input", null, {
      id: "compare-file",
      type: "file",
      accept: ".fgllmcompare",
      hidden: "",
    });
    file.onchange = async () => {
      try {
        if (file.files[0])
          await this.loadReport(
            await verifyReportIdentities(
              parseReport(await file.files[0].text()),
            ),
          );
        await this.persist();
        await this.refreshReports();
      } catch (e) {
        this.status(e.message);
      } finally {
        file.value = "";
      }
    };
    reportbar.append(file);
    this.sharedControls = el("div", null, {
      class: "comparison-shared-controls",
    });
    this.sharedControls.append(reportbar);
    this.host.append(this.sharedControls);
    this.$("reports").onchange = async () => {
      try {
        const r = await BrowserComparisonStore.open(this.$("reports").value);
        if (r)
          await this.loadReport(
            await verifyReportIdentities(parseReport(JSON.stringify(r))),
          );
      } catch (e) {
        this.status(e.message);
      }
    };
    const filters = el("div", null, { class: "compare-filters" });
    filters.append(
      select("Trial", "compare-trial", [["all", "All trials"]], "all"),
      select(
        "Method",
        "compare-method",
        [
          ["all", "All methods"],
          ...Object.entries(METHOD_STYLE).map(([k, v]) => [k, v.label]),
        ],
        "all",
      ),
      select(
        "Termination",
        "compare-status-filter",
        [
          ["full", "Full answers · EOS + cap"],
          ["eos", "EOS only"],
          ["partial", "Partial only"],
          ["all", "All nonempty traces"],
        ],
        "full",
      ),
      select(
        "Population",
        "compare-pool",
        [
          ["both", "Both populations"],
          ["archive", "Archived answers"],
          ["retained", "Retained population"],
        ],
        "both",
      ),
      select(
        "Measure",
        "compare-metric",
        Object.entries(METRICS).filter(
          ([k]) => !["grade", ...CRITERIA].includes(k),
        ),
        "mean",
      ),
      select(
        "Paired baseline",
        "compare-baseline",
        Object.entries(METHOD_STYLE).map(([k, v]) => [k, v.label]),
        "independent_tokens",
      ),
      select(
        "Embedding distance",
        "compare-distance",
        [
          ["cosine", "Cosine"],
          ["l2", "L2"],
        ],
        "cosine",
      ),
      select(
        "Attempt inspector",
        "compare-attempt",
        [["default", "Successful methods + live preview"]],
        "default",
      ),
    );
    this.sharedControls.append(filters);
    for (const k of Object.keys(DEFAULT_FILTERS)) {
      const id = k === "status" ? "status-filter" : k;
      this.$(id).onchange = () => {
        this.filters[k] = this.$(id).value;
        this.selection = null;
        this.page = 0;
        this.compute();
        this.saveSoon();
        if (k === "trial") this.schedulePreview();
      };
    }
    this.sharedControls.append(
      el("p", "", {
        id: "compare-status",
        role: "status",
        "aria-live": "polite",
      }),
      el("p", "", { id: "compare-storage", class: "hint", role: "status" }),
    );
    this.host.append(el("div", null, { id: "compare-results" }));
    this.evaluationHost.append(
      el("h2", "Answer evaluation"),
      el(
        "p",
        "Evaluate the selected answers with Gemini Flash, then inspect scores and compare complete traces.",
        { class: "hint" },
      ),
    );
    const sources = el("div", null, { class: "compare-filters" });
    sources.append(
      select(
        "Evaluation source",
        "compare-evaluation-source",
        [["", "Current Fractal run"]],
        "",
      ),
      button("Benchmark sampling", () => this.onBenchmark()),
    );
    const keyLabel = el("label", "OpenRouter session key"),
      keyInput = el("input", null, {
        id: "compare-evaluation-key",
        type: "password",
        autocomplete: "off",
        placeholder: "Paste your OpenRouter API key",
      });
    keyLabel.append(keyInput);
    sources.append(keyLabel);
    this.evaluationHost.append(sources);
    keyInput.oninput = () => {
      this.setKey(keyInput.value.trim());
      this.schedulePreview();
    };
    this.$("evaluation-source").onchange = () => {
      try {
        const id = this.$("evaluation-source").value;
        if (id !== "report") this.onBenchmarkSource(id);
      } catch (e) {
        this.status(e.message);
      }
    };
    this.buildGrading();
    const mode = select(
      "Evaluation mode",
      "compare-evaluation-mode",
      [
        ["absolute", "Absolute grades"],
        ["pairwise", "Pairwise ranking"],
      ],
      "absolute",
    );
    this.evaluationHost.insertBefore(mode, this.gradingSection);
    this.rankingHost = el("section", null, {
      id: "compare-ranking",
      hidden: "",
    });
    this.evaluationHost.append(this.rankingHost);
    this.evaluationHost.append(
      el("div", null, {
        id: "compare-judge-charts",
        class: "compare-chart-grid",
      }),
      el("div", null, { id: "compare-judge-summary" }),
      el("section", null, { id: "compare-traces" }),
      el("section", null, { id: "compare-pins" }),
    );
    this.ranking = new RankingView(this, this.rankingHost);
    this.$("evaluation-mode").onchange = () => {
      this.setEvaluationMode(this.$("evaluation-mode").value);
      if (this.report) {
        this.report.evaluation_mode = this.evaluationMode;
        this.saveSoon();
      }
    };
  }
  setEvaluationMode(mode) {
    this.evaluationMode = mode;
    this.$("evaluation-mode").value = mode;
    this.gradingSection.hidden = mode === "pairwise";
    this.$("judge-charts").hidden = this.$("judge-summary").hidden =
      mode === "pairwise";
    this.rankingHost.hidden = mode !== "pairwise";
    if (mode === "pairwise") {
      this.ranking.sync();
      this.ranking.schedulePreview();
    }
  }
  updateCurrent(record, identity = this.currentIdentity) {
    this.current = record;
    this.currentIdentity = identity;
    if (this.benchmarkId || this.importedReport) return;
    if (this.displayedCurrentIdentity !== identity) {
      this.displayedCurrentIdentity = identity;
      this.grader?.stop();
      this.ranking?.stop();
      this.report = null;
      this.prepared = null;
      this.pins = [];
      this.selection = null;
      this.data = null;
      this.filters.trial = "all";
      this.filters.method = "all";
      this.filters.attempt = "default";
      this.epoch++;
      this.$("method").value = "all";
    }
    const signature = record
      ? `${record.created_at}:${record.nodes.length}:${record.snapshots.length}:${record.requests?.length}:${record.run?.stop_reason}`
      : "empty";
    if (this.currentRecord === record && this.currentSignature === signature)
      return;
    this.currentRecord = record;
    this.currentSignature = signature;
    this.source = record ? { kind: "recording", record } : null;
    this.revision++;
    if (this.active) this.compute(true);
    else this.sourceDirty = true;
  }
  setActive(active) {
    this.active = active;
    if (active) {
      this.configSummary();
      if (this.sourceDirty) {
        this.sourceDirty = false;
        this.compute(true);
      } else if (!this.data && this.source) this.compute(true);
    }
  }
  configSummary() {
    let config;
    try {
      config =
        this.source?.kind === "benchmark"
          ? this.source.manifest.settings.config
          : (this.source?.record?.config ?? this.getConfig());
    } catch {
      return;
    }
    this.$("config").textContent =
      `${config.algorithm === "graph" ? "Graph" : "Wave"} · ${config.model} · ${config.walkers} walkers · temperature ${config.temperature} · ${config.chunk_tokens}-token chunks · ${config.sequence_tokens}-token cap · ${objectiveLabel(config)} objective`;
  }
  async selectBenchmark(id, live = false) {
    this.grader?.stop();
    this.ranking?.stop();
    this.prepared = null;
    this.report = null;
    this.importedReport = false;
    this.benchmarkId = id;
    this.pins = [];
    this.selection = null;
    this.filters.attempt = "default";
    this.filters.trial = "all";
    this.filters.method = "all";
    this.$("method").value = "all";
    this.epoch++;
    this.job++;
    this.loadingSource = false;
    const epoch = this.epoch;
    if (!id) {
      this.currentSignature = null;
      this.currentRecord = null;
      this.updateCurrent(this.current);
      if (!this.source) {
        this.data = null;
        this.render();
      }
      return;
    }
    this.loadingSource = true;
    this.data = null;
    this.status("Loading saved benchmark…");
    try {
      const store = await BrowserBenchmarkStore.open(id),
        events = await store.readEvents();
      if (epoch !== this.epoch) return;
      this.source = {
        kind: "benchmark",
        manifest: store.manifest,
        events,
        live,
      };
      this.revision++;
      this.compute(true);
    } catch (e) {
      this.status(e.message);
    } finally {
      if (epoch === this.epoch) {
        this.loadingSource = false;
        if (this.readAgain) {
          const next = this.readAgain;
          this.readAgain = null;
          this.refreshBenchmark(next.id, next.live);
        }
      }
    }
  }
  async refreshBenchmark(id, live) {
    if (id !== this.benchmarkId || this.readPending || this.loadingSource) {
      if (id === this.benchmarkId) this.readAgain = { id, live };
      return;
    }
    this.readPending = true;
    const epoch = this.epoch;
    try {
      const store = await BrowserBenchmarkStore.open(id),
        events = await store.readEvents(this.source?.events?.at(-1)?.seq ?? 0);
      if (epoch !== this.epoch) return;
      if (!this.source || this.source.kind !== "benchmark") return;
      this.source.events.push(...events);
      this.source.live = live;
      this.revision++;
      this.compute(false, events);
    } catch (e) {
      this.status(e.message);
    } finally {
      this.readPending = false;
      if (this.readAgain) {
        const next = this.readAgain;
        this.readAgain = null;
        this.refreshBenchmark(next.id, next.live);
      }
    }
  }
  compute(full = false, events) {
    if (!this.source) {
      this.data = null;
      this.status(
        "Generate or import a Fractal recording, or start a benchmark.",
      );
      this.render();
      return;
    }
    this.configSummary();
    this.status("Updating comparisons…");
    const message = {
      id: ++this.job,
      revision: this.revision,
      filters: this.filters,
      grades: this.gradeLookup(),
    };
    if (full || this.sourceDirty) {
      message.source = this.source;
      this.sourceDirty = false;
    } else if (events) {
      message.events = events;
      message.live = this.source.live;
    }
    this.worker.postMessage(message);
  }
  gradeLookup() {
    return gradesForSession(
      this.report?.evaluations.find(
        (s) => s.id === this.report.selected_evaluation,
      ),
    );
  }
  async ensureReport() {
    if (!this.source) throw Error("There is no source recording to save");
    if (!this.report) this.report = createReport(this.source, this.filters);
    this.report.source = structuredClone(this.source);
    delete this.report.source.live;
    this.report.view = { ...this.filters };
    this.report.evaluation_mode = this.evaluationMode ?? "absolute";
    return this.report;
  }
  async persist() {
    await this.ensureReport();
    try {
      await BrowserComparisonStore.save(this.report);
      this.$("storage").textContent =
        "Comparison report saved in this browser.";
    } catch (e) {
      this.$("storage").textContent =
        `Storage failed: ${e.message}. Export the report to retain the in-memory data.`;
      throw e;
    }
  }
  saveSoon() {
    clearTimeout(this.saveTimer);
    if (this.source)
      this.saveTimer = setTimeout(
        () =>
          this.persist()
            .then(() => this.refreshReports())
            .catch(() => {}),
        500,
      );
  }
  async refreshReports() {
    try {
      const list = await BrowserComparisonStore.list(),
        selected = this.report?.id ?? "";
      this.$("reports").replaceChildren(
        new Option("Select a report", ""),
        ...list
          .sort((a, b) => b.created_at - a.created_at)
          .map(
            (r) =>
              new Option(
                `${new Date(r.created_at).toLocaleString()} · ${r.source.kind === "recording" ? "Fractal recording" : "Benchmark"} · ${r.id.slice(0, 8)}`,
                r.id,
              ),
          ),
      );
      this.$("reports").value = selected;
    } catch (e) {
      this.$("storage").textContent =
        `Report storage unavailable: ${e.message}`;
    }
  }
  async loadReport(report) {
    this.grader?.stop();
    this.ranking?.stop();
    this.prepared = null;
    this.report = report;
    this.setEvaluationMode(report.evaluation_mode ?? "absolute");
    this.source = report.source;
    this.filters = { ...report.view };
    if (["grade", ...CRITERIA].includes(this.filters.metric))
      this.filters.metric = "mean";
    this.benchmarkId = null;
    this.importedReport = true;
    this.onCurrentSource();
    this.epoch++;
    this.revision++;
    this.pins = [];
    this.selection = null;
    for (const k of Object.keys(DEFAULT_FILTERS))
      this.$(k === "status" ? "status-filter" : k).value = this.filters[k];
    this.renderSessions();
    const selected = report.evaluations.find(
      (s) => s.id === report.selected_evaluation,
    );
    if (selected) {
      this.$("judge-model").value = selected.profile.model;
      for (const k of CRITERIA)
        this.$(`rubric-${k}`).value = selected.profile.rubric[k];
      this.$("reference").value = selected.profile.reference;
    }
    this.compute(true);
  }
  async export() {
    try {
      await this.ensureReport();
      download(
        exportReport(this.report),
        `llm-comparison-${this.report.id}.fgllmcompare`,
      );
    } catch (e) {
      this.status(e.message);
    }
  }
  selectKeys(keys) {
    this.selection = new Set(keys);
    this.page = 0;
    this.renderTraces();
    this.onEvaluation();
    this.$("traces").scrollIntoView({ block: "start", behavior: "smooth" });
  }
  render() {
    this.ranking?.sync();
    if (this.panel === "evaluation") this.refreshEvaluationSources();
    const out = this.$("results");
    out.replaceChildren();
    const d = this.data;
    this.$("judge-charts").replaceChildren();
    this.$("judge-summary").replaceChildren();
    this.$("method-legend").replaceChildren(legend(d?.methods ?? []));
    this.$("export").disabled = !this.source;
    if (!d) {
      out.append(
        el(
          "p",
          "Your Fractal results will appear here as soon as a generation boundary is recorded.",
          { class: "compare-empty" },
        ),
      );
      this.renderTraces();
      this.renderSessions();
      this.schedulePreview();
      return;
    }
    const trial = this.filters.trial;
    this.$("judge-summary").append(
      el("h3", "Judge assessments · trial-weighted means"),
      table(
        [
          "Method / population",
          "Trials",
          "Graded observations",
          "Overall / 100",
          ...CRITERIA.map((k) => `${k} / 4`),
        ],
        d.summaries.map((s) => {
          const groups = d.groups.filter(
            (g) => g.method === s.method && g.pool === s.pool,
          );
          const average = (key) => {
            const values = groups
              .map((g) => describe(g.traces, key).mean)
              .filter(finite);
            return values.length
              ? values.reduce((a, b) => a + b, 0) / values.length
              : null;
          };
          return [
            METHOD_STYLE[s.method].label + " · " + poolName(s.pool),
            s.trials,
            `${groups.reduce((n, g) => n + describe(g.traces, "grade").count, 0)}/${s.count}`,
            ...["grade", ...CRITERIA].map((k) => fmt(average(k))),
          ];
        }),
      ),
    );
    this.$("trial").replaceChildren(
      new Option("All trials", "all"),
      ...d.trials.map((t) => new Option(`Trial ${t + 1}`, String(t))),
    );
    this.$("trial").value = trial;
    const attempt = this.filters.attempt;
    this.$("attempt").replaceChildren(
      new Option("Successful methods + live preview", "default"),
      ...d.attempts.map(
        (r) =>
          new Option(
            `Trial ${r.trial + 1} · ${METHOD_STYLE[r.method].label} · attempt ${r.attempt} · ${r.status}`,
            r.id,
          ),
      ),
    );
    this.$("attempt").value = attempt;
    out.append(
      el(
        "p",
        "Archived answers count each generated endpoint once, including discarded branches. Retained population counts the last saved slots, including clones. Identical independently sampled text remains a separate observation. Full answers include EOS and sequence caps; these are reported separately.",
        { class: "hint" },
      ),
    );
    if (this.filters.attempt !== "default")
      out.append(
        el(
          "p",
          "Attempt inspector · Partial evidence from this attempt is isolated from the successful-method comparison.",
          { class: "compare-notice" },
        ),
      );
    if (this.filters.attempt !== "default") {
      const run = sourceRuns(this.source).find(
          (r) => r.id === this.filters.attempt,
        ),
        evidence = el("details");
      evidence.append(
        el("summary", "Attempt requests, accepted responses and errors"),
        el(
          "pre",
          JSON.stringify(
            {
              accepted: run?.accepted,
              requests: run?.requests,
              errors: run?.errors,
            },
            null,
            2,
          ),
        ),
      );
      out.append(evidence);
    }
    if (!d.rows.length)
      out.append(
        el(
          "p",
          "No eligible answers match these filters. Choose partial traces or inspect an unfinished attempt.",
          { class: "compare-empty" },
        ),
      );
    const numeric = button(
      "Export displayed numeric data",
      () =>
        download(
          JSON.stringify(
            {
              filters: d.filters,
              selected_keys: this.selection ? [...this.selection] : null,
              pinned_keys: this.pins,
              summaries: d.summaries,
              groups: d.groups.map(({ traces, diversity, ...g }) => ({
                ...g,
                diversity: { ...diversity, pairs: undefined },
              })),
              traces: d.rows,
              progress: d.histories,
              compute: d.efficiency,
              projection: d.projection,
              distributions: d.distributions,
            },
            null,
            2,
          ),
          "comparison-data.json",
        ),
      "compare-numeric-export",
    );
    out.append(numeric);
    out.append(el("h3", METRICS[this.filters.metric]));
    out.append(
      table(
        [
          "Method / population",
          "Trials",
          "Observations",
          "Measured",
          "Mean",
          "Median",
          "SD",
          "IQR",
          "Δ paired baseline (95% CI)",
          "Unique text",
          "Mean distance / coverage",
        ],
        d.summaries.map((s) => [
          button(`${METHOD_STYLE[s.method].label} · ${poolName(s.pool)}`, () =>
            this.selectKeys(
              d.rows
                .filter((r) => r.method === s.method && r.pool === s.pool)
                .map((r) => r.key),
            ),
          ),
          String(s.trials),
          String(s.count),
          `${s.measured_count}/${s.count} observations · ${s.measured_trials}/${s.trials} trials`,
          fmt(s.mean),
          fmt(s.median),
          fmt(s.sd),
          `${fmt(s.q25)} – ${fmt(s.q75)}`,
          `${fmt(s.difference.value)}${s.difference.interval ? ` [${s.difference.interval.map(fmt).join(", ")}]` : ""} · ${s.difference.trials} paired trials`,
          `${s.unique} · ${fmt(s.uniqueness == null ? null : s.uniqueness * 100)}%`,
          `${fmt(s.diversity)} · ${s.embedding_count}/${s.count} embedded`,
        ]),
      ),
    );
    out.append(
      el(
        "p",
        "Statistics and plotted distributions give each measured trial equal weight. Paired differences use matching trials; 95% bootstrap intervals resample trials and require at least two. Medians, SD and IQR are averages of within-trial statistics. Likelihood is returned under each method’s generation settings; temperature zero is not a common-temperature rescore. Reward is the configured final objective relative to the empty root, not answer correctness.",
        { class: "hint" },
      ),
    );
    const chartGrid = el("div", null, { class: "compare-chart-grid" });
    out.append(chartGrid);
    for (const pool of [...new Set(d.groups.map((g) => g.pool))]) {
      const groups = d.groups.filter((g) => g.pool === pool),
        dist = d.distributions[pool].selected,
        label = poolName(pool);
      chart(chartGrid, {
        title: `${label} · distribution`,
        xLabel: METRICS[this.filters.metric],
        yLabel: "Probability mass",
        series: dist.hist,
        kind: "histogram",
        domain: dist.domain,
        zeroY: true,
        onSelect: (p) => this.selectKeys(p.keys),
      });
      chart(chartGrid, {
        title: `${label} · cumulative distribution`,
        xLabel: METRICS[this.filters.metric],
        yLabel: "Cumulative probability",
        series: dist.cdf,
        kind: "step",
        zeroY: true,
        onSelect: (p) => this.selectKeys(p.keys),
      });
      const pairs = d.distributions[pool].pairs,
        pairCount = groups.reduce((s, g) => s + g.diversity.pairs.length, 0),
        total = groups.reduce((s, g) => s + g.diversity.total, 0);
      chart(chartGrid, {
        title: `${label} · embedding diversity`,
        xLabel: `${this.filters.distance} distance`,
        yLabel: "Probability mass",
        series: pairs.hist,
        kind: "histogram",
        domain: pairs.domain,
        zeroY: true,
        onSelect: (p) => this.selectKeys(p.keys),
        note: `${pairCount.toLocaleString()} / ${total.toLocaleString()} pairs. ${groups.some((g) => g.diversity.sampled) ? "Deterministic uniform sample; nearest-neighbor values are sample-based upper bounds." : "All eligible pairs."} Single-answer groups have no within-run diversity. ${groups
          .map((g) => g.diversity.reason)
          .filter(Boolean)
          .join("; ")}`,
      });
      const nearest = d.distributions[pool].nearest;
      chart(chartGrid, {
        title: `${label} · nearest neighbor`,
        xLabel: `${this.filters.distance} distance`,
        yLabel: "Cumulative probability",
        series: nearest.cdf,
        kind: "step",
        zeroY: true,
        onSelect: (p) => this.selectKeys(p.keys),
      });
      const rows = d.rows.filter((r) => r.pool === pool),
        seriesFor = (x, y) =>
          d.methods.map((method) => ({
            method,
            points: rows
              .filter((r) => r.method === method)
              .map((r) => ({
                x: r[x],
                y: r[y],
                keys: [r.key],
                label: `Trial ${r.trial + 1}; weight ${r.weight}`,
              })),
          }));
      chart(chartGrid, {
        title: `${label} · reward and length`,
        xLabel: "Generated tokens",
        yLabel: "Full-trace reward",
        series: seriesFor("tokens", "reward"),
        kind: "scatter",
        onSelect: (p) => this.selectKeys(p.keys),
        note: "Each marker is an endpoint; retained weights are shown in the trace table.",
      });
      chart(chartGrid, {
        title: `${label} · distance to temperature zero`,
        xLabel: "Mean token log likelihood",
        yLabel: `${this.filters.distance} distance`,
        series: seriesFor("mean", "greedy_distance"),
        kind: "scatter",
        onSelect: (p) => this.selectKeys(p.keys),
        note: "Distances to the temperature-zero endpoint from the same trial. Missing or incompatible embeddings are unavailable.",
      });
      if (this.report?.selected_evaluation) {
        const grades = d.distributions[pool].grade;
        chart(this.$("judge-charts"), {
          title: `${label} · judge assessments`,
          xLabel: "Judge overall score (0–100)",
          yLabel: "Cumulative probability",
          series: grades.cdf,
          kind: "step",
          onSelect: (p) => this.selectKeys(p.keys),
          zeroY: true,
        });
        for (const x of ["mean", "nearest"])
          chart(this.$("judge-charts"), {
            title: `${label} · judge score vs ${x === "mean" ? "likelihood" : "diversity"}`,
            xLabel: METRICS[x],
            yLabel: "Judge overall score",
            series: seriesFor(x, "grade"),
            kind: "scatter",
            onSelect: (p) => this.selectKeys(p.keys),
          });
      }
    }
    chart(chartGrid, {
      title: "Shared embedding projection · PCA",
      xLabel: "Principal component 1",
      yLabel: "Principal component 2",
      kind: "scatter",
      series: d.methods.map((method) => ({
        method,
        points: d.projection.points
          .filter((p) => p.method === method)
          .map((p) => ({ ...p, keys: [p.key] })),
      })),
      onSelect: (p) => this.selectKeys(p.keys),
      note:
        d.projection.reason ??
        `Shared two-dimensional projection of unique endpoints. ${this.filters.distance === "cosine" ? "Vectors normalized before PCA. " : ""}Distances use original embeddings, not this projection.`,
    });
    const histories = [...new Set(d.histories.map((h) => h.run_id))];
    const curves = (key, work = true) =>
      histories.map((id) => {
        const rows = d.histories.filter((h) => h.run_id === id);
        return {
          method: rows[0].method,
          label: `Trial ${rows[0].trial + 1}`,
          points: rows.map((h) => ({
            x: work ? h.generated_tokens : h.step,
            y: h[key],
          })),
        };
      });
    for (const [key, title, y] of [
      ["best_mean", "Best observed full answer", "Mean token log likelihood"],
      ["completed", "Completed answers", "EOS + capped endpoints"],
      ["eos", "Model-finished answers", "EOS endpoints"],
      ...(this.report?.selected_evaluation
        ? [["best_grade", "Best graded answer", "Judge overall score"]]
        : []),
    ])
      chart(key === "best_grade" ? this.$("judge-charts") : chartGrid, {
        title,
        xLabel: "Generated-token work at saved boundary",
        yLabel: y,
        series: curves(key),
        kind: "step",
        note: "One curve per method and trial. Answers appear only at their recorded generation boundary. Historical recordings without accepted-work counters use committed generation.",
      });
    for (const [key, title] of [
      ["clones", "Clones per boundary"],
      ["clone_rate", "Clone rate"],
      ["concentration", "Retained population concentration"],
      ["distinct", "Distinct retained endpoints"],
    ])
      chart(chartGrid, {
        title,
        xLabel: "Saved boundary",
        yLabel: title,
        series: curves(key, false),
        kind: "line",
      });
    out.append(
      el("h3", "Completion and retained multiplicity"),
      table(
        [
          "Method / trial / population",
          "EOS",
          "Capped",
          "Partial",
          "Empty",
          "Exact-text duplicates",
          "Concentration",
        ],
        d.groups.map((g) => [
          `${METHOD_STYLE[g.method].label} · ${g.trial + 1} · ${poolName(g.pool)}`,
          g.outcomes.eos,
          g.outcomes.capped,
          g.outcomes.partial,
          g.outcomes.empty,
          g.total - g.unique,
          fmt(g.concentration),
        ]),
      ),
    );
    for (const pool of [...new Set(d.groups.map((g) => g.pool))])
      chart(chartGrid, {
        title: `${poolName(pool)} · termination outcomes`,
        xLabel: "Outcome: 1 EOS · 2 cap · 3 partial · 4 empty",
        yLabel: "Mean within-trial fraction",
        kind: "scatter",
        zeroY: true,
        series: d.methods.map((method) => {
          const groups = d.groups.filter(
            (g) => g.pool === pool && g.method === method,
          );
          return {
            method,
            points: ["eos", "capped", "partial", "empty"].map((key, i) => {
              const values = groups
                .map((g) => {
                  const n = Object.values(g.outcomes).reduce(
                    (a, b) => a + b,
                    0,
                  );
                  return n ? g.outcomes[key] / n : null;
                })
                .filter(finite);
              return {
                x: i + 1,
                y: values.length
                  ? values.reduce((a, b) => a + b, 0) / values.length
                  : null,
                label: key,
              };
            }),
          };
        }),
      });
    const multiplicity = d.rows.filter((r) => r.pool === "retained");
    chart(chartGrid, {
      title: "Retained endpoint multiplicity",
      xLabel: "Endpoint index (within method)",
      yLabel: "Retained slots",
      kind: "scatter",
      series: d.methods.map((method) => ({
        method,
        points: multiplicity
          .filter((r) => r.method === method)
          .map((r, i) => ({ x: i + 1, y: r.weight, keys: [r.key] })),
      })),
      onSelect: (p) => this.selectKeys(p.keys),
      zeroY: true,
    });
    out.append(
      el("h3", "Generation, embedding and scoring work"),
      table(
        [
          "Method / trial",
          "Status / stop reason",
          "EOS target",
          "Generated tokens",
          "Provider generation tokens",
          "Gen. requests",
          "Gen. cost",
          "Embed. requests",
          "Embed. cost",
          "Scoring requests",
          "Scoring input / output tokens",
          "Scoring cost",
          "Scoring time (ms)",
          "Gen. request time (ms)",
          "Embed. request time (ms)",
          "Wall time (ms)",
        ],
        d.efficiency.map((e) => [
          `${METHOD_STYLE[e.method].label} · ${e.trial + 1}`,
          `${e.status} · ${e.stop_reason}`,
          fmt(e.completion_target),
          fmt(e.generated_tokens),
          fmt(e.generation.total_tokens.total),
          e.generation_requests,
          fmt(e.generation.cost.total),
          e.embedding_requests,
          fmt(e.embedding.cost.total),
          e.scoring_requests,
          `${fmt(e.scoring.prompt_tokens.total)} / ${fmt(e.scoring.completion_tokens.total)}`,
          fmt(e.scoring.cost.total),
          fmt(e.scoring_request_ms),
          fmt(e.generation_request_ms),
          fmt(e.embedding_request_ms),
          fmt(e.elapsed_ms),
        ]),
      ),
    );
    out.append(
      el(
        "p",
        "Costs are reported USD. Unreported usage, cost or timing remains unavailable; request times sum overlapping requests and differ from elapsed wall time. Preflight and grading are excluded from generation totals.",
        { class: "hint" },
      ),
    );
    for (const [key, title] of [
      ["cost", "Reported cost (USD)"],
      ["time", "Request timing (ms)"],
    ])
      for (const type of ["generation", "embedding"])
        chart(chartGrid, {
          title: `${type === "generation" ? "Generation" : "Embedding"} · ${title}`,
          xLabel: "Trial",
          yLabel: title,
          kind: "scatter",
          zeroY: true,
          series: d.methods.map((method) => ({
            method,
            points: d.efficiency
              .filter((e) => e.method === method)
              .map((e) => ({
                x: e.trial + 1,
                y:
                  key === "cost" ? e[type].cost.total : e[`${type}_request_ms`],
              })),
          })),
        });
    this.renderTraces();
    this.renderSessions();
    this.schedulePreview();
  }
  renderTraces() {
    const host = this.$("traces");
    host.replaceChildren();
    const d = this.data;
    if (!d) {
      this.$("pins").replaceChildren();
      return;
    }
    const bar = el("div", null, { class: "section-title" });
    bar.append(
      el("h3", "Trace browser"),
      button("Clear chart selection", () => {
        this.selection = null;
        this.page = 0;
        this.renderTraces();
      }),
    );
    host.append(bar);
    const controls = el("div", null, { class: "compare-filters" }),
      sort = select(
        "Sort by",
        "compare-sort",
        [
          ["mean", "Mean log likelihood"],
          ["reward", "Full-trace reward"],
          ["tokens", "Length"],
          ["grade", "Judge overall"],
          ["weight", "Multiplicity"],
          ["trial", "Trial"],
        ],
        this.sort,
      );
    sort.querySelector("select").onchange = (e) => {
      this.sort = e.target.value;
      this.renderTraces();
    };
    controls.append(
      sort,
      button(this.desc ? "Descending ↓" : "Ascending ↑", () => {
        this.desc = !this.desc;
        this.renderTraces();
      }),
    );
    host.append(controls);
    let rows = d.rows.filter(
      (r) => !this.selection || this.selection.has(r.key),
    );
    rows = [...rows].sort((a, b) => {
      const x = a[this.sort],
        y = b[this.sort];
      return finite(x) && finite(y)
        ? (x - y) * (this.desc ? -1 : 1)
        : finite(x)
          ? -1
          : finite(y)
            ? 1
            : a.key.localeCompare(b.key);
    });
    const pageSize = 100;
    this.page = Math.min(
      this.page,
      Math.max(0, Math.ceil(rows.length / pageSize) - 1),
    );
    host.append(
      el(
        "p",
        `${rows.length} rows · ${new Set(rows.map((r) => r.key)).size} endpoints · Page ${this.page + 1} of ${Math.max(1, Math.ceil(rows.length / pageSize))}. Select Pin A and Pin B to compare any two traces.`,
        { class: "hint" },
      ),
    );
    host.append(
      table(
        [
          "Method / trial",
          "Population",
          "Endpoint",
          "Termination",
          "Slots",
          "Tokens",
          "Mean log likelihood",
          "Full-trace reward",
          "Judge",
          "Text",
          "Compare",
        ],
        rows
          .slice(this.page * pageSize, (this.page + 1) * pageSize)
          .map((r) => {
            const actions = el("div", null, { class: "compare-pin-actions" });
            for (const i of [0, 1])
              actions.append(
                button(
                  `Pin ${i ? "B" : "A"}`,
                  () => {
                    this.pins[i] = r.key;
                    this.renderPins();
                  },
                  null,
                ),
              );
            const trace = button(r.text || "(empty)", () => {
              this.pins[0] = r.key;
              this.renderPins();
            });
            trace.className = "compare-text-button";
            return [
              `${METHOD_STYLE[r.method].label} · ${r.trial + 1}`,
              poolName(r.pool),
              r.node_id,
              termination(r),
              r.weight,
              r.tokens,
              fmt(r.mean),
              fmt(r.reward),
              fmt(r.grade),
              trace,
              actions,
            ];
          }),
      ),
    );
    const pages = el("div", null, { class: "toolbar" }),
      prev = button("Previous", () => {
        this.page--;
        this.renderTraces();
      }),
      next = button("Next", () => {
        this.page++;
        this.renderTraces();
      });
    prev.disabled = this.page === 0;
    next.disabled = (this.page + 1) * pageSize >= rows.length;
    pages.append(prev, next);
    host.append(pages);
    this.renderPins();
  }
  renderPins(source = this.source) {
    const host = this.$("pins");
    host.replaceChildren();
    if (!source) return;
    const runs = sourceRuns(source),
      comparison = traceComparison(runs, this.pins.filter(Boolean));
    if (!comparison.traces.length) return;
    host.append(el("h3", "Pinned trace comparison"));
    if (comparison.traces.length === 2) {
      const [a, b] = comparison.traces;
      const ra = runs.find((r) => a.key.startsWith(r.id + "/")),
        rb = runs.find((r) => b.key.startsWith(r.id + "/"));
      const compatible =
        ra.config.embedding_model === rb.config.embedding_model &&
        ra.config.embedding_input === rb.config.embedding_input;
      host.append(
        el(
          "p",
          `Shared recorded ancestry: ${comparison.shared_ancestry_tokens} tokens · Matching text prefix: ${comparison.matching_text_characters} characters · ${this.filters.distance} embedding distance: ${fmt(compatible ? distance(a.node.embedding, b.node.embedding, this.filters.distance) : null)}. Matching text alone does not imply shared ancestry.`,
          { class: "compare-notice" },
        ),
      );
    }
    const grid = el("div", null, { class: "compare-chart-grid" });
    host.append(grid);
    for (const t of comparison.traces) {
      const card = el("article", null, { class: "compare-trace-card" });
      card.append(
        el("h4", `${METHOD_STYLE[t.method].label} · ${t.key}`),
        el(
          "p",
          `${termination(t.node)} · ${t.node.tokens} tokens · total log likelihood ${fmt(t.node.logp)} · mean ${fmt(t.node.logp / t.node.tokens)}`,
        ),
        button("Copy full text", async () => {
          try {
            await navigator.clipboard.writeText(t.text);
          } catch (e) {
            this.status(`Copy failed: ${e.message}`);
          }
        }),
        el("pre", t.text, { class: "compare-full-text" }),
      );
      const tokens = el("div", null, { class: "tokens" });
      renderTokens(tokens, t.tokens, (token) =>
        this.status(JSON.stringify(token)),
      );
      card.append(tokens);
      const detail = el("details"),
        summary = el("summary", "Tokens, bytes and chunk boundaries");
      detail.append(
        summary,
        table(
          [
            "Index",
            "Chunk",
            "Token text",
            "UTF-8 bytes",
            "Log probability",
            "Probability",
            "Cumulative reward",
          ],
          t.tokens.map((token) => [
            token.index,
            token.chunk,
            token.text,
            token.bytes?.join(" ") ?? "Unavailable",
            fmt(token.logprob),
            fmt(Math.exp(token.logprob)),
            fmt(token.reward),
          ]),
        ),
      );
      card.append(detail);
      const chunks = el("details");
      chunks.append(
        el("summary", "Incremental chunk rewards"),
        table(
          ["Chunk", "Cumulative tokens", "Incremental reward"],
          t.chunks.map((c) => [c.id, c.tokens, fmt(c.reward)]),
        ),
      );
      card.append(chunks);
      const grade = this.gradeLookup()[t.key];
      if (grade)
        card.append(
          el(
            "p",
            `Judge assessment: ${fmt(grade.overall)} / 100 · ${CRITERIA.map((k) => `${k} ${fmt(grade.scores[k])}/4`).join(" · ")}\n${grade.explanation}`,
          ),
        );
      grid.append(card);
    }
    const plots = el("div", null, { class: "compare-chart-grid" });
    host.append(plots);
    for (const [key, label] of [
      ["reward", "Cumulative objective"],
      ["logp", "Cumulative log likelihood"],
    ])
      chart(plots, {
        title: label,
        xLabel: "Token index",
        yLabel: label,
        series: comparison.traces.map((t) => ({
          method: t.method,
          label: t.key,
          points: t.tokens.map((token) => ({ x: token.index, y: token[key] })),
        })),
      });
  }
  buildGrading() {
    const section = el("section", null, { class: "compare-grading" });
    section.append(
      el("h3", "Grade answers · optional"),
      el(
        "p",
        "Judge assessments are separate from likelihood. The grading model defaults to the latest Gemini Flash alias and remains independent from generation. Only prompt, answer, rubric and optional reference are sent; method names and sampling history are hidden.",
        { class: "hint" },
      ),
    );
    const fields = el("div", null, { class: "compare-filters" }),
      model = el("label", "Grading model"),
      input = el("input", null, {
        id: "compare-judge-model",
        value: DEFAULT_GRADING_MODEL,
        placeholder: "OpenRouter model ID",
        autocomplete: "off",
      });
    model.append(input);
    fields.append(
      select(
        "Saved grading session",
        "compare-judge-session",
        [["", "No judge scores"]],
        "",
      ),
    );
    section.append(fields);
    const details = el("details");
    details.append(el("summary", "Advanced grading settings"), model);
    for (const k of CRITERIA) {
      const label = el("label", k[0].toUpperCase() + k.slice(1)),
        area = el("textarea", DEFAULT_RUBRIC[k], {
          id: `compare-rubric-${k}`,
          rows: 2,
        });
      label.append(area);
      details.append(label);
    }
    const reference = el("label", "Reference answer (optional)");
    reference.append(el("textarea", "", { id: "compare-reference", rows: 3 }));
    details.append(reference);
    section.append(details);
    const actions = el("div", null, { class: "compare-filters" }),
      limit = el("label", "Maximum answer requests this action");
    limit.append(
      el("input", null, {
        type: "number",
        min: 1,
        max: 10000,
        value: 100,
        id: "compare-judge-limit",
      }),
    );
    actions.append(
      limit,
      button(
        "Refresh estimate",
        () => this.previewGrading(),
        "compare-judge-preview",
      ),
      button(
        "Evaluate with Gemini Flash",
        () => this.startGrading(),
        "compare-judge-start",
      ),
      button("Pause", () => this.grader?.pause(), "compare-judge-pause"),
      button(
        "Continue",
        () => this.grader?.continue(),
        "compare-judge-continue",
      ),
      button("Stop", () => this.grader?.stop(), "compare-judge-stop"),
    );
    section.append(
      actions,
      el(
        "p",
        "Click Evaluate to assess missing or failed answers with the selected judge.",
        { id: "compare-judge-status", role: "status", "aria-live": "polite" },
      ),
      el("p", "", { id: "compare-judge-estimate", role: "status" }),
      el("div", null, { id: "compare-judge-results" }),
    );
    this.gradingSection = section;
    this.evaluationHost.append(section);
    for (const id of ["pause", "continue", "stop"])
      this.$(`judge-${id}`).disabled = true;
    this.$("judge-session").onchange = () => {
      if (!this.report) return;
      this.report.selected_evaluation = this.$("judge-session").value || null;
      const s = this.report.evaluations.find(
        (s) => s.id === this.report.selected_evaluation,
      );
      if (s) {
        this.$("judge-model").value = s.profile.model;
        for (const k of CRITERIA)
          this.$(`rubric-${k}`).value = s.profile.rubric[k];
        this.$("reference").value = s.profile.reference;
      }
      this.prepared = null;
      this.schedulePreview();
      this.compute();
      this.saveSoon();
    };
    for (const field of [
      input,
      ...details.querySelectorAll("textarea"),
      this.$("judge-limit"),
    ])
      field.oninput = () => {
        this.prepared = null;
        this.schedulePreview();
      };
  }
  schedulePreview() {
    if (this.evaluationMode === "pairwise") return;
    if (this.panel !== "evaluation" || this.grader || this.gradingStarting)
      return;
    clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => this.previewGrading(), 400);
  }
  async previewGrading({ forStart = false } = {}) {
    const ticket = (this.previewTicket = (this.previewTicket || 0) + 1);
    try {
      if (this.grader)
        throw Error(
          "Stop the current grading action before changing its configuration",
        );
      if (!this.source) throw Error("Generate or select a dataset first");
      const source = this.source,
        trial = this.filters.trial;
      const candidates = await gradingCandidates(source, trial),
        profile = {
          model: this.$("judge-model").value,
          rubric: Object.fromEntries(
            CRITERIA.map((k) => [k, this.$(`rubric-${k}`).value]),
          ),
          reference: this.$("reference").value,
        };
      this.$("judge-start").textContent =
        profile.model === DEFAULT_GRADING_MODEL
          ? "Evaluate with Gemini Flash"
          : "Evaluate with selected judge";
      if (!candidates.length)
        throw Error(
          "No completed answers in the selected trials. Generate or select a dataset with finished or capped answers first.",
        );
      if (!this.getKey())
        throw Error(
          `${candidates.length} distinct answers available. Enter your OpenRouter session key to evaluate them.`,
        );
      const routeKey = JSON.stringify(profile);
      const prepared =
        !forStart &&
        this.previewRoute?.key === routeKey &&
        Date.now() - this.previewRoute.time < 60000
          ? this.previewRoute.prepared
          : await prepareJudge(this.getKey(), profile);
      this.previewRoute = { key: routeKey, time: Date.now(), prepared };
      const existing = this.report?.evaluations.find(
          (s) => s.id === prepared.id,
        ),
        pending = candidates.filter(
          (c) => existing?.results[c.id]?.status !== "valid",
        ),
        limit = Number(this.$("judge-limit").value);
      if (!Number.isSafeInteger(limit) || limit < 1 || limit > 10000)
        throw Error("Choose a request limit between 1 and 10,000");
      if (
        source !== this.source ||
        trial !== this.filters.trial ||
        ticket !== this.previewTicket
      )
        throw Error("Source or grading settings changed; evaluate again");
      this.prepared = {
        ...prepared,
        candidates,
        source: this.source,
        trial: this.filters.trial,
        limit,
      };
      const estimate = estimateGrading(prepared, pending, limit);
      this.$("judge-estimate").textContent =
        `${candidates.length} distinct prompt/answer pairs across methods and both populations; ${pending.length} missing or failed. This action: up to ${Math.min(limit, pending.length)} answer requests, concurrency 2, temperature 0, 1,024 output tokens each. Provider: ${prepared.profile.provider}. ${estimate === null ? "Cost estimate unavailable." : `Conservative token-cost estimate: $${fmt(estimate)} (UTF-8 byte estimate for input).`} Provider transport retries may add requests; actual usage is recorded.`;
      return this.prepared;
    } catch (e) {
      if (ticket === this.previewTicket) {
        this.$("judge-estimate").textContent = e.message;
        this.prepared = null;
      }
      if (forStart) throw e;
    }
  }
  async startGrading() {
    if (this.gradingStarting || this.grader || this.ranking?.busy) return;
    this.gradingStarting = true;
    clearTimeout(this.previewTimer);
    const sourceEpoch = this.epoch;
    this.gradingControls(true);
    try {
      this.$("judge-status").textContent =
        "Preparing judge and checking the provider…";
      const prepared = await this.previewGrading({ forStart: true });
      await this.ensureReport();
      if (sourceEpoch !== this.epoch)
        throw Error("Source changed before grading started");
      let s = this.report.evaluations.find((s) => s.id === prepared.id);
      if (!s) {
        s = {
          id: prepared.id,
          profile: prepared.profile,
          requested_model: prepared.requested_model,
          endpoint: prepared.endpoint,
          created_at: Date.now(),
          status: "idle",
          candidates: [],
          results: {},
          requests: [],
          request_starts: [],
        };
        this.report.evaluations.push(s);
      }
      for (const c of prepared.candidates) {
        const prior = s.candidates.find((p) => p.id === c.id);
        if (prior)
          prior.occurrences = [
            ...new Set([...prior.occurrences, ...c.occurrences]),
          ];
        else s.candidates.push(c);
      }
      this.report.selected_evaluation = s.id;
      await this.persist();
      const savedReport = this.report;
      this.grader = new GradingController(this.getKey(), s, {
        persist: async () => {
          try {
            await BrowserComparisonStore.save(savedReport);
          } catch (e) {
            this.$("storage").textContent =
              `Storage failed: ${e.message}. Export the in-memory report.`;
            throw e;
          }
        },
        onStatus: (text) => {
          if (this.report === savedReport) {
            this.$("judge-status").textContent = text;
            this.compute();
          }
        },
      });
      this.gradingControls(true);
      await this.grader.run(
        prepared.limit,
        prepared.candidates.map((c) => c.id),
      );
      this.$("judge-status").textContent =
        `Grading ${s.status}. Valid grades are retained. Click Evaluate to retry missing or failed grades.`;
    } catch (e) {
      this.$("judge-status").textContent = e.message;
    } finally {
      this.grader = null;
      this.gradingStarting = false;
      this.prepared = null;
      this.gradingControls(false);
      this.renderSessions();
      this.compute();
      this.refreshReports();
    }
  }
  gradingControls(running) {
    for (const id of ["preview", "start", "model", "limit", "session"])
      this.$(`judge-${id}`).disabled = running;
    for (const k of CRITERIA) this.$(`rubric-${k}`).disabled = running;
    this.$("reference").disabled = running;
    for (const id of ["pause", "continue", "stop"])
      this.$(`judge-${id}`).disabled = !running;
    this.$("reports").disabled = this.$("import").disabled = running;
    for (const id of ["evaluation-source", "evaluation-key", "trial"])
      if (this.$(id)) this.$(id).disabled = running;
    this.$("evaluation-mode").disabled = running;
  }
  renderSessions() {
    const sessions = this.report?.evaluations ?? [],
      id = this.report?.selected_evaluation ?? "";
    this.$("judge-session").replaceChildren(
      new Option("No judge scores", ""),
      ...sessions.map(
        (s) =>
          new Option(
            `${s.profile.model} · ${s.profile.provider} · ${s.id.slice(0, 8)}`,
            s.id,
          ),
      ),
    );
    this.$("judge-session").value = id;
    const host = this.$("judge-results");
    host.replaceChildren();
    const s = sessions.find((s) => s.id === id);
    if (!s) return;
    const usage = usageTotals(
        s.requests.filter((r) => r.path === "chat/completions"),
      ),
      valid = Object.values(s.results).filter(
        (r) => r.status === "valid",
      ).length;
    host.append(
      el(
        "p",
        `${valid}/${s.candidates.length} distinct answers assessed · Judge cost $${fmt(usage.cost.total)} · ${s.requests.filter((r) => r.path === "chat/completions").length} provider requests. Missing criteria have no fabricated overall score.`,
        { class: "hint" },
      ),
    );
    const detail = el("details");
    detail.append(el("summary", "Grading responses, usage and errors"));
    detail.ontoggle = () => {
      if (detail.open && detail.children.length === 1)
        detail.append(
          el(
            "pre",
            JSON.stringify(
              { profile: s.profile, results: s.results, requests: s.requests },
              null,
              2,
            ),
          ),
        );
    };
    host.append(detail);
    for (const result of Object.values(s.results).filter(
      (r) => r.status !== "valid",
    ))
      host.append(
        el(
          "p",
          `Evaluation failed: ${result.error || "Invalid judge response"}. Click Evaluate to retry.`,
          { class: "compare-notice" },
        ),
      );
  }
}
