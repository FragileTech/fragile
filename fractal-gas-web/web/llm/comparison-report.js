import { importRecording } from "./recording.js";
import {
  validateArchive,
  validateManifest,
  processBenchmark,
} from "./benchmark-data.js";
import {
  computeComparison,
  DEFAULT_FILTERS,
  METRICS,
  METHOD_STYLE,
  sourceRuns,
  endpointWeights,
} from "./comparison-metrics.js";
import {
  gradingProfile,
  validateGrade,
  gradesForSession,
  digest,
} from "./grading.js";
import {
  validateRankingSession,
  verifyRankingIdentities,
} from "./ranking-validation.js";
import { rankingTables } from "./ranking-statistics.js";

export const COMPARISON_VERSION = 2;
export function validateView(input = {}) {
  const v = { ...DEFAULT_FILTERS, ...input };
  if (
    !["all", "full", "eos", "partial"].includes(v.status) ||
    !["both", "archive", "retained"].includes(v.pool) ||
    !["cosine", "l2"].includes(v.distance) ||
    !Object.hasOwn(METRICS, v.metric) ||
    !(v.method === "all" || Object.hasOwn(METHOD_STYLE, v.method)) ||
    !Object.hasOwn(METHOD_STYLE, v.baseline) ||
    !(
      v.trial === "all" ||
      (Number.isSafeInteger(Number(v.trial)) && Number(v.trial) >= 0)
    ) ||
    typeof v.attempt !== "string"
  )
    throw Error("Invalid comparison view settings");
  return Object.fromEntries(Object.keys(DEFAULT_FILTERS).map((k) => [k, v[k]]));
}
export function validateSource(source) {
  if (source?.kind === "recording")
    return {
      kind: "recording",
      record: importRecording(JSON.stringify(source.record)),
    };
  if (source?.kind === "benchmark") {
    const manifest = validateManifest(source.manifest);
    validateArchive(manifest, source.events);
    return { kind: "benchmark", manifest, events: source.events };
  }
  throw Error("Comparison report has no supported source snapshot");
}
export function createReport(source, view = {}) {
  return {
    format: "fgllmcompare",
    version: COMPARISON_VERSION,
    id: crypto.randomUUID(),
    created_at: Date.now(),
    source: structuredClone(source),
    view: validateView(view),
    evaluations: [],
    rankings: [],
    evaluation_mode: "absolute",
  };
}
function rejectCredentials(value) {
  if (!value || typeof value !== "object") return;
  for (const [key, item] of Object.entries(value)) {
    if (/^(api[_-]?key|authorization|access_token|apiKey)$/i.test(key))
      throw Error("Reports must not contain credentials");
    rejectCredentials(item);
  }
}
export function parseReport(text) {
  const r = JSON.parse(text);
  if (
    r?.format !== "fgllmcompare" ||
    ![1, COMPARISON_VERSION].includes(r.version) ||
    typeof r.id !== "string" ||
    !/^[\w-]{1,100}$/.test(r.id) ||
    !Number.isFinite(r.created_at) ||
    !Array.isArray(r.evaluations)
  )
    throw Error("Unsupported comparison report");
  rejectCredentials(r);
  const source = validateSource(r.source),
    view = validateView(r.view);
  const runs = sourceRuns(source),
    occurrences = new Map(
      runs.flatMap((run) =>
        run.nodes.map((n) => [
          `${run.id}/${n.id}`,
          [run.config.prompt, n.text],
        ]),
      ),
    );
  const sessions = new Set();
  for (const s of r.evaluations) {
    if (
      typeof s.id !== "string" ||
      sessions.has(s.id) ||
      !Array.isArray(s.candidates) ||
      !s.results ||
      !Array.isArray(s.requests) ||
      !Array.isArray(s.request_starts)
    )
      throw Error("Invalid grading session");
    sessions.add(s.id);
    gradingProfile(s.profile);
    const candidates = new Set();
    for (const c of s.candidates) {
      if (
        typeof c.id !== "string" ||
        candidates.has(c.id) ||
        typeof c.prompt !== "string" ||
        typeof c.answer !== "string" ||
        !Array.isArray(c.occurrences) ||
        !c.occurrences.length
      )
        throw Error("Invalid grading candidate");
      candidates.add(c.id);
      for (const key of c.occurrences)
        if (
          JSON.stringify(occurrences.get(key)) !==
          JSON.stringify([c.prompt, c.answer])
        )
          throw Error("Grade does not match the source answer");
    }
    for (const [id, result] of Object.entries(s.results)) {
      if (
        !candidates.has(id) ||
        !["valid", "failed", "interrupted", "running", "received"].includes(
          result.status,
        )
      )
        throw Error("Invalid grading result");
      if (result.status === "valid") {
        const validated = validateGrade({
          scores: result.grade?.scores,
          explanation: result.grade?.explanation,
        });
        if (validated.overall !== result.grade.overall)
          throw Error("Invalid overall grade");
      }
    }
  }
  const rankings = r.rankings ?? [];
  if (
    !Array.isArray(rankings) ||
    new Set(rankings.map((s) => s.id)).size !== rankings.length
  )
    throw Error("Invalid ranking sessions");
  for (const s of rankings) validateRankingSession(s, validateSource);
  if (![undefined, "absolute", "pairwise"].includes(r.evaluation_mode))
    throw Error("Invalid evaluation mode");
  if (r.selected_ranking && !rankings.some((s) => s.id === r.selected_ranking))
    throw Error("Unknown selected ranking");
  return {
    format: r.format,
    version: r.version,
    id: r.id,
    created_at: r.created_at,
    source,
    view,
    evaluations: r.evaluations,
    selected_evaluation: r.selected_evaluation ?? null,
    rankings,
    selected_ranking: r.selected_ranking ?? null,
    evaluation_mode: r.evaluation_mode ?? "absolute",
  };
}
export function exportReport(report) {
  return JSON.stringify(parseReport(JSON.stringify(report)));
}
export async function verifyReportIdentities(report) {
  for (const s of report.rankings ?? []) await verifyRankingIdentities(s);
  for (const s of report.evaluations) {
    if (s.id !== (await digest(gradingProfile(s.profile))))
      throw Error("Grading configuration identity mismatch");
    for (const c of s.candidates)
      if (c.id !== (await digest([c.prompt, c.answer])))
        throw Error("Grading answer identity mismatch");
  }
  return report;
}
export function processReport(report) {
  const r = parseReport(JSON.stringify(report)),
    source = r.source;
  let tables;
  if (source.kind === "benchmark")
    tables = processBenchmark(source.manifest, source.events);
  else {
    const run = sourceRuns(source)[0],
      link = { run_id: run.id, method: "fractal", trial: 0, attempt: 1 };
    tables = {
      runs: [{ ...link, config: run.config, status: run.status }],
      trajectories: [...endpointWeights(run, "archive")].map(([node_id]) => ({
        ...link,
        trajectory_id: `current/${node_id}`,
        node_id,
        status: run.nodes[node_id].status,
        tokens: run.nodes[node_id].tokens,
      })),
      nodes: [],
      tokens: [],
      requests: run.requests.map((r) => ({ ...link, ...r })),
      snapshots: [],
      clone_decisions: [],
    };
    for (const n of run.nodes) {
      const { token_data, ...node } = n;
      tables.nodes.push({ ...link, ...node });
      token_data.forEach((t, i) =>
        tables.tokens.push({ ...link, node_id: n.id, index: i, ...t }),
      );
    }
    for (const s of run.snapshots) {
      const { decisions, ...snapshot } = s;
      tables.snapshots.push({ ...link, ...snapshot });
      for (const d of decisions ?? [])
        tables.clone_decisions.push({ ...link, step: s.step, ...d });
    }
  }
  const session = r.evaluations.find((s) => s.id === r.selected_evaluation),
    lookup = gradesForSession(session);
  const data = computeComparison(source, r.view, lookup);
  const distributionRows = [];
  for (const [pool, measures] of Object.entries(data.distributions))
    for (const [measure, value] of Object.entries(measures))
      for (const kind of ["hist", "cdf"])
        for (const series of value[kind])
          for (const point of series.points)
            distributionRows.push({
              pool,
              measure,
              kind,
              method: series.method,
              ...point,
            });
  const rankingData = {};
  for (const s of r.rankings)
    for (const [name, rows] of Object.entries(rankingTables(s)))
      (rankingData[name] ??= []).push(...rows);
  return {
    ...tables,
    ...rankingData,
    metrics: data.summaries.map((row) => ({ metric: r.view.metric, ...row })),
    comparison_distributions: distributionRows,
    comparison_projection: data.projection.points,
    comparison_traces: data.rows,
    comparison_progress: data.histories,
    comparison_compute: data.efficiency,
    grading_sessions: r.evaluations.map(({ candidates, results, ...s }) => s),
    grades: r.evaluations.flatMap((s) =>
      s.candidates.map((c) => ({ session_id: s.id, ...c, ...s.results[c.id] })),
    ),
  };
}
