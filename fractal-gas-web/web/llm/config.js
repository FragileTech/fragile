export const DEFAULTS = Object.freeze({
  prompt: "Write a short explanation of why the sky is blue.",
  model: "qwen/qwen3.5-35b-a3b",
  embedding_model: "openai/text-embedding-3-small",
  algorithm: "wave",
  objective: "beam",
  beam_alpha: 0.6,
  xed_direction: "maximize",
  scoring_model: "Qwen/Qwen3.5-9B",
  embedding_input: "generated",
  distance_metric: "cosine",
  walkers: 8,
  chunk_tokens: 32,
  sequence_tokens: 256,
  concurrency: 4,
  iterations: 32,
  seed: 7,
  temperature: 1,
  distance_coef: 1,
  reward_coef: 1,
  max_walkers: 256,
});
export function configuration(input = {}) {
  const out = {};
  for (const key of Object.keys(DEFAULTS))
    out[key] = input[key] ?? DEFAULTS[key];
  for (const [key, choices] of Object.entries({
    algorithm: ["wave", "graph"],
    objective: ["total", "mean", "beam", "xed"],
    xed_direction: ["maximize", "minimize"],
    embedding_input: ["generated", "prompt"],
    distance_metric: ["l2", "cosine"],
  }))
    if (!choices.includes(out[key])) throw new Error(`Invalid ${key}`);
  for (const [key, lo, hi] of [
    ["walkers", 2, 1024],
    ["chunk_tokens", 1, 4096],
    ["sequence_tokens", 1, 32768],
    ["concurrency", 1, 32],
    ["iterations", 1, 10000],
    ["seed", 0, 2147483647],
    ["max_walkers", 2, 4096],
  ]) {
    const n = Number(out[key]);
    if (!Number.isInteger(n) || n < lo || n > hi)
      throw new Error(`${key} must be between ${lo} and ${hi}`);
    out[key] = n;
  }
  for (const [key, max] of [
    ["temperature", 2],
    ["beam_alpha", 2],
    ["distance_coef", 10],
    ["reward_coef", 10],
  ]) {
    out[key] = Number(out[key]);
    if (!Number.isFinite(out[key]) || out[key] < 0 || out[key] > max)
      throw new Error(`Invalid ${key}`);
  }
  for (const key of ["prompt", "model", "embedding_model", "scoring_model"])
    if (typeof out[key] !== "string" || !out[key].trim())
      throw new Error(`Enter ${key.replaceAll("_", " ")}`);
  if (out.max_walkers < out.walkers)
    throw new Error("Graph population cap must cover the starting walkers");
  if (out.chunk_tokens > out.sequence_tokens)
    throw new Error("Tokens per step must not exceed the sequence cap");
  return out;
}
// Imports retain the defaults in force when the original recording was written.
export function recordedConfiguration(input = {}) {
  return configuration({ ...input, objective: input.objective ?? "total" });
}
export function selectedScore(node, config) {
  const c =
    typeof config === "string" ? { objective: config } : (config ?? DEFAULTS);
  if (!node.tokens) return 0;
  switch (c.objective) {
    case "mean":
      return node.logp / node.tokens;
    case "beam":
      return node.logp / node.tokens ** (c.beam_alpha ?? 0.6);
    case "xed":
      if (!node.xed || !(node.xed.tokens > 0)) throw Error("Missing XED score");
      return (
        (node.xed.conditional_logp - node.xed.baseline_logp) / node.xed.tokens
      );
    default:
      return node.logp;
  }
}
export function objective(node, config) {
  const value = selectedScore(node, config);
  return config?.objective === "xed" && config.xed_direction === "minimize"
    ? -value
    : value;
}
export function objectiveLabel(config) {
  if (config?.objective === "xed")
    return `Mean XED · ${config.xed_direction} · ${config.scoring_model}`;
  if (config?.objective === "beam")
    return `Beam-style · α = ${config.beam_alpha}`;
  return config?.objective === "total"
    ? "Total log probability (legacy)"
    : "Negative mean Xent";
}
export function embeddingText(config, text) {
  return config.embedding_input === "prompt"
    ? `${config.prompt}\n\n${text}`
    : text;
}
export function bestNode(nodes, mode) {
  const candidates = nodes.filter((n) => n.tokens > 0 && n.committed !== false);
  const completed = candidates.filter((n) => n.status !== 0);
  const maxDepth = candidates.reduce(
    (depth, n) => Math.max(depth, n.tokens),
    0,
  );
  return (
    completed.length
      ? completed
      : candidates.filter((n) => n.tokens === maxDepth)
  ).reduce(
    (best, n) =>
      !best || objective(n, mode) > objective(best, mode) ? n : best,
    null,
  );
}
