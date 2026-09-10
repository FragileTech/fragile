export const STOP_LABELS = Object.freeze({
  eos_target: "Completed — EOS target reached",
  token_budget: "Generated-token budget reached",
  no_active_branches: "No active branches remain",
  iteration_limit: "Iteration limit reached",
});

export function initialRun(config) {
  return {
    completion_target: config.walkers,
    eos_node_ids: [],
    token_budget: config.walkers * config.sequence_tokens,
    generated_tokens: 0,
    stop_reason: null,
  };
}

export function formatRunProgress(run) {
  if (!run) return "";
  return `${run.eos_node_ids.length}/${run.completion_target} EOS completions · ${run.generated_tokens}/${run.token_budget} generated tokens${run.stop_reason ? ` · ${STOP_LABELS[run.stop_reason]}` : ""}`;
}

export function stoppingReason(config, run, snapshot, nodes, steps) {
  if (run.eos_node_ids.length >= run.completion_target) return "eos_target";
  if (run.generated_tokens >= run.token_budget) return "token_budget";
  if (
    snapshot &&
    !snapshot.walkers.some(
      (w) => w.alive && w.node !== null && nodes[w.node]?.status === 0,
    )
  )
    return "no_active_branches";
  if (steps >= config.iterations) return "iteration_limit";
  return null;
}

// Reservations include every in-flight request. EOS responses stop dispatch
// immediately; the saved EOS identities and run ending are committed atomically.
export class RunControl {
  constructor(recording) {
    this.recording = recording;
    this.reserved = 0;
    this.completions = recording.data.run.eos_node_ids.length;
  }
  reserve(maxTokens) {
    const run = this.recording.data.run;
    if (run.stop_reason || this.completions >= run.completion_target) return 0;
    const count = Math.max(
      0,
      Math.min(
        maxTokens,
        run.token_budget - run.generated_tokens - this.reserved,
      ),
    );
    this.reserved += count;
    return count;
  }
  settle(count, result) {
    this.reserved -= count;
    if (result?.finish_reason === "stop") this.completions++;
  }
}
