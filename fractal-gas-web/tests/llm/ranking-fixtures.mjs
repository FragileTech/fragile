import { DEFAULTS } from "../../web/llm/config.js";
import { RANKING_CRITERIA } from "../../web/llm/ranking-math.js";
import { DEFAULT_RUBRIC } from "../../web/llm/grading.js";
import { DEFAULT_OVERALL } from "../../web/llm/ranking-data.js";
export function rankingSource(count = 12) {
  const root = {
    id: 0,
    parent: null,
    text: "",
    tokens: 0,
    logp: 0,
    status: 0,
    embedding: [],
    token_data: [],
  };
  const nodes = [
    root,
    ...Array.from({ length: count }, (_, i) => {
      const text = `Answer ${String(i + 1).padStart(3, "0")} 🐈`;
      const token_data = Array.from(text).map((text) => ({
        text,
        bytes: [...new TextEncoder().encode(text)],
        logprob: -0.1,
      }));
      return {
        id: i + 1,
        parent: 0,
        text,
        tokens: token_data.length,
        logp: -0.1 * token_data.length,
        status: i % 2 ? 1 : 2,
        embedding: [i, 1],
        token_data,
        duration: token_data.length,
        actual_tokens: token_data.length,
        action: i,
        reward: -0.1,
      };
    }),
  ];
  for (const n of nodes.slice(1))
    n.finish_reason = n.status === 1 ? "stop" : "length";
  return {
    kind: "recording",
    record: {
      format: "fgllm",
      version: 1,
      engine: "fgllm-1",
      config: {
        ...DEFAULTS,
        objective: "mean",
        walkers: count,
        chunk_tokens: nodes[1].tokens,
        sequence_tokens: nodes[1].tokens,
      },
      metadata: { dimensions: 2 },
      nodes,
      snapshots: [
        {
          iteration: 1,
          node_count: nodes.length,
          walkers: nodes.slice(1).map((n, i) => ({
            node: n.id,
            slot: i,
            parentSlot: i,
            alive: true,
            leaf: true,
            cloned: false,
            score: 1,
            fitness: 1,
          })),
        },
      ],
      requests: [],
      attempts: [],
      errors: [],
    },
  };
}
export function rankingJudge({
  failures = 0,
  invalid = false,
  delay = 0,
  wait = null,
} = {}) {
  let active = 0,
    peak = 0;
  const calls = [];
  return {
    calls,
    get peak() {
      return peak;
    },
    fetch: async (url, options = {}) => {
      if (!options.body)
        return new Response(
          JSON.stringify(
            url.endsWith("/models")
              ? { data: [{ id: "google/gemini-3.8-flash", created: 100 }] }
              : {
                  data: {
                    endpoints: [
                      {
                        tag: "google-ai-studio",
                        status: 0,
                        supported_parameters: [
                          "structured_outputs",
                          "response_format",
                          "temperature",
                          "max_tokens",
                        ],
                        pricing: { prompt: "0.000001", completion: "0.000002" },
                      },
                    ],
                  },
                },
          ),
          { status: 200 },
        );
      const body = JSON.parse(options.body);
      calls.push(body);
      active++;
      peak = Math.max(peak, active);
      await wait?.();
      if (delay) await new Promise((r) => setTimeout(r, delay));
      active--;
      if (failures-- > 0)
        return new Response(
          JSON.stringify({ error: { message: "Try again" } }),
          { status: 429 },
        );
      const user = JSON.parse(body.messages[1].content),
        verdict = Object.fromEntries(
          RANKING_CRITERIA.map((k) => [
            k,
            {
              verdict:
                user.candidate_A === user.candidate_B
                  ? "tie"
                  : user.candidate_A > user.candidate_B
                    ? "A"
                    : "B",
              explanation: "Fixture text comparison",
            },
          ]),
        );
      return new Response(
        JSON.stringify({
          id: `pair-${calls.length}`,
          model: body.model,
          provider: "Google",
          choices: [
            {
              finish_reason: "stop",
              message: {
                content: invalid ? "{invalid" : JSON.stringify(verdict),
              },
            },
          ],
          usage: { prompt_tokens: 25, completion_tokens: 100, cost: 0.001 },
        }),
        { status: 200 },
      );
    },
  };
}
export const preparedJudge = {
  profile: {
    model: "google/gemini-3.8-flash",
    provider: "google-ai-studio",
    rubric: DEFAULT_RUBRIC,
    reference: "",
    overall: DEFAULT_OVERALL,
  },
  requested_model: "~google/gemini-flash-latest",
  endpoint: { pricing: { prompt: "0.000001", completion: "0.000002" } },
};
