import { GAME_FORMAT } from "../../web/llm/games.js";
export const game = {
  version: 1,
  format: GAME_FORMAT,
  id: "fixed-game",
  title: "The missing key",
  background: "A detective searches a café.",
  target: "Café ☀ — 蓝色.",
};
export function gameEcho(body) {
  const end = body.prompt.lastIndexOf("</think>\n\n") + "</think>\n\n".length;
  const prefix = body.prompt.slice(0, end),
    target = [...body.prompt.slice(end)];
  const context = prefix
    .split("Additional context:\n")[1]
    ?.split("<|im_end|>")[0];
  const lp =
    context === undefined
      ? -2
      : context.length === 0
        ? -4
        : context.length <= 2
          ? -1
          : -6;
  return {
    model: body.model,
    prompt: [
      {
        text: body.prompt,
        logprobs: {
          tokens: [prefix, ...target],
          token_ids: [1, ...target.map((c) => c.codePointAt(0) + 10)],
          token_logprobs: [null, ...target.map(() => lp)],
        },
      },
    ],
    usage: {
      prompt_tokens: target.length + 12,
      completion_tokens: body.max_tokens,
    },
  };
}
export function gameScoringFetch() {
  const calls = [];
  return {
    calls,
    fetch: async (_url, init) => {
      const body = JSON.parse(init.body);
      calls.push(body);
      return { ok: true, status: 200, json: async () => gameEcho(body) };
    },
  };
}
