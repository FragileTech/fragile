import {
  CONTINUATION_PROBE_PROMPT,
  CONTINUATION_PROBE_TEXT,
} from "../../web/llm/openrouter.js";
import { DEFAULTS } from "../../web/llm/config.js";

export function completion(text, logprob = -0.25, finish = "length") {
  const tokens = Array.from(text).map((token) => ({
    token,
    bytes: [...new TextEncoder().encode(token)],
    logprob,
  }));
  return {
    id: "test-response",
    model: "deepseek/deepseek-v4-flash",
    provider: "Test",
    choices: [
      {
        message: { content: text },
        logprobs: { content: tokens },
        finish_reason: finish,
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: tokens.length },
  };
}
export function fakeOpenRouter({ tag = "test", provider = "Test" } = {}) {
  let calls = 0,
    active = 0,
    peak = 0;
  const requests = [];
  const fetch = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    requests.push({ url, body });
    let data;
    if (url.endsWith("/endpoints"))
      data = {
        data: {
          endpoints: [
            {
              tag,
              provider_name: provider,
              status: 0,
              context_length: 100000,
              supported_parameters: [
                "logprobs",
                "max_tokens",
                "temperature",
                "reasoning",
              ],
            },
          ],
        },
      };
    else if (url.endsWith("models?output_modalities=embeddings"))
      data = {
        data: [
          {
            id: "openai/text-embedding-3-small",
            name: "Small",
            context_length: 8192,
          },
        ],
      };
    else if (url.endsWith("/models"))
      data = {
        data: [
          {
            id: DEFAULTS.model,
            name: "Default",
            supported_parameters: ["logprobs"],
          },
          {
            id: "deepseek/deepseek-v4-flash",
            name: "Flash",
            supported_parameters: ["logprobs"],
          },
        ],
      };
    else if (url.endsWith("/embeddings"))
      data = {
        data: body.input
          .map((s, index) => ({
            index,
            embedding: [
              s.length + 1,
              1 + ([...s].reduce((v, c) => v + c.charCodeAt(0), 0) % 7),
              2,
            ],
          }))
          .reverse(),
      };
    else {
      const id = ++calls;
      active++;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, id % 3));
      active--;
      const letter = String.fromCharCode(97 + (id % 26));
      data = completion(letter.repeat(body.max_tokens), -0.05 * (1 + (id % 7)));
      if (body.messages[0].content === CONTINUATION_PROBE_PROMPT) {
        if (
          body.messages[1] &&
          body.model.startsWith("deepseek/") &&
          !body.messages[1].prefix
        )
          throw Error("DeepSeek continuation requires prefix: true");
        if (body.messages[1] && tag === "alibaba" && !body.messages[1].partial)
          throw Error("Alibaba continuation requires partial: true");
        const prefix = body.messages[1]?.content ?? "";
        data = completion(
          CONTINUATION_PROBE_TEXT.slice(
            prefix.length,
            prefix.length + body.max_tokens,
          ),
        );
      }
      data.model = body.model;
      data.provider = provider;
    }
    return {
      ok: true,
      status: 200,
      json: async () => data,
      headers: { get: () => null },
    };
  };
  return {
    fetch,
    requests,
    get peak() {
      return peak;
    },
  };
}
