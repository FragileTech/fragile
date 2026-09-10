// Supplied-text scoring: neither term is taken from generation log probabilities.
export const SCORING_FORMAT = "qwen-nonthinking-empty-user-v1";
export const TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a";
let tokenizerPromise;
export async function loadScoringTokenizer() {
  if (!tokenizerPromise)
    tokenizerPromise = (async () => {
      const { Tokenizer } = await import("./vendor/tokenizer/tokenizers.mjs");
      const read = async (file) => {
        const url = new URL(`./vendor/tokenizer/${file}`, import.meta.url);
        if (url.protocol === "file:")
          return JSON.parse(
            await (await import("node:fs/promises")).readFile(url, "utf8"),
          );
        const r = await fetch(url);
        if (!r.ok)
          throw Error("Missing bundled scoring tokenizer; rebuild the LLM lab");
        return r.json();
      };
      return new Tokenizer(
        await read("tokenizer.json"),
        await read("tokenizer_config.json"),
      );
    })();
  return tokenizerPromise;
}
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });
export function scoringPrefix(question) {
  return `<|im_start|>user\n${question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}
// Qwen token strings may use the reversible GPT-2 byte alphabet.
const alphabet = [...Array(256).keys()].filter(
  (b) => (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174,
);
const chars = [...alphabet];
for (let b = 0, n = 0; b < 256; b++)
  if (!alphabet.includes(b)) {
    alphabet.push(b);
    chars.push(256 + n++);
  }
const byteMap = new Map(
  chars.map((c, i) => [String.fromCodePoint(c), alphabet[i]]),
);
function tokenBytes(tokens, expected) {
  const direct = tokens.map((t) => encoder.encode(t));
  if (tokens.join("") === expected) return direct;
  const encoded = tokens.map((t) =>
    t.startsWith("<|") && t.endsWith("|>")
      ? encoder.encode(t)
      : Uint8Array.from(
          [...t].map((c) => {
            if (!byteMap.has(c))
              throw Error("Unsupported scorer token encoding");
            return byteMap.get(c);
          }),
        ),
  );
  if (
    decoder.decode(Uint8Array.from(encoded.flatMap((b) => [...b]))) !== expected
  )
    throw Error("Scorer tokens do not reconstruct the supplied text");
  return encoded;
}
export function parsePromptScore(response, prefix, answer, tokenizer) {
  let prompt = response?.prompt;
  // Some Together routes use the OpenAI completion echo shape and omit IDs.
  // Recover IDs only from the pinned tokenizer after verifying EVERY prompt token.
  if (!prompt?.length && tokenizer) {
    const choice = response?.choices?.[0],
      p = choice?.logprobs;
    const encoded = tokenizer.encode(prefix + answer, {
      add_special_tokens: false,
    });
    const count = encoded.ids.length;
    if (
      response?.choices?.length !== 1 ||
      !choice.text?.startsWith(prefix + answer) ||
      response.usage?.prompt_tokens !== count ||
      !Array.isArray(p?.tokens) ||
      !Array.isArray(p?.token_logprobs) ||
      p.tokens.length !== p.token_logprobs.length ||
      p.tokens.length < count ||
      encoded.ids.some(
        (id, i) =>
          tokenizer.decode([id], { skip_special_tokens: false }) !==
          p.tokens[i],
      )
    )
      throw Error(
        "Echoed probabilities do not match the pinned scoring tokenizer",
      );
    prompt = [
      {
        text: prefix + answer,
        logprobs: {
          tokens: encoded.tokens,
          token_ids: encoded.ids,
          token_logprobs: p.token_logprobs.slice(0, count),
        },
      },
    ];
  }
  if (
    !Array.isArray(prompt) ||
    prompt.length !== 1 ||
    prompt[0].text !== prefix + answer
  )
    throw Error("Scorer did not echo the exact supplied prompt");
  const p = prompt[0].logprobs;
  if (
    !Array.isArray(p?.tokens) ||
    !Array.isArray(p.token_ids) ||
    !Array.isArray(p.token_logprobs) ||
    p.tokens.length !== p.token_ids.length ||
    p.tokens.length !== p.token_logprobs.length ||
    p.tokens.some((t) => typeof t !== "string")
  )
    throw Error("Scorer did not return complete prompt token probabilities");
  const spans = tokenBytes(p.tokens, prefix + answer);
  const boundary = encoder.encode(prefix).length;
  let offset = 0;
  const data = [];
  for (let i = 0; i < spans.length; i++) {
    const end = offset + spans[i].length;
    if (offset < boundary && end > boundary)
      throw Error("Scorer token crosses the answer boundary");
    if (offset >= boundary) {
      const logprob = p.token_logprobs[i],
        id = p.token_ids[i];
      if (
        !spans[i].length ||
        !Number.isSafeInteger(id) ||
        id < 0 ||
        !Number.isFinite(logprob) ||
        logprob > 0 ||
        logprob <= -9999
      )
        throw Error("Missing or invalid answer-token probability");
      data.push({ id, bytes: [...spans[i]], logprob });
    }
    offset = end;
  }
  if (!data.length) throw Error("Scorer returned no answer tokens");
  return data;
}
export function validateXed(node, config) {
  if (!node.tokens) {
    if (node.xed != null)
      throw Error("Empty sequence must not contain XED tokens");
    return;
  }
  const x = node.xed;
  if (
    !x ||
    x.model !== config.scoring_model ||
    x.format !== SCORING_FORMAT ||
    !Number.isSafeInteger(x.tokens) ||
    x.tokens < 1
  )
    throw Error("Invalid XED scoring metadata");
  for (const term of ["conditional", "baseline"]) {
    const ts = x[term];
    if (
      !Array.isArray(ts) ||
      ts.length !== x.tokens ||
      ts.some(
        (t) =>
          !Number.isSafeInteger(t.id) ||
          t.id < 0 ||
          !Number.isFinite(t.logprob) ||
          t.logprob > 0 ||
          t.logprob <= -9999 ||
          !Array.isArray(t.bytes) ||
          !t.bytes.length ||
          t.bytes.some((b) => !Number.isInteger(b) || b < 0 || b > 255),
      )
    )
      throw Error("Invalid XED token probabilities");
    if (
      decoder.decode(Uint8Array.from(ts.flatMap((t) => t.bytes))) !==
        node.text ||
      !Number.isFinite(x[term + "_logp"]) ||
      Math.abs(ts.reduce((s, t) => s + t.logprob, 0) - x[term + "_logp"]) > 1e-8
    )
      throw Error("XED score does not match recorded answer");
  }
  if (
    x.conditional.some(
      (t, i) =>
        t.id !== x.baseline[i].id ||
        JSON.stringify(t.bytes) !== JSON.stringify(x.baseline[i].bytes),
    )
  )
    throw Error("XED terms use different answer tokens");
}
export class TogetherScorer {
  #key;
  constructor(
    key,
    {
      signal,
      concurrency = 4,
      fetchImpl = globalThis.fetch.bind(globalThis),
      onRequest = () => {},
      onRequestStart = () => {},
    } = {},
  ) {
    if (!key) throw Error("Enter a Together API key to use XED");
    this.#key = key;
    this.signal = signal;
    this.fetch = fetchImpl;
    this.onRequest = onRequest;
    this.onRequestStart = onRequestStart;
    this.limit = concurrency;
    this.running = 0;
    this.queue = [];
    this.cache = new Map();
    this.maxTokens = 0;
  }
  async request(config, prefix, answer, context) {
    if (this.running >= this.limit)
      await new Promise((resolve) => this.queue.push(resolve));
    else this.running++;
    const start = Date.now(),
      logical_request_id = crypto.randomUUID();
    const detail = {
      ...context,
      logical_request_id,
      provider: "together",
      path: "scoring/completions",
    };
    let response,
      body,
      started = false;
    try {
      this.signal?.throwIfAborted();
      const request = {
        model: config.scoring_model,
        prompt: prefix + answer,
        echo: true,
        logprobs: 1,
        max_tokens: this.maxTokens,
        temperature: 1,
        top_p: 1,
        top_k: 0,
        repetition_penalty: 1,
        stream: false,
      };
      await this.onRequestStart({ ...detail, request, status: "started" });
      started = true;
      response = await this.fetch("https://api.together.ai/v1/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.#key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
        signal: this.signal
          ? AbortSignal.any([this.signal, AbortSignal.timeout(90000)])
          : AbortSignal.timeout(90000),
      });
      body = await response.json();
      await this.onRequest({
        ...detail,
        status: response.ok ? "ok" : "error",
        http_status: response.status,
        elapsed_ms: Date.now() - start,
        response: body,
        usage: body.usage,
      });
      if (!response.ok) {
        const error = new Error(
          `Together scoring failed (HTTP ${response.status}): ${String(body.error?.message ?? body.error ?? "request rejected").replaceAll(this.#key, "[redacted]")}`,
        );
        error.zeroUnsupported =
          this.maxTokens === 0 &&
          [400, 422].includes(response.status) &&
          /max_tokens|max_completion_tokens/.test(JSON.stringify(body));
        throw error;
      }
      if (body.model && body.model !== config.scoring_model)
        throw Error("Scorer returned a different model");
      const tokenizer =
        !body.prompt?.length && config.scoring_model === "Qwen/Qwen3.5-9B"
          ? await loadScoringTokenizer()
          : null;
      return parsePromptScore(body, prefix, answer, tokenizer);
    } catch (error) {
      if (!response && started)
        await this.onRequest({
          ...detail,
          status: "network_error",
          elapsed_ms: Date.now() - start,
        });
      throw error;
    } finally {
      const next = this.queue.shift();
      if (next) next();
      else this.running--;
    }
  }
  async score(config, answer, context = {}) {
    if (!answer) return null;
    const key = JSON.stringify([
      config.scoring_model,
      SCORING_FORMAT,
      config.prompt,
      answer,
    ]);
    if (!this.cache.has(key)) {
      const task = (async () => {
        // Both requests consume the same global scoring-concurrency limit.
        const results = await Promise.allSettled([
          this.request(config, scoringPrefix(config.prompt), answer, {
            ...context,
            term: "conditional",
          }),
          this.request(config, scoringPrefix(""), answer, {
            ...context,
            term: "baseline",
          }),
        ]);
        for (const r of results) if (r.status === "rejected") throw r.reason;
        const [conditional, baseline] = results.map((r) => r.value);
        const xed = {
          model: config.scoring_model,
          format: SCORING_FORMAT,
          tokenizer_revision:
            config.scoring_model === "Qwen/Qwen3.5-9B"
              ? TOKENIZER_REVISION
              : null,
          tokens: conditional.length,
          conditional,
          baseline,
          conditional_logp: conditional.reduce((s, t) => s + t.logprob, 0),
          baseline_logp: baseline.reduce((s, t) => s + t.logprob, 0),
        };
        validateXed({ tokens: 1, text: answer, xed }, config);
        return xed;
      })();
      this.cache.set(key, task);
      task.catch(() => this.cache.delete(key));
    }
    return structuredClone(await this.cache.get(key));
  }
  async prepare(config) {
    const probe = async () => {
      for (const text of ["Blue", "Café ☀ — 蓝色", "A sentence ends here."])
        await this.score(
          { ...config, prompt: "Explain why the sky is blue." },
          text,
          { phase: "scoring_probe" },
        );
    };
    try {
      await probe();
    } catch (error) {
      if (!error.zeroUnsupported) throw error;
      this.maxTokens = 1;
      this.cache.clear();
      await probe();
    }
    this.cache.clear();
    return {
      provider: "together",
      model: config.scoring_model,
      format: SCORING_FORMAT,
      tokenizer_revision:
        config.scoring_model === "Qwen/Qwen3.5-9B" ? TOKENIZER_REVISION : null,
      baseline: "Same assistant scaffold with an empty user question",
      max_tokens: this.maxTokens,
    };
  }
}
