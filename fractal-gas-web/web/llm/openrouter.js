const BASE = "https://openrouter.ai/api/v1/";
const decoder = new TextDecoder("utf-8", { fatal: true });
export const CONTINUATION_PROBE_TEXT =
  "A copper telescope rests beside the northern window while a small wooden clock ticks quietly above the door. Outside, pale clouds drift across the valley and the last train crosses the old stone bridge.";
export const CONTINUATION_PROBE_PROMPT =
  "Copy the following text exactly, with no introduction, quotation marks, or formatting:\n\n" +
  CONTINUATION_PROBE_TEXT;
export function parseCompletion(response, maxTokens) {
  const choice = response?.choices?.[0],
    content = choice?.message?.content;
  const tokens = choice?.logprobs?.content;
  if (
    response?.choices?.length !== 1 ||
    typeof content !== "string" ||
    !Array.isArray(tokens)
  )
    throw new Error(
      "Provider did not return text with token log probabilities. Select a compatible endpoint.",
    );
  if (
    choice.message.reasoning ||
    choice.message.reasoning_content ||
    choice.message.reasoning_details?.length ||
    (response.usage?.completion_tokens_details?.reasoning_tokens ?? 0) > 0
  )
    throw new Error(
      "Provider generated reasoning tokens despite reasoning being disabled.",
    );
  if (!["stop", "length"].includes(choice.finish_reason))
    throw new Error(`Unsupported finish reason: ${choice.finish_reason}`);
  if (
    tokens.length > maxTokens ||
    (!tokens.length && choice.finish_reason !== "stop")
  )
    throw new Error("Provider returned an invalid chunk length");
  const count = response.usage?.completion_tokens;
  // Some endpoints count an unreported EOS in billing. It is never assigned
  // an invented probability or added to the sequence token count.
  if (
    count != null &&
    (!Number.isInteger(count) ||
      count < tokens.length ||
      count > maxTokens ||
      count - tokens.length > (choice.finish_reason === "stop" ? 1 : 0))
  )
    throw new Error(
      `Provider token usage (${count}) does not match ${tokens.length} scored tokens (${choice.finish_reason})`,
    );
  const clean = tokens.map((t) => {
    if (
      typeof t.token !== "string" ||
      !Number.isFinite(t.logprob) ||
      t.logprob > 0 ||
      t.logprob <= -9999
    )
      throw new Error("Missing or unavailable token log probability");
    if (
      t.bytes != null &&
      (!Array.isArray(t.bytes) ||
        t.bytes.some((b) => !Number.isInteger(b) || b < 0 || b > 255))
    )
      throw new Error("Invalid token bytes");
    return { text: t.token, bytes: t.bytes ?? null, logprob: t.logprob };
  });
  let reconstructed;
  if (clean.every((t) => t.bytes !== null))
    reconstructed = decoder.decode(
      Uint8Array.from(clean.flatMap((t) => t.bytes)),
    );
  else reconstructed = clean.map((t) => t.text).join("");
  if (reconstructed !== content)
    throw new Error("Token probabilities do not align with returned text");
  return {
    text: content,
    token_data: clean,
    logp: clean.reduce((v, t) => v + t.logprob, 0),
    finish_reason: choice.finish_reason,
    usage: response.usage ?? {},
    request_id: response.id ?? null,
    returned_model: response.model ?? null,
    returned_provider: response.provider ?? null,
  };
}
export async function mapConcurrent(items, limit, fn, signal) {
  const results = new Array(items.length);
  let next = 0,
    failure;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (!failure) {
        if (signal?.aborted) {
          failure = signal.reason ?? new Error("Stopped");
          break;
        }
        const index = next++;
        if (index >= items.length) break;
        try {
          results[index] = await fn(items[index], index);
        } catch (e) {
          failure = e;
        }
      }
    }),
  );
  if (failure) throw failure;
  return results;
}
export class OpenRouter {
  #key;
  constructor(
    key,
    {
      fetchImpl = globalThis.fetch.bind(globalThis),
      signal,
      onRequest = () => {},
      onRequestStart = () => {},
      onProviderAttempt = () => {},
      onProviderResponse = () => {},
      onGenerationResponse = () => {},
      sleep,
    } = {},
  ) {
    if (!key) throw new Error("Enter your OpenRouter API key");
    this.#key = key;
    this.fetch = fetchImpl;
    this.signal = signal;
    this.onRequest = onRequest;
    this.onRequestStart = onRequestStart;
    this.onProviderAttempt = onProviderAttempt;
    this.onProviderResponse = onProviderResponse;
    this.onGenerationResponse = onGenerationResponse;
    this.sleep =
      sleep ??
      ((ms) =>
        new Promise((resolve, reject) => {
          const abort = () => {
            clearTimeout(timer);
            reject(signal.reason ?? new Error("Stopped"));
          };
          const timer = setTimeout(() => {
            signal?.removeEventListener("abort", abort);
            resolve();
          }, ms);
          if (signal?.aborted) abort();
          else signal?.addEventListener("abort", abort, { once: true });
        }));
  }
  async request(path, body, context = {}) {
    const logical_request_id =
      context.logical_request_id ?? crypto.randomUUID();
    const detail = { ...context, logical_request_id };
    await this.onRequestStart({
      ...detail,
      path,
      time: Date.now(),
      request: body ?? null,
    });
    for (let attempt = 0; attempt < 3; attempt++) {
      this.signal?.throwIfAborted();
      await this.onProviderAttempt({
        ...detail,
        path,
        attempt,
        time: Date.now(),
        request: body ?? null,
      });
      const start = Date.now();
      let response, data;
      try {
        response = await this.fetch(BASE + path, {
          method: body ? "POST" : "GET",
          // Catalog GETs are public. Avoid an unnecessary credentialed CORS
          // preflight: the embeddings catalog has no OPTIONS handler.
          headers: body
            ? {
                Authorization: `Bearer ${this.#key}`,
                "Content-Type": "application/json",
              }
            : {},
          ...(body ? { body: JSON.stringify(body) } : {}),
          signal: this.signal
            ? AbortSignal.any([this.signal, AbortSignal.timeout(90000)])
            : AbortSignal.timeout(90000),
        });
        data = await response.json();
      } catch (e) {
        await this.onRequest({
          ...detail,
          path,
          attempt,
          status: "network_error",
          elapsed_ms: Date.now() - start,
        });
        // Ambiguous network failures can already have incurred a generation charge.
        throw new Error(
          this.signal?.aborted
            ? "Stopped"
            : "OpenRouter connection failed or timed out; reset to retry.",
        );
      }
      await this.onProviderResponse({
        ...detail,
        path,
        attempt,
        status: response.status,
        response: data,
      });
      await this.onRequest({
        ...detail,
        path,
        attempt,
        status: response.status,
        elapsed_ms: Date.now() - start,
        request_id: data?.id ?? null,
        model: data?.model ?? body?.model ?? null,
        provider: data?.provider ?? null,
        usage: data?.usage ?? null,
      });
      if (response.ok && !data.error) return data;
      if (
        [429, 500, 502, 503, 504, 529].includes(response.status) &&
        attempt < 2
      ) {
        const retry = Number(response.headers?.get("retry-after"));
        await this.sleep(
          Math.min(
            10000,
            Number.isFinite(retry) && retry > 0
              ? retry * 1000
              : 500 * 2 ** attempt,
          ),
        );
        continue;
      }
      const message = String(
        data?.error?.message ?? "Request failed",
      ).replaceAll(this.#key, "[redacted]");
      throw new Error(`OpenRouter ${response.status}: ${message}`);
    }
  }
  async catalogs() {
    const [models, embeddings] = await Promise.all([
      this.request("models"),
      this.request("models?output_modalities=embeddings"),
    ]);
    return { models: models.data ?? [], embeddings: embeddings.data ?? [] };
  }
  async prepare(config, { pinnedProvider } = {}) {
    const [catalogs, providers] = await Promise.all([
      this.catalogs(),
      this.request(`models/${encodeURI(config.model)}/endpoints`),
    ]);
    const model = catalogs.models.find((m) => m.id === config.model);
    if (!model || model.reasoning?.mandatory)
      throw new Error(
        "Select a model supporting generation with reasoning disabled",
      );
    const embedding = catalogs.embeddings.find(
      (m) => m.id === config.embedding_model,
    );
    if (!embedding) throw new Error("Selected embedding model is unavailable");
    const endpoints = providers.data?.endpoints?.filter(
      (e) =>
        (!pinnedProvider ||
          (e.tag === pinnedProvider.tag &&
            e.provider_name === pinnedProvider.name)) &&
        e.status === 0 &&
        ["logprobs", "max_tokens", "temperature"].every((p) =>
          e.supported_parameters?.includes(p),
        ),
    );
    if (!endpoints?.length)
      throw new Error(
        "No available endpoint supports the required token log probabilities",
      );
    // This route passed live prose continuation and cloning checks. Preference
    // does not bypass the capability probes, including on later runs.
    if (config.model === "qwen/qwen3.5-35b-a3b")
      endpoints.sort(
        (a, b) => Number(b.tag === "alibaba") - Number(a.tag === "alibaba"),
      );
    this.embeddingLimit = embedding.context_length;
    if (!Number.isFinite(this.embeddingLimit))
      throw new Error("Provider did not supply context limits");
    this.embeddingCache = new Map();
    this.dimensions = 0;
    // Validate the complete API path before spending on a population. Store probe
    // metadata separately; probe text never becomes a search candidate.
    const probe = {
      ...config,
      prompt: CONTINUATION_PROBE_PROMPT,
      temperature: 0,
    };
    let first, second, third;
    const rejected = [];
    // Discovery is bounded. Once a route passes these checks it stays pinned:
    // population transitions never fall back to a different provider.
    for (const endpoint of endpoints.slice(0, 4)) {
      this.route = { tag: endpoint.tag, name: endpoint.provider_name };
      this.generationLimit = endpoint.context_length;
      this.reasoningSupported =
        endpoint.supported_parameters.includes("reasoning");
      try {
        if (!this.route.tag || !Number.isFinite(this.generationLimit))
          throw new Error(
            "Provider did not supply routing or context metadata",
          );
        first = await this.generate(probe, "", 8);
        if (!first.text) throw new Error("Generation probe returned no text");
        if (first.finish_reason === "stop")
          throw new Error(
            "Generation probe ended before continuation could be checked",
          );
        if (!CONTINUATION_PROBE_TEXT.startsWith(first.text))
          throw new Error("Provider failed the exact-output generation probe");
        second = await this.generate(probe, first.text, 8);
        if (
          !second.text ||
          !CONTINUATION_PROBE_TEXT.slice(first.text.length).startsWith(
            second.text,
          )
        )
          throw new Error(
            "Provider restarted or changed the answer instead of continuing the assistant prefix",
          );
        if (second.finish_reason === "stop")
          throw new Error(
            "Continuation probe ended before a second inherited chunk could be checked",
          );
        const continuedPrefix = first.text + second.text;
        third = await this.generate(probe, continuedPrefix, 8);
        if (
          !third.text ||
          !CONTINUATION_PROBE_TEXT.slice(continuedPrefix.length).startsWith(
            third.text,
          )
        )
          throw new Error(
            "Provider failed continuation from the complete inherited prefix",
          );
        break;
      } catch (error) {
        this.signal?.throwIfAborted();
        if (/OpenRouter (401|402):/.test(error.message)) throw error;
        const failure = { provider: this.route, message: error.message };
        rejected.push(failure);
        await this.onRequest({
          path: "capability_probe",
          status: "rejected",
          ...failure,
        });
        first = second = third = null;
      }
    }
    if (!first || !second || !third)
      throw new Error(
        `No compatible continuation route: ${rejected.map((r) => `${r.provider.name}: ${r.message}`).join("; ")}`,
      );
    await this.embed(config.embedding_model, [first.text]);
    return {
      provider: this.route,
      dimensions: this.dimensions,
      embedding_context_length: this.embeddingLimit,
      generation_context_length: this.generationLimit,
      probe: {
        first,
        second,
        third,
        continuation_check: "prose_suffix_v1",
        expected_text: CONTINUATION_PROBE_TEXT,
      },
      rejected_routes: rejected,
    };
  }
  async generate(config, prefix, count, context = {}) {
    context = {
      ...context,
      logical_request_id: context.logical_request_id ?? crypto.randomUUID(),
    };
    // UTF-8 bytes give a conservative bound for the supported byte-tokenized
    // text models. Never silently discard the original context.
    if (
      new TextEncoder().encode(config.prompt + prefix).length + count + 256 >
      this.generationLimit
    )
      throw new Error(
        "Prompt and continuation exceed the conservative generation context limit",
      );
    const messages = [{ role: "user", content: config.prompt }];
    if (prefix)
      messages.push({
        role: "assistant",
        content: prefix,
        // Providers need their native unfinished-turn marker as well as the
        // complete prefix. Probes still reject routes that ignore the marker.
        ...(config.model.startsWith("deepseek/") ? { prefix: true } : {}),
        ...(config.model.startsWith("qwen/") && this.route.tag === "alibaba"
          ? { partial: true }
          : {}),
      });
    const body = {
      model: config.model,
      messages,
      max_tokens: count,
      temperature: config.temperature,
      top_p: 1,
      logprobs: true,
      stream: false,
      ...(this.reasoningSupported ? { reasoning: { enabled: false } } : {}),
      provider: {
        only: [this.route.tag],
        allow_fallbacks: false,
        require_parameters: true,
      },
    };
    const response = await this.request("chat/completions", body, context);
    await this.onGenerationResponse({ ...context, count, response });
    const result = {
      ...parseCompletion(response, count),
      logical_request_id: context.logical_request_id,
    };
    if (
      result.returned_provider &&
      result.returned_provider !== this.route.name
    )
      throw new Error("OpenRouter changed the pinned generation provider");
    return result;
  }
  async embed(model, texts, context = {}) {
    const unique = [...new Set(texts)].filter(
      (t) => !this.embeddingCache.has(model + "\0" + t),
    );
    for (const text of unique)
      if (!text || new TextEncoder().encode(text).length > this.embeddingLimit)
        throw new Error(
          "Embedding input exceeds the conservative context limit or is empty",
        );
    for (let offset = 0; offset < unique.length; offset += 32) {
      const input = unique.slice(offset, offset + 32);
      const result = await this.request(
        "embeddings",
        {
          model,
          input,
          encoding_format: "float",
        },
        context,
      );
      if (!Array.isArray(result.data) || result.data.length !== input.length)
        throw new Error("Incomplete embedding batch");
      const seen = new Set();
      const pending = [];
      for (const row of result.data) {
        if (
          !Number.isInteger(row.index) ||
          row.index < 0 ||
          row.index >= input.length ||
          seen.has(row.index)
        )
          throw new Error("Invalid embedding response order");
        seen.add(row.index);
        const v = row.embedding;
        if (
          !Array.isArray(v) ||
          !v.length ||
          v.length > 65536 ||
          v.some((x) => !Number.isFinite(Math.fround(x))) ||
          !v.some((x) => Math.fround(x) !== 0)
        )
          throw new Error("Invalid embedding vector");
        if (!this.dimensions) this.dimensions = v.length;
        if (v.length !== this.dimensions)
          throw new Error("Embedding dimensions changed during the run");
        pending.push([input[row.index], v]);
      }
      for (const [text, vector] of pending)
        this.embeddingCache.set(model + "\0" + text, vector);
    }
    return texts.map((t) => this.embeddingCache.get(model + "\0" + t));
  }
}
