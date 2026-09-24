import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, join } from "node:path";
import { parseEnv } from "node:util";
import { pairManifests, runGamePair } from "../web/llm/game-benchmark.js";
import { collectRuns } from "../web/llm/benchmark-data.js";
import { bestNode, selectedScore } from "../web/llm/config.js";
import { scoringPrefix } from "../web/llm/scoring.js";
import {
  CONTINUATION_PROBE_PROMPT,
  CONTINUATION_PROBE_TEXT,
} from "../web/llm/openrouter.js";
import { DiskBenchmarkStore, lockDirectory } from "./llm-benchmark-storage.mjs";

const MODEL = "Qwen/Qwen3.5-9B";
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

class TogetherGeneration {
  constructor(key, embeddings, runner) {
    this.key = key;
    this.embeddings = embeddings;
    this.runner = runner;
  }
  get dimensions() {
    return this.embeddings.dimensions;
  }
  get embeddingCache() {
    return this.embeddings.embeddingCache;
  }
  set embeddingCache(value) {
    this.embeddings.embeddingCache = value;
  }
  embed(model, texts) {
    return this.embeddings.embed(model, texts);
  }
  async generate(config, prefix, count, context = {}) {
    if (config.model !== MODEL || config.scoring_model !== MODEL)
      throw Error("Generation and scoring must use the same supported model");
    const logical_request_id = crypto.randomUUID();
    const request = {
      model: MODEL,
      prompt: scoringPrefix(config.prompt) + prefix,
      echo: false,
      logprobs: 1,
      max_tokens: count,
      temperature: config.temperature,
      top_p: 1,
      top_k: 0,
      repetition_penalty: 1,
      stream: false,
    };
    const detail = {
      ...context,
      logical_request_id,
      provider: "together",
      path: "generation/completions",
    };
    await this.runner.emit("request_start", { ...detail, request, status: "started" });
    let body;
    for (let attempt = 0; attempt < 6; attempt++) {
      const started = Date.now();
      let response;
      try {
        response = await fetch("https://api.together.ai/v1/completions", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.key}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(request),
          signal: AbortSignal.timeout(20000),
        });
        body = await response.json();
      } catch (error) {
        await this.runner.emit("request", {
          ...detail,
          attempt,
          status: "network_error",
          elapsed_ms: Date.now() - started,
        });
        if (attempt === 5 || !["AbortError", "TimeoutError", "TypeError"].includes(error.name))
          throw error;
        await sleep(Math.min(16000, 2000 * 2 ** attempt));
        continue;
      }
      const event = {
        ...detail,
        attempt,
        status: response.status,
        elapsed_ms: Date.now() - started,
        request_id: body.id ?? null,
        model: body.model ?? MODEL,
        usage: body.usage ?? null,
      };
      await this.runner.emit("request", event);
      this.runner.record?.append("requests", event);
      if (response.ok && !body.error) break;
      if (attempt === 5 || ![429, 500, 502, 503, 504].includes(response.status))
        throw Error(`Together generation HTTP ${response.status}: ${String(body.error?.message ?? "request rejected").replaceAll(this.key, "[redacted]")}`);
      await sleep(Math.min(16000, 2000 * 2 ** attempt));
    }
    const choice = body.choices?.[0];
    await this.runner.emit("provider_response", {
      ...detail,
      response: body,
    });
    const tokens = choice?.logprobs?.tokens?.slice();
    const probs = choice?.logprobs?.token_logprobs?.slice();
    // Together reports the terminal token's probability but omits it from text.
    // At an exact token cap, it can label that same terminal token as "length".
    let terminalToken = false;
    if (
      ["<|im_end|>", "<|endoftext|>"].includes(tokens?.at(-1)) &&
      tokens?.slice(0, -1).join("") === choice.text
    ) {
      tokens.pop();
      probs.pop();
      terminalToken = true;
    }
    if (
      body.model !== MODEL ||
      body.choices?.length !== 1 ||
      !["stop", "length"].includes(choice?.finish_reason) ||
      typeof choice.text !== "string" ||
      !Array.isArray(tokens) ||
      !Array.isArray(probs) ||
      tokens.length !== probs.length ||
      tokens.length > count ||
      tokens.join("") !== choice.text ||
      probs.some((p) => !Number.isFinite(p) || p > 0 || p <= -9999)
    )
      throw Error("Together generation returned misaligned token probabilities");
    return {
      text: choice.text,
      token_data: tokens.map((text, i) => ({ text, bytes: null, logprob: probs[i] })),
      logp: probs.reduce((sum, p) => sum + p, 0),
      finish_reason: terminalToken ? "stop" : choice.finish_reason,
      usage: body.usage ?? {},
      request_id: body.id ?? null,
      returned_model: body.model,
      returned_provider: "Together",
      logical_request_id,
    };
  }
  async prepare(config) {
    const catalog = await this.embeddings.request("models?output_modalities=embeddings");
    const embedding = catalog.data?.find((model) => model.id === config.embedding_model);
    if (!Number.isFinite(embedding?.context_length))
      throw Error("Embedding route has no context limit");
    this.embeddings.embeddingLimit = embedding.context_length;
    this.embeddingCache = new Map();
    this.embeddings.dimensions = 0;
    const probe = { ...config, prompt: CONTINUATION_PROBE_PROMPT, temperature: 0 };
    let prefix = "";
    const chunks = [];
    for (let i = 0; i < 3; i++) {
      const chunk = await this.generate(probe, prefix, 8);
      if (
        chunk.finish_reason !== "length" ||
        !chunk.text ||
        !CONTINUATION_PROBE_TEXT.slice(prefix.length).startsWith(chunk.text)
      )
        throw Error("Together failed the exact assistant-prefix continuation probe");
      chunks.push(chunk);
      prefix += chunk.text;
    }
    await this.embed(config.embedding_model, [chunks[0].text]);
    return {
      provider: { tag: "together-completions", name: "Together" },
      dimensions: this.dimensions,
      embedding_context_length: this.embeddings.embeddingLimit,
      generation_context_length: 32768,
      probe: { chunks, continuation_check: "prose_suffix_v1" },
      rejected_routes: [],
    };
  }
}

async function summarize(stores, output) {
  const summary = [];
  for (const store of stores) {
    const { runs, metadata } = collectRuns(store.manifest, await store.readEvents());
    const mode = store.manifest.settings.config.game_mode;
    for (const run of runs.values()) {
      if (run.status !== "completed") continue;
      const best = bestNode(run.nodes, run.config);
      summary.push({
        mode,
        method: run.method,
        status: run.status,
        stop_reason: run.stop_reason,
        score: best ? selectedScore(best, run.config) : null,
        best_context: best?.text ?? null,
        best_context_tokens: best?.tokens ?? null,
        finished_candidates: run.nodes.filter((node) => node.tokens > 0 && node.status === 1).length,
        generated_tokens: run.generated_tokens,
        request_count: run.requests.length,
        provider: metadata?.provider?.name,
        generator: run.config.model,
        scorer: run.config.scoring_model,
        selection_policy: "nonempty_model_stop_only",
      });
    }
  }
  await writeFile(join(output, "summary.json"), JSON.stringify(summary, null, 2) + "\n", { mode: 0o600 });
  console.log(JSON.stringify(summary, null, 2));
}

async function main() {
  const [settingsPath, outputPath, option] = process.argv.slice(2);
  if (!settingsPath || !outputPath || (option && option !== "--resume"))
    throw Error("Usage: node tools/llm-game-same-model.mjs SETTINGS_JSON OUTPUT_DIR [--resume]");
  const resume = option === "--resume";
  const output = resolve(outputPath);
  const env = { ...parseEnv(await readFile(".env", "utf8")), ...process.env };
  if (!env.OPENROUTER_API_KEY || !env.TOGETHER_API_KEY)
    throw Error("OPENROUTER_API_KEY and TOGETHER_API_KEY are required");
  const settings = JSON.parse(await readFile(resolve(settingsPath), "utf8"));
  settings.config.model = MODEL;
  settings.config.scoring_model = MODEL;
  await mkdir(output, { recursive: true });
  if (!resume)
    await writeFile(join(output, "settings.json"), JSON.stringify(settings, null, 2) + "\n", { flag: "wx", mode: 0o600 });
  await lockDirectory(output, async () => {
    const stores = resume
      ? await Promise.all(["unsurprising", "surprising"].map((mode) => DiskBenchmarkStore.open(join(output, mode), { recover: true })))
      : await Promise.all(pairManifests(settings).map((h) => DiskBenchmarkStore.create(join(output, h.settings.config.game_mode), h)));
    let nextScoreRequestAt = 0;
    const resilientScoreFetch = async (url, init) => {
      for (let attempt = 0; attempt < 3; attempt++) {
        const now = Date.now();
        const ready = Math.max(now, nextScoreRequestAt);
        nextScoreRequestAt = ready + 1000;
        if (ready > now) await sleep(ready - now);
        try {
          return await fetch(url, { ...init, signal: AbortSignal.any([init.signal, AbortSignal.timeout(20000)]) });
        } catch (error) {
          if (attempt === 2 || !["AbortError", "TimeoutError", "TypeError"].includes(error.name))
            throw error;
        }
      }
    };
    await runGamePair(stores, env.OPENROUTER_API_KEY, {
      togetherKey: env.TOGETHER_API_KEY,
      retryIncomplete: resume,
      onStatus: (status) => console.log(status),
      onRunner: (runner) => {
        runner.api = new TogetherGeneration(env.TOGETHER_API_KEY, runner.api, runner);
        runner.scoringFetch = resilientScoreFetch;
      },
    });
    await summarize(stores, output);
  });
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
