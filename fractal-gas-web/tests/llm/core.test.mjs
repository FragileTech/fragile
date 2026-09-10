import test from "node:test";
import assert from "node:assert/strict";
import {
  configuration,
  objective,
  bestNode,
  embeddingText,
} from "../../web/llm/config.js";
import {
  OpenRouter,
  parseCompletion,
  mapConcurrent,
} from "../../web/llm/openrouter.js";
import {
  Recording,
  importRecording,
  RECORDING_LIMIT,
} from "../../web/llm/recording.js";
import { TokenEnvironment } from "../../web/llm/environment.js";
import { NativeLlm } from "../../web/llm/native.js";
import { completion, fakeOpenRouter } from "./fixtures.mjs";

test("token parsing preserves UTF-8 and rejects missing/sentinel/misaligned data", () => {
  assert.equal(parseCompletion(completion("é🙂"), 2).text, "é🙂");
  const split = completion("🙂");
  split.choices[0].logprobs.content = [
    { token: "�", bytes: [240, 159], logprob: -1 },
    { token: "�", bytes: [153, 130], logprob: -2 },
  ];
  split.usage.completion_tokens = 2;
  assert.equal(parseCompletion(split, 2).logp, -3);
  for (const bad of [
    completion("a", -9999),
    completion("a", NaN),
    completion("abc"),
  ])
    assert.throws(() => parseCompletion(bad, 1));
  const missing = completion("a");
  missing.choices[0].logprobs = null;
  assert.throws(() => parseCompletion(missing, 1));
  const mismatch = completion("a");
  mismatch.choices[0].message.content = "b";
  assert.throws(() => parseCompletion(mismatch, 1));
});
test("configuration excludes secrets and separates conditioning from embedding inputs", () => {
  const c = configuration({ apiKey: "secret" });
  assert.equal(c.apiKey, undefined);
  assert.equal(embeddingText(c, "text"), "text");
  assert.equal(
    embeddingText({ ...c, embedding_input: "prompt" }, "text"),
    c.prompt + "\n\ntext",
  );
  assert.throws(() => configuration({ distance_metric: "wrong" }));
  assert.equal(
    bestNode(
      [
        { id: 1, tokens: 3, logp: -1, status: 0 },
        { id: 2, tokens: 4, logp: -2, status: 0 },
      ],
      "total",
    ).id,
    2,
  );
});
test("OpenRouter pins routing, conditions every request, deduplicates and reorders embeddings", async () => {
  const fake = fakeOpenRouter(),
    api = new OpenRouter("secret", { fetchImpl: fake.fetch });
  const c = configuration({ model: "deepseek/deepseek-v4-flash" });
  await api.prepare(c);
  await api.generate(c, "existing prefix", 2);
  const body = fake.requests.findLast((r) =>
    r.url.endsWith("chat/completions"),
  ).body;
  assert.deepEqual(body.messages, [
    { role: "user", content: c.prompt },
    { role: "assistant", content: "existing prefix", prefix: true },
  ]);
  assert.deepEqual(body.provider.only, ["test"]);
  assert.equal(body.provider.require_parameters, true);
  assert.equal(body.reasoning.enabled, false);
  const [a, b, again] = await api.embed(c.embedding_model, [
    "alpha",
    "beta",
    "alpha",
  ]);
  assert.deepEqual(a, again);
  assert.equal(a[0], 6);
  assert.equal(b[0], 5);
  const n = fake.requests.length;
  await api.embed(c.embedding_model, ["alpha"]);
  assert.equal(fake.requests.length, n);
  await assert.rejects(api.embed(c.embedding_model, ["x".repeat(8193)]));
});
test("retries are bounded, failures do not fabricate results, cancellation stops queued jobs", async () => {
  let attempts = 0;
  const api = new OpenRouter("secret", {
    sleep: async () => {},
    fetchImpl: async () => ({
      ok: false,
      status: 429,
      json: async () => {
        attempts++;
        return {};
      },
      headers: { get: () => null },
    }),
  });
  await assert.rejects(api.request("embeddings", {}));
  assert.equal(attempts, 3);
  const abort = new AbortController();
  let count = 0;
  await assert.rejects(
    mapConcurrent(
      [1, 2, 3, 4],
      1,
      async () => {
        count++;
        abort.abort(new Error("Stop"));
      },
      abort.signal,
    ),
  );
  assert.equal(count, 1);
});
for (const algorithm of ["wave", "graph"])
  for (const mode of ["total", "mean"])
    for (const distance of ["l2", "cosine"])
      test(`actual WASM ${algorithm}/${mode}/${distance}: cloning, objectives and recording roundtrip`, async () => {
        const fake = fakeOpenRouter({ tag: "alibaba", provider: "Alibaba" });
        const c = configuration({
          algorithm,
          objective: mode,
          distance_metric: distance,
          walkers: 8,
          chunk_tokens: 2,
          sequence_tokens: 8,
          concurrency: 3,
          max_walkers: 32,
        });
        const requestStarts = [];
        const record = new Recording(c),
          api = new OpenRouter("never-export-this", {
            fetchImpl: fake.fetch,
            onRequestStart: (r) => requestStarts.push(r),
            onRequest: (r) => record.append("requests", r),
          });
        record.metadata(await api.prepare(c));
        const env = new TokenEnvironment(c, api, record);
        const native = await NativeLlm.create(
          { ...c, dimensions: api.dimensions },
          (r) => env.transition(r),
        );
        try {
          for (let i = 0; i < 4; i++) env.commit(await native.advance());
          for (const snapshot of record.data.snapshots)
            for (const w of snapshot.walkers)
              if (w.node !== null)
                assert.ok(
                  Math.abs(
                    w.score - objective(record.data.nodes[w.node], mode),
                  ) < 1e-5,
                  `native score ${w.score} must match the complete inherited sequence objective`,
                );
          assert.ok(record.data.nodes.length > 8);
          const continuations = requestStarts.filter(
            (r) => r.path === "chat/completions" && r.source > 0,
          );
          assert.ok(continuations.length > 0);
          for (const r of continuations) {
            assert.deepEqual(
              r.request.messages,
              [
                { role: "user", content: c.prompt },
                {
                  role: "assistant",
                  content: record.data.nodes[r.source].text,
                  partial: true,
                },
              ],
              "Every request must use the entire selected donor sequence, without another chat turn",
            );
          }
          assert.ok(fake.peak <= 3);
          assert.ok(
            record.data.snapshots
              .slice(1)
              .some((s) => s.walkers.some((w) => w.cloned)),
          );
          for (const n of record.data.nodes.slice(1)) {
            const parent = record.data.nodes[n.parent];
            assert.ok(n.text.startsWith(parent.text));
            assert.equal(n.tokens, parent.tokens + n.token_data.length);
            assert.ok(
              Math.abs(
                n.logp -
                  parent.logp -
                  n.token_data.reduce((s, t) => s + t.logprob, 0),
              ) < 1e-8,
            );
            assert.equal(
              objective(n, mode),
              mode === "mean" ? n.logp / n.tokens : n.logp,
            );
          }
          const exported = record.export();
          assert.ok(!exported.includes("never-export-this"));
          assert.deepEqual(importRecording(exported).nodes, record.data.nodes);
          const corrupt = JSON.parse(exported);
          corrupt.nodes[1].token_data[0].bytes = [0];
          assert.throws(
            () => importRecording(JSON.stringify(corrupt)),
            /align/,
          );
          const last = record.data.snapshots.at(-1);
          env.recording.data.nodes.forEach((n) => {
            if (n.status) assert.ok(n.tokens <= 8);
          });
          assert.ok(
            last.walkers.every(
              (w) => w.node === null || record.data.nodes[w.node],
            ),
          );
        } finally {
          native.close();
        }
      });
test("embedding failure does not commit partial population; recorded attempts survive", async () => {
  const record = new Recording(configuration({ walkers: 2, chunk_tokens: 1 }));
  const api = {
    dimensions: 2,
    generate: async () => ({
      text: "a",
      token_data: [{ text: "a", bytes: [97], logprob: -1 }],
      logp: -1,
      finish_reason: "length",
    }),
    embed: async () => {
      throw new Error("Embedding failed");
    },
  };
  const env = new TokenEnvironment(record.data.config, api, record);
  await assert.rejects(
    env.transition([
      { source: 0, action: 1, duration: 1 },
      { source: 0, action: 2, duration: 1 },
    ]),
  );
  assert.equal(record.data.nodes.length, 1);
  assert.equal(record.data.snapshots.length, 0);
  assert.equal(record.data.attempts.length, 2);
  record.size = RECORDING_LIMIT;
  assert.throws(() => record.commit([], { walkers: [] }));
});

test("recorded actions replay exactly; fresh actions sample and empty stops stay ineligible", async () => {
  const c = configuration({ walkers: 2, chunk_tokens: 1, sequence_tokens: 3 });
  const recording = new Recording(c);
  let samples = 0;
  const api = {
    dimensions: 2,
    generate: async () => {
      samples++;
      return parseCompletion(completion("", 0, "stop"), 1);
    },
    embed: async (_model, texts) => {
      assert.deepEqual(texts, []);
      return [];
    },
  };
  const env = new TokenEnvironment(c, api, recording);
  const request = { source: 0, action: 1, duration: 1 };
  const rows = await env.transition([request, request]);
  assert.equal(samples, 1);
  assert.deepEqual(rows[0], rows[1]);
  assert.equal(rows[0].status, 1);
  assert.equal(rows[0].tokens, 0);
  assert.equal(bestNode(env.pending, "total"), null);
  assert.deepEqual(await env.transition([request]), [rows[0]]);
  assert.equal(samples, 1);
  await env.transition([{ ...request, action: 2 }]);
  assert.equal(samples, 2);
});

test("cancelling a native operation preserves the committed boundary and rejects overlapping calls", async () => {
  const c = configuration({ walkers: 2, chunk_tokens: 1, sequence_tokens: 8 });
  const fake = fakeOpenRouter(),
    abort = new AbortController(),
    record = new Recording(c);
  const api = new OpenRouter("key", {
    fetchImpl: fake.fetch,
    signal: abort.signal,
  });
  record.metadata(await api.prepare(c));
  const env = new TokenEnvironment(c, api, record);
  const native = await NativeLlm.create(
    { ...c, dimensions: api.dimensions },
    (r) => env.transition(r),
  );
  try {
    env.commit(await native.advance());
    const boundary = JSON.stringify(record.data.snapshots);
    let began;
    const started = new Promise((r) => {
      began = r;
    });
    api.generate = async () => {
      began();
      return new Promise((_resolve, reject) =>
        abort.signal.addEventListener(
          "abort",
          () => reject(abort.signal.reason),
          { once: true },
        ),
      );
    };
    const operation = native.advance();
    await started;
    await assert.rejects(native.advance(), /busy/);
    assert.throws(() => native.close(), /pending/);
    abort.abort(new Error("Stopped"));
    await assert.rejects(operation, /Stopped/);
    assert.equal(JSON.stringify(record.data.snapshots), boundary);
    assert.equal(env.pending.length, 0);
    await assert.rejects(native.advance());
  } finally {
    native.close();
  }
});

test("token counts, stop reasons and endpoint support fail explicitly", async () => {
  const wrong = completion("ab");
  wrong.usage.completion_tokens = 4;
  assert.throws(() => parseCompletion(wrong, 4), /usage/);
  const stopped = completion("a", -1, "stop");
  stopped.usage.completion_tokens = 2;
  assert.equal(parseCompletion(stopped, 2).logp, -1);
  for (const reason of ["content_filter", "tool_calls", null]) {
    wrong.choices[0].finish_reason = reason;
    assert.throws(() => parseCompletion(wrong, 4), /finish reason/);
  }
  const fake = fakeOpenRouter();
  const api = new OpenRouter("key", {
    fetchImpl: async (url, options) => {
      if (url.endsWith("/endpoints"))
        return {
          ok: true,
          status: 200,
          json: async () => ({ data: { endpoints: [] } }),
        };
      return fake.fetch(url, options);
    },
  });
  await assert.rejects(api.prepare(configuration()), /No available endpoint/);
  assert.ok(fake.requests.every((r) => !r.url.endsWith("chat/completions")));
});

test("preflight records incompatible routes and pins a validated endpoint", async () => {
  const fake = fakeOpenRouter(),
    rejected = [];
  const api = new OpenRouter("key", {
    onRequest: (r) => {
      if (r.path === "capability_probe") rejected.push(r);
    },
    fetchImpl: async (url, options) => {
      const response = await fake.fetch(url, options);
      if (url.endsWith("/endpoints")) {
        const data = await response.json();
        data.data.endpoints.unshift({
          ...data.data.endpoints[0],
          tag: "broken",
          provider_name: "Broken",
        });
        return { ...response, json: async () => data };
      }
      if (
        url.endsWith("chat/completions") &&
        JSON.parse(options.body).provider.only[0] === "broken"
      ) {
        const data = completion("a");
        data.usage.completion_tokens = 3;
        return { ...response, json: async () => data };
      }
      return response;
    },
  });
  const metadata = await api.prepare(configuration());
  assert.equal(metadata.provider.tag, "test");
  assert.equal(metadata.rejected_routes.length, 1);
  assert.equal(rejected.length, 1);
  await api.generate(configuration(), "prefix", 1);
  assert.deepEqual(fake.requests.at(-1).body.provider.only, ["test"]);
});

test("routes without reasoning support receive no reasoning parameter", async () => {
  const fake = fakeOpenRouter();
  const api = new OpenRouter("test", {
    fetchImpl: async (url, options) => {
      const response = await fake.fetch(url, options);
      if (!url.endsWith("/endpoints")) return response;
      const data = await response.json();
      data.data.endpoints.forEach((e) => {
        e.supported_parameters = e.supported_parameters.filter(
          (p) => p !== "reasoning",
        );
      });
      return { ...response, json: async () => data };
    },
  });
  const metadata = await api.prepare(configuration());
  assert.ok(metadata.probe.third.text);
  assert.ok(
    fake.requests
      .filter((r) => r.url.endsWith("chat/completions"))
      .every((r) => !Object.hasOwn(r.body, "reasoning")),
  );
});

test("the default prefers Alibaba but still verifies complete Qwen partial prefixes", async () => {
  const fake = fakeOpenRouter({ tag: "alibaba", provider: "Alibaba" });
  const api = new OpenRouter("test", {
    fetchImpl: async (url, options) => {
      const response = await fake.fetch(url, options);
      if (!url.endsWith("/endpoints")) return response;
      const data = await response.json();
      data.data.endpoints.unshift({ ...data.data.endpoints[0], tag: "other" });
      return { ...response, json: async () => data };
    },
  });
  const metadata = await api.prepare(configuration());
  assert.equal(metadata.provider.tag, "alibaba");
  assert.equal(metadata.probe.continuation_check, "prose_suffix_v1");
  const requests = fake.requests.filter((r) =>
    r.url.endsWith("chat/completions"),
  );
  assert.equal(requests.length, 3);
  assert.deepEqual(requests[2].body.messages[1], {
    role: "assistant",
    content: metadata.probe.first.text + metadata.probe.second.text,
    partial: true,
  });
});

for (const algorithm of ["wave", "graph"])
  test(`${algorithm}: cloning a multichunk donor sends its entire ancestry as an unfinished answer`, async () => {
    const c = configuration({
      algorithm,
      walkers: 8,
      chunk_tokens: 2,
      sequence_tokens: 16,
      iterations: 30,
      max_walkers: 32,
    });
    const fake = fakeOpenRouter({ tag: "alibaba", provider: "Alibaba" });
    const starts = [],
      record = new Recording(c);
    const api = new OpenRouter("test", {
      fetchImpl: fake.fetch,
      onRequestStart: (r) => starts.push(r),
    });
    record.metadata(await api.prepare(c));
    const env = new TokenEnvironment(c, api, record);
    const native = await NativeLlm.create(
      { ...c, dimensions: api.dimensions },
      (r) => env.transition(r),
    );
    try {
      while (!record.data.run.stop_reason) env.commit(await native.advance());
      const d = importRecording(record.export());
      const clones = d.snapshots
        .flatMap((s) => s.decisions)
        .filter(
          (decision) =>
            decision.cloned &&
            d.nodes[decision.donor]?.parent > 0 &&
            d.nodes[decision.result]?.parent === decision.donor,
        );
      assert.ok(
        clones.length > 0,
        "Exercise an actual clone after at least two inherited chunks",
      );
      for (const decision of clones) {
        const donor = d.nodes[decision.donor],
          child = d.nodes[decision.result];
        const request = starts.find(
          (r) => r.source === donor.id && r.action === child.action,
        );
        assert.deepEqual(request.request.messages, [
          { role: "user", content: c.prompt },
          { role: "assistant", content: donor.text, partial: true },
        ]);
        assert.equal(
          child.text,
          donor.text + child.token_data.map((t) => t.text).join(""),
        );
      }
    } finally {
      native.close();
    }
  });
