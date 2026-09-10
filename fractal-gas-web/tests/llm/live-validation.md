# LLM Lab live acceptance

Validated on 2026-09-09 with an authorized local OpenRouter key. No credentials
are stored in the recordings or this report.

## Continuation and EOS lifecycle (version 3)

The default is `qwen/qwen3.5-35b-a3b`, with the Alibaba route (`alibaba`)
preferred and revalidated on every run. Alibaba requires `partial: true` on
the final assistant message. That message contains the complete selected
sequence, including all inherited chunks after cloning.

The preflight checks three eight-token chunks of a fixed prose passage at
temperature zero. Each must match the expected next text exactly; the third
request contains the first two chunks as its prefix. No EOS response is
continued. The saved check identifier is `prose_suffix_v1`.

Browser-worker acceptance used the default sky prompt, total likelihood,
generated-only embeddings, cosine distance, four walkers, 32 tokens per step,
a 256-token sequence cap, concurrency two, 24 iterations, Graph cap 32,
temperature 0.6, seed 7 and coefficients 1. Embeddings used
`openai/text-embedding-3-small` with 1536 dimensions.

| Algorithm | Steps | Nodes | Newly scored tokens / budget | EOS / target | Clones | Maximum depth | Ending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Wave | 5 | 20 | 547 / 1024 | 4 / 4 | 6 | 143 | `eos_target` |
| Graph | 22 | 32 | 1024 / 1024 | 0 / 4 | 29 | 128 | `token_budget` |

Both runs had zero errors and continuous inherited text. The tests checked
outgoing requests against their saved source sequences, rejected requests from
terminal nodes, exercised clones from multichunk prefixes, and checked that no
chunk restarted the opening paragraph. Token probabilities stayed aligned with
the returned text. Version 3 export/import preserved ancestry and run progress.
Run and Step were disabled after each ending. Wave achieved its completion
target; Graph retained partial traces at its exact token budget.

A separate native/WASM acceptance run used the same settings: Wave reached four
EOS completions in five steps with 562 tokens and four clones. Graph stopped at
1024 tokens after 19 steps with 29 clones and maximum depth 96. These are dated
observations, not guarantees about future provider behavior.

Reproduce with a running local lab:

```sh
LLM_LAB_URL=http://127.0.0.1:8080/llm/ npm --prefix fractal-gas-web run test:llm-live
```

This opt-in command uses paid requests. It reads `OPENROUTER_API_KEY` from the
process environment or repository `.env`, enters it into the browser, and saves
credential-free recordings as `/tmp/llm-eos-live-wave.fgllm` and
`/tmp/llm-eos-live-graph.fgllm`. It is excluded from automatic test runs.

## Correction to earlier acceptance reports

Earlier version 1 and version 2 checks used DeepSeek Flash through StreamLake
and Parasail. Those runs verified recording structure, token alignment, clone
ancestry and diagnostics. They did **not** establish correct continuation:
inspection showed restarted answer chunks despite full donor prefixes being
sent. Their weak continuation probe accepted any different response, and even
continued an initial EOS response. The earlier preference for StreamLake has
been removed.

Adding DeepSeek's `prefix: true` flag was insufficient: StreamLake received
both the full prefix and the flag but still restarted answers. DeepSeek remains
selectable and must pass the new checks. A numbered-list probe also gave false
confidence on Qwen through Venice: it continued the list while restarting
ordinary prose. The current prose check and real sky-response acceptance cover
that observed failure.

Historical version 1/2 recordings remain unchanged and readable. Their absent
progress fields do not acquire version 3 limits retroactively. Shared-core tests
continue to cover seeded algorithm behavior, clone selection, trees, tensor
operations and planner regressions. Deterministic browser checks cover both
algorithms, EOS latching, playback, decisions, filters, comparisons, PNG export,
mobile layout and culling in a 10,000-node history.


## Xent, beam and mean XED scoring validation

Together `Qwen/Qwen3.5-9B` was live-tested using the local `.env` credential (never recorded). The route rejects `max_tokens: 0` and succeeds with the adapter's one-token fallback. It supplies prompt log probabilities in `choices[0].logprobs` with the full echoed text, not `prompt[0]`, and omits token IDs. The adapter matches the complete input against Qwen's pinned tokenizer revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, reconstructs split Unicode bytes, and excludes the one output token.

Three preflight strings (including accented text, a sun symbol and Chinese) passed in both contexts. A separate explanatory answer scored successfully: 15 scorer tokens, conditional log probability −42.909924621380924, baseline −38.994436236680485, mean XED −0.26103255898002925. This validates mechanics, not answer quality or support for other models. Live checks here exercised the hosted scoring adapter; generation/analysis/benchmark browser regressions use deterministic provider fixtures.
