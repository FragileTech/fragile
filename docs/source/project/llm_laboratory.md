(sec-llm-laboratory)=
# LLM Lab

:::{div} feynman-prose
LLM Lab grows several token sequences from the same prompt. Each walker carries
one complete generated prefix. A step extends those prefixes, scores the new
tokens, and embeds the resulting text. Wave or Graph then uses reward and
embedding distance to choose which branches supply the next continuations.
Cloning carries the donor's complete prefix, score, and ancestry together.
The next request supplies that selected prefix as the assistant text to
continue, including every earlier chunk in the donor's sequence.

Open the `/llm/` application alongside {doc}`control_laboratory` and
{doc}`optimization_laboratory`. **Generation** contains the model and search
settings, run controls, and leading traces. **Analysis** lets you follow branches,
replay recorded iterations, and inspect the decisions behind cloning. **Benchmark**
compares full answers, distributions, and computational work across sampling
methods. **Evaluation**, the fourth tab, contains optional model grading, judge
score charts, the trace browser, and pinned answer comparisons. Benchmark and
Evaluation share the selected source, trials, and saved report, including a
standalone current Fractal recording. Inspect the text, token probabilities,
and clone history before
interpreting the search's best score. A high likelihood measures the selected
model's preference for the generated text; it does not establish that the answer
is correct or useful.
:::

(sec-llm-laboratory-start)=
## Start a small run

:::{div} feynman-prose
From the repository root, launch the lab with the command below. The default
port is 8080; open [http://localhost:8080/llm/](http://localhost:8080/llm/).
:::

```bash
make llm-lab
```

:::{div} feynman-prose
Enter an OpenRouter API key and a prompt. The key stays in browser memory for
the session and is excluded from recordings. Generation and embedding requests
go to OpenRouter; the prompt and whichever text you select for embeddings are
sent with those requests. Optional Mean XED scoring also sends the question and
generated answers to Together AI, using a separate session key. Negative mean
Xent and beam-style scoring need no additional model requests.

The initial generation model is `qwen/qwen3.5-35b-a3b`, with Alibaba (`alibaba`)
preferred during route discovery. The embedding model is
`openai/text-embedding-3-small`. Model identifiers are editable; DeepSeek remains
selectable. Every selected route must supply token log probabilities and pass
the continuation checks at the start of each run. The lab then pins the provider
and requests the required parameters explicitly. A model appearing in a catalog
does not establish that its endpoint implements the needed continuation
behavior. See OpenRouter's
[assistant-prefill reference](https://openrouter.ai/docs/api_reference/overview#assistant-prefill).

Before generating the population, the lab tries up to four endpoints that
advertise token log probabilities. A candidate must pass an exact-copy prose
probe at temperature zero, using three requests of up to eight tokens each.
The first must return the beginning of the expected prose. The second must
match the next part exactly when given the first chunk as its assistant prefix.
The third receives both preceding chunks and must match the next text after
that complete prefix. The first two responses must end at their token limits;
a model stop is never continued. Restarting, repeating the prefix, or changing
the expected text rejects the route for that run. Rejected routes and their
reasons are recorded.

The first successful route is pinned for the run, with no provider fallback
during population generation. One embedding probe follows the successful
generation checks. These checks use your API key and consume provider usage;
their probe data and usage are recorded separately from the search population.
Probe metadata stores the expected text and the check identifier
`prose_suffix_v1`, so a saved check can be inspected directly.

Earlier live Wave and Graph runs on 2026-09-09 used DeepSeek Flash through
StreamLake (`streamlake/fp8`) and `openai/text-embedding-3-small`. Their checks
validated recording structure, clone ancestry, and `.fgllm` export/import,
but the original continuation probe failed to detect restarted answers.
Inspecting saved chunks showed correct donor prefixes followed by newly
returned text that began the answer again. Those runs therefore did not
establish correct continuation. Supplying DeepSeek's explicit prefix flag
also proved insufficient: StreamLake received it but still intermittently
restarted the probe. A later list probe gave false confidence on Qwen through
Venice, which continued numbered lines but restarted prose. The three-chunk
prose probe supersedes both weaker checks.

On 2026-09-09, Qwen through Alibaba passed the three-chunk prose check twice
with its required partial-message flag. Live Wave and Graph runs on the
sky-color prompt used 4 initial walkers, 32 tokens per chunk, a 256-token
sequence cap, a 24-iteration limit, concurrency 2, a Graph population cap of
32, temperature 0.6, and seed 7. Wave saved four EOS completions after five
iterations and four clone events, using 562 of its 1,024 generated-token
allowance. Graph ended with `token_budget` after 19 committed iterations and
29 clone events, using exactly 1,024 tokens. It had no EOS completions; its
deepest trace contained 96 tokens, and unfinished traces remained partial.
Graph therefore exercised the budget ending without claiming completion-target
success or marking branches capped prematurely.

Inspection of both runs confirmed that outgoing continuation requests
contained complete donor prefixes and returned chunks continued the inherited
prose coherently. Wave's completed answers and Graph's partial traces were
preserved. This is evidence for those tested requests; each new run still
revalidates the selected route.

Start with **Wave** and press **Step**. Inspect the generated tokens, then step
again to check continuation from the same prefix. Press **Run** to advance
repeatedly. **Pause** finishes the current iteration before pausing. **Stop**
cancels pending work and retains the last completed population. **Reset** starts
a fresh run with the selected configuration. Changes to the prompt, models,
scoring, or search settings require a new run. Once the run reaches one of its
normal stopping conditions, **Run** and **Step** stay disabled until **Reset**.
:::

:::{div} feynman-added
| Control | Initial value | What it controls |
|---|---|---|
| Algorithm | Wave | Fixed population or Graph's growing search |
| Walkers | 8 | Initial population size and saved EOS completion target |
| Tokens per step | 32 | Maximum new tokens requested per continuation |
| Sequence cap | 256 | Maximum generated tokens in a branch |
| Concurrency | 4 | Simultaneous generation requests |
| Iteration limit | 32 | Maximum search iterations |
| Temperature | 1 | Generation sampling parameter |
| Reward coefficient | 1 | Reward contribution to cloning fitness |
| Distance coefficient | 1 | Diversity contribution to cloning fitness |
| Graph population cap | 256 | Maximum Graph population |
| Objective | Negative mean Xent | Full-sequence score used to guide cloning and rank answers |
| Beam α | 0.6 | Length exponent, adjustable from 0 to 2 when beam scoring is selected |
| XED direction | Maximize | Whether larger or smaller mean XED guides selection |
| XED scoring model | `Qwen/Qwen3.5-9B` | Together model used for both supplied-text evaluations |
| Embedding input | Generated sequence only | Text represented by the observation vector |
| Distance | Cosine | Observation comparison used for diversity |
:::

:::{div} feynman-prose
The seed controls the lab's own sampling choices. Remote generation need not
reproduce identical tokens from the same seed, even when a provider accepts a
seed parameter. Keep the recorded responses when you need to inspect exactly
what happened.

A request can return fewer tokens than the chunk limit. A model stop records
an end-of-sequence (EOS) completion; reaching the sequence cap marks a branch
capped. Reaching only the
per-step token limit leaves the branch available for continuation. The trace
records the actual generated-token count and finish reason separately.
:::

(sec-llm-laboratory-stopping)=
### Keep the answer when a branch finishes

:::{div} feynman-prose
Think of a walker slot as a place to do more work, and its saved trace as the
work already done. When the provider returns `finish_reason: "stop"`, that
branch has finished. The lab keeps its text, token probabilities, score, and
ancestry in an immutable EOS node. It never asks the model to continue that
node. Cloning can then assign the available slot to an unfinished branch,
carrying that branch's actual prefix into the next request. This reuse does
not erase the answer that freed the slot.

The target is **N saved EOS completions**, where **N** is the initial walker
count. These accumulate over the run; the current walkers need not all finish
in the same iteration. Each distinct saved EOS node counts once. Replaying a
cached transition or copying a finished trace does not add a completion, but
two independent generations of identical text count separately. A stop with
zero new tokens still records a completion. If its complete answer is empty,
it counts toward the target but is ineligible for **Best**.

Reaching the target stops queued requests from starting. Requests already in
flight finish and are preserved, so the saved total can exceed N. The progress
display shows saved EOS completions against the target. A capped branch also
keeps its trace, but it does not count as an EOS completion. If no unfinished
branches remain, the run ends with the evidence it has; it does not restart
the prompt to manufacture more candidates.
:::

:::{div} feynman-prose
There is also a shared allowance of **N × sequence cap** newly generated
scored tokens. Suppose you start eight walkers with a 256-token sequence cap.
The run may generate at most 2,048 such tokens, even if Graph grows beyond
eight slots. A ten-token prefix copied into four slots still costs ten
generated tokens. Inherited prefixes and cached transition replays consume
no additional allowance. Probes and embeddings are outside this counter;
tokens generated on a branch that selection later discards still count.

Concurrent requests reserve their token allowances before starting. Each
request fits both the branch's remaining sequence cap and the unreserved run
allowance. A shorter response releases unused reservations. The final request
may therefore ask for fewer tokens than the configured chunk size. Reaching
the shared allowance preserves unfinished traces as partial; it does not turn
them into EOS completions or claim that they reached their own sequence caps.

After committing an iteration, the lab checks the reasons below in order.
The first applicable reason becomes the recorded ending, shown alongside
generated-token usage and EOS progress. Only `eos_target` means the requested
number of completed answers was obtained. The other endings still preserve
the run for inspection and export. Reset before starting another run.
:::

:::{div} feynman-added
| Priority | Recorded reason | What ended the run |
|---|---|---|
| 1 | `eos_target` | At least N distinct EOS nodes were saved |
| 2 | `token_budget` | The N × sequence-cap generated-token allowance was used |
| 3 | `no_active_branches` | No unfinished branches remain available for continuation |
| 4 | `iteration_limit` | The configured number of search iterations was reached |
:::

(sec-llm-laboratory-scoring)=
## Read likelihood with its context

:::{div} feynman-prose
Every generation request includes the original prompt. After the first chunk,
the complete generated prefix is supplied as the assistant prefill. After
cloning, this is the donor's full sequence, including its inherited text.
The provider must also recognize that this final assistant message is
unfinished. For Qwen, the lab sets `partial: true` on that message, as required
by Alibaba's [partial-mode API](https://www.alibabacloud.com/help/en/model-studio/partial-mode).
For `deepseek/*` models, it sets `prefix: true`. These flags construct explicit
continuation requests; the observed DeepSeek restarts show why sending a flag
alone is insufficient evidence that a route honors it. There is no extra user
instruction between chunks.

A new token's probability uses both the prompt and the preceding generated
text. Only newly returned tokens add to the recorded score; sending the
prefix again does not score it again. The prose-suffix preflight checks that
the selected route honors continuation on its test text. Inspect the actual
chunk boundaries as well: correct ancestry alone does not establish that the
provider continued the answer as requested.
:::

:::{prf:definition} LLM Lab objectives
:label: def-llm-laboratory-objectives

For a generated sequence of $n>0$ scored tokens, let

$$
L_n=\sum_{t=1}^{n}\log p(x_t\mid\text{prompt},x_{<t}).
$$

**Beam-style length normalization**, the default for new generation runs, has
score $S=L_n/n^\alpha$, with $0\leq\alpha\leq2$ and default $\alpha=0.6$.
At $\alpha=0$ it equals total log likelihood; at $\alpha=1$ it equals
**negative mean Xent**, $S=L_n/n$, measured in nats per generated token.
Legacy **Total likelihood** uses $S=L_n$, in nats.

For **Mean XED**, let a fixed scoring model $J$ tokenize the complete answer
into $m>0$ answer tokens $y_1,\ldots,y_m$. Let $c(q)$ be the recorded assistant
scoring format containing the original question $q$, and let $c(\varnothing)$
be the identical format with an empty question. Define

$$
L_{J,\mathrm{conditional}}
 =\sum_{t=1}^{m}\log p_J(y_t\mid c(q),y_{<t}),\qquad
L_{J,\mathrm{baseline}}
 =\sum_{t=1}^{m}\log p_J(y_t\mid c(\varnothing),y_{<t}),
$$

$$
S_{\mathrm{XED}}
 =\frac{L_{J,\mathrm{conditional}}-L_{J,\mathrm{baseline}}}{m}.
$$

Mean XED is measured in nats per scorer token. Both evaluations must use
identical answer token IDs. Prompt and formatting tokens do not enter either
sum, and scorer tokens need not agree with generator tokens.

The engine maximizes a cumulative **utility** $U$. For negative mean Xent,
beam scoring, and total likelihood, $U=S$. For XED, $U=S_{\mathrm{XED}}$ when
maximizing and $U=-S_{\mathrm{XED}}$ when minimizing. An environment transition
earns $U(\text{child})-U(\text{parent})$. The empty sequence has internal utility
zero and is excluded from Best-answer selection.
:::

:::{div} feynman-prose
Think of Xent as the model's surprise bill for the words it sees. A familiar
word in a familiar context adds a small charge; an unlikely word adds a larger
one. Every extra token adds another nonnegative charge, so comparing whole
bills automatically disadvantages longer answers. Negative mean Xent compares
the charge per token and changes its sign so that larger values are preferred.
The beam exponent lets you adjust how strongly length enters this comparison.
It is a decoding preference, not a test of factual accuracy.

Now XED asks a different question: how much did supplying this question change
the scorer's surprise about this answer? If an answer has conditional log
likelihood −12 and baseline log likelihood −24 across six scorer tokens, its
mean XED is 2 nats per token. Maximizing favors that increase in predictability;
minimizing favors the reverse. The displayed XED remains 2 in either mode.
Only the internal utility changes sign. The baseline still contains the
recorded assistant format, so this is a comparison with an empty question in
that format, not an unconditional probability over arbitrary text.

Rewards compare complete prefixes. Adding the means of separate chunks would
give chunk size an unintended role in selection. Cloning therefore carries
the donor's complete text and cumulative utility; the next transition compares
its child's utility with that donor's value.
:::

:::{div} feynman-prose
**Best** selects among nonempty finished or capped traces using the configured
utility. Until one exists, the displayed best partial trace is selected from
the greatest reached token depth and is labelled partial. Empty roots and
empty completed answers are ineligible. Wave's historical elite reinjection
is disabled for this lab. Terminal branches are also ineligible for native
best-walker protection, so Graph can recycle their slots. **Best** still
selects from the immutable recording archive, which keeps previously observed
candidates after recycling.

Generation scores use the provider's returned token log probabilities under
the chosen model and request settings. Responses must contain finite, valid
probabilities aligned with the generated text. Missing or placeholder values
cause an error. The lab never substitutes zero for a missing probability and
never invents an unavailable end-of-sequence probability. Termination remains
a separate recorded fact.
:::

(sec-llm-laboratory-xed)=
### Score supplied answers with Together

:::{div} feynman-prose
Choose **Mean XED** to reveal its direction, scoring model, and Together API-key
controls. Both XED evaluations use the same model and a fixed non-thinking
assistant format. The initial scorer is `Qwen/Qwen3.5-9B`. Together's
[completions API](https://docs.together.ai/reference/completions) documents
`echo` and `logprobs` for supplied-text scoring, but that interface alone does
not establish that a particular hosted model supports the required fields.
The lab probes the selected scorer before starting XED generation. It checks
answer boundaries, Unicode text, coverage of supplied-token probabilities,
and finite values. An unsupported scorer stops preparation; it does not cause
a silent switch to another objective. Live support requires a successful
model-specific probe; the model name is not evidence of a verified route.
:::

:::{div} feynman-prose
Live checks on 2026-09-09 verified supplied-text scoring for Together's
`Qwen/Qwen3.5-9B`. The endpoint rejected `max_tokens: 0`; the one-token fallback
worked. Three exact supplied-text probes, including Unicode, passed, and both
XED terms were scored successfully for a short answer. This establishes the
scoring behavior of those requests. It does not establish answer quality or
support for another model.

The endpoint returned echoed probabilities under `choices`, omitted token
IDs, and displayed some split Unicode tokens with replacement characters.
Those displayed fragments cannot reconstruct the original token bytes. For
this model, the lab therefore uses a bundled Qwen tokenizer to reconstruct
exact IDs and bytes locally, checking every prompt token and the prompt-token
count against the response before accepting its probabilities. The tokenizer
runtime is `@huggingface/tokenizers` version `0.2.0`; the Qwen assets are pinned
to revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Approximately 12.8 MB of
tokenizer data loads only when an XED response needs this reconstruction. This
local work makes no additional model requests. A mismatch fails scoring.
:::

:::{div} feynman-prose
Every distinct new nonempty prefix is scored before it enters the engine,
so XED influences which walkers clone as well as which answer leads the final
ranking. This requires two supplied-text evaluations, including the complete
answer each time. Scoring asks for zero new tokens when supported; otherwise
it requests one and ignores that output when computing XED. Its usage is still
recorded. The lab limits scoring concurrency and caches results by model,
format, question, and complete answer. Unchanged text and cloned prefixes
reuse cached scores. Independent EOS nodes remain distinct completions even
when their identical text shares a scoring result.

Scoring requests, latency, provider usage, and errors are separate from
generation and embedding measurements. They do not consume the search's
newly generated-token allowance, but they do incur provider work and cost.
If scoring fails after generation succeeds, the accepted generation and any
available scoring evidence survive alongside the last committed population.
The run reports an error rather than admitting an unscored prefix into the
search. XED is optional; the default beam objective and optional negative mean
Xent objective use the generation probabilities already returned.
:::

(sec-llm-laboratory-embeddings)=
## Choose what distance sees

:::{div} feynman-prose
**Generated sequence only** embeds the complete generated prefix. **Prompt plus
generated sequence** embeds the original prompt followed by two newlines and
that prefix. This switch changes the observation used for diversity. Both
choices still condition generation and token probabilities on the original
prompt.

Cosine distance compares vector directions, with values from 0 to 2 after
numerical clamping. L2 compares their Euclidean separation and can also respond
to vector magnitudes. Embedding similarity is a property of the selected model;
it does not certify that two answers have the same meaning or factual content.
Use the text inspector to check what a distance difference represents in your
run.

The shared core defaults to **L2** for existing consumers. LLM Lab explicitly
selects **Cosine**, and its distance selector also permits L2. This setting is
shared by Wave and Graph's observation-diversity calculations; it does not
redefine the Euclidean gas's physical geometry. Saved configurations retain
the selected metric.

The adapter batches repeated embedding inputs, deduplicates exact matches, and
caches results by model and input text. It checks vector dimensions and finite,
nonzero values. Empty initial prefixes use a sentinel observation until text
exists. Context-limit checks use conservative UTF-8 byte bounds rather than a
provider tokenizer, so they can reject some text that the model would accept.
Oversized inputs produce an error instead of silent truncation, so the distance
never quietly switches to representing only part of a branch. See
OpenRouter's [embedding API](https://openrouter.ai/docs/api_reference/embeddings)
for the request and response format.
:::

(sec-llm-laboratory-traces)=
## Inspect and save the evidence

:::{div} feynman-prose
Select a trace in **Generation**, or a node in **Analysis**, to read its complete
text and token details. Compare the selected objective, total and mean log
likelihood, token count, and termination status. XED displays its raw score and
selected direction; internal utility is labelled separately. The population table describes the current
search; the archive also retains branches that were discarded. A reused walker
slot is not a new identity for an old branch: immutable generation nodes and
clone events preserve lineage independently of slots.

The token inspector preserves the complete Unicode text. A provider token can
contain only part of a UTF-8 character, so a colored text span need not correspond
to one token. Use the token details to inspect exact token records, bytes, and
log probabilities; do not count visible characters to infer the token count.
The full-sequence score counts each recorded token once even when several token
records contribute to one displayed character.

Use comparison to keep two selected traces side by side. Read their shared
prefix, the point of divergence, and the remaining text alongside their lengths
and scores. A difference in total likelihood includes the effect of different
lengths. The comparison does not rescore text or ask the provider to evaluate
one branch under the other branch's context.

Recordings retain generated token text and bytes, log probabilities,
embeddings, parents, companion choices, cloning decisions, population
snapshots, model/provider settings, request identifiers, usage, timing, and
errors. XED recordings additionally retain scorer identity, format version,
conditional and baseline answer-token probabilities, scorer-token count,
direction, and scoring usage. Those saved measurements reproduce XED offline.
These records make it possible to follow both the provider's output
and the algorithm's subsequent selection. Temporary provider failures use
bounded retries. Invalid responses or exhausted retries stop the run without
turning a failed request into a scored continuation.

Use **Export** to save a versioned `.fgllm` recording and **Import** to inspect
one offline. New recordings use version 3 and retain run progress through
`completion_target`, `eos_node_ids`, `token_budget`, `generated_tokens`, and
`stop_reason`. The stop reason is null until a normal stopping condition is
reached. These counters describe the entire run, including saved completions
whose walker slots have since been reused. Imported version 3 recordings show
the saved stopping reason alongside EOS and generated-token progress.

Versions 1 and 2 remain readable without an API key. They retain their original
information without applying the new run limits retroactively. Legacy `total`
and `mean` configurations keep their original rankings and rewards; an imported
configuration missing its objective retains the historical total default.
New configurations default to `objective: "beam"` with `beam_alpha: 0.6`.
Beam and XED settings are additive version 3 fields, including `beam_alpha`, `xed_direction`, and
`scoring_model`; importing does not request missing scores. Version 2 adds
measured decision data from the cloning stage; version 1 shows absent decision
metrics as **Not recorded**. Importing replays recorded data and does not resume
paid generation. Both provider API keys are excluded, but the prompt and generated text
remain part of the recording. The 64 MiB recording limit bounds stored history;
the run stops cleanly before exceeding it. Export a useful result before
resetting the session. The Analysis PNG export saves an image of the view;
retain the `.fgllm` file as well when you need the underlying data.
:::

(sec-llm-laboratory-benchmarks)=
## Compare sampling in Benchmark

:::{div} feynman-prose
Suppose branching finds an interesting answer. How much of that result came
from sharing promising prefixes, and how much came from simply asking the
model several times? The benchmark records both experiments and puts their
answers side by side. Each trial runs the selected **Wave** or **Graph** search,
the selected independent baseline or baselines, and one answer at temperature
zero. Every method receives the same prompt, generation model, pinned provider,
continuation chunk size, and sequence cap. Independent sampling uses the
configured temperature. Its trajectories continue their own prefixes without
cloning, donor selection, or sharing sampled responses, even when two prefixes
happen to contain identical text.

The comparison mode determines what is held equal. **Same population** starts
one independent answer per initial Fractal walker and continues each until
model completion or the sequence cap. **Same generated-token budget** first
measures the tokens actually generated by Fractal, then spends that allowance
on independent trajectories. Completed independent trajectories are replaced
with fresh ones while budget remains. **Both** collects both baselines using
the same Fractal run and temperature-zero answer for that trial.

The Fractal method uses the same EOS target, token reservations, and stopping
rules as interactive generation. N is fixed by the initial walker count, even
when Graph grows. All four normal stopping reasons finish the benchmark method
with a completed status, while only `eos_target` establishes completion-target
success. Read the saved reason to distinguish obtaining N EOS answers from
ending at a budget, branch-exhaustion, or iteration limit. Errors and user
cancellation retain their separate statuses, accumulated run progress, and
preserved partial work.
:::

:::{div} feynman-prose
Keep the token counter in mind. If four walkers inherit the same ten-token
prefix, cloning has copied access to that prefix; it has not generated forty
new tokens. The matching budget counts each realized continuation once. It
excludes inherited prefixes, cached transition replays, preflight probes,
embeddings, XED scoring, and retries that produced no accepted output.
Concurrent requests
reserve their allowances before starting, and the final request can use a
shorter chunk. A trajectory cut off by the shared budget remains partial in
the saved data. For independent baselines, three consecutive rounds without
generated tokens end the method with an explicit incomplete status.

This budget matches generated-token counts. Requests also resend prompt and
prefix text, and Fractal uses embeddings. XED adds supplied-text scoring
requests, so equal
generated-token counts do not imply equal provider cost or elapsed time. The archive retains request
usage and timing so those quantities can be examined separately. An unavailable
provider measurement is recorded as unavailable, rather than zero. Token
likelihood still measures the model's preference under the request settings.
In particular, the temperature-zero answer's returned likelihood is not a
rescore at the independent sampler's temperature. Optional model grading is
a separate operation, with its own measurements and cost.
:::

(sec-llm-laboratory-benchmark-browser)=
### Run and save in the browser

:::{div} feynman-prose
Open the full-width **Benchmark** tab after Generation and Analysis. Select the
matching mode and the number of repetitions; the defaults are token matching
and one trial. **Start** captures the current lab configuration. Fractal runs
first within each trial, followed by the baselines in the recorded fixed
order. Methods run sequentially, while requests within a method use the
configured concurrency. Preflight validates the common provider route,
including generation at temperature zero, and stores probe usage separately.
Each trial records its algorithm seed derived from the configured seed and
trial index. These seeds do not guarantee repeatable remote model responses.
The Fractal method's status displays the same EOS progress, generated-token
allowance, and recorded stopping reason as Generation.

**Pause** waits at a completed generation boundary; **Continue** continues the
active attempt. **Stop** cancels pending work and preserves what has already
been saved. IndexedDB stores request outcomes and completed boundaries
incrementally. The saved-benchmark selector retrieves earlier data, while
**Export** and **Import** transfer a portable `.fgllmbench` archive. Export
before clearing browser storage or moving to another browser profile.

After an interruption or reload, completed methods remain complete. Explicitly
retry an unfinished method to start a new attempt of that method from its
beginning; the partial attempt remains recorded separately. Continuing a saved
benchmark revalidates its pinned route and skips completed methods. A storage
failure stops further scheduling, leaving saved data available for export.
:::

(sec-llm-laboratory-benchmark-cohorts)=
### Choose the answers being compared

:::{div} feynman-prose
The source selector starts at **Current Fractal run**. Its metrics and figures
work immediately, including for an imported recording; a baseline is not
required. Starting or importing a benchmark selects that experiment. Results
update as generation boundaries are saved. Select trials, methods, and
termination states to choose the evidence shown. Generation settings remain
in Generation, accessible from the configuration summary.

**Archived answers** and **Retained population** answer different questions.
The archive counts each generated endpoint once, including answers subsequently
discarded by Fractal. Separate independent generations remain separate
observations even if their text is identical. The retained view keeps the last
Wave population or Graph frontier slots, including clone multiplicity; Graph
interior nodes and unused slots are excluded. For independent sampling it uses
the latest endpoint of each trajectory. Both views appear together by default.

Full-answer comparisons initially include nonempty EOS and sequence-capped
traces, with their counts shown separately. Use EOS-only or partial filters
when appropriate. If the current run has only partial traces, the tab shows a
labelled partial preview. Failed and interrupted attempts are excluded from
the main comparison; the attempt inspector exposes their preserved evidence.
An absent baseline or missing measurement stays unavailable rather than
appearing as zero.
:::

(sec-llm-laboratory-benchmark-figures)=
### Read distributions and inspect their traces

:::{div} feynman-prose
Start with the summary table, then inspect the distribution behind a mean.
Reward and length plots share bins and axes across methods. The default score
is mean token log likelihood; the configured full-trace objective, total
likelihood, and NLL are also available. Full-trace reward is the utility at
the endpoint relative to the empty root. Under XED minimization, its sign is
therefore opposite to the raw XED score. It is separate from a single chunk's
objective increment.

Diversity distributions use endpoint embeddings within each method, trial,
and population. They include pairwise cosine or L2 distances, nearest-neighbor
distance, and exact-text duplication. Retained copies keep their population
weight. One temperature-zero answer has no within-run pairwise diversity;
distances from other answers to that answer are a different comparison.
Groups use all pairs up to 50,000 and a deterministic uniform pair sample
beyond that, with sampled and total counts displayed. The shared PCA picture
is only a two-dimensional projection: reported distances use the original
embeddings. Cosine mode normalizes vectors before projection. Incompatible or
missing embeddings cannot support a shared distance comparison.

Completion and compute views show EOS, capped, partial, and empty outcomes,
along with stopping reasons, generated-token work, provider usage, requests,
cost, and timing. Generation, embedding, XED scoring, and grading measurements
remain
separate. Benchmark descriptions retain the objective, beam exponent, XED
direction, and scorer identity so results with different scoring settings can
be distinguished. XED curves use saved scored-node boundaries; they do not
pretend that scorer tokens align with generator tokens. Progress curves place each answer at the boundary where it was
recorded; later answers cannot improve an earlier budget's result. Fractal
structure views show cloning, distinct retained endpoints, concentration, and
shared ancestry. With repeated trials, summaries give each trial equal
weight. Differences use paired trials, and bootstrap intervals resample whole
trials rather than cloned answers or distance pairs. A single trial has no
such uncertainty interval.

Select a histogram range, scatter point, or table row to open the corresponding
traces in **Evaluation**. Pin two answers from any methods to compare complete
text, token bytes
and probabilities, chunk boundaries, cumulative reward, termination, embedding
distance, and available grades. Recorded shared ancestry differs from text
that merely matches. Sort the trace table, copy an answer, export displayed
numeric data, or download a chart image. Keep the underlying recording or
comparison report when you need to reproduce the figure.
:::

(sec-llm-laboratory-benchmark-grading)=
### Evaluate answers with Gemini Flash or another model

:::{div} feynman-prose
Likelihood and embedding distance cannot tell you whether an explanation is
correct. Open **Evaluation** after Benchmark to request an optional judge
assessment. You can evaluate the current Fractal recording without running a
benchmark. The tab also contains the trace browser, pinned comparisons, saved
grading sessions, judge-score charts, and judge-specific summary columns.
Dataset and trial selections stay shared with Benchmark when you switch tabs.

Press **Evaluate with Gemini Flash** to prepare the route and start grading.
The default requested model is `~google/gemini-flash-latest`. The lab resolves
that alias through OpenRouter's catalog to the newest stable standard Gemini
Flash release before looking up its endpoints. Lite, image, audio/TTS, batch,
and preview variants are excluded. The concrete model and a compatible
provider are pinned for the grading session; the requested alias is recorded
separately. This matters when an alias has no endpoint listing of its own.
Older alias-based grading sessions remain available for inspection.

The OpenRouter session-key field is accessible in Evaluation; credentials are
not saved in reports or browser storage. Advanced settings let you select a
different judge independently from the generation model, edit guidance for
correctness, relevance, completeness, and clarity, and supply a reference
answer if useful. Each criterion receives a score from 0 to 4 or an unavailable
result. An overall
score from 0 to 100 is calculated with equal weights only when all four criteria
were assessed.

The judge sees the prompt, candidate answer, rubric, and optional reference.
It does not receive sampling-method names, likelihoods, temperatures, or clone
history. Candidate text is material to evaluate, not instructions to follow.
For the selected trials, identical prompt/answer pairs share one assessment
within a grading configuration. Grading uses the session key, a separately pinned compatible
endpoint, temperature zero, at most two concurrent requests, and at most 1,024
output tokens per response. Structured results are validated before becoming
grades.

The distinct-answer count, request limit, and available cost estimate appear
automatically without paid grading requests. There is no required preview
step: the evaluation button starts missing or failed assessments within the
displayed limit. Pause, continue, stop, and retry controls preserve valid
assessments. Missing credentials, unavailable routes, rejected requests,
truncated responses, and invalid grades appear beside the controls so you can
address the cause and retry; a failed assessment never becomes a valid score.
Changing the concrete judge model, provider, rubric, or reference creates a
separate grading session. Raw responses, short explanations, errors, usage,
timing, and grading coverage stay
with that session. Score distributions and grade-versus-likelihood or diversity
plots describe the judge's assessments; inspect the explanations and coverage
when interpreting them. Opening saved comparisons never starts grading requests.
:::

(sec-llm-laboratory-pairwise-ranking)=
### Rank answers with paired judge comparisons

:::{div} feynman-prose
Choose **Pairwise ranking** in Evaluation when the useful question is which
of two answers the judge prefers. **Absolute grades** remains available for
the original scoring workflow. A pairwise session freezes the current source,
selected trials, rubric, judge and provider, sampling seed, and request budget.
Its candidates are nonempty EOS or sequence-capped answers from completed
methods. Every pair addresses the same prompt; scores from separate prompts
do not form a common leaderboard.

The judge assesses correctness, relevance, completeness, clarity, and overall
quality separately. Each verdict is **A better**, **B better**, **tie**, or
**cannot assess**, with a short explanation. A tie is evidence of equivalence
under the rubric. An unassessable result supplies no preference evidence for
that category. Overall quality has its own editable rubric, initially
prioritizing correctness and relevance before completeness and clarity. It
is not an average of the four category ratings.

Each selected pair is shown in both presentation orders. The judge sees the
prompt, answers, rubric, and optional reference, without method names,
likelihoods, temperatures, or ancestry. Both verdicts remain inspectable. If
swapping the answers changes the preference, the record shows that disagreement;
it does not turn it into an explicit tie. The two orders form one paired
evidence unit, rather than two independent judges.

Start with **Run pairwise evaluation** after inspecting the allocation and
available cost estimate. The default judge uses the resolved Gemini Flash
route, temperature zero, concurrency two, and a 2,048-token response cap.
Structured responses are validated locally. A session has a default ceiling
of **600 provider POST attempts**, including reversed presentations and retries.
That ceiling never grows automatically. Pause, continue, stop, and explicit
retry preserve completed judgments; an extension records a new budget linked
to the earlier session without rewriting its conclusions.
:::

#### Spend comparisons on three different questions

:::{div} feynman-prose
Imagine drawing a line between two answers whenever the judge compares them.
A connected network lets shared opponents tell us something about a pair
that never met. A network with two disconnected islands cannot establish which
island contains the better answers. Even within one connected network, a
single thin connection can leave substantial uncertainty.

The default allocation reserves 50% of planned comparisons for fitting the
ranking, 40% for a randomized method audit, and 10% for held-out ranking
validation. With Fractal alone, the allocation is 80% ranking and 20%
validation. The ranking cohort is a reproducible sample balanced across
methods and trials, capped at 200 distinct answers and approximately one
answer per three available ranking pairs. The interface reports its coverage;
answers outside that cohort remain unranked.

Ranking first builds connected coverage, targeting four distinct opponents per
answer. It then selects batches of ten pairs: eight seek to reduce rating
uncertainty, while two explore randomly. These adaptive choices answer
“which comparison would teach the ranking model most?” They do not provide
a representative estimate of how often one sampling method beats another.

The method audit asks that second question by sampling occurrence pairs
uniformly without replacement within each trial, population, and
Fractal-versus-baseline contrast. A third set, the held-out validation pairs,
is sampled before fitting and asks whether the fitted model predicts new
verdicts. Audit and validation identities are kept out of adaptive training.
The fit is frozen before their outcomes are revealed, and those outcomes do
not refit the reported validated ranking. When a small dataset cannot support
disjoint connected training and validation sets, ranking coverage takes
priority and validation is explicitly unavailable.
:::

#### Read ratings as estimates of judge preference

:::{div} feynman-prose
The ranking model gives each answer a strength for each category. The
regularized Davidson model turns differences in strength into probabilities
of a win, tie, or loss; it also estimates a presentation-position effect.
All training results are fitted together, so the rating does not depend on
the order in which requests finished. Strengths have zero mean and are shown
on an Elo scale centered at 1500. The numerical origin is arbitrary: rating
differences carry the information.

Regularization keeps a sparsely compared or undefeated answer from acquiring
an extreme strength on scant evidence. Versioned Gaussian priors use standard
deviations 2 for answer strength, 1.5 for the log-tie parameter, and 1 for the
position effect. Each presentation has half weight; an unassessable category
contributes none. Connectivity is checked separately by category, because a
network connected for clarity might have too little assessable correctness
evidence. Disconnected components cannot support cross-component rankings.

The tables show rating intervals, rank intervals, top-five membership
probabilities, and distinct opponents. Pair predictions distinguish observed
judgments from inferred win/tie/loss probabilities. Approximate posterior
intervals come from a Laplace approximation with 2,000 seeded draws. Inspect
the convergence and prior-sensitivity diagnostics: a narrow-looking interval
from an unstable or prior-dominated fit is not strong evidence.

Held-out log loss, Brier score, and calibration compare predictions with
unseen judgments and with a constant-outcome predictor. Position disagreement,
missing assessments, and contradictory cycles are also reported. A scalar
rating cannot exactly represent every cycle of preferences. These checks
help reveal failures of that approximation; they do not certify that the
judge knows the correct answer. Statistical precision concerns this judge,
rubric, and evidence, not factual truth.
:::

#### Compare methods without counting clones as new evidence

:::{div} feynman-prose
An identical prompt and answer share one ranking candidate even if generation
produced that text several times. Those occurrences retain their original
population weights. Ten retained copies can matter greatly to the question
“what answer would a random retained slot give me?” They do not amount to
ten independent opinions about the quality of that text.

The primary method statistic is **Fractal preference share**: a Fractal win
contributes 1, an explicit tie contributes one half, and a loss contributes 0.
The interface also reports win, tie, loss, and assessment coverage separately.
Archived answers and retained populations remain separate comparisons. Each
trial is evaluated first, and trials receive equal weight in the summary.
Unassessable results remain missing; worst-case bounds show what happens if
their unknown scores are all 0 or all 1.

The randomized finite-source audit uses fixed-sample bounds for bounded
outcomes, with Bonferroni control over the reported method contrasts,
populations, and criteria. These bounds are conditional on the saved answers
and judge protocol. They do not say how a fresh generation trial will behave.
Trial-level variation and approximate trial-cluster bootstrap intervals address
that separate question when repeated trials exist; a single trial has no
generalization interval.

Start with method comparisons, then inspect the rating table, prediction
matrix, coverage, and validation diagnostics. Open an answer or pair in the
existing trace browser to read the complete text and both judge presentations.
An optional, separately budgeted second-model or blinded human audit samples
20 completed pairs. Its judgments and agreement remain separate from primary
ratings, so disagreement is visible instead of being hidden by pooling judges.
:::

(sec-llm-laboratory-benchmark-cli)=
### Run from the command line

:::{div} feynman-prose
The command-line runner uses the same benchmark generation and validation
code. Save a JSON configuration such as the following as `config.json`.
The nested `config` object accepts the lab's existing settings; omitted
settings use their usual defaults. Here two small trials each collect both
independent baselines. Supply the API key through `OPENROUTER_API_KEY` in the
shell environment; keep credentials out of configuration files and archives.
For XED, also set `TOGETHER_API_KEY`, select `"objective": "xed"`, and choose
`"xed_direction": "maximize"` or `"minimize"`. The default `scoring_model` is
`"Qwen/Qwen3.5-9B"`. Beam scoring uses `"objective": "beam"` and a
`"beam_alpha"` between 0 and 2, defaulting to 0.6. Omitting the objective in a
new run chooses this beam-style objective.
:::

```json
{
  "config": {
    "prompt": "Explain why the sky is blue in three sentences.",
    "algorithm": "wave",
    "walkers": 8,
    "chunk_tokens": 8,
    "sequence_tokens": 24,
    "iterations": 3,
    "temperature": 1,
    "seed": 7
  },
  "comparison": "both",
  "repetitions": 2
}
```

:::{div} feynman-prose
Run from the repository root. The output directory contains an append-only
journal and an atomically updated manifest. Resuming preserves completed
methods; `--retry-incomplete` explicitly authorizes a fresh attempt of an
interrupted method. Only one writer may own a benchmark at a time.
:::

```bash
npm --prefix fractal-gas-web run benchmark:llm -- run --config config.json --output outputs/my-benchmark
npm --prefix fractal-gas-web run benchmark:llm -- resume --output outputs/my-benchmark --retry-incomplete
```

(sec-llm-laboratory-benchmark-storage)=
### Process the saved records

:::{div} feynman-prose
A `.fgllmbench` file is a versioned JSONL archive: a manifest followed by
incremental records using the same schema in the browser and command-line
runner. It retains resolved settings, provider metadata, trial and attempt
identities, stop reasons, requests and retries, generated prefixes and parents,
token text and bytes, log probabilities, embeddings, Fractal population
snapshots, run progress, and clone decisions. XED scorer configuration,
measurements, and separate usage survive journaling and portable export. The journal and portable export
preserve the Fractal completion target, saved EOS node identities, token
allowance, actual generated-token count, and stopping reason. Accepted responses
are retained even if an interruption prevents them from entering a committed
population. This lets a
later reader distinguish generated evidence from the algorithm's subsequent
selection.

Processing validates the archive and writes deterministic JSONL tables for
runs, trajectories, nodes, tokens, requests, snapshots, and clone decisions.
Stable identifiers link the tables. Derived lengths, likelihoods, negative
log likelihoods, and usage totals come from the saved records; processing
requires neither an API key nor network access. Use an exported archive as
the input:
:::

```bash
npm --prefix fractal-gas-web run benchmark:llm -- process --input archive.fgllmbench --output outputs/processed
```

:::{div} feynman-prose
Export a `.fgllmcompare` report to keep an immutable source snapshot together
with grading sessions and view settings. Its source can be a standalone
Fractal recording or a benchmark archive. Browser storage retains the report,
and importing it reproduces the comparison offline. Existing `.fgllm` and
`.fgllmbench` exports remain available. The same `process` command accepts a
comparison report and exports metrics, evaluations, and grades alongside the
underlying data tables. Processing reads saved grades without requesting new
ones; paid grading is a browser action.
:::

:::{div} feynman-prose
Version 2 comparison reports additionally preserve pairwise ranking sessions:
candidate and occurrence identities, frozen sampling plans and partitions,
request attempts, both presentation orders, validated judgments, model
settings, and fit provenance. Browser storage journals ranking work
incrementally and prevents concurrent writers. A storage failure stops new
requests while leaving preserved evidence exportable. Reopening a session
does not silently repeat a completed judgment.

The same offline `process` command exports ranking candidates, pairs,
judgments, ratings, method comparisons, validation, and audit tables. Derived
ranking results are recomputed from saved judgments using the shared numerical
implementation; no judge request is needed to reproduce them. Version 1
comparison reports remain readable, as do the existing recording and benchmark
formats. Ranking request costs remain separate from absolute grading,
generation, embeddings, and scoring. Credentials are excluded from exports;
the saved prompts, answers, and judge explanations remain part of the report.
:::

```bash
npm --prefix fractal-gas-web run benchmark:llm -- process --input report.fgllmcompare --output outputs/comparison
```

:::{div} feynman-prose
Fractal attempts can also be extracted as ordinary `.fgllm` recordings for the
existing lab recording tools. New version 3 extractions keep their run progress;
version 1 and version 2 `.fgllm` compatibility is unchanged. The existing
64 MiB bound applies to each Fractal recording;
benchmark storage accumulates trials incrementally instead of imposing that
bound on the entire experiment. Archives exclude credentials but include the
prompt and generated text. Keep the archive when exact responses, partial
attempts, and provider measurements matter to a later comparison.
:::

(sec-llm-laboratory-analysis)=
## Follow branches in Analysis

:::{div} feynman-prose
Start with one recorded iteration and select a branch you recognize from the
text inspector. **Wave** displays the immutable generation tree: each node is
a saved sequence prefix, and an edge follows a real continuation from its
parent. Cloning can make several population slots share the same prefix. That
does not create additional generated text, so slot multiplicity and clone
events are displayed separately from generation ancestry.

**Graph** displays the actual recorded graph slots and their parent-slot links
at the selected iteration. Several slots may refer to the same sequence record;
they remain distinct graph slots. Changing iterations can change this graph.
Use the recorded slot identity when reading Graph's structure and the immutable
sequence identity when reading the generated text. The two answer different
questions about the same search.

Selecting a Graph decision opens the generation tree so you can inspect its
pre-cloning participants, including prefixes that have left the stored graph.
The stored Graph view and its parent-slot links remain accessible separately.
:::

(sec-llm-laboratory-analysis-navigation)=
### Move through the view and the recording

:::{div} feynman-prose
Choose the cumulative-token layout to place a prefix according to how many
tokens it contains. Choose the iteration layout to see when it entered the
recording. A long chunk and a short chunk can occur in the same iteration, so
these axes need not place nodes in the same order or at the same separation.
The other axis arranges branches for readability; separation on the canvas is
not an embedding distance.

Pan and zoom to inspect a crowded branch, use the minimap to move across the
view, and collapse subtrees to keep their surrounding ancestry readable.
Filters restrict the displayed candidates; text search helps find a phrase or
branch of interest. These operations change the view, not the saved population,
the generated text, or the cloning algorithm.

Scrub the recorded-iteration timeline to inspect an earlier population. Playback
advances through saved iterations without generating tokens. **Follow latest**
returns to the newest recorded iteration and follows new snapshots from a live
run. Looking at an older iteration does not restore an earlier engine state.
Use Generation's **Pause** or **Stop** when you want to control ongoing API work;
Analysis playback and generation are separate controls.
:::

(sec-llm-laboratory-analysis-metrics)=
### Read the metric before reading the color

:::{div} feynman-prose
The initial color metric is **Selected objective**. Read its formula and
legend before comparing node colors; XED also shows the selected optimization
direction. A beam score is not a log probability of an event and must not be
exponentiated into a purported answer probability. Likelihood,
incremental reward, distance, and fitness measure different things. In
particular, a high cloning fitness need not identify the most likely sequence:
fitness also incorporates the configured diversity contribution.
:::

:::{div} feynman-added
| Quantity | What it measures |
|---|---|
| Token log probability | The provider's natural logarithm of the probability of one generated token in its prompt and prefix context |
| Token probability | The exponential of that token's log probability; a value between 0 and 1 |
| Total log likelihood / `logp` | The sum of token log probabilities along the complete generated prefix |
| Mean token log likelihood | Total log likelihood divided by the number of generated tokens |
| Total / mean NLL | The negative of the corresponding log likelihood; smaller values are better for the selected likelihood objective |
| Selected objective | Negative mean Xent, beam-normalized likelihood, or raw mean XED under the saved configuration |
| Internal utility | The value maximized by the engine; the negative of raw XED only when XED minimization is selected |
| Chunk objective increment | The child's utility minus its parent's utility, after the actual returned chunk |
| Distance | The recorded observation distance to the sampled distance companion at the decision stage |
| Fitness | The selection value computed by the algorithm at that decision stage, after its normalization and configured coefficients |
:::

:::{div} feynman-prose
The chunk objective increment uses the tokens actually returned. In total mode
it equals that chunk's sum of log probabilities. In mean mode it is the change
in the whole prefix's mean, which can be positive even though every individual
token log probability is nonpositive. Beam scoring uses the same full-prefix
comparison with its selected exponent. XED uses the two complete scorer
evaluations and the chosen utility sign. No objective divides by the requested
chunk size or assigns reward to tokens that were never returned.

Decision inspection identifies the evaluated prefix, its distance companion,
its proposed cloning donor, and the resulting prefix. Distance companions and
cloning donors are separate samples. The measured distance, normalized values,
fitness, clone score, random draw, and protection flags belong to the recorded
pre-cloning decision. A clone proposal and an applied clone can differ when
the algorithm protects a row or rejects a donor. Read the final clone outcome
alongside those flags.

The resulting population may contain different prefixes from those evaluated
before cloning. Assigning its old fitness to a newly generated prefix would
tell the wrong story. Analysis keeps these stages separate. Missing legacy
decision metrics are marked **Not recorded**. **Unmeasured at this step** means
that the prefix has no recorded evaluation at or before the selected playback
step. Neither case substitutes zero or reconstructs a measurement from a later
population. Sequence scores and chunk increments that follow directly from
saved token records remain available in all supported recording versions. XED
requires its saved scorer measurements; older recordings do not acquire them
by importing.
:::

(sec-llm-laboratory-engine)=
## How the token environment reuses the core

:::{div} feynman-prose
Wave and Graph are the shared native Fractal implementations. The
`LlmEnvironment` adapter supplies the snapshot-based `BatchEnv` contract:
state restoration, sampled continuation actions, observations, incremental
rewards, terminal flags, and actual transition durations. Here duration means
new generated tokens. A state references an immutable sequence record, so
cloning restores the complete prefix and its score history together. JavaScript
calculates the selected cumulative utility and passes it explicitly through
the transition bridge and native snapshots. Native rewards use the difference
between child and source utility. Undispatched rows keep their source utility,
zero reward, and zero generated-token duration.

OpenRouter requests run in the browser worker. The standalone WebAssembly
build uses Asyncify to wait for asynchronous generation, embedding, and optional
XED scoring without copying the algorithms into JavaScript. Only one native engine operation can
be pending at a time; network concurrency occurs inside that operation.
Embedding-model discovery uses OpenRouter's public
`/api/v1/models?output_modalities=embeddings` catalog because the separate
`/api/v1/embeddings/models` route does not permit browser CORS access.

Fresh action identifiers request independent continuations. Once a transition
has been realized, its source state, action, duration, and configuration identify
the saved result, allowing replay without another API call. This is the
environment contract needed by future FMC and Jump Wave integration; the
first lab exposes Wave and Graph. A live acceptance check must verify actual
log probabilities and prefill continuation over at least two chunks on the
selected route before treating that route as supported.
:::

:::{div} feynman-prose
Run the focused native and adapter checks from the repository root with the
command below. Automated checks exercise the environment and recorded response
handling; live provider compatibility still requires a session key and actual
continuation responses.
:::

```bash
make llm-test
```
