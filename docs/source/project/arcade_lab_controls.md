(sec-arcade-lab-controls)=
# Arcade Lab controls, settings, and diagnostics

:::{div} feynman-prose
Arcade Lab has one game that is actually being played and, depending on the
algorithm, a crowd of candidate games under consideration. A **walker** is a
complete emulator snapshot: its memory, its reward history, and the small
piece of game-specific bookkeeping needed to continue from that exact moment.
It is a complete state, far richer than a dot wandering over a map.

This distinction is the key to the whole page. **Wave** and **Graph** display a
leading search walker. **FMC** and **Jump Wave** display the one committed game,
while their walkers search possible continuations in the background. The
planner may spend a long time examining futures while the committed game
holds its current frame. Then it chooses an action or path, plays it, and searches
again from the new emulator state. See {doc}`arcade_laboratory` for the broad
picture and {doc}`arcade_lab_environments` for game-specific state and reward
details.
:::

:::{div} feynman-added
| Question | Read this control or display |
|---|---|
| Which future is being copied? | **Virtual reward**, the cloning signal; **Cumulative reward** measures game progress. |
| What is actually on the screen? | **Best walker** for Wave/Graph; **Played game** for FMC/Jump Wave. |
| How many game copies act now? | **Walkers (N)** for Wave and the planners; Graph uses it as starting walkers and minimum leaves. |
| How large may the Graph become? | **Max walkers**, its cap on retained emulator-state nodes. |
| Did the game really run this many frames? | **Env frames**, the exact count returned by the emulator steps. |
:::

(sec-arcade-lab-algorithms)=
## Choose an algorithm

:::{div} feynman-prose
The four buttons are four ways of spending the same basic resource: complete
game continuations. The action vocabulary is discrete, and an action is held
for several emulator frames. The interesting part is what happens after the
population has produced a mixture of good, bad, diverse, and dead timelines.

**Wave** keeps a fixed population. Every walker samples an action, holds it for
its sampled `dt`, and steps. The cloning operation then copies promising
timelines onto less promising ones. This is the simplest view of the Fractal
Gas: a fixed crowd of alternative plays.

**Graph** keeps the history instead of overwriting it. A visited emulator state
becomes a node, and the active leaves are the states from which new alternatives
can be extended. Each graph update steps the leaves selected for cloning; the
tree can therefore grow while the number of active leaves remains
near the requested minimum. Each node retains a full emulator state, which is
why its memory cost is much larger than a single dot suggests.

**FMC** and **Jump Wave** use an `ArcadePlanner`. They search from the current
committed game while keeping candidate states as search snapshots. FMC chooses
the most represented first action among surviving search branches, then advances
the committed game by that one action and replans. Jump Wave can execute a common
initial path shared by the surviving branches, or the entire winning branch
when its consensus option is cleared. In both cases, the bright game view is
the committed emulator, distinct from the best search snapshot.
:::

:::{div} feynman-added
| Algorithm | Population meaning | What is committed to the played game? | Special readout |
|---|---|---|---|
| **Wave** | Fixed `N` walkers; all act each iteration. | Wave displays the leading search walker; its state stays in the search population. | Best walker screen and swarm plots. |
| **Graph** | `N` initial walkers and minimum leaves; nodes grow up to **Max walkers**. | Graph displays the leading search walker; its state stays in the search population. | `Nodes / leaves / stepped`, plus parent lines on the map. |
| **FMC** | Search population of `N` candidate continuations. | One voted first action, then replan. | Played game, search depth, played score, and played frames. |
| **Jump Wave** | Search population of `N` candidate continuations. | Shared prefix by default; full best path or a fallback under the rules below. | Same planner readouts, with the current execution mode. |
:::

:::{figure} ../../_static/arcade_lab/arcade-planner.png
:alt: Separation of a committed arcade game from candidate search walkers and selected actions.
:class: feynman-added

The planner's search population branches from a frozen committed game, then
sends the selected action or path alone back to the game being played.
:::

:::{div} feynman-prose
Here is the picture to keep in mind for a planner. Put one emulator on the
table—the committed game—and make copies of it for an experiment. The copies
may run ahead, die, discover a useful route, or disagree about the next button.
Those copies leave the emulator on the table unchanged. After the planner
selects an executable action sequence, the committed game advances.

This is why **Max reward** can rise while the displayed Sonic or Mario screen
holds steady during several search iterations. The search is making progress
in possible futures. The separate **Played frames** counter moves when the
committed game executes an action. A search result can also be good and remain
unselected: the planner applies the full algorithmic selection rule with the
population context.
:::

(sec-arcade-lab-planner-settings)=
## Planner settings

:::{div} feynman-prose
The **Planning** section appears for FMC and Jump Wave. **Search horizon** is
the normal number of search iterations before a decision can be made. An
iteration samples and executes one action per active search walker, so the
horizon is a depth in action edges; the actual time covered by an edge depends
on its sampled `dt`.

FMC uses the population at the normal horizon to vote on its next action. Jump
Wave's checked **Stop at first bifurcation** setting means “execute the exact
ancestral prefix shared by the surviving final walkers.” If they disagree at
the first edge, the shared prefix is empty, so Jump Wave keeps searching. A
single surviving branch shares its path with itself; shared-prefix agreement
means identical ancestry among the surviving branches.

**Maximum search horizon** matters only for Jump Wave while that checkbox is
checked. `0` means twice the normal horizon, capped at `4096`. An explicit value
must be at least the normal horizon. If the search reaches that limit before a
shared prefix appears, the implementation falls back to one positive-duration
action from the best available branch. If all search walkers are dead, it
likewise uses a one-action best-path fallback when a path exists. With the
checkbox cleared, Jump Wave executes
the selected best branch's full recorded path. Each selected action still needs
an execution step; the browser callback advances along a long path one action at
a time.
:::

:::{div} feynman-added
| Setting | Browser default and UI range | Applies to | Change behavior |
|---|---|---|---|
| **Search horizon** | `32`; integer `1`–`4096` | FMC and Jump Wave | Live; discards the pending search/plan and starts planning from the current committed game. |
| **Stop at first bifurcation** | Checked | Jump Wave only | Live; the checked state uses the shared prefix, and a cleared checkbox uses the full selected best path. |
| **Maximum search horizon** | `0` (automatic); integer `0`–`4096` | Jump Wave with the checkbox checked | Live; nonzero values must be at least Search horizon. |
:::

:::{note}
:class: feynman-added

Changing a planner setting preserves the current game. It invalidates the queued
search or trajectory and replans from the current committed emulator. A change
to the console, start level, algorithm, observation mode, walker count, Graph
cap, or seed does rebuild the run.
:::

(sec-arcade-lab-observations)=
## Choose RAM, RGB, Gray, or Coords

:::{div} feynman-prose
The observation is what the algorithm uses when it asks whether two walkers are
far apart. It can differ from the human view and from the reward signal.
**RAM** exposes emulator memory. **RGB** exposes screen pixels with three values
per pixel. **Gray** reduces the screen to one luminance value per pixel.
**Coords** is a hand-written compact state representation for the games where
the implementation knows useful coordinates.

Coords is a compact, hand-written state representation. It compares the pieces
of state that matter for reachability: position,
momentum, room, level, or other game variables. Mario, Sonic, and dedicated
Montezuma default to it; generic Atari defaults to RAM because each Atari ROM
has its own state layout. In generic Atari, selecting Coords deliberately
aliases RAM.

Changing the observation mode restarts the run because it changes the vector
used for distance and the state representation allocated to every walker. RGB
and Gray can be useful experiments, but their large vectors make the same
population much more expensive than Coords.
:::

:::{div} feynman-added
| Console | Default | RAM | RGB / Gray | Coords |
|---|---|---:|---|---|
| **Mario / NES** | Coords | `2048` values | `256×240×3` / `256×240` | 10 raw values: position, vertical state, level identifiers, time, velocities, sub-area, and power-up. |
| **Generic Atari / ALE** | RAM | `128` values | Dimensions reported by ALE; the common frame is `160×210` | Same `128`-value RAM vector supplies the state representation. |
| **Sonic / Genesis** | Coords | `65536` values (64 KiB work RAM) | `320×224×3` / `320×224` | 6 scaled values: position, velocities, ground speed, and zone/act progress. |
| **Montezuma / dedicated ALE** | Coords | `128` values | `160×210×3` / `160×210` | 8 raw values: pyramid-global position, room, local position, level, inventory bitmask, and lives. |
:::

:::{div} feynman-prose
The table contains an easy trap. A pixel observation is large because the
machine has many pixels; the observation itself carries pixels, while
interpretation requires a separate model or adapter. A Coords observation is
small because the adapter has selected a few RAM-derived variables. The reward
adapter still runs separately. Coords keeps score and distance as separate
signals, and RGB supplies screen pixels as the observation; learned visual
recognition requires a model that uses those pixels.

The exact tuple contents are documented in {doc}`arcade_lab_environments`.
There, the Mario identifiers are described as raw NES values, Sonic's first
five quantities are scaled by `256`, and Montezuma's global coordinates place
local room positions into the 24-room pyramid. Those details matter when
interpreting a distance coefficient: the units are representation-dependent.
:::

(sec-arcade-lab-swarm-settings)=
## Set the swarm and Graph population

:::{div} feynman-prose
The browser-facing default is **Walkers (N) = 48**, with an integer range from
`2` to `1024`. In Wave, FMC, and Jump Wave this is the number of complete game
copies in the current search population. In Graph, the same input is relabeled
**Leaves (start = min leaves)**: it supplies the starting walkers and the
minimum number of active leaves the tree tries to maintain.

Graph has another number because its history is retained. **Max walkers** is a
cap on graph nodes, including currently stepping leaves. The UI chooses a
console-specific default: Mario `4000`, generic Atari `20000`, Sonic `150`, and
dedicated Montezuma `20000`. The input range is `2`–`100000`, but the effective
cap may be lower when browser WebAssembly memory reaches its capacity for
another full emulator state. The ready message reports that effective cap.
Sonic's smaller default is intentional: Genesis state blobs are comparatively
expensive.

**Seed** defaults to `7`. It drives random action choices, frame skips, and
clone decisions. It is applied when the run is built, so changing it restarts
the run. **Elite walkers** defaults to `2`, accepts `0`–`16`, and is hidden for
Graph. In Wave, FMC, and Jump Wave, the best walkers ever found by cumulative
reward are kept in an elite buffer and re-injected on later iterations. Set it
to zero to disable that buffer. Graph exposes its best state through the
retained tree, with its elite state represented there.
:::

:::{div} feynman-added
| Setting | Default and range | Meaning | Change behavior |
|---|---|---|---|
| **Walkers (N)** / Graph **Leaves** | `48`; integer `2`–`1024` | Wave/planner population, or Graph starting walkers and minimum leaves. | Restart required. |
| **Max walkers** | Mario `4000`, Atari `20000`, Sonic `150`, Montezuma `20000`; integer `2`–`100000` | Graph's maximum retained nodes; the effective value may be memory-capped. | Restart required; Graph only. |
| **Seed** | `7`; nonnegative integer | Random actions, inclusive frame skips, and cloning choices. | Restart required. |
| **Elite walkers** | `2`; integer `0`–`16` | Best-ever cumulative-reward walkers re-injected by Wave/FMC/Jump Wave. `0` disables. | Live for those three algorithms; the next iteration uses the new count. |
:::

(sec-arcade-lab-fitness)=
## Understand fitness and visit controls

:::{div} feynman-prose
There are two meters in this experiment. The environment produces a cumulative
reward: game score for generic Atari, and shaped game progress for Mario, Sonic,
and dedicated Montezuma. The cloning operation combines that number with other
signals to build a **virtual reward**, the selection signal that says which
timeline should be copied.

In the current implementation the virtual reward is a product of population-
rescaled factors:

\[
  V = D^{a}\,R^{b}\,Q^{c}.
\]

`D` is observation distance from a randomly chosen companion, `R` is the
cumulative reward signal, and `Q` is the visit factor when visit reward is on.
The coefficients are the exponents **Distance coef** `a`, **Reward coef** `b`,
and **Visit coef** `c`; each defaults to `1`, has range `0`–`3`, and moves in
steps of `0.05`. The backend rescales each current population signal before
forming the product, so these exponents act on population-rescaled signals. Raw
pixel distances and raw score units pass through that rescaling before
exponentiation. A zero coefficient removes that factor's variation. Higher
Distance coef favors diversity, higher Reward coef favors accumulated progress,
and higher Visit coef favors the exploration signal.

The rescaling is asymmetric: a signal with zero spread contributes the neutral
factor `1`; standardized values above and below the population mean are mapped
differently. This is why a virtual reward can change when the population's
spread changes while the underlying game score stays fixed. A low-fitness walker
can be cloned onto a higher-fitness companion, and dead walkers are always
eligible to clone. Read **Virtual reward** as “what the resampler preferred,”
and **Cumulative reward** as “what the game scored.”
:::

:::{div} feynman-added
| Control | Default and range | Applicability and meaning | Change behavior |
|---|---|---|---|
| **Distance coef** | `1.00`; `0`–`3`, step `0.05` | Exponent on rescaled observation distance; higher values preserve diversity. | Live. |
| **Reward coef** | `1.00`; `0`–`3`, step `0.05` | Exponent on rescaled cumulative environment reward; higher values exploit progress. | Live. |
| **Visit coef** | `1.00`; `0`–`3`, step `0.05` | Exponent on the visit factor, when visit reward is enabled and visit controls are available. | Live. |
:::

### Visit reward, pooling, and erasure

:::{div} feynman-prose
Visit controls are specific to **Coords** on Mario, Sonic, and dedicated
Montezuma. The implementation maintains visit counts while their reward term is
inactive. That makes a useful ablation possible: set the term to **Off** while
history continues accumulating, then turn it back on with the earlier visits
preserved.

**Visit reward** is On by default for Graph and Off by default for Wave, FMC,
and Jump Wave. On means that walkers in less-visited coordinate cells receive a
larger visit factor in virtual reward. Off makes the visit factor neutral; the
counters remain intact. **Visit pooling (px)** defaults to `5`, with range
`1`–`1024`. The backend stores counts at the underlying pixel/key resolution but
sums a square `B×B` window, where `B` is the pooling value, before using the
count in fitness or the heatmap. A larger window makes novelty coarser; the
per-pixel history remains intact when this setting changes.

**Erase coef** defaults to `0.05`, with range `0`–`1` in steps of `0.01`. At
each iteration stored counts decay by this coefficient. Zero preserves stored
counts indefinitely; a larger coefficient lets old exploration become novel
again more quickly. The visit grid is part of the current run, so **Reset**
clears it.
Changing any of these three settings is live. The map's **visits** toggle is a
display choice and is also remembered by the browser; it asks the worker to
send the current blocks while the view is on.
:::

:::{div} feynman-added
| Setting | Browser default and range | What it changes |
|---|---|---|
| **Visit reward** | Graph **On**; Wave/FMC/Jump Wave **Off** | Whether visit novelty affects virtual reward. Counts continue in either state. |
| **Visit pooling (px)** | `5`; integer `1`–`1024` | Side length of the square sum-pooling window for the visit term and heatmap. |
| **Erase coef** | `0.05`; `0`–`1`, step `0.01` | Per-iteration decay of stored visit counts. |
:::

:::{figure} ../../_static/arcade_lab/graph-visits.png
:alt: Retained Graph state tree, walker leaves, and visit-count heatmap over an arcade level map.
:class: feynman-added

Graph parent links show the alternatives that were retained, while the heatmap
shows the accumulated coordinate visits used by the optional novelty term. The
two are related views built from distinct data structures.
:::

:::{div} feynman-prose
There is a subtle but important separation here. A Graph node is a full emulator
state with an ancestry link. A visit cell stores a count attached to a
coordinate key. The graph can remember a branch that later becomes dead, while
the visit grid can say that a region has been explored many times. Each view
offers a partial description of the run, while the committed game in a planner
remains a separate state.
:::

(sec-arcade-lab-reward-terms)=
## Tune the game reward terms

:::{div} feynman-prose
The **Reward terms** panel is game-specific. Every slider below is live and
changes rewards earned from the next transitions onward. It leaves the
cumulative reward already carried by a walker unchanged, so historical plot
points and an old elite's history retain their original meanings. For a planner,
changing reward weights also discards the pending search/trajectory and replans
from the current committed game; the game continues from its present state.

The words “game reward” need one qualification. Generic Atari passes the ALE
score delta through directly. Mario, Sonic, and dedicated Montezuma use the
shaped reward adapters below. In every case this is the cumulative environment
reward shown by the reward plot, before the virtual-fitness rescaling and
coefficients are applied.
:::

:::{div} feynman-added
| Console | Term (UI label) | Default; UI range and step | Applicability and effect |
|---|---|---|---|
| Mario | **X progress** (`x`) | `1`; `0`–`5`, step `0.1` | Signed accepted horizontal progress per pixel. It also scales the shortcut-distance payout when a pipe/warp returns Mario farther along. |
| Mario | **Time penalty** (`time`) | `1`; `0`–`5`, step `0.1` | Penalty for the in-game clock decreasing. `0` removes time pressure; the flagpole countdown is treated as forward progress. |
| Mario | **Death penalty** (`death`) | `25`; `0`–`100`, step `1` | One-off penalty while dying or dead. It is inside the per-frame clip. |
| Mario | **Per-frame clip** (`clip`) | `15`; `1`–`100`, step `1` | Clips the combined x, time, and death contribution to plus/minus this value. Flag and area bonuses are added after the clip. |
| Mario | **Flag bonus** (`flag`) | `500`; `0`–`2000`, step `10` | One-off reward when the flagpole is grabbed. The wrapper remains active after flag capture. |
| Mario | **Area bonus** (`area`) | `100`; `0`–`500`, step `5` | One-off reward for the first entry into each pipe, warp, bonus, or intro area in the stage. |
| Sonic | **X progress** (`dx`) | `1`; `0`–`5`, step `0.1` | Signed horizontal progress. The adapter holds this term at zero while a boss is loaded; the guard filters implausible jumps beyond its range. |
| Sonic | **Rings** (`rings`) | `3`; `0`–`20`, step `0.5` | Signed ring change: gains reward and lost rings penalize. A hit that drops rings therefore matters. |
| Sonic | **Score** (`score`) | `0.5`; `0`–`5`, step `0.1` | Multiplier on in-game score changes from enemies, monitors, boss hits, and tallies. |
| Sonic | **Exploration bonus** (`cell`) | `500`; `0`–`2000`, step `10` | One-off reward for a new `64×64` map cell in a walker's lineage and current act. |
| Sonic | **Extra-life bonus** (`life`) | `1000`; `0`–`5000`, step `50` | Reward for gaining a life, including a 1-up or a 100-ring life. Life loss ends that walker; this term rewards life gains. |
| Sonic | **Boss hit bonus** (`boss`) | `2000`; `0`–`10000`, step `100` | Reward per boss hit point removed; Sonic 1 bosses use eight hit points. |
| Sonic | **Act clear bonus** (`act`) | `5000`; `0`–`20000`, step `100` | One-off reward when zone/act progress advances. It is outside the Sonic x-distance calculation. |
| Montezuma | **Game score** (`score`) | `1`; `0`–`5`, step `0.1` | Multiplier on ALE score delta from keys, doors, jewels, enemies, and other game events. |
| Montezuma | **New-room bonus** (`room`) | `500`; `0`–`2000`, step `10` | One-off reward for entering a new one of the 24 rooms in a lineage on the current temple level. The reward is zero during dying and for terminal room 8. |
| Generic Atari | — | ALE score delta only | Uses the selected game's ALE score delta as its sole reward-term signal; the interface presents the score adapter as its sole reward signal. |
:::

:::{div} feynman-prose
The large values are deliberate. Sonic's act-clear and boss terms are meant to
make finishing and fighting visible to the selection signal; Montezuma's room
bonus is meant to keep a lineage exploring. Mario's clip keeps one ordinary
frame's contribution bounded, while its flag and area events are added after the
clip so important transitions retain their full effect.

These weights have console-specific meanings. A value of `500` means
something different when it pays for a Mario flag, a Sonic map cell, or a
Montezuma room. When comparing runs, record the console, ROM, start level, and
the complete reward-term vector.
:::

(sec-arcade-lab-kinetics)=
## Set the frame skip: dt min and dt max

:::{div} feynman-prose
At each walker step, the algorithm chooses one discrete action and holds it for
an integer number of emulator frames. **dt min** defaults to `6`; **dt max**
defaults to `30`. Both accept integers from `1` to `120`, and `dt min` stays at
or below `dt max`. The implementation samples uniformly, inclusively, between
the two endpoints. At roughly 60 frames per second, the defaults mean about
`0.1` to `0.5` seconds per sampled action.

This is a control-resolution choice. Smaller values let the swarm change its
mind more often but require more iterations to cross a level. Larger values
move farther per iteration but make a mistaken button hold harder to undo. A
terminal event can end a hold early, so the actual number of frames can fall
below the sampled number. In planner mode, the recorded edge duration is used
when the action or path is committed.

Both controls are live. A planner setting change invalidates its current search
so the next search samples with the new interval; Wave and Graph use the new
interval on later iterations. The **Mean frame skip (dt)** plot reports the
mean of the steps that actually ran, and **Env frames** reports the exact
emulator-frame total.
:::

:::{div} feynman-added
| Control | Default; range | Interpretation |
|---|---|---|
| **dt min** | `6`; integer `1`–`120` | Smallest inclusive action hold. |
| **dt max** | `30`; integer `1`–`120` | Largest inclusive action hold. Must be at least dt min. |
:::

(sec-arcade-lab-display)=
## Read the screen, map, plots, and statistics

:::{div} feynman-prose
The display is a report from the current algorithm; each algorithm defines its
own camera. Wave and Graph label the screen **Best walker** and render the leading
search state. FMC and Jump Wave label it **Played game** and render the
committed state. Their plots can still change while that game view is paused,
because the search population is doing work elsewhere.

The maps are also implementation-specific. Mario uses a supplied full-level
map and overlays the swarm. Sonic's UI stitches downsampled camera tiles from
the walkers, so dark territory marks regions this swarm has yet to visit. Montezuma
builds a 24-room pyramid from valid HUD-cropped room frames as walkers enter
rooms. Generic Atari presents its screen and plots as its display views; Mario,
Sonic, and dedicated Montezuma provide the map views.
In every map, magenta dots are alive walkers, gray dots are dead walkers, and
the gold ring marks the best walker. Graph draws parent links and larger leaf
markers, showing retained ancestry across branches.

The optional visit view changes the base map to grayscale and draws pooled visit
blocks with a fire scale. Blocks with zero recorded visits are transparent;
walkers remain on top. This count heatmap represents recorded visits; future
prediction belongs to a separate model. It can be viewed for any solver that is
counting visits in Coords mode,
although Graph is the solver for which the visit term is enabled by default.
:::

:::{div} feynman-added
| Display | Meaning |
|---|---|
| **Cumulative reward** plot | Max and mean cumulative environment reward. It is shaped game reward for Mario/Sonic/Montezuma and ALE score reward for generic Atari. |
| **Virtual reward** plot | Max and mean cloning fitness after distance/reward/visit factors are rescaled and exponentiated; **Cumulative reward** reports game score/reward. |
| **Cloned walkers (%)** | Percentage of the current update population counted as cloned; dead walkers are always clone candidates. High values usually mean stronger concentration on a few timelines. |
| **Alive walkers** | Walkers still active after the latest update. Wave and Graph can replace dead walkers by cloning while a live companion exists. |
| **Mean frame skip (dt)** | Mean number of frames actually held in the update, normally between dt min and dt max but shortened by terminal events. |
| **Graph size** | Graph-only time series of total nodes, leaves, and walkers that stepped. |
| **Iteration** | Completed algorithm iterations. Planner execution messages update execution state while the plots retain the current search iteration. |
| **World** | Mario/Atari-style world-stage value, Sonic zone abbreviation plus act, or Montezuma room, level, and lives. Planner mode uses the committed game's location. |
| **Max reward / Mean reward** | Highest and average cumulative environment reward in the current reported population. |
| **Alive** | Alive count over the current population; Graph's population can change, so read its displayed denominator to account for the current `N`. |
| **Nodes** | Graph-only `nodes / leaves / stepped` for the latest update. |
| **Iterations/s** | Three-second sliding average of worker step messages; planner search messages contribute, and execution-only messages update execution state separately. |
| **Env frames** | Exact total emulator frames actually run since reset, summed over walker steps. |
| **Planner / phase** | `planning`, `playing`, or ended; identifies whether the committed game is still or executing a selected plan. |
| **Search depth** | Current depth of the planner search. It can reset when a decision is committed or when a live planner setting invalidates the pending search. |
| **Played score** | The committed emulator's display score; for games represented through accumulated reward, the adapter reports accumulated reward. |
| **Played frames** | Exact frames executed by the committed game, separate from the many search-walker frames. |
:::

:::{div} feynman-prose
The two reward plots answer different questions. **Cumulative reward** asks,
“What has this timeline earned under the current game reward adapter?”
**Virtual reward** asks, “How attractive is this timeline to the cloning rule,
after the population's distance, reward, and optional novelty signals have been
rescaled?” A swarm can raise virtual reward through diversity while cumulative
game reward is flat. Conversely, a high-scoring walker can be cloned away if the
current population-level fitness comparison selects other timelines.

Planner plots are sampled on search advances. Executing a selected action
updates **Played score** and **Played frames**, while the swarm plots retain the
current search point. This is why the planner status can say **Playing**
while the plot lines hold still. The screen, the planner counters, and the
population plots are three synchronized, distinct views.
:::

(sec-arcade-lab-lifecycle)=
## Reset, terminal states, and reproducibility

:::{div} feynman-prose
**Start** begins the worker loop or resumes it. **Pause** stops scheduling new
steps and preserves the current Wave population, Graph tree, planner search, or
partly executed trajectory. **Reset** rebuilds the active environment with the
same selected settings and seed, clears the swarm/tree or planner state, clears
visit counts, maps, plots, and readouts, and leaves the run paused. It is the
right response to an all-dead stop or a completed committed game.

Wave and Graph stop when every current walker is dead: the population then has
zero live companions for useful cloning. A dead walker can be copied while
some live walker remains, so the alive plot may dip and recover. A planner is
different. Its search population can hit the all-dead fallback while the
committed game is still valid; the planner may play one fallback action and
search again. The committed game ends when the actual environment reports a
hard terminal state.

The game adapters define those terminal events. Mario stops on dying, falling
or dead player states, or game over; flag capture leaves this wrapper running.
Sonic treats any life loss or zero lives as terminal for that walker.
Generic Atari marks life loss as done for the walker and can recover it while
lives remain, but game over is hard. Dedicated Montezuma follows the Atari
life rule except that entering room 8 is deliberately hard terminal and earns a
zero room bonus. An action hold can therefore finish before its sampled `dt`.
:::

:::{div} feynman-prose
For a reproducible run, write down the ROM identity and build, console and
game, Mario World/Stage or Sonic Zone/Act, algorithm, observation mode, walker
count, Graph cap, seed, elite count, fitness coefficients, visit settings,
reward-term vector, and `dt` range. For planners also record horizon,
consensus, and maximum horizon. The browser derives its parallel worker count
from the machine's hardware concurrency (capped at eight), so include the
browser/device when comparing throughput or exact traces.

With the same ROM bytes, WebAssembly build, starting selectors, active settings,
and seed, the implementation's random action, frame-skip, and cloning choices
are intended to replay the same experiment. A seed fixes those random choices
while wall-clock speed remains a runtime property, and a different browser
build, ROM, observation mode, or population changes the sequence of emulator
states. A reward-term edit is also a new reward boundary: it affects future
transitions and leaves history unchanged.
:::

:::{div} feynman-added
| Setting class | Controls |
|---|---|
| **Restart required** | Console/game, Mario World/Stage, Sonic Zone/Act, algorithm, RAM/RGB/Gray/Coords, Walkers (N), Graph Max walkers, and Seed. |
| **Live** | Distance/Reward/Visit coefficients, dt min/max, Elite walkers where applicable, Visit reward, pooling, erase, and all visible game reward terms. |
| **Live but replans** | Search horizon, Jump Wave consensus, maximum search horizon, and planner reward/fitness changes. The committed game is retained; pending search state is discarded. |
| **Display only** | Map zoom/fit, map resize, plots/sidebar layout, and the map visits toggle. The visits toggle is remembered locally by the browser. |
:::

(sec-arcade-lab-troubleshooting)=
## Troubleshoot the common surprises

:::{div} feynman-prose
When the page reports **Not cross-origin isolated**, the server configuration
usually causes the status. Serve the repository's browser app with
`fractal-gas-web/serve.py` and reload; that app server supplies the headers
needed by WebAssembly threads, while direct HTML loading and plain static
hosting use response paths that omit those headers.
If the status remains on ROM loading, use the permitted local ROM, unlock the
encrypted vault, or upload a permitted Sonic ROM. A generic Atari game and the
dedicated Montezuma button are separate experiments: choosing Montezuma in the
Atari game picker selects the generic Atari experiment; the dedicated Montezuma
button supplies the room-aware map and room reward.

If Graph stops at a lower node count than **Max walkers**, read the effective
cap shown beneath that control. Each node stores a complete emulator state, and
the browser may cap the request for memory. Lowering the cap or choosing a
smaller observation is the relevant fix. The map and visit controls appear with
Mario, Sonic, or dedicated Montezuma plus Coords observation; generic Atari
uses its RAM state and standard screen and plot views.

If all walkers die, the population has reached a terminal state; rendering
continues to report that state.
Reset first. For a controlled experiment, lower `dt max`, reduce a population
that is exhausting memory, or inspect the reward terms and terminal game state.
If the planner seems to pause, look at **Planner** and **Search depth**:
a long horizon or Jump Wave's extension can spend many worker steps planning
before any committed action is played. Lower the horizon for a quick diagnostic.

Finally, a dark Sonic map marks unexplored fog; a gray dot marks a dead walker;
and a rising Virtual reward signals cloning preference while Cumulative reward
tracks game score. These three interpretations account for most apparent
contradictions between the screen, map, and plots.
:::

:::{warning}
:class: feynman-added

Compare cumulative-reward magnitudes across Mario, Sonic, generic Atari, and
Montezuma alongside the reward adapter and slider values. The virtual-reward
plot requires even more context: it is a population-rescaled selection signal
whose value depends on the current distance, reward, and visit distributions.
:::
