(sec-arcade-lab-environments)=
# Arcade Lab environments

:::{div} feynman-prose
The Arcade Lab has four buttons, but they are not four skins on the same
environment. They are four different machines with four different ideas of what
the state of a game is. Mario exposes a little NES machine, Atari exposes the
Arcade Learning Environment (ALE), Sonic exposes a Genesis machine, and
Montezuma uses ALE again with extra knowledge of that particular game.

The planner still asks every machine the same basic questions: *What actions can
I take? What happened after I held one for a few frames? How much reward did that
produce? Is this walker finished?* The answers are where the important
differences live. This page is the map of those answers.
:::

(sec-arcade-common-controls)=
## Controls shared by the four modes

:::{div} feynman-prose
Start with one useful distinction. The picture on the screen is what *you* see;
the observation is what the swarm uses to decide whether two possible futures are
similar. The reward is a third thing: it is the score signal used to rank those
futures. These can agree, but they do not have to. A player can see a key on the
screen, for example, while a Coords observation records only the player, room,
inventory, and lives; the reward still comes from the game's score and the
room-entry rule.

Every walker is a complete copy of the emulator, not merely a dot moving over a
picture. On one iteration the walker samples an action and holds it for a random
number of emulator frames between **dt min** and **dt max**. A death or game over
can stop that hold early. The screen then shows either the best walker or the
committed played game, depending on the selected algorithm; it is not a promise
that every walker has exactly the displayed frame.

Changing the console, game, observation mode, or start-level selector creates a
new run. **Reward terms**, where present, are different: moving one of those
sliders changes rewards earned from that point onward, while reward already
carried by a walker is not recomputed.
:::

:::{div} feynman-added
| UI choice | What it changes | What it does not change |
|---|---|---|
| **Mario** | Loads the NES environment and its Super Mario Bros ROM. | The common swarm, plots, and planner controls. |
| **Atari** | Loads a selected generic ALE game and its bundled ROM. | It does not add game-specific coordinates or a map. |
| **Sonic** | Loads the Genesis environment, a Sonic ROM, and a selected zone/act; an uploaded ROM is stored in this browser's IndexedDB. | The observation is still only the chosen mode; the fog map is a separate view. |
| **Montezuma** | Loads Montezuma's Revenge through ALE with room-aware state and reward logic. | It is not the same as choosing Montezuma in the generic Atari list. |
| **RAM / RGB / Gray / Coords** | Changes the vector on which walker distances are measured. | The environment's reward and game dynamics. |
:::

### Observation modes in one sentence each

:::{div} feynman-prose
The four labels are deliberately blunt. **RAM** means the emulator's memory
bytes. **RGB** means the current screen's red, green, and blue values laid out as
one long vector. **Gray** means the same screen reduced to luminance. **Coords**
means a small state tuple chosen for that particular game. Pixel values and RAM
bytes are exposed as floating-point numbers, but their values are still the
original 0--255-style integers unless a mode says otherwise.

The practical consequence is easy to miss: Coords is not automatically “better
vision.” It is a deliberate state representation that makes distances cheap and
usually makes the game's useful geometry visible to the swarm. The reward is
computed separately, so turning on Coords does not secretly replace the game's
score with a coordinate reward.
:::

:::{div} feynman-added
| Environment | Default | Coords contents | Other dimensions |
|---|---|---|---|
| Mario / NES | **Coords** | 10 values: level position, vertical state, level identifiers, clock, velocities, sub-area, and power-up state. | RAM: 2,048; RGB: $256\times240\times3$; Gray: $256\times240$. |
| Generic Atari / ALE | **RAM** | **Exactly the same 128-byte RAM vector as RAM**; generic ALE has no universal coordinate schema. | RGB and Gray use the screen dimensions reported by ALE, commonly $160\times210$. |
| Sonic / Genesis | **Coords** | 6 scaled values: position, horizontal/vertical velocity, ground speed, and zone/act progress. | RAM: 64 KiB work RAM; RGB: $320\times224\times3$; Gray: $320\times224$. |
| Montezuma / ALE dedicated | **Coords** | 8 values: pyramid-global position, room, local position, level, inventory bitmask, and lives. | RAM: 128; RGB and Gray use the ALE frame, $160\times210$ here. |
:::

:::{note}
:class: feynman-added
The observation vector is the object used for distance and diversity. It is not
the same thing as the `Best walker` canvas, and it is not a preprocessed neural
network input. If you want normalized pixels or a learned encoder, that is a
different experiment.
:::

(sec-arcade-mario-nes)=
## Mario / NES

:::{div} feynman-prose
Mario is the cleanest place to see what a map-aware arcade environment is doing.
The NES emulator runs Super Mario Bros directly, while the swarm keeps many
complete copies of the game. The map below the screen is not another emulator
view: it is a full-level drawing with the walkers' positions painted on top.
That lets you see the gas spreading through a level even when the screen follows
only one camera window.
:::

### Selectors and start state

:::{div} feynman-prose
Choose **Mario**, then choose **World** from 1 through 8 and **Stage** from 1
through 4. The selectors are the familiar player-facing names; the emulator
does the small boot-time RAM operation needed to start directly in that level.
Changing either selector restarts the run from a fresh paused state. The bundled
Super Mario Bros ROM is loaded for the page; on a deployment that keeps ROMs in
the encrypted local vault, unlock it once in the browser.

There is no separate Mario map selector. The chosen world and stage determine
both the initial game and the full-level map shown before the first iteration.
When walkers later reach other stages, the map follows the leading walker's
current stage.
:::

### Mario observations

:::{div} feynman-prose
In **Coords** mode, think of the tuple as a pocket notebook carried by Mario.
The first entry says how far along the level he is; the next entries say where he
is vertically and how he is moving; the remaining entries identify the local
area and a few pieces of game state. It is enough to tell “running right on the
ground in 1-1” from “falling in a pipe room,” without comparing 184,320 pixel
numbers.

The values are intentionally raw. In particular, the level identifiers are the
NES's RAM values, while the **World** and **Stage** controls and the map labels
are one-based. Do not use the tuple as a pretty display string. It is a compact
coordinate chart for distances.
:::

:::{div} feynman-added
| Coords entry | Meaning |
|---|---|
| `x` | Mario's horizontal position in level/world pixels. |
| `y_pixel` | Player vertical pixel position on the current screen. |
| `y_viewport` | Vertical viewport state used to distinguish normal play from falling/off-screen states. |
| `world`, `stage` | Raw level identifiers from NES RAM. |
| `time` | The in-game decimal clock. |
| `h_velocity`, `v_velocity` | Signed horizontal and vertical velocity proxies. |
| `sub_area` | The current sub-area/area byte. |
| `power_up` | The power-up state byte. |
:::

**RAM** is the 2 KiB NES CPU RAM. **RGB** is the $256\times240$ framebuffer,
with three values per pixel. **Gray** is one luminance value per pixel. All
three are valid ways to make the swarm compare states, but RGB and Gray are
large; Coords is the fast default for this environment.

### Mario actions, reward, and termination

:::{div} feynman-prose
Mario's action set is a small, intentional vocabulary rather than every possible
combination of controller bits. The swarm can stand still, walk or run left and
right, jump, and combine movement with the A/B buttons. Holding an action for
several frames is what gives an action its physical meaning; a single “right”
choice is not one pixel of motion.

The default reward is shaped toward finishing a level. Each ordinary frame pays
for small rightward progress and charges for clock time running down. A large
position jump is treated as an emulator transition rather than trusted as real
motion, which prevents a bad RAM read from becoming a giant reward. Death carries
a penalty, and the ordinary per-frame sum is clipped to $[-15,15]$ by default.

There are two bonuses outside that clip. Grabbing the flag pays $500 once, and
entering a previously unseen area pays $100 once. The second rule matters because
some useful Mario routes go through a pipe: the player may move into a side room,
appear to move backward, and then return much farther along the level. The
environment pays the signed skipped distance on that return, so the pipe is not
made invisible to the optimizer.
:::

:::{div} feynman-added
| Event or term | Default behavior |
|---|---|
| Horizontal progress | Signed change in level `x`; changes larger than 5 pixels in one frame are ignored as a transition/glitch guard. |
| Clock | A decrease in the in-game clock is a penalty. During the flagpole countdown it is treated as forward progress so the swarm can cross into the next level. |
| Death | $-25$ before the default per-frame clip; the default clip therefore limits the visible one-frame penalty to $-15$. |
| New area | $+100$ once per previously unseen area in the current stage, including pipe/warp/bonus-room transitions. |
| Flag | $+500$ once when Mario grabs the flag. |
| Termination | Dying, falling/off-screen, a dead player state, or the game's lives-exhausted marker. Grabbing the flag alone does **not** terminate this wrapper. |
:::

The flag behavior is a deliberate Lab choice. In the original single-stage
interface, the flag ends the episode; here the walker continues through the
castle sequence so “finishing” is not treated like a dead branch that must be
cloned away. A step stops early when one of the actual death/game-over conditions
fires, and the NES environment does not emit a separate truncation signal.

### The Mario level map

:::{div} feynman-prose
The Mario map is a static full-level image for every world/stage pair. One map
pixel represents one level/world pixel, and the map is 224 pixels high because
the 8-pixel top and bottom overscan are left out. Magenta dots are alive walkers,
gray dots are dead walkers, and the gold-ringed red dot is the best walker. In
Graph mode, thin lines join nodes to their parents, so the picture is also a
record of the explored state tree.

The map's dots come from RAM-derived positions, not from computer vision. That is
why a dot can be useful even when the game screen is scrolling, and also why it
should not be read as a pixel-perfect outline of Mario's sprite.
:::

(sec-arcade-generic-atari)=
## Generic Atari / ALE

:::{div} feynman-prose
Generic Atari is the honest baseline: hand the Arcade Learning Environment a
ROM, read its machine, and use the score that the game itself reports. There is
no universal “Atari position” because Breakout, Q*bert, and Seaquest do not put
their meaningful state in the same RAM addresses. The environment therefore
refuses to invent one.
:::

### Selector and game identity

Choose **Atari**, then choose a game from the **Game** selector. The default is
**Ms. Pac-Man**. The page bundles a broad list of ALE-supported Atari 2600 games,
including **Breakout**, **Montezuma's Revenge**, **Pong**, **Q*bert**, and many
others. Changing the game loads its ROM and restarts the run.

There are no **World/Stage** or **Zone/Act** selectors in this mode. A game starts
where its own reset starts it. If you choose **Montezuma's Revenge** from this
generic list, it remains generic Atari: you get generic ALE behavior, generic
observation semantics, no pyramid map, and raw score reward. Choose the separate
**Montezuma** console button to activate the dedicated logic described below.

### Observations and actions

:::{div} feynman-prose
The default **RAM** observation is exactly 128 Atari RAM bytes. That is compact,
but it is not a semantic dictionary: byte 42 might matter in one game and be
ordinary scenery state in another. This is the important alias in the UI:
**Coords is RAM for generic Atari**. It is not a guessed tuple of player x and
y, because ALE cannot supply one tuple that is correct for every ROM.

**RGB** and **Gray** use the current screen returned by ALE. The screen size is
game/core data rather than a promise made by this page; the common Atari frames
are typically $160\times210$. Pixel modes make the observation much larger, but
they are useful when visual state is the question you want to study.

Actions come from ALE's minimal action set for the selected game. The set can
therefore differ between games, and the Arcade Lab does not pretend that action
index 3 means the same button in every ROM. The planner chooses one of those
actions and holds it for the selected frame skip.
:::

### Score and termination

:::{div} feynman-prose
The reward is the game's own score change, summed over the frames in the action
hold. There is no Atari reward-term panel and no generic progress shaping. This
is useful precisely because it is plain: if the score rises, the game says that
something valuable happened. It also means that raw reward scales are not
comparable between different Atari games.

ALE life handling needs one careful sentence. A life loss marks that walker done
so the swarm avoids spending its next action in a bad state. If the game still
has lives, the death is recoverable and the swarm may revive that walker from a
live companion when necessary. The final game over remains a hard death. Games
without an ALE life counter do not get this soft-death behavior.
:::

:::{div} feynman-added
| Item | Generic ALE behavior |
|---|---|
| Reward | ALE score delta; no hand-designed coordinate, room, or novelty bonus. |
| Frame hold | The selected ALE action is applied for `dt` frames, stopping early at game over or a life loss. |
| Life loss | `done` for that walker; recoverable when lives remain and the game is still playable. |
| Game over | Hard termination; a step cannot continue through it. |
| Truncation | ALE's truncation information is kept distinct from the game-over/life-loss `done` flag. |
| Map | No level map panel in generic Atari. Inspect the current best-walker screen and the reward/alive plots. |
:::

:::{figure} ../../_static/arcade_lab/atari-breakout.png
:alt: Atari Breakout frame with the paddle below the ball and the generic ALE arcade run readouts beside it.
:class: feynman-added

Breakout is a useful generic-ALE example: the environment can display the game
and optimize its score without claiming to know a universal Atari coordinate map.
:::

### Generic-ALE caveats

:::{div} feynman-prose
There are three traps here. First, a generic Atari **Coords** run is not a cheap
player-coordinate run; it is the same 128-byte RAM vector. Second, a life-loss
`done` is not necessarily the end of the game, so a low alive count can recover
on the next iteration. Third, score meanings differ wildly. A score of 100 in
one ROM is not evidence of equal progress in another.

The emulator instances are restored from complete serialized ALE state, including
the emulator's random state. Sticky actions are disabled in the Lab wrapper. This
makes a copied walker a genuine continuation of the copied game state, but it
does not make two different Atari games physically comparable.
:::

(sec-arcade-sonic-genesis)=
## Sonic / Genesis

:::{div} feynman-prose
Sonic is a different kind of difficulty from Mario. The world is wide, the
camera hides most of it, and momentum makes a position without a velocity almost
useless. The Genesis environment therefore records position *and* momentum in
Coords mode, and it builds a second map from what the swarm has actually seen.
That second qualification is the whole point of the fog of war.
:::

### Zone, act, and ROM selectors

Choose **Sonic**, then choose a **Zone** and **Act**. The menu offers Green Hill,
Marble, Spring Yard, Labyrinth, Star Light, and Scrap Brain, with acts 1 through
3. The menu order is for humans; the ROM uses internal zone identifiers. A small
ROM-specific exception is preserved: Scrap Brain Act 3 is loaded through the
ROM's Labyrinth Act 4 slot.

Sonic runs from a **local Genesis ROM**. The page first looks for a local bundled
`sonic.rom`; on a hosted build it may unlock the encrypted local copy in the
browser. If that is unavailable, use **Sonic ROM** to upload a `.md`, `.bin`,
`.gen`, or `.smd` file. The uploaded ROM is kept in this browser's IndexedDB for
later visits; it is not a server-side ROM service. Replacing the ROM or
changing Zone/Act restarts the run.

### Sonic observations

:::{div} feynman-prose
Sonic's Coords tuple is designed around reachability. Position tells you where
Sonic is, but the velocity entries tell you what kind of future that position can
produce: a runner on a ramp, a character in a jump, and a character standing
still are not interchangeable states. The first five entries are scaled by 256,
roughly putting pixels and subpixel speeds into tile-sized numbers. The final
entry is a progress label scaled by 100 so changing acts is unmistakable to the
distance calculation.

Notice what is absent: score, rings, and lives. They still affect reward, but
putting them into the distance tuple would make the reward signal do the same job
twice and could make a high-score but physically identical state look far away.
:::

:::{div} feynman-added
| Coords entry | Meaning |
|---|---|
| 1 | `x / 256`, horizontal position in the level. |
| 2 | `y / 256`, vertical position. |
| 3 | `x velocity / 256`. |
| 4 | `y velocity / 256`. |
| 5 | Ground speed/inertia ` / 256`. |
| 6 | `(zone * 3 + act) * 100`, a coarse zone/act progress coordinate. |
:::

**RAM** is the 64 KiB Genesis work RAM exposed by the core. **RGB** and
**Gray** use the $320\times224$ Genesis frame. Pixel modes show the current
camera window, not the whole level.

### Sonic actions, reward, and termination

:::{div} feynman-prose
Sonic has eight actions. The fourth action is **Right + Down**, which the Lab
uses as a rolling action; it replaces the Genesis core's otherwise unhelpful
upward/camera-tilt slot. Holding B means a higher jump, but a jump requires a
button-press edge. Consequently, if one action hold ended with B down, the first
frame of the next consecutive B hold releases B before holding it again. This
looks like a fussy implementation detail until you see chained planner actions
turning one long press into no new jumps.

The default reward has several jobs. Rightward progress pays per pixel, ring
changes pay with a signed weight, score changes pay a smaller amount, new 64 by
64 pixel cells pay an exploration bonus once per lineage and act, extra lives
pay, boss hit points pay heavily, and entering a later act pays a large clear
bonus. The x-progress term is suppressed while a boss is loaded because the
camera is locked there; otherwise the planner would learn to press against a
wall instead of hitting the boss. A jump or a loop can therefore be valuable
even when its immediate x coordinate goes backward.

Any life loss ends that walker. Reaching a later act is rewarded but is not itself
terminal: the committed game can continue into the next act until a life is
lost. A step stops early at that life loss.
:::

:::{div} feynman-added
| Reward term | Default |
|---|---:|
| Signed x progress | $1$ per accepted pixel of change; changes larger than 5 pixels in one frame are ignored by the transition/glitch guard. |
| Rings | $+3$ per ring gained and $-3$ per ring lost; a hit that drops rings is therefore penalized. |
| Game score | $+0.5$ per score point. |
| New exploration cell | $+500$ once for a new $64\times64$ map cell in the walker's lineage and current act. |
| Extra life | $+1000$ for a gained life. |
| Boss damage | $+2000$ per boss hit point removed; Sonic 1 bosses have eight hit points. |
| Act clear | $+5000$ when the zone/act progress coordinate advances. |
| Termination | Any life loss or a zero-lives state. |
:::

### The Sonic fog-of-war map

:::{div} feynman-prose
The Sonic map is not downloaded as a complete blueprint. Each walker sends back
the part of its $320\times224$ camera frame it saw, downsampled by eight, along
with the camera position. The UI stitches those $40\times28$ tiles into a
per-zone/act canvas. Areas no walker has seen remain dark.

This is a useful picture of exploration, but it is not omniscient ground truth.
The map can have holes, and a dark region means “not revealed by this swarm,” not
“empty level.” The fixed Sonic HUD is masked while tiles are stitched so score
text does not become a fake platform. Magenta and gray walker dots, the gold
best-walker marker, and Graph parent lines are drawn on top of the revealed map.
The zoom buttons are there because a long level quickly becomes wider than the
panel.
:::

:::{figure} ../../_static/arcade_lab/sonic-fog.png
:alt: Sonic level map assembled from explored camera tiles, with dark unrevealed regions and magenta walker positions over the revealed terrain.
:class: feynman-added

The fog map records the swarm's knowledge of the selected Sonic zone and act.
Unexplored level remains dark until a walker carries its camera there.
:::

### Sonic caveats

:::{div} feynman-prose
The map and the observation use different coordinates. Coords stores the
player's level position and velocity; the fog map uses the camera position to
place screen tiles. A player can be in the same level x-coordinate while the
camera is showing a different vertical slice, so do not infer camera position
from the six-element observation.

The exploration bonus belongs to a walker's lineage. When a walker clones a
companion, it inherits that companion's remembered cells; it cannot repeatedly
collect the same novelty bonus merely by copying itself and walking in circles.
The memory is cleared when the act changes. This is why the bonus can encourage
backtracking without becoming an infinite reward faucet.
:::

(sec-arcade-montezuma)=
## Montezuma / ALE with dedicated logic

:::{div} feynman-prose
Montezuma is the Atari exception that earns its own button. The emulator is still
ALE, but the Lab knows enough of Montezuma's Revenge's RAM layout to name the
room, project Panama Joe into that room, remember which rooms a lineage has
visited, and stop the swarm from wasting its budget in a known dead end. This is
game-specific knowledge; it is not a generic Atari feature.
:::

### Selector and initial state

Choose **Montezuma** in the console selector. The UI loads the bundled
`montezuma_revenge` Atari ROM and turns on the dedicated logic automatically. The
generic Atari **Game** selector is not used here, and there is no separate room
selector: the game begins from its reset state, initially in room 1 of the first
temple level (shown as **L1** in the readout).

There are no World/Stage or Zone/Act controls for this mode. The current room,
level, inventory, and lives are read from each walker's emulator state after the
step.

### Montezuma observations

:::{div} feynman-prose
Montezuma's Coords vector has two kinds of position. `room`, `x`, and `y` tell
you the game's local coordinates. `global_x` and `global_y` place that local
position into the temple pyramid, with one room-width of separation between
adjacent room cells. Thus a state in room 0 and a state in room 1 are not
mistaken for the same location merely because their local x values match.

The Atari RAM y-coordinate grows upward, while a screen image grows downward.
The UI converts it when it places the walker dot on the room image. Inventory is
a bitmask, not a count: different bits represent carried objects such as keys,
the torch, sword, and hammer.
:::

:::{div} feynman-added
| Coords entry | Meaning |
|---|---|
| `global_x`, `global_y` | Position in the 9-by-4 pyramid canvas, in room-image pixels. |
| `room` | Montezuma room number. The initial room is 1. |
| `x`, `y` | Local RAM position in the room; RAM `y` grows upward. |
| `level` | Temple level identifier, stored zero-based and shown as L1, L2, and so on. |
| `inventory` | Object inventory bitmask. |
| `lives` | Remaining lives. |
:::

**RAM** and generic **Coords** are different here. RAM remains 128 bytes, while
dedicated Coords is the eight-value tuple above. **RGB** and **Gray** use ALE's
$160\times210$ frame; the top 50 rows are the HUD, and the map uses the remaining
$160\times160$ room image.

### Montezuma actions, reward, and termination

:::{div} feynman-prose
Actions come from ALE's minimal action set for Montezuma, just as they do for a
generic Atari game. What changes is what happens after the action. The reward is
the ALE score delta plus a one-time new-room bonus, $500 by default, for each
room first entered by that walker's lineage on the current temple level. The
start room is marked as already visited at reset, so standing still at the start
does not manufacture a bonus. The visited-room mask resets when the temple level
changes.

Now the sharp edge: **entering room 8 terminates that walker**. Room 8 is treated
as a dead-end/death room by the dedicated logic, so it receives no new-room bonus
and is not a recoverable life loss. This is intentionally stronger than merely
letting the game animate a death; the swarm should spend its search budget on the
rest of the temple rather than repeatedly rediscovering a room the Lab has marked
as terminal.

Other life losses follow ALE's normal recoverable rule when lives remain. The
final life, a real game over, and room 8 are hard terminations. The action hold
ends as soon as one of those conditions is reached.
:::

:::{div} feynman-added
| Event or term | Dedicated Montezuma behavior |
|---|---|
| Game score | ALE score delta multiplied by the **Game score** weight; default multiplier $1$. |
| New room | One $+500$ bonus per newly entered room in the lineage and level; no bonus during a death animation. |
| Room 8 | Hard termination on entry; no room bonus. |
| Life loss | Walker is done; recoverable if lives remain, except in room 8. |
| Final game over | Hard termination. |
| Reward panel | **Game score** and **New-room bonus** can be tuned live. |
:::

### The 24-room pyramid map

:::{div} feynman-prose
The initial temple level has **24 rooms** arranged in a 9-column by 4-row
pyramid. The rows contain 3, 5, 7, and 9 rooms, numbered as follows:
:::

:::{div} feynman-added
| Pyramid row | Rooms, left to right |
|---|---|
| Top row | 0, 1, 2 |
| Second row | 3, 4, 5, 6, 7 |
| Third row | 8, 9, 10, 11, 12, 13, 14 |
| Bottom row | 15, 16, 17, 18, 19, 20, 21, 22, 23 |
:::

:::{div} feynman-prose
The UI starts with the pyramid's room outlines and numbers. When an alive walker
stands in a room, its HUD-cropped $160\times160$ image can be revealed in that
cell. Unvisited rooms remain dark; the dots show where the current swarm is;
the gold frame marks the best walker's room. The image capture rejects black or
blue transition screens and avoids replacing a real room with the temporary
palette flash shown when an item is collected. This is why a room may be
revealed a little after the first transition, but the map is less likely to
freeze a misleading frame.
:::

:::{figure} ../../_static/arcade_lab/montezuma-pyramid.png
:alt: Montezuma temple pyramid map with 24 numbered room cells, several revealed Atari room images, and swarm positions overlaid on the rooms.
:class: feynman-added

The dedicated Montezuma map arranges the first temple level's 24 rooms as a
pyramid. Dark numbered cells have not yet been revealed by the swarm; room 8 is
visible as a map cell but is terminal when entered.
:::

### Montezuma caveats

:::{div} feynman-prose
Do not confuse three numbers in the readout: the room number identifies a cell
of the pyramid, the level identifies which temple level is active, and lives
identify the remaining game attempts. The map currently shows one level's
pyramid at a time, so a walker in another level is not silently placed into the
wrong room cell.

Also keep the two Montezuma entry points separate. **Atari → Game → Montezuma's
Revenge** is a valid generic ALE experiment, but it deliberately has generic
128-byte RAM semantics, raw score reward, and no pyramid map. **Montezuma** is the
dedicated experiment: Coords has room geometry, new-room reward is available, the
pyramid is built by the swarm, and room 8 is terminal.
:::

(sec-arcade-environment-choice)=
## Which mode should you use?

:::{div} feynman-prose
Choose the environment by the question, not by the popularity of the game. If
you want a known level with an explicit full map and a compact state, start with
Mario in Coords mode. If you want to measure a planner against a game's own
score, choose generic Atari and keep its game identity fixed. If momentum and
partially revealed long levels are the subject, choose Sonic. If room discovery,
inventory, and controlled exploration are the subject, choose the dedicated
Montezuma button.

And one final warning about comparisons: a Coords vector is a local language for
one environment. Sonic's scaled velocity and Montezuma's pyramid-global x are
not quantities to compare numerically with Mario's raw NES pixels. Compare the
behavior the representation makes visible, and report the environment, ROM,
selectors, observation mode, reward weights, and termination rule alongside any
result.
:::
