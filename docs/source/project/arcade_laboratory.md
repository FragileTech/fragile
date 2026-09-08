(sec-arcade-laboratory)=
# Arcade Lab

:::{div} feynman-prose
Arcade Lab is a laboratory for watching a search process happen in public. The
browser runs many complete copies of an arcade game. Each copy—called a
**walker**—chooses discrete actions, advances for a small number of frames,
records what it earned, and may later be copied by another walker. The pictures
on the screen show the current evidence, not a policy pretending to know the
answer.

Open [Arcade in the browser](https://fragiletech.github.io/fragile/web/). The page puts Mario, Atari, Sonic, and
Montezuma behind one interface. It runs the Fractal Gas machinery and emulator
cores as WebAssembly; parallel workers let a crowd of game copies move together.
The useful habit is to watch the swarm and the reward plots at the same time. A
single exciting score can be luck. A population that repeatedly finds and
preserves progress is evidence of a different kind.

Arcade Lab sits next to the continuous-control and optimization laboratories.
Here the state is not a point on a smooth surface: it is an emulator snapshot,
and a one-frame mistake can change everything.
:::

(sec-arcade-laboratory-start)=
## Open the browser laboratory

:::{div} feynman-prose
From the repository root, serve the browser build with the command below, then
open the local address. The small server supplies the cross-origin-isolation
headers required by `SharedArrayBuffer` and WebAssembly threads; opening
`index.html` directly, or using an ordinary static server, is not enough. This
command serves the existing `fractal-gas-web/web` assets. Rebuilding the
WebAssembly artifacts follows the Emscripten workflow in `fractal-gas-web/README.md`.
:::

```bash
make web
# open http://localhost:8000/web/
```

:::{note}
:class: feynman-added

On a hosted build, the ROM vault may ask for one password. The browser decrypts
the bundled ROM locally and remembers the successful unlock in that browser;
the ROM is not sent to a server. Sonic also accepts a one-time local upload of
your own Genesis ROM. Local development can use the plaintext ROM files when
they are present.
:::

(sec-arcade-laboratory-mental-model)=
## The mental model: a crowd of timelines

:::{div} feynman-prose
Imagine putting 48 identical game machines on a table and giving each one a
slightly different sequence of button presses. After a short while, some
machines have moved farther, found a ring, entered a new room, or simply stayed
alive. Arcade Lab does the bookkeeping for this crowd. A walker is not just a
dot on a map: it carries the emulator state, its reward history, and any
game-specific information needed to continue from that exact moment.

Now comes the important operation: cloning is copying a **timeline**, not merely
copying a position. A promising walker can become the starting point for
another attempt, while the other attempts continue to explore. In **Wave**, the
population stays fixed and every walker steps on each iteration. In **Graph**,
visited states remain as nodes in a growing tree, and active leaves are the
states from which new alternatives can be extended. The graph is therefore a
record of alternatives, not a single path with the past erased.

**FMC** and **Jump Wave** use the same search population differently: they look
ahead, then advance one committed game and search again from the resulting
emulator snapshot. That distinction matters when reading the screen. Wave and
Graph show a leading search walker; the planners show the game being played,
while the plots still describe the search population.
:::

(sec-arcade-laboratory-environments)=
## Four environments, four kinds of difficulty

:::{div} feynman-prose
The four buttons are not four skins on one benchmark. They expose different
state variables, reward signals, and map constructions. The defaults are chosen
to make the state useful to the swarm: Mario, Sonic, and Montezuma start in
**Coords** mode, while generic Atari starts in **RAM** mode because it has no
game-specific coordinate tuple.
:::

:::{div} feynman-added
| Setup | Browser runtime | Default observation and reward | Map in the app |
|---|---|---|---|
| **Mario** | NES, `nes-py`; World/Stage selectors | Coords; shaped progress with time, death, area, and flag terms | Full level map with the swarm overlaid |
| **Generic Atari** | Atari 2600, ALE; bundled game picker, default Ms. Pac-Man | RAM; the selected game's own score | No level map panel |
| **Sonic** | Sega Genesis, Genesis Plus GX; Zone/Act selectors | Coords; shaped progress through the act, rings, score, lives, and completion | Fog of war assembled from walker views |
| **Montezuma** | Atari/ALE with room-aware logic | Coords; score plus a new-room bonus | A 24-room pyramid assembled as rooms are found |
:::

:::{div} feynman-prose
The table also tells you why raw reward values should not be compared across
games. Mario's reward is not Atari's score, and Montezuma's room bonus is an
explicit exploration signal. Likewise, the map is not always a supplied level
image: Sonic's map is stitched from the swarm's downsampled views, and
Montezuma's rooms appear as walkers discover and render them. In both cases,
dark or missing territory means “not seen yet,” not “the game has no level
there.”
:::

(sec-arcade-laboratory-controls)=
## Controls that change the experiment

:::{div} feynman-prose
Start with **Wave**. Each iteration gives every walker a random action and holds
it for `dt` frames, with `dt` drawn uniformly between **dt min** and **dt max**
(the defaults are 6 and 30 frames, roughly 0.1 to 0.5 seconds at 60 frames per
second). The cloning step uses virtual reward: a combination of how well a
walker has done and how far it is from its comparison walker, with an optional
visit-count term. Thus a good score and a good virtual reward are related, but
they are not the same measurement.

**Observation** chooses what the distance calculation sees: console RAM, RGB
pixels, grayscale pixels, or a compact Coords tuple. It does not change the
game's controls. Changing observation, algorithm, game, or start level restarts
the run because these choices change the state that is being copied. The seed,
walker count, elite buffer, fitness coefficients, visit settings, and frame
range are the experimental knobs; keep them with a result you want to repeat.

The reward-term sliders for Mario, Sonic, and Montezuma apply to rewards earned
from that point onward. Reward already banked by a walker keeps its old
weighting, so a live change does not rewrite history. **Pause** preserves the
current search or trajectory. **Reset** reconstructs the run with the same
settings and seed. The **Env frames** readout counts frames actually emulated,
including steps that end early because of death or life loss.
:::

:::{warning}
:class: feynman-added

Do not read the **Virtual reward** plot as the game's score. Virtual reward is
the selection signal used for cloning; **Cumulative reward** is the reward
accumulated by each walker. A swarm can improve the selection signal by
spreading out, while a game score can remain flat. Read both plots together.
:::

(sec-arcade-laboratory-guide)=
## Follow the Arcade Lab guide

:::{div} feynman-prose
Use the short guide pages when you want to turn the picture into an experiment:

- {doc}`arcade_lab_getting_started`: open the app, run the first swarm, and read
  the screen, map, plots, and status values.
- {doc}`arcade_lab_environments`: learn the game-specific observations, rewards,
  start-level controls, maps, and termination behavior.
- {doc}`arcade_lab_controls`: work through algorithms, planning, observations,
  fitness, visits, frame skips, seeds, and live reward settings.

For a first session, choose Mario in Coords mode, leave Wave selected, press
Start, and watch how the dots and plots change together. Then switch to Graph
and ask what the retained branches reveal; after that, try a planner and notice
the difference between searching and committing an action. That sequence makes
the controls answer a concrete question instead of becoming a collection of
knobs.
:::
