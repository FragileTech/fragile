(sec-control-lab-scenes)=
# Scenes, agents, and the editor

:::{div} feynman-prose
A scene describes the equipment on the table: bodies, their actuators, walls, targets,
and the rules that award reward. The running world describes what that equipment is
currently doing. This distinction explains the editor's most important behavior:
changing a scene rebuilds the experiment from its initial conditions. Moving a kart
in the editor changes its starting position; it does not continue its current drive.

Start with {doc}`control_lab_getting_started` for installation and a first run.
This page covers constructing the experiment. {doc}`control_lab_controls` covers
operating its controller, and {doc}`control_lab_replay` covers preserving and revisiting
its motion. The original {doc}`control_laboratory` provides further technical reference material.
:::

(sec-lab-scenes-presets)=
## Choose an experiment

:::{div} feynman-prose
Use **Environment** to choose one of six tasks. **Racing** offers six kart circuits
through its separate **Select track** dropdown.
Loading a preset starts a fresh world and clears the editor's undo history.
The **SCORE** panel shows a task counter, which
is different from the accumulated reward that the controller optimizes. Reward can
include progress, collisions, and formation penalties as well as completed objectives.
:::

:::{div} feynman-added
| Environment | What to try | Main score |
|---|---|---|
| **Asteroid harvesting** | Guide the tug, acquire cargo with its automatic tether, and bring ore into the delivery base. | Cargo deliveries |
| **Ants & drops** | Coordinate 1–128 harvesters or drones collecting food; defaults to 48 harvesters. | Food collected |
| **Tandem flight** | Guide a pair through ordered checkpoint zones while maintaining formation. | Gates crossed |
| **Collaborative mining** | Move a heavy shared load with two thrusters and their tethers. | Cargo deliveries |
| **Thinking graphs** | Use the harvesting task to inspect alternative futures, cloning, and ancestry. | Cargo deliveries |
| **Racing** | Choose a circuit with **Select track**, then drive its checkpoint zones in order. | Laps completed; next checkpoint |
:::

### Configure Ants & Drops

:::{div} feynman-prose
Choose **Ants & drops**, then use **Vehicle type** and **Vehicle count** beneath
**Environment**. Choose either **Harvesters** or **Drones** for the whole group and
enter a whole-number count from 1 to 128. The default is 48 harvesters. Each type
uses its own physics and visual model, with three action channels per vehicle.
The count determines how many vehicles move in the arena; **Walkers** determines
how many possible futures the planner considers.

Changing either vehicle setting rebuilds the original preset, clears the current
run and editor history, and leaves the world paused. **Reset** retains the current
scene and vehicle configuration. Switching to another environment and back retains
your vehicle selections within the tab session.

The arena has 24 pickup slots. Collect a drop and that slot returns after three
simulation seconds at a seeded random playable position. This repeats indefinitely:
an empty arena can mean that all 24 slots are waiting to return. Pausing the world
also pauses their timers.
:::

### Choose a kart circuit

:::{div} feynman-prose
Choose **Racing** in **Environment**, then use **Select track** to choose a circuit.
The tracks are ordered from Easy to Hard. Start with Violet Circuit or Roots Oval to get a feel for steering. Then try Fearless
Circuit: a corner now sets up the next one, so entering too quickly can leave you
poorly placed for the following bend. The Hard circuits add close hairpins or
obstacles that leave less room to recover. Every circuit uses the same kart physics;
the difficulty labels describe the route you must drive.

The preview beneath **Select track** shows the active scene's outline, difficulty,
racing direction, and checkpoint count. Historical circuits also link to a reference
video. Look at that outline before driving: a long straight followed by a tight bend
calls for a different approach from an open oval. Selecting another circuit starts
a fresh run and updates the checkpoint count. Imported scenes and replay archives
keep their own geometry, so the preview follows the scene you actually loaded.

The five historical layouts were reconstructed from Sergio Hernandez's kart videos,
with proportions traced by hand and scaled uniformly. Their distances are simulation
units, not surveyed track measurements. Sepang Kart uses the go-kart layout shown in
the videos, not the Formula One circuit. Labyrinths, caves, and non-racing arenas are
outside this circuit collection.
:::

:::{div} feynman-added
| Track | Difficulty | What to try | Checkpoints per lap |
|---|---|---|---|
| **Violet Circuit** | Easy | Learn the controls on the broad original lab circuit. | 16 |
| **Roots Oval** | Easy | Practise braking and steering on the historical oval. | 50 |
| **Fearless Circuit** | Medium | Connect bends and negotiate a deep hairpin. | 130 |
| **Sepang Kart** | Hard | Tackle the historical go-kart layout's close hairpins. | 151 |
| **Original Obstacle Circuit** | Hard | Steer around bollards and edge intrusions. | 133 |
| **Fearless Obstacle Field** | Hard | Thread the islands before returning to linked hairpins. | 128 |
:::

### Drive a circuit

:::{div} feynman-prose
Select a kart circuit, enable **Keyboard control**, and click the world so that a
number field is no longer receiving keystrokes. Hold **W** to accelerate,
use **A/D** to steer, **S** for reverse throttle, and **Space** to brake. Use **Follow
agent** for a close view, and **2D / 3D** to change the camera angle. Select a controller
and press **Run experiment** when you want the planner to drive.

The gold checkpoint is the next target. A checkpoint counts when the kart's centre
enters its circular zone; passing through a decorative arch is not a separate timing
measurement. After all zones in the selected circuit, the lap counter increases and
the target returns to checkpoint 1. Waiting at the finish or visiting a later
checkpoint first does not complete a lap. The zones do not impose a crossing direction. In ordinary live play,
completing a lap does not stop the world; each preset's experiment success criterion
is its checkpoint count, meaning one lap (16 gates for Violet Circuit).
See {doc}`control_lab_experiments` for episode limits.

The walls are collidable but nonlethal in these kart presets. The asphalt, painted guide,
curbs, and gantry help you see the course; native boundary and hole polygons define
its physical limits. Replaying a world restores the gate counter, lap display, and
highlighted target along with the kart's movement.
:::

:::{figure} ../../_static/control_lab/kart-detail.png
:alt: Purple kart on the Violet Circuit asphalt with striped curbs and the laboratory controls visible.
:class: feynman-added

Follow the kart to inspect its steering and motion while keeping checkpoint progress visible.
:::

(sec-lab-scenes-editing)=
## Place and move objects

:::{div} feynman-prose
Press **Edit scene** to pause control and open the editor. Closing it with **×** closes
the panel; it does not resume the simulation. Each committed edit recompiles the scene,
resets motion and counters, and starts new recording and planning histories. Export a
run before editing if you want to keep the preceding experiment.

With **Select & move**, click a body, food pickup, delivery base, checkpoint, or gravity
well. **Shift-click** adds or removes an entity from the selection. Drag a selected
entity to move the whole selection by the same displacement; release to commit. Click
empty space to clear the selection. The last selected entity is the one shown in the
property editor. Selection does not itself modify the scene.

The viewport supports wheel zoom and middle-button, right-button, or **Alt-drag**
panning. Panning releases camera following. **Follow agent** follows a selected body,
or the first controlled body when no body is selected; **Whole arena** recentres and
resets zoom. Camera changes do not affect physics. Outside the editor, clicking the
world selects nearby recorded search nodes when rollout paths are visible, rather
than selecting bodies for editing.
:::

:::{div} feynman-added
| Tool | Gesture and result |
|---|---|
| **Agent** | Choose **Agent type**, then click a starting position. |
| **Asteroid** | Click to place a passive, six-vertex convex cargo body of mass 3 kg. |
| **Food pickup** | Click to place a pickup with radius 0.4 m. |
| **Delivery base** | Click to place a delivery zone with radius 2.5 m. |
| **Checkpoint gate** | Click to append a checkpoint with radius 2.5 m to the ordered gate list. |
| **Gravity well** | Click to place a source with strength 25 m³/s² and softening 3 m. |
| **Tether · click source, then cargo** | Click two distinct bodies; creates a spring with stiffness 25 N/m and damping 6 N·s/m. The second body need not be cargo. |
| **Draw hole / pillar** | Click at least three vertices, then **Finish polygon** to append a hole. |
| **Draw outer boundary** | Click at least three vertices, then **Finish polygon** to replace the outside boundary. |
:::

:::{div} feynman-prose
Switching tools discards an unfinished polygon or first tether endpoint. Polygon
vertices use world coordinates in metres. The compiler closes the ring; do not repeat
the first vertex. Boundaries may be concave but cannot cross themselves. Holes must
lie inside the outer boundary without intersecting other rings or nesting inside
another hole. Body and zone centres must lie in playable space. Dynamic body hulls,
in contrast, must be convex and have no more than 32 vertices.

Tethers and polygon vertices are not selectable with **Select & move**. Edit their
parameters and coordinates in the complete scene JSON. Creating a tether gives it an
initial rest length from the placement distance; automatic cargo hooking and breaking
parameters are additional JSON fields, not separate placement tools.
:::

(sec-lab-scenes-properties)=
## Edit properties and manage revisions

:::{div} feynman-prose
Select an entity and choose one of two ways to change it. **Selected entity** contains
that one entity's JSON; **Apply entity** replaces that object. The numeric fields below
it show resolved values, including an agent type's defaults. **Apply properties**
writes those displayed values back to the selected entity. With multiple entities
selected, these two operations still affect only the last selected one.

Numeric properties recurse through nested objects and arrays, including actuator
parameters. They show units for position (m), velocity (m/s), angle and steering limit
(rad), mass (kg), thrust (N), torque (N·m), spring stiffness (N/m), and damping
(N·s/m). Values must be finite. Native compilation applies the physical range checks.
Strings, booleans, adding new fields, and collision `vertices` require JSON editing.
For example, change `controlled`, `cargo`, an actuator's `kind`, or a visual model name
in JSON. The numeric editor is not a complete scene schema.

**Duplicate selection** offsets copies by +2 m in both coordinates. If both endpoints
of a tether are selected bodies, their connecting tether is duplicated with corrected
body indices. A tether to an unselected body is not copied. **Delete** removes every
selected entity, removes tethers attached to deleted bodies, and remaps remaining
body indices. Removing or reordering gates also changes their visitation sequence.

**Undo** and **Redo** move through scene definitions, recompiling each one. The editor
keeps up to 40 previous scene revisions; a new edit clears the redo branch. They do
not rewind executed motion. The main **↺** reset button resets the current edited
scene using the current seed; it does not reload the shipped preset. Choose the
preset in **Environment** to reload its file and clear editor history.
:::

(sec-lab-scenes-types)=
## Define reusable agents and test their actuators

:::{div} feynman-prose
The supplied scenes carry an `agent_types` catalog. **Agent type** offers the Kestrel
vector rocket, Mite electric kart, Wisp holonomic drone, and Kestrel independent
thrusters, plus any scene-specific types such as the racing kart. A type contains
`physics` and `visual` defaults. A body's `agent_type` selects them; fields on that
body override the defaults. A type's `extends` names another type in the same scene.
The compiler rejects unknown types and inheritance cycles.

Select a body, enter a unique **Template name**, and press **Save selection as agent
type**. The editor saves its resolved physical and visual configuration in this
scene's catalog, omitting its starting position, velocity, and angle. The existing
body remains as it was. Select the new type and use **Agent** to place another copy.
Export JSON to preserve the template; saving it does not update the shared catalog
file or other presets.

Overrides are shallow. Replacing `actuator` replaces that entire nested object, and
replacing a hull or thruster array replaces the whole array. When customizing a kart,
include every actuator setting you intend to preserve. Applying numeric properties
also writes inherited values onto the instance; later changes to the template will
not override those explicit instance values.
:::

### Actions without a planner

:::{div} feynman-prose
Expand **Actuator channels** to see sliders for every compiled channel, labelled by
body index and channel name. Their limits come from the engine. Set the joint action,
then press **Apply action · 1 frame**. This pauses automated control and advances the
authoritative world by one physics frame. Sliders cover all bodies, independently of
which entity is selected, and support registered custom actuators.

Keyboard input applies to the selected body, or the first controlled body if no body
is selected. Selecting passive cargo sends no keyboard thrust to another body. **W/S**
map to thrust, throttle, or body-local x force; **A/D** map to torque or steering;
**Q/E** map to body-local y force; **Space** maps to brake. Values are clamped to
channel bounds: **S** cannot reverse a forward-only vector rocket. Independent
`thruster_0`, `thruster_1`, and similar channels currently require the sliders in the
live lab. Unknown custom channel names also need sliders or a custom keyboard mapping.

While mapped keys are held, the keyboard adapter requests two physics frames at
30 Hz and stops automated control. Releasing all keys stops these requests, so this
mode does not automatically coast forward in time. To inspect coasting, apply a
zero-input frame with the sliders. Keyboard input is ignored while an input, text
area, or select element has focus. Manual actions return the display to the live
world; to drive from a replay frame, first use **Continue here**.
:::

(sec-lab-scenes-json)=
## Import, export, and compile a complete scene

:::{div} feynman-prose
**Export JSON ↓** saves the current scene definition, including templates and edits.
It does not save the current body poses or a recording. **Import JSON ↑** reads a
complete scene file as an undoable scene edit. **Edit complete scene JSON** opens the
whole definition; press **Compile scene** to apply it. The scene must contain at least
one body, and the native compiler validates geometry, physical parameters, actuator
definitions, and references. Read the reported error if compilation fails, correct
the JSON, or use Undo to return to a previous definition.

A scene file and a world snapshot solve different problems. The JSON builds the
laboratory; a `.fgcs` snapshot restores a compatible world's mutable state. Editing
the scene changes snapshot compatibility. Use run archives when you need scene,
settings, and recorded motion together; see {doc}`control_lab_replay`.

Here is a complete small kart scene. Paste it into the complete JSON editor, compile,
then use the sliders or keyboard to approach the gate at `[24, 12]`. The omitted
boundary defaults to the rectangular `size`. This example supplies its own type,
so it does not depend on whichever preset you had open.
:::

```json
{
  "version": 1,
  "name": "Kart workshop",
  "task": "navigation",
  "size": [32, 24],
  "physics": {"dt": 0.016666666666666666, "lethal_walls": false},
  "agent_types": {
    "workshop_kart": {
      "label": "Workshop kart",
      "physics": {
        "controlled": true,
        "radius": 0.65,
        "mass": 1,
        "thrust": 12,
        "drag": 0.7,
        "actuator": {
          "kind": "kart",
          "wheelbase": 1,
          "steering_limit": 0.6,
          "lateral_grip": 14,
          "brake_deceleration": 18
        }
      },
      "visual": {"model": "kart", "color": "#c69bd9", "scale": 1}
    }
  },
  "bodies": [{"agent_type": "workshop_kart", "position": [8, 12], "angle": 0}],
  "gates": [{"position": [24, 12], "radius": 2}],
  "evaluation": {"metric": "gates", "target": 1}
}
```

:::{div} feynman-prose
With only one gate, remaining inside it earns another gate count on each frame.
Use several separated gates for an ordered route. The example is an actuator workshop,
not a lap-counting circuit.
:::

(sec-lab-scenes-extension)=
## Add a preset and separate appearance from mechanics

:::{div} feynman-prose
To make an edited scene available in **Environment**, export it and save the file as
`fractal-gas-web/web/lab/scenarios/workshop.json`. Append
`{"id": "workshop", "label": "Kart workshop"}` to `scenario-catalog.json` in the lab
folder. Keep IDs unique, beginning with a lowercase letter and using only lowercase
letters, digits, underscores, or hyphens. Reload the page to fetch the catalog. The
picker and **All preset scenes** experiment option use that catalog automatically.
Include an `evaluation` metric and target when the scene has a useful default success
criterion. `presentation` can choose score and progress readouts; it does not change
reward or termination behavior.

An agent's `visual.model` selects `rocket`, `kart`, `drone`, or the declarative `kit`.
Color, scale, and kit parts affect appearance. Physical `radius` or convex `vertices`,
mass, and the actuator determine collisions and motion. Making a model twice as large
does not double its collision hull. Enable **Collision geometry** to compare them.
Likewise, the angled camera draws a three-dimensional view of planar physics.

The optional `environment.kind` selects a registered environment renderer (`arena`
by default, or `circuit`). Circuit centreline and start decoration belong to this
visual metadata; actual walls still come from `boundary` and `holes`, and targets
from `gates`. If you reshape a circuit, update its decorative guide as well. New
actuators, models, environment renderers, reward logic, and controllers have separate
registration points described in {doc}`control_lab_architecture`. Add behavior at the
appropriate interface so that every planner can continue using the same batch-state
and bounded-action contract.
:::
