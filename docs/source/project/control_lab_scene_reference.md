(sec-control-lab-scene-reference)=
# Scene JSON reference

:::{div} feynman-prose
A scene file is a recipe for an experiment. It says where the equipment starts, how
it moves, what counts as an event, and how the display describes those events. It
does not contain the motion you have already recorded. Keep that distinction in
mind when editing: **Compile scene** builds a fresh world from this recipe.

Use {doc}`control_lab_scenes` for the complete editor walkthrough and
{doc}`control_lab_tasks` for the task tutorials. This page is the field-by-field
companion: use it when you know what you want to change but need the exact spelling,
units, limits, or interaction with another option. The defaults below are compiler
defaults, not necessarily the values chosen by a shipped preset.
:::

(sec-lab-json-first-scene)=
## Compile a small, self-contained scene

:::{div} feynman-prose
Open **Edit scene**, click **Edit complete scene JSON**, replace the text with this
example, and press **Compile scene**. Click the world after enabling **Keyboard
control**, then use **W** to accelerate, **A/D** to steer, **S** for reverse throttle,
and **Space** to brake. The first target is at `[24, 12]`; the second is at `[8, 12]`.
After visiting both, the display counts one circuit and targets the first again.
The experiment goal is two gate events. Ordinary live driving continues after that
point; a batch experiment can stop successfully there.

The initial position is outside both gates. Two separated gates are deliberate:
with one gate, a stationary vehicle inside it receives another gate event every
frame. Gates detect occupancy of the next target, not a directional crossing of a
finish line. They should also be separated enough that a vehicle cannot occupy
successive targets without moving.
:::

```json
{
  "version": 1,
  "name": "Two-gate kart workshop",
  "task": "navigation",
  "size": [32, 24],
  "bodies": [
    {
      "position": [14, 12],
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
        "yaw_response": 12,
        "brake_deceleration": 18
      },
      "visual": {"model": "kart", "color": "#c69bd9", "scale": 1}
    }
  ],
  "gates": [
    {"position": [24, 12], "radius": 2},
    {"position": [8, 12], "radius": 2}
  ],
  "presentation": {
    "task_label": "Two-gate workshop",
    "score": {"metric": "gates", "label": "Circuits", "divisor": 2},
    "progress": {"metric": "gates", "label": "Next gate", "cycle": 2}
  },
  "evaluation": {"metric": "gates", "target": 2}
}
```

:::{div} feynman-prose
For the longer editor exercises, download the
{download}`foraging starter <../../_static/control_lab/examples/foraging-starter.json>` and
{download}`finished arena <../../_static/control_lab/examples/foraging-finished.json>`, the
{download}`cargo starter <../../_static/control_lab/examples/cargo-starter.json>` and
{download}`finished delivery course <../../_static/control_lab/examples/cargo-finished.json>`,
or the {download}`kart starter <../../_static/control_lab/examples/kart-starter.json>` and
{download}`finished checkpoint course <../../_static/control_lab/examples/kart-finished.json>`.
Use **Import JSON ↑** to load them; use **Export JSON ↓** to save your edited recipe.
:::

(sec-lab-json-world)=
## World, geometry, and numerical settings

:::{div} feynman-prose
Physical coordinates are planar `[x, y]` vectors in metres. Angle zero points along
positive x; positive angles rotate toward positive y. The angled camera adds visual
depth but does not introduce vertical physics. Numbers are finite JSON numbers;
booleans are `true` and `false`, not strings. An omitted optional number uses the
listed default. Use omission rather than `null` when sharing files between the
native compiler and browser renderer.
:::

:::{div} feynman-added
| Root field | Type and default | Meaning and limits |
|---|---|---|
| `version` | Number, `1` | Only version 1 is supported. |
| `name` | String, `"Untitled experiment"` | Human-readable scene name. |
| `description` | Optional string | Text shown beneath the environment controls; defaults to `"Custom continuous-control experiment."` when absent or empty. |
| `task` | String, `"navigation"` | Presentation defaults and the special `"tandem"` formation term; see reward semantics below. It is not a preset-file identifier. |
| `size` | Two numbers, `[64, 44]` | Arena extent in metres; each dimension is in `[4, 4096]`. |
| `boundary` | Array of `[x, y]`; rectangle if omitted | Outer playable polygon. Default corners are `[0,0]`, `[width,0]`, `[width,height]`, `[0,height]`. |
| `holes` | Array of polygon arrays, `[]` | Excluded regions inside the boundary. |
| `bodies` | Array of body objects; required | Between 1 and 4096 bodies, including passive cargo. |
| `agent_types` | Object, `{}` | Up to 256 named reusable body definitions. |
| `bases`, `gates`, `pickups`, `refineries` | Arrays, each `[]` | Cargo-body delivery, ordered checkpoint, food, and vehicle unloading zones. Pickups are limited to 4096 slots. |
| `cargo` | Optional object; disabled when omitted | Enables limited pickup storage on each controlled vehicle. Requires at least one refinery; distinct from body `cargo: true`. |
| `gravity`, `tethers` | Arrays, each `[]` | Force sources and body-to-body springs. |
| `physics`, `rewards` | Objects, `{}` | Numerical integration and reward settings. |
| `formation_distance` | Number, `3` | Desired pair separation in metres, `[0.1, 1000]`; formation interpretation for larger groups appears below. |
| `respawn_seconds` | Number, `4` | Pickup cooldown in simulation seconds, `[0, 10000]`. Cargo uses its own `respawn` flag instead. |
| `presentation`, `evaluation` | Optional objects | Display and batch-experiment success settings. |
| `environment`, `circuit` | Optional objects | Rendering and circuit-preview metadata. |
| `extensions` | Array, `[]` | Definitions for native extensions already registered in the engine build. |
:::

### Polygon rules

:::{div} feynman-prose
Write each ring as at least three non-collinear points, in either winding order.
The compiler closes the ring and normalizes its winding. A repeated final copy of
the first point is accepted, but is unnecessary. Do not repeat adjacent vertices or
cross edges. The native limit is 10000 vertices per boundary ring; the circuit
renderer and preview support at most 4096. Use the smaller limit for rendered tracks.

All boundary and hole coordinates must fit inside `size`. Every hole must lie
inside the outer polygon. Rings cannot intersect or touch, and holes cannot nest.
A hole is solid excluded space, not another room. Body and zone centres must lie in
playable space; place them with clearance from edges, because centre validation does
not guarantee that the entire hull or zone fits. Dynamic hull vertices are a
different kind of geometry: they are local coordinates around a body and must form
a convex polygon.

For example, add `"holes": [[[12, 8], [16, 8], [16, 10], [12, 10]]]` to a sufficiently
large scene to make a rectangular obstruction. First check that no starting body or
zone centre occupies it. An invisible collision obstacle is usually a hole whose
visual outline you have overlooked; turn on **Collision geometry** to inspect it.
:::

### Physics options

:::{div} feynman-added
| `physics` field | Default and supported interval | Effect |
|---|---|---|
| `dt` | `1/60` s; `[0.0001, 0.1]` | Simulation time per frame. In JSON, write a decimal, not the expression `1/60`. |
| `substeps` | Integer `4`; `[1, 32]` | Divide each frame into this many integration/collision substeps. More substeps increase work. |
| `solver_iterations` | Integer `8`; `[1, 32]` | Contact-impulse solver iterations per substep. |
| `lethal_walls` | Boolean `false` | A controlled body's wall collision can mark the whole world dead. |
| `lethal_bodies` | Boolean `false` | A body-body contact involving a controlled body can mark the whole world dead. |
:::

:::{div} feynman-prose
A larger `dt` advances more simulated time per frame; it also changes the numerical
experiment. It does not simply speed up playback. Strong forces, light bodies,
stiff springs, and large steps can be a difficult numerical combination even when
each setting individually passes validation. Begin with the defaults, change one
quantity, and inspect a few manual frames before starting a large planning run.
:::

(sec-lab-json-bodies)=
## Bodies and reusable agent types

:::{div} feynman-prose
Every entry in `bodies` is a dynamic physical body. A passive asteroid still moves,
collides, and responds to gravity; `controlled: false` only removes actuator input.
For fixed obstacles use boundary geometry or holes. Indices used by tethers refer
to this complete array, starting at zero, including passive bodies.
:::

:::{div} feynman-added
| Body field | Default; supported interval | Meaning |
|---|---|---|
| `agent_type` | Optional string | Named entry in this scene's `agent_types`. |
| `position` | `[0, 0]` | Initial world position in metres. Supply an interior point explicitly. |
| `velocity` | `[0, 0]` | Initial world velocity in m/s; components must fit finite float32. |
| `angle` | `0`; `[-10000, 10000]` | Initial orientation in radians. |
| `omega` | `0`; `[-1000, 1000]` | Initial angular velocity in rad/s. |
| `radius` | `0.5`; `[0.01, 100]` | Circle radius in metres when no `vertices` are supplied. |
| `vertices` | Optional polygon | Convex local hull with 3–32 vertices. Overrides circle geometry and derives the physical bounding radius. |
| `mass` | `1`; `[0.001, 100000]` | Kilograms. There is no zero-mass static-body mode. |
| `inertia` | Computed; `[0.000001, 10^12]` | Rotational inertia in kg·m². Circle default is `mass × radius² / 2`; polygon default comes from its geometry. |
| `drag` | `0.15`; `[0, 100]` | Linear velocity decay rate in s⁻¹. A substep of duration `h` multiplies velocity by `exp(-drag × h)`. |
| `angular_drag` | `2`; `[0, 100]` | Angular velocity decay rate in s⁻¹. |
| `thrust` | `12`; `[0, 100000]` | Force scale in newtons for vector, kart, and holonomic actuators. |
| `torque` | `8`; `[0, 100000]` | Torque scale in N·m for vector and holonomic actuators. |
| `restitution` | `0.25`; `[0, 1]` | Contact bounce coefficient. |
| `friction` | `0.3`; `[0, 2]` | Contact friction coefficient. |
| `controlled` | Boolean `false` | Compile action channels for this body. |
| `cargo` | Boolean `false` | Eligible for delivery and automatic cargo hooking. |
| `respawn` | Boolean `false` | Respawn delivered cargo at a random clear position throughout the playable map. |
| `actuator` | Vector actuator if omitted | Applied only to controlled bodies; options below. |
| `visual` | Optional object | Appearance metadata; does not replace the physical hull. |
:::

### Hulls and initial conditions

:::{div} feynman-prose
For a small rectangular body, use
`"vertices": [[-0.8, -0.4], [0.8, -0.4], [0.8, 0.4], [-0.8, 0.4]]`.
These are offsets from the body origin before its `angle` rotates them. Centre a
custom hull sensibly around `[0, 0]`; the compiler does not recenter it for you.
Winding is normalized, but a concave hull is rejected. Increasing `visual.scale`
only enlarges the drawing. Change `radius` or `vertices` to enlarge collisions.

On cargo delivery, `respawn: false` makes the cargo inactive. With `true`, the same
cargo body normally respawns immediately at a seeded random clear position
throughout the playable map, outside bases and with clearance from walls, holes,
and active bodies. It retains its configured `angle`, with zero linear and angular
velocity. This deliberately does not restore its initial `velocity` or `omega`.
Tethers targeting that cargo detach. If 256 placement attempts find no clear
position, the cargo stays delivered and inactive and retries on the next frame,
without counting another delivery.
:::

### Type definitions and inheritance

:::{div} feynman-added
| Type field | Meaning |
|---|---|
| Object key | Unique nonempty name used by `agent_type`, such as `"workshop_drone"`. |
| `label` | Display label used by the agent picker; falls back to the type name. |
| `extends` | Optional parent type name in the same scene. Unknown parents and cycles are errors. |
| `physics` | Object of body-field defaults, including `controlled`, shape, and `actuator`. Do not put `agent_type` here; use `extends`. |
| `visual` | Object of appearance defaults. |
:::

:::{div} feynman-prose
Resolution proceeds from parent type to child type to body instance. Overlays are
shallow: a child's `actuator` replaces the parent's entire actuator object. Arrays
such as `vertices`, `thrusters`, and visual `parts` replace the whole inherited array.
Visual object fields are overlaid separately by the browser. Type definitions do
not load other files; an exported scene must carry every type that its bodies use.

This fragment defines a drone type and a heavier child. Include it under the root
`agent_types`, and use the body fragment under `bodies`. The child's mass changes
without changing its actuator. By contrast, adding `"actuator": {"kind": "kart"}`
to the child would replace the whole holonomic actuator.
:::

```json
{
  "agent_types": {
    "workshop_drone": {
      "label": "Workshop drone",
      "physics": {
        "controlled": true,
        "mass": 1,
        "radius": 0.6,
        "actuator": {"kind": "holonomic"}
      },
      "visual": {"model": "drone", "color": "#6ffff1"}
    },
    "heavy_drone": {
      "extends": "workshop_drone",
      "label": "Heavy workshop drone",
      "physics": {"mass": 3}
    }
  },
  "bodies": [{"agent_type": "heavy_drone", "position": [8, 8]}]
}
```

(sec-lab-json-actuators)=
## Actuators and action channels

:::{div} feynman-prose
The actuator translates a bounded action into forces and torques. Changing its
`kind` can change both the meaning and number of controls. **Actuator channels**
shows the actual compiled channels for all controlled bodies in body-array order.
Use its sliders and **Apply action · 1 frame** to test a change before asking a
planner to use it. Channel values are dimensionless and clamped to their bounds.
:::

:::{div} feynman-added
| `actuator.kind` | Channels and bounds | Physical interpretation |
|---|---|---|
| `"vector"` (default) | `thrust` `[0,1]`; `torque` `[-1,1]` | Forward force along the body's heading, plus independent rotational torque. No reverse thrust. |
| `"kart"` | `throttle` `[-1,1]`; `steering` `[-1,1]`; `brake` `[0,1]` | Signed forward force, speed-dependent turning, lateral grip, and longitudinal braking. |
| `"holonomic"` | `force_x`, `force_y`, `torque`, all `[-1,1]` | Two body-local force components, scaled by body `thrust`, plus rotational torque. These are not fixed world axes. |
| `"thrusters"` | `thruster_0`, `thruster_1`, … | One channel per thruster; `[0,1]` normally or `[-1,1]` when reversible. |
:::

### Kart parameters

:::{div} feynman-added
| `actuator` field for `kind: "kart"` | Default; range | Meaning |
|---|---|---|
| `wheelbase` | `1` m; `[0.01,100]` | Turning length scale. Target yaw rate is longitudinal speed divided by wheelbase, multiplied by the tangent of steering angle. |
| `steering_limit` | `0.55` rad; `[0.01,1.4]` | Maximum signed steering angle at full input. |
| `lateral_grip` | `12` s⁻¹; `[0,100]` | Rate that removes lateral sliding velocity. Zero removes this grip force. |
| `yaw_response` | `12` s⁻¹; `[0,100]` | Rate at which angular velocity approaches the steering target. |
| `brake_deceleration` | `16` m/s²; `[0,1000]` | Full-brake longitudinal deceleration; braking is limited to avoid reversing longitudinal velocity in one substep. |
:::

:::{div} feynman-prose
A stationary kart does not turn merely because steering is nonzero: its target yaw
rate depends on longitudinal speed. Reverse speed also reverses that turning
relation. Body `thrust` sets its engine force, while body `torque` does not set kart
steering strength; use `yaw_response`. Braking acts along the kart's forward axis;
`lateral_grip` handles sideways motion. This explains why increasing the brake
setting alone does not cure a sideways slide.
:::

### Independent thrusters

:::{div} feynman-added
| Field in each `actuator.thrusters` entry | Default; range | Meaning |
|---|---|---|
| `position` | `[0,0]`; components `[-100000,100000]` | Body-local mounting point in metres. Its offset creates a torque arm. |
| `direction` | `[1,0]`; components `[-100000,100000]` | Nonzero body-local direction, normalized by the compiler. Its magnitude does not increase force. |
| `force` | `1` N; `[0,100000]` | Full-power force of this thruster; independent of body `thrust`. |
| `reversible` | Boolean `false` | Allow negative channel input to reverse force. |
:::

:::{div} feynman-prose
Supply between 1 and 32 thrusters. In this two-thruster fragment, equal inputs give
forward force with cancelling torque; unequal inputs turn the body. The live
keyboard does not map `thruster_0` and `thruster_1`; use their sliders. Explicit
thruster forces and mounting points determine torque, independently of body `torque`.
:::

```json
{
  "kind": "thrusters",
  "thrusters": [
    {"position": [-0.5, -0.4], "direction": [1, 0], "force": 6},
    {"position": [-0.5, 0.4], "direction": [1, 0], "force": 6}
  ]
}
```

(sec-lab-json-mechanics)=
## Zones, gravity, tethers, and reward

:::{div} feynman-prose
The scene's objects activate its mechanics. Putting food and gates in the same
scene enables both event systems, although the progress incentive gives priority
to the gates. Renaming `task` to `"harvest"` does not create cargo or delivery bases.
The important question is what arrays and body flags you have actually supplied.
:::

### Zones and event rules

:::{div} feynman-added
| Field in `bases`, `gates`, `pickups`, or `refineries` | Default; range | Meaning |
|---|---|---|
| `position` | `[0,0]` | Centre in metres; must be inside playable space. Supply an interior point. |
| `radius` | `1` m; `[0.01,1000]` | Event radius. Editor placement tools choose their own larger or smaller starting values. |
:::

:::{div} feynman-prose
A **base** delivers any active cargo whose centre lies strictly inside its radius.
A tug need not enter the base itself, and a tether is not a delivery prerequisite.
A **gate** counts when a controlled body's centre lies strictly inside its next
zone. Each controlled body has its own ordered gate counter; the displayed `gates`
metric sums their gate events. The sequence wraps around indefinitely and checks
one gate per controlled body per frame. Neither rule adds the body's radius.

A **pickup** is collected when a controlled body approaches within the sum of its
bounding radius and the pickup radius. Only one vehicle gets that slot's event in
a frame. Collection starts a cooldown of `max(dt, respawn_seconds)`. At the end,
the slot moves to a seeded random playable position; after up to 64 unsuccessful
placement attempts it uses the configured initial position. The timer-expiration
frame does not also collect the newly respawned slot. These are simulation timers:
pausing pauses them. Cargo `respawn` has no cooldown; it attempts placement on
delivery and retries on subsequent frames if necessary.
:::

### Vehicle storage and unloading refineries

:::{div} feynman-prose
Root `cargo` configures pickup storage for every controlled vehicle. It is a
different mechanism from a body marked `cargo: true`: a full harvester carries a
quantity internally, while an asteroid is a separate physical body. Refineries
unload vehicle storage; bases deliver separate cargo bodies. Neither zone
substitutes for the other.
:::

:::{div} feynman-added
| Root `cargo` field | Default; supported range | Meaning |
|---|---|---|
| `capacity` | Integer `5`; `[1,10000]` | Number of pickups that fill each controlled vehicle. |
| `unload_seconds` | `2` s; `[0.01,10000]` | Time spent inside a refinery to discharge a complete load. |
| `full_reward` | Current `rewards.pickup`; `[0,10000]` | Additional reward when a vehicle first fills its storage. |
:::

:::{div} feynman-prose
Add this fragment to a scene with controlled vehicles and pickups. Put the refinery
centre in playable space. **Unloading refinery** also places a refinery in the
editor, but the root `cargo` object is what enables vehicle storage.
:::

```json
{
  "cargo": {"capacity": 5, "unload_seconds": 2, "full_reward": 10},
  "refineries": [{"position": [8, 8], "radius": 3}]
}
```

:::{div} feynman-prose
Each collected pickup adds one storage unit. Reaching capacity awards `full_reward`
and switches that vehicle into its return phase. It cannot collect again until
completely empty. A partly filled vehicle still in its collecting phase cannot
unload: it must fill first. An empty `cargo: {}` enables the default five-unit
capacity and therefore still requires a refinery. Omit `cargo` to retain unlimited
pickup collection without a return cycle.

A returning vehicle unloads when its centre is inside or exactly on a refinery's
radius. The amount per frame is `capacity × dt / unload_seconds`, capped by the
remaining load. It receives `rewards.delivery × discharged_amount / capacity` as
reward, so a complete load earns one delivery reward in total. Discharging the
last amount increments the global `deliveries` counter once and returns the vehicle
to collecting. Remaining inside multiple refineries does not multiply the rate.

Leaving the zone pauses discharge while retaining the remaining load and return
phase. Returning resumes it; pickup collection remains blocked throughout the
interruption. Discharge is processed before pickup collection each frame: a
newly filled vehicle cannot start discharging until a later frame, while a vehicle
that becomes empty may collect again in the same frame. A complete unload takes
approximately `unload_seconds`, rounded to frame resolution. Very small remaining
amounts are snapped to zero to finish the cycle.

Refinery capacity is shared without a queue or exclusive docking slot. Storage
changes neither body mass nor physical hull. Each controlled vehicle has four
additional state values: load, return phase, delivered units, and full-load cycles.
The selected-vehicle cargo status helps distinguish **Collecting** from **Return /
unload**. Built-in score and evaluation metrics still use `pickups` and `deliveries`;
there is no built-in metric named `cargo` or `refined`. When a scene combines
refineries with cargo-body delivery bases, `deliveries` counts both completed
vehicle unloads and delivered cargo bodies.
:::

### Gravity fields

:::{div} feynman-added
| Field in `gravity` entry | Default; range | Meaning |
|---|---|---|
| `position` | `[0,0]` | Source location in metres. Unlike zone centres, this need not lie inside playable space. |
| `strength` | `10` m³/s²; `[-100000,100000]` | Positive attracts; negative repels; zero has no force. |
| `softening` | `2` m; `[0.01,1000]` | Smooths the force close to the source. |
:::

:::{div} feynman-prose
For displacement `d = source_position - body_position`, gravity adds acceleration
`strength × d / (|d|² + softening²)^(3/2)`. Every active body receives this acceleration,
regardless of mass. The glowing source is a marker, not automatically a solid
obstacle. Add a hole if you also want an impassable central region. Larger softening
spreads and weakens the near-source field; it is not a collision radius.
:::

### Tethers

:::{div} feynman-added
| Field in `tethers` entry | Default; range | Meaning |
|---|---|---|
| `a` | Integer `0`; valid body index | Source body in the complete `bodies` array. |
| `b` | Integer `-1`; `-1` or valid body index | Initial target; `-1` means initially detached. Cannot equal `a`. |
| `rest_length` | Initial endpoint distance if attached, otherwise `2` m; `[0,10000]` | Unstressed spring length. |
| `stiffness` | `25` N/m; `[0,1000000]` | Spring restoring strength. |
| `damping` | `6` N·s/m; `[0,100000]` | Resistance to relative motion along the tether. |
| `break_force` | `500` N; `[0,10^12]` | Breaks when the required substep impulse magnitude exceeds this force times substep duration. |
| `hook_range` | `2` m; `[0,1000]` | Automatic attachment checks centre distance strictly below this range. |
| `automatic` | Boolean `false` | Detached tether can acquire the closest active cargo. |
:::

:::{div} feynman-prose
For an automatic tug, start with
`{"a": 0, "b": -1, "automatic": true, "hook_range": 3}`.
When it hooks cargo, the runtime rest length becomes the current centre distance,
with a minimum of 0.1 m. Thus `rest_length` is the initial spring setting, not a way
to force every later automatic attachment to a fixed length. Automatic tethers can
reacquire after breaking. Two tugs can attach to the same cargo; there is no
exclusive ownership rule. Keep sources distinct from cargo targets.

A tether is a spring joining centres, not a rope with editable hull attachment
points. Its restoring action can push as well as pull. A nonautomatic tether with
`b: -1` remains detached. A fixed initial target may be any distinct body, not just
cargo. Reordering `bodies` requires updating every tether index; the editor's delete
and duplicate operations do the corresponding remapping for you.
:::

### Reward settings and priority

:::{div} feynman-added
| `rewards` field | Default; range | Contribution |
|---|---|---|
| `progress` | `1`; `[0,1000]` | Multiplies the frame's change in progress potential, including formation shaping. |
| `distance_squared` | `1`; `[0,1000]` | Multiplies the mean squared displacement of controlled vehicles in each physics frame, measured in m². |
| `collision` | `2`; `[0,10000]` | Penalty for qualifying contacts involving controlled bodies; a collision need not be lethal. |
| `pickup` | `10`; `[0,10000]` | Reward per collected food slot. |
| `delivery` | `100`; `[0,10000]` | Reward per cargo-body delivery, or total reward distributed across unloading one full vehicle load. |
| `gate` | `30`; `[0,10000]` | Reward per controlled-body gate event. |
| `formation` | `0.15`; `[0,1000]` | Weight of radial formation error inside the potential for `task: "tandem"`. |
:::

:::{div} feynman-prose
The progress potential is the negative target distance, averaged over controlled
bodies. Its target selection follows this priority: the next **gate**, if any gates
exist; otherwise the nearest **refinery** while a vehicle is in its cargo return
phase; otherwise the nearest active **pickup**, if pickup slots exist; otherwise,
when **bases** exist, either the nearest active cargo or the attached cargo's distance
to the nearest base. Refinery distance is measured to the zone edge and is zero inside it, while
other target distances use centres. A returning vehicle keeps its refinery target
even after partially unloading. Gates still take priority over that return target.
A temporarily empty pickup field does not switch to cargo-body navigation. With no relevant target the distance contribution is zero.

For exact `task: "tandem"` and more than one controlled body, the potential also
subtracts `rewards.formation` times each body's radial error from the group centroid.
The desired radius is `formation_distance / 2`. For two bodies this encourages the
specified pair separation. For three or more it encourages a radius around the
centroid, not every pairwise distance. This term rewards improvement in formation
through the change in potential; it is not a standalone fixed penalty every frame.
Setting `rewards.progress` to zero also disables its effect.

Reward values use the scene's chosen reward scale: event coefficients are reward
per event, and progress converts a distance change into reward. Contact penalties
can accumulate across substeps and repeated contacts; they are not a one-time fee
for an entire crash. Progress is measured before event bookkeeping updates the next
target. Gate, food, and delivery bonuses are then added independently. A large task
counter and a low total reward are therefore compatible.

The `distance_squared` term pays for motion in any direction. In each
physics frame, measure each controlled vehicle's displacement $(\Delta x_i,
\Delta y_i)$ and add

$$
r_{\mathrm{distance},t}
= w_{\mathrm{distance}}\frac{1}{N}
\sum_{i=1}^{N}\left((\Delta x_i)^2+(\Delta y_i)^2\right),
$$

where $N$ is the number of controlled vehicles and $w_{\mathrm{distance}}$ is
`rewards.distance_squared`. Stationary vehicles contribute zero, and cargo bodies
are excluded. With no controlled vehicles, the contribution is zero.
The runtime measures this motion before scene mechanics and respawns, so a respawn
does not earn a travel bonus. It sums these frame rewards; it does not square the
total path length. For example, two frames with a 1 m displacement each contribute
$2w_{\mathrm{distance}}$, whereas one frame with a 2 m displacement contributes
$4w_{\mathrm{distance}}$. Changing the physics timestep therefore changes this
reward's scale, even for the same path and speed. The coefficient converts m² per
frame into reward per frame. Its default is one, so motion is valuable even away
from a target. Set it explicitly to zero to disable this bonus. Older scenes that
omit `distance_squared` also receive the default weight of one.

The Lab's **Reward terms** panel is expanded by default. Use its synchronized
sliders and numeric inputs to set these weights and `cargo.full_reward`, the bonus
for first filling vehicle storage. The same panel exposes **Diversity coefficient**
and **Reward coefficient**, which control FMC selection.

Edits remain pending until you press **Apply settings**. This shared button applies
both the coefficients and reward weights, keeps the current physical state,
discards previous plans, replans, and starts a new recording. If the simulation was
running, it resumes running. Recordings and exports retain the settings actually
applied rather than pending edits. **Reset defaults** stages the engine defaults,
including one for `distance_squared`; press **Apply settings** to use them. The
formation control still acts through the progress potential: increasing it has no
effect while `progress` is zero.
:::

(sec-lab-json-appearance)=
## Appearance and circuit metadata

:::{div} feynman-prose
Physical fields determine the experiment. Appearance fields help you read it.
Change a model's color freely, but use **Collision geometry** after changing its
size or shape: visible wings, wheels, and kit parts do not become new collision
surfaces. The **Visual style** picker is a separate application setting; adding an
arbitrary `style` key to scene JSON does not configure that picker.
:::

### Body visuals and declarative model kits

:::{div} feynman-added
| `visual` field | Values and defaults | Effect |
|---|---|---|
| `model` | `"rocket"`, `"kart"`, `"drone"`, `"harvester"`, or `"kit"` | Selects a registered model. Without a visual object, controlled forage bodies use kart fallback, other controlled bodies rocket fallback. With a visual object but no model, the model factory defaults to rocket. |
| `color` | Color value; use a CSS hex string such as `"#6ffff1"` | Sets the identification ring under authored asset vehicles, whose bodywork retains its palette. Procedural model factories use it as their supplied color; kit parts inherit it unless they specify `color`. Omitted colors use renderer defaults. |
| `scale` | `1`; `(0,100]` | Positive visual scale multiplier. Body display also scales by configured radius (fallback 0.5) divided by 0.8. |
| `parts` | Required for `model: "kit"`; 1–128 entries | Declarative visual components, described below. |
:::

:::{div} feynman-prose
The controlled-body layer uses these vehicle models. Passive cargo is drawn as ore;
it does not become a driven kart merely by adding `visual.model`. The renderer's
radius-based scaling uses the resolved scene field; a native convex hull may derive
a different bounding radius. Supply a sensible radius for appearance as well when
using custom vertices, and inspect the overlay.
:::

:::{div} feynman-added
| Kit part field | Values/default | Meaning |
|---|---|---|
| `shape` | Required: `"box"`, `"sphere"`, `"cylinder"`, `"cone"`, `"ring"` | Primitive geometry. |
| `size` | `[1,1,1]`; positive finite entries ≤100 | Box: three dimensions; sphere: radius; cylinder/cone: radius and height; ring: major and tube radius. One to three entries, with at least the number needed by that shape. |
| `position` | `[0,0,0]` | Three finite local visual coordinates. |
| `rotation` | `[0,0,0]` | Three finite local Euler angles in radians. |
| `color` | Parent color | Per-part color override. |
| `emissive` | False if omitted | Use a glowing material. |
| `motion` | Optional `"thrust"`, `"steer"`, `"wheel"`, `"rotor"` | Visual animation driven by displayed simulation time, speed, and input. |
:::

:::{div} feynman-prose
A minimal visual kit is
`{"model": "kit", "parts": [{"shape": "box", "size": [1.2, 0.6, 0.3]}]}`.
Thrust parts appear with positive thrust and stretch; steering parts rotate around
local z, wheels around local y, and rotors around local z. These animation tags do
not add actuator channels or torque. The visual model has three coordinates because
it is a 3D drawing of a planar body.
:::

### Environment renderer

:::{div} feynman-added
| `environment` field | Values/default | Meaning |
|---|---|---|
| `kind` | `"arena"` by default, or `"circuit"` | Registered environment renderer. Arena needs no additional metadata. |
| `centerline` | Required for circuit: 3–4096 finite `[x,y]` points | Closed decorative route guide; not a collision boundary or checkpoint generator. |
| `width` | Required for circuit: `[1,100]` m | Decorative circuit/start-line width. Actual road bounds remain `boundary` and `holes`. |
| `start.position` | Optional start object; position required within it | Finite `[x,y]` location of the checkered start decoration. |
| `start.angle` | `0` rad | Orientation of start decoration. Supply explicitly for the preview arrow too. |
| `sponsor.position` | Optional sponsor object; position required within it | Finite `[x,y]` placement of the Fragile logo. |
| `sponsor.size` | `11`; `(0,100]` | Side length of its decorative square. |
:::

:::{div} feynman-prose
The circuit renderer requires an explicit valid `boundary` even though native
physics can supply a default rectangle. Its centerline is drawn as a closed loop.
Start-line metadata does not move the agent or award laps. Set body starting poses
and actual `gates` separately. When reshaping a track, update all three: physical
bounds, checkpoint positions, and decorative centerline.
:::

:::{div} feynman-added
| `circuit` field | Meaning |
|---|---|
| `id` | Associates the scene with a known **Select track** catalog entry. It does not load or replace geometry by itself. |
| `name` | Preview name; otherwise scene name or `"Custom circuit"`. |
| `difficulty` | `"Easy"`, `"Medium"`, or `"Hard"`; otherwise `"Unrated"`. |
| `direction` | `"clockwise"` or `"counterclockwise"`; otherwise no direction label. This does not enforce gate-crossing direction. |
| `sources` | Array of source metadata. The preview uses the first entry's `url` when it is a valid HTTPS URL. |
:::

(sec-lab-json-presentation-validation)=
## Score, experiment goals, and error recovery

:::{div} feynman-prose
There are three different questions here. The **score** says what the display is
counting. **Reward** tells the planner how its simulated actions performed. The
**evaluation goal** tells a batch experiment when to call a trial successful. None
of these should be inferred from the scene's title.
:::

### Presentation fields

:::{div} feynman-added
| Field | Meaning and validation |
|---|---|
| `presentation.task_label` | Text above the viewport describing the task. |
| `presentation.score.metric` | Built-ins: `"reward"`, `"deliveries"`, `"pickups"`, `"gates"`. Unknown metric reads as zero. |
| `presentation.score.label` | Score label text. Supply it with a custom score object. |
| `presentation.score.divisor` | Finite positive number, default `1`. Displayed score is `floor(metric / divisor)`. |
| `presentation.progress.metric` | Same built-in metrics, optionally displayed as a cycling next-target label. |
| `presentation.progress.label` | Text such as `"Checkpoint"`. |
| `presentation.progress.cycle` | Required positive integer when progress is present. Display is `(metric modulo cycle) + 1` out of `cycle`. |
:::

:::{div} feynman-prose
Default task presentation is `forage` → food collected, `tandem` → gates crossed,
`navigation` → gates crossed, and `harvest` → cargo deliveries. Other task strings
use the navigation presentation fallback. Overrides are shallow: if you replace
`score`, supply its `metric` and `label` together. For a one-agent 16-gate circuit,
use `score.divisor: 16` and `progress.cycle: 16`; this displays completed laps and
the next checkpoint. With multiple agents, the global gates metric is a sum, so
that same display does not represent each driver's individual lap.

The `reward` presentation metric reads the native most-recent step result; it is
not the accumulated experiment reward. Dividing and flooring also hides fractional
values. For accumulated results, inspect the experiment report described in
{doc}`control_lab_experiments`.
:::

### Experiment success and termination

:::{div} feynman-added
| Field | Meaning |
|---|---|
| `evaluation.metric` | `"deliveries"`, `"pickups"`, `"gates"`, `"survival"`, or `"reward"`. |
| `evaluation.target` | Required finite number strictly greater than zero. Event targets may be fractional, but integer counts reach them only at the next whole event. |
:::

:::{div} feynman-prose
A batch experiment's explicitly supplied goal overrides scene `evaluation`. Without
either, the goal is survival for the episode's maximum frame count. Survival counts
frames advanced during that episode, not seconds. Reward evaluation accumulates
reward during the episode. Event evaluations read world counters, including any
counts already present when starting from a recorded root.

The experiment advances one frame at a time within each chosen action, checking
success, death, and the frame limit. A dead world does not count as successful even
if it also reaches the target. In ordinary live operation, collecting the target
number of objects or finishing a lap does not itself terminate the world. Lethal
collisions can do so. For runtime controls, recording, and comparing trials, see
{doc}`control_lab_controls`, {doc}`control_lab_replay`, and
{doc}`control_lab_experiments`.
:::

### Extensions and capacity limits

:::{div} feynman-prose
JSON cannot install executable behavior. Custom `actuator.kind`, `visual.model`,
`environment.kind`, and extension `kind` values must already be registered in the
respective native or browser application. The native world-extension registry has
no default built-in kinds. Adding `{"extensions": [{"kind": "wind"}]}` to an
ordinary build therefore fails; it does not create wind. See
{doc}`control_lab_architecture` for the registration interfaces.

The native scene file is limited to 8 MiB and JSON nesting to 64 levels. Native
parsing rejects duplicate object keys and non-finite numbers; avoid duplicate keys
even in the browser, where ordinary JSON parsing keeps the last occurrence. The
combined action space is limited to
8192 channels. A registered actuator can expose 1–64 channels and up to 256 initial
state values; its channels need nonempty names and finite increasing bounds. These
plugin limits do not enlarge the built-in 32-thruster limit. There may be at most
64 world extensions, each with at most 4096 initial state values and 4096 observation
values. Combined auxiliary state and extension observations are each limited to
65536 values, and total world state to 100000 float32 words. Enabled vehicle
storage adds four state words per controlled body and counts toward that total. A scene below the body
limit may still exceed one of these combined limits.
:::

### Diagnose a failed edit

:::{div} feynman-prose
Save a known-good export before large JSON changes. Parsing, native compilation,
renderer construction, and experiment-goal validation happen in different places.
The complete JSON dialog catches syntax errors and its missing-body check locally.
A valid JSON edit can close the dialog and subsequently report an asynchronous
worker or renderer error. The dialog closing is not proof that the experiment is
ready. Inspect the application status and wait for the rebuilt world before driving.

Native validation uses the fields it knows; it is not a strict schema rejecting
every unknown key. A misspelled optional field may have no effect rather than
produce an error. Keep exact names from these tables, inspect the exported scene,
and test the resulting behavior. Passing compilation checks parameter ranges and
geometry; it does not guarantee a useful objective or numerical stability.
:::

:::{div} feynman-added
| Symptom or error | What to correct |
|---|---|
| JSON syntax error | Remove comments and trailing commas; quote keys and strings; use decimal numbers rather than formulas. |
| `Scene needs at least one body` / `Scene requires 1–4096 bodies` | Restore a nonempty `bodies` array. For interactive control include at least one controlled body. |
| `Expected a number` / `Expected a boolean` | Replace strings such as `"3"` or `"false"` with correctly typed JSON values. |
| `Scene parameter outside supported range` | Compare every changed field with its table; verify units. |
| `Body centre outside playable region` / `Zone outside playable region` | Move the centre away from outer edges and holes; enlarging `size` alone does not enlarge an explicit boundary. |
| Boundary, ring intersection, or nested-hole error | Remove crossing/touching edges, repeated adjacent vertices, or nested holes. |
| `Dynamic hull must be convex` | Replace the concave local hull with a convex shape; use holes for concave fixed scenery. |
| `Cargo collection requires a refinery zone` | Add an interior `refineries` zone, or omit root `cargo` to disable vehicle storage. |
| Vehicle cannot collect or unload | A full or partially discharged vehicle must finish unloading; a partly filled collecting vehicle must fill first. Inspect the cargo phase and refinery centre-distance condition. |
| Unknown or cyclic agent type | Embed the referenced definition and repair `extends`; a catalog entry in another scene is not available automatically. |
| Unknown actuator/model/environment/extension | Use a built-in name or load a build that registers that feature. |
| Visual scale, kit dimensions, or circuit geometry error | Correct renderer fields as well as native fields; include explicit circuit bounds and centerline. |
| Invalid score divisor/progress cycle | Use a finite positive divisor and positive integer cycle. |
| Invalid episode success criterion | Set a supported evaluation metric and finite positive target. |
| Numerical-limit error after compilation | Return to a stable revision; reduce excessive force or stiffness, increase mass where appropriate, or use a smaller step and test manually. |
| Score advances while nothing moves | Check overlapping/one-gate routes; cargo initially inside a base also counts as a delivery. |
:::

:::{div} feynman-prose
After an unsuccessful applied edit, try **Undo** to recompile the preceding scene
revision. If the application cannot recover, reload and import your saved valid
export, or reselect a shipped environment. Scene Undo restores definitions; it does
not restore the run you had before editing. Export recordings separately before
changing the experiment itself.
:::
