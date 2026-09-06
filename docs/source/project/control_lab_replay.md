(sec-control-lab-replay)=
# Record, replay, and continue experiments

:::{div} feynman-prose
There are two different histories to inspect. The kart followed one physical
trajectory, but the planner considered many trajectories before choosing it.
World replay answers “what moved?” The exploration record answers “what futures
were considered?” A saved physical state and a saved planner checkpoint answer
two further questions: where to restart the world, and where to resume a search.
Keep these objects separate and the recording controls become much easier to use.

Begin with {doc}`control_lab_getting_started` if you have not run the lab yet.
{doc}`control_lab_controls` explains the diagnostics, and
{doc}`control_lab_experiments` explains synchronized comparisons from a shared
root. This page covers the recording controls below the main world view.
:::

(sec-lab-replay-objects)=
## Choose what you want to preserve

:::{div} feynman-added
| Object | What it preserves | How you use it |
|---|---|---|
| **World recording** | Initial world and captured physical frames, applied actions, decision IDs, segment labels, and markers. | Seek or play actual movement; restore a frame with Continue here. |
| **Exploration record** | A recorded decision's search tree, its root snapshot, branch actions, and associated diagnostics. | Inspect alternatives and reconstruct one branch with Replay branch. |
| **World snapshot** | One authoritative native world, including all future-affecting environmental state. | Save state / Load state; requires the matching scene. |
| **Planner checkpoint** | The controller's resumable search state plus its authoritative root and saved settings. | Save planner checkpoint / Load checkpoint; then Step, or Advance Wave population for a Wave checkpoint. |
:::

:::{div} feynman-prose
The world state includes positions, velocities, angles, angular velocities,
body flags, tether connections and lengths, pickup positions and respawn timers,
task counters, environmental randomness, and any registered mutable extension
fields. The scene owns static geometry and physical parameters. Those definitions
are included once in a run archive rather than copied into each recorded frame.

World replay does not restore the planner's population, its internal random
stream, warm-start memory, or wall-clock scheduling. A picture of a kart at a
bend cannot tell you which candidate action sequences its planner was halfway
through evaluating. That is what the separate planner checkpoint is for.
:::

(sec-lab-replay-world)=
## Replay actual world movement

:::{div} feynman-prose
World recording starts with the initial state and captures each executed physics
frame during planned control, real-time control, manual keyboard input, and
single-frame editor actions. It continues when **Tree → Off** is selected.
The world slider counts stored samples, including the initial state and any
explicit restoration samples; its index is not necessarily the simulation tick.

1. Pause a running experiment, then drag the slider under **WORLD REPLAY**.
   Seeking also pauses continuous control. The viewport changes to **WORLD REPLAY**
   and shows the selected frame's bodies, task counters, and visual animation.
2. Press **Play world** to animate the recording. Choose **¼×**, **½×**, **1×**,
   **2×**, or **4×** from the speed selector. **Pause replay** stops playback;
   reaching the last frame also stops it. Playing again at the end starts from
   the beginning.
3. Use the camera, **Follow agent**, and diagnostic layers while watching.
   Playback reads stored states; it does not run physics or ask the planner to
   reconstruct the movement. Rendering can skip displayed samples at higher
   playback speeds without removing them from the recording.
4. Press **Back to live** to return to the paused authoritative world. This
   does not resume control. Press **Run experiment** or **Step** when ready.

The lab displays the matching recorded decision tree and planning diagnostics
when that decision is available. For device-backed recordings it can load an
older tree from storage. A frame without a retained tree still has complete
world motion. Do not interpret a missing search overlay as a missing world state.

Pressing **Run experiment**, **Step**, or applying manual input while looking at
replay returns to the authoritative live world. To make the selected replay frame
the next starting point, use **Continue here** first.
:::

### Continue from a selected frame

:::{div} feynman-prose
Pause replay on the desired frame and press **Continue here**. The worker restores
that frame's packed world state, invalidates pending plans, and appends a segment
named **Continued from replay**. It remains paused. Press **Step** or **Run
experiment** to plan from this restored world.

The earlier recording is retained; continuation appends another segment rather
than deleting the future you just watched. Physical ticks may jump backward at
the cut. Playback presents the stored frames across that cut without inventing a
flight or drive between the two endpoints. Snapshot loads, search-branch replay,
Wave selections, and planner restoration also create labeled segments.

Continuation does not promise the same future controller decisions as the
original run. The world is restored, while the current controller may have its
own episode memory and the live decision counter continues. Use a planner
checkpoint for resumable search, or a fixed-seed comparison from this root for
an explicit comparison of future behavior.
:::

(sec-lab-replay-tree)=
## Inspect thinking traces and replay an alternative

:::{div} feynman-prose
The slider under **EXPLORATION RECORD** selects a recorded decision. Moving it
pauses control and changes the displayed search tree and diagnostics. It does
not by itself move the physical bodies to that decision's root. Use the world
slider when you want the corresponding executed movement.

With **Rollout paths** visible and scene editing closed, click near a recorded
branch position in the viewport to choose a node, or enter its ID in **Node**.
The click picker searches within 1.5 world units of recorded controlled-body
positions. Node IDs belong to the selected tree, so select the decision first.

Press **Replay branch** to restore the tree's native root and execute the action
sequence along the ancestry leading to that node. This changes the authoritative
world. The worker records the branch's physical frames in a new segment named
**Search branch / node …**, then pauses at the resulting world. Use **Play world**
to watch that alternative or **Step** to plan from its endpoint. Branch replay
reconstructs physics; ordinary world playback only reads recorded states.

FMC and direct Wave exploration provide native ancestry trees. iCEM and MPPI
provide candidate clouds and world motion but do not export search trees. Their
lack of branches does not prevent recording or replaying their executed actions.
:::

:::{div} feynman-added
| Tree control | Effect |
|---|---|
| **Pruned** | Record the search, removing orphan leaves while protecting current walkers, elites, and the ancestry they require. This is the default. |
| **Full** | Keep all recorded search nodes within each decision; tree memory grows more quickly. |
| **Off** | Disable native search-tree recording; executed world frames still record. |
| **Keep all decisions** | For an in-memory run, retain decisions until the separate 64 MiB tree budget is reached instead of retaining a recent window. |
:::

:::{div} feynman-prose
Changing **Tree** rebuilds the current scene and clears the current session's
history. Choose it before collecting the run. **Keep all decisions** changes
retention going forward; it cannot bring back decisions already discarded.

By default the in-memory exploration record keeps at most 32 recent decisions
and at most 64 MiB of tree arrays and roots. A single tree larger than that budget
is rejected. With **Keep all decisions**, reaching the budget stops control with
an export message rather than silently evicting old decisions. A device-backed
run stores recorded trees on disk while keeping a bounded recent window in
memory; the Keep all checkbox does not turn that window into unlimited RAM.

The renderer limits path drawing to roughly 50,000 segments. A large tree can
therefore contain more branches than are drawn. This display limit does not prune
the underlying tree or alter native simulation.
:::

(sec-lab-replay-files)=
## Save and open portable files

:::{div} feynman-added
| File | Save / open controls | Contents and boundary |
|---|---|---|
| `.fgcs` | **Save state / Load state** | Compact native world snapshot, with format header, scene fingerprint, and payload checksum. No scene JSON, recording, or planner population. |
| `.fgclab` | **Export run / Open run** for an in-memory recording | Version 2 JSON archive containing scene, controller settings, seed, world motion, segment/event metadata, and retained exploration entries. Older version 1 tree-only archives still import. |
| `.fgcrec` | **Export run / Open run** for a device-backed recording | Version 3 chunked archive with scene/settings metadata, compressed world frames, and stored tree/checkpoint objects. Import creates a new device recording. |
| `.fgcp` | **Save planner checkpoint / Load checkpoint** | Version 1 checkpoint envelope containing scene, controller settings, seed, decision counter, world root, and the selected controller's checkpoint. |
:::

### Save a world snapshot

:::{div} feynman-prose
Pause control and press **Save state** to download the authoritative world's
`.fgcs` file. **Save state does not save the frame currently displayed by replay.**
It asks the simulation worker for its current state. To save a replay frame,
seek to it, press **Continue here**, wait for the restored world to appear, and
then press **Save state**. Saving a snapshot does not itself pause a running
simulation, so pause first when the exact capture point matters.

Use **Load state** with the same compiled scene. Successful loading pauses control,
restores the snapshot, clears pending plans, and creates a **Restored snapshot**
segment. A snapshot contains environmental randomness and task progress but
neither static scene definitions nor planner memory. Scene changes can invalidate
its fingerprint; restore the matching scene JSON or open a complete run archive
when transferring an experiment between sessions.
:::

### Export or reopen a run

:::{div} feynman-prose
Pause control, then press **Export run**. The active recording type determines
the extension: `.fgclab` for memory, `.fgcrec` for device storage. A memory archive
contains only the search decisions still retained, while its world motion covers
the recorded session. Device export waits for queued writes and includes stored
objects. It gathers compressed chunks into the portable file without expanding
the entire recording, but still needs memory for those compressed chunks.

Press **Open run**, choose either supported run file, and wait for the scene to
load. The importer restores the scene and recorded controller settings, opens the
recorded motion at its first frame, and places the paused authoritative world at
the recording's last frame. **Back to live** shows that endpoint; **Continue here**
chooses an earlier sample instead. A version 1 tree-only archive has no executed
world stream to play; inspect its decisions and use Replay branch as appropriate.

Run and checkpoint settings include controller parameters and seed. The exported
settings object does not include the live **Clock** or **Worker threads** selector,
so check those explicitly before continuing an imported experiment. Camera pose,
keyboard state, layer visibility, and playback speed are also interface state,
not restored physical state.

The memory importer accepts archive versions 1 and 2 and limits JSON text to
192 MiB. The device importer accepts version 3, limits the file to 1 GiB, and
checks chunk order, sizes, shapes, and checksums before keeping the imported run.
Motion payloads use CRC32; native `.fgcs` snapshots use their native checksum and
scene validation. These checks catch corruption and incompatibility, not malicious
authorship. Keep the matching engine build for reproducible continuation and
planner checkpoints; exported world motion itself is read directly for playback.
:::

(sec-lab-replay-device)=
## Keep long runs on the device

:::{div} feynman-prose
For a longer experiment, enable **Store long runs on this device · applies on
reset**, then press **↺ Reset**. Toggling the checkbox does not migrate the recording
already in progress. The new run writes to this browser's IndexedDB database;
there is no account or remote upload.

World samples are stored in chunks of 256 frames. Full chunks queue immediately;
incomplete chunks flush every five seconds and when the page becomes hidden.
**Save recording** explicitly flushes the incomplete chunk and waits for writes,
then reports **Recording saved on this device**. Use it before closing a valuable
run. An abrupt close can lose recently unflushed frames. Without device storage,
Save recording tells you to enable it and reset.

Storage uses gzip when browser compression is available and raw bytes otherwise,
with CRC32 checks on decompression. Ordinary scrubbing keeps a target cache of
eight durable chunks, fetching older chunks asynchronously. Unsaved chunks and
protected current/read chunks can temporarily increase memory use. The world
readout distinguishes **KiB resident** from the number of frames **stored**;
resident memory is not the full on-disk archive size.

Press **Saved runs** to flush the current recording and open the device library.
Each entry shows the scene name, stored frame count, and update time. **Open**
loads that recording and closes the dialog. **Delete** removes its metadata and
chunks; deletion is disabled for the currently active recording. Open a different
run or reset first if you intend to delete that one. The dialog's **×** closes it.

Browser storage belongs to an origin: changing host, protocol, or port can show
a different library. It is subject to browser quota and eviction. Export a
`.fgcrec` file for a portable copy, and use Open run to import it elsewhere.
Importing a device archive requires writable local browser storage.
:::

### Recording budgets and write failures

:::{div} feynman-prose
The default in-memory motion recorder has its own 64 MiB payload budget, separate
from the exploration budget. It stops recording/control at capacity and does not
evict early world frames. Its packed frame size is
`4 * (scene_words + action_dim + 1)` bytes: world data, action values, and a decision
ID. The one-kart circuit uses 80 bytes per sample, plus the initial snapshot and
container overhead. Rendering assets are not stored in every frame.

Device recording removes that fixed total motion-payload limit, but bounds pending
resident data. The guard is the larger of 32 MiB and twice the eight-chunk cache
size for the current frame layout. If writes fall behind, the lab pauses with
**Recording storage cannot keep up. Wait for writes, then continue.** Let pending
writes finish and save before continuing. An actual storage write failure marks
that recording as failed; later appends do not silently pretend to persist. Resolve
the storage problem and start a fresh recording. Export previously saved data when
possible. These payload limits do not include all JavaScript, compression,
export-buffer, native, or GPU memory.
:::

(sec-lab-replay-checkpoints)=
## Resume the planner rather than only its world

:::{div} feynman-prose
Press **Save planner checkpoint** after the world is ready. This pauses control
and waits for a complete incremental iteration boundary, then downloads `.fgcp`.
For a device-backed run it also stores the checkpoint object with the recording.
The save operation aligns the planner root with the paused authoritative world.
If real-time planning was targeting a future root, that search can restart at the
paused world before it is checkpointed.

Press **Load checkpoint**, choose the file, and wait for the restoration message.
The lab reloads its scene and controller settings. **Step** then continues the
saved search rather than beginning an unrelated search from the visible pose.
Checkpoints preserve controller-specific data: FMC's population, ancestry, elites,
and random state, or the shooting controllers' partial rollouts, optimizer state,
randomness, and warm-start sequences. Use the same compatible engine/backend and
controller implementation for checkpoint continuation.

A checkpoint made after **Advance Wave population** preserves that active Wave
population. On load, the message explicitly says to use **Advance Wave population**
to continue it. This is different from requesting an ordinary planned Step.
Checkpoints do not contain the entire prior replay timeline; loading starts a
new recording with a labeled restoration segment. Opening a `.fgcrec` with stored
checkpoint objects does not automatically resume one of them: the explicit UI
resume workflow uses the downloaded `.fgcp` file.
:::

(sec-lab-replay-markers)=
## Mark an event and practice a complete replay

:::{div} feynman-prose
Delivery, pickup, gate, and terminal-event counters generate markers when their
values increase during normal captured movement. Segment restorations are labeled
separately so a restored counter is not mistaken for a newly achieved event.
Choose **Jump to event…** to pause control/playback and seek to a marker. The menu
shows the most recent 500 events; earlier frames remain available on the slider.

To annotate a moment, seek to it, type an **Event note**, and press **Add marker**.
While viewing live motion, the marker uses the latest recorded frame. Notes are
limited to 120 characters and default to **Marker** when empty. Markers appear in
exports; flush a device recording after adding notes to persist its metadata.

For a short end-to-end check, select the racing preset, choose reproducible mode
and six Action frames, and reset. Press Step twice. You now have the initial
sample plus twelve captured physical frames. Seek to the first sample: the kart
returns visually to its starting pose and the lap counter is zero. Play the short
recording, then seek back and press Continue here. The tick returns to the selected
world's tick, and the timeline gains a new segment without deleting the first run.

Press Save state to capture that restored starting world. Press Step, then Load
state with that file: the restored pose and checkpoint progress return. Export
the run and reopen it to recover both recorded motion segments. This exercise
checks world restoration; do not use it as a claim that an uncheckpointed planner
must choose the same subsequent actions. For measured alternative futures from
one selected world, continue with {doc}`control_lab_experiments`.
:::
