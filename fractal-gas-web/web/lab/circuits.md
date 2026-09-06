# Historical kart circuits

The laboratory includes five video-based reconstructions and the original Violet
Circuit. Choose **Racing** in **Environment**, then choose a circuit with
**Select track**. All use the same Mite R actuator,
vehicle dimensions, grip, steering, braking, integration step, and contact rules.
Difficulty describes geometry, not a change in vehicle behavior.

| Circuit ID | Difficulty | Checkpoints | Driving character | Reference frame |
| --- | --- | ---: | --- | --- |
| `racing` | Easy | 16 | Wide, consistent beginner oval | Original laboratory scene |
| `racing-roots` | Easy | 50 | Broad irregular oval with generous clearance | [Video 13, 0:00](https://www.youtube.com/watch?v=A5EeIxZ07nM&t=0s) |
| `racing-fearless` | Medium | 130 | Deep hairpin and linked direction changes | [Video 04, 0:00](https://www.youtube.com/watch?v=XE8raxj07gA&t=0s) |
| `racing-sepang` | Hard | 151 | Eleven bends, close hairpins, diagonal straight | [Video 03, 0:00](https://www.youtube.com/watch?v=a7JxjwaCvDU&t=0s) |
| `racing-original` | Hard | 133 | Narrow notches, hooked turn, staggered obstacles | [Video 02, 0:05](https://www.youtube.com/watch?v=Lmnah2pPcMk&t=5s) |
| `racing-obstacle-field` | Hard | 128 | Fearless perimeter with a widened obstacle section | [Video 14, 0:00](https://www.youtube.com/watch?v=X-qwUjhKPLQ&t=0s) |

## Inventory and fidelity

The [channel](https://www.youtube.com/@SergioHernandezCerezo/videos) inventory is
recorded in [kart-video-inventory.json](../../tools/kart-video-inventory.json).
It groups the numbered kart series and the two later lap videos by shared layout.
Controller, grip, and collection variants do not create additional circuits.
Labyrinths, caves, branched collection arenas, combat arenas, and asteroid fields
are excluded. The review used titles and representative opening frames; it was
not a frame-by-frame audit of every compilation on the channel.

[Editable traces](../../tools/kart-circuits.json) record source URLs, timestamps,
screen-coordinate points, uniform scale factors, and uncertainty for each layout.
Each generated scene also carries its sources, direction, and uncertainty in
optional `circuit` metadata. Names other than Sepang are descriptive library names,
not claims about historical venue names. Sepang is the **go-kart layout**.

The reconstructions retain visible corner sequences, proportions, infield shapes,
and stationary obstacles. Curves interpolate hand-picked points and suppress
small raster irregularities. Sepang uses a traced centerline with an approximate
constant road width. Obstacle radii and shallow edge notches are estimates.
The obstacle field's ordered route chooses one continuous passage between islands;
other local passages remain physically open. Dimensions are simulation units,
not measured real-world distances. These are approximate reconstructions rather
than recovered original assets.

## Geometry and progression

The renderer and native collision engine consume the same boundary and hole
polygons. The route supplies spawn heading and ordered proximity checkpoints.
Each historical checkpoint disk lies inside the road and is disjoint from the
other checkpoint disks. Violet retains its established adjacent disk overlap for
compatibility. A lap requires the full ordered sequence, ending at the finish;
these are circular proximity zones, not directional timing lines.

The preview reads the active scene's geometry, including imported scenes and
replay archives. Loading a circuit uses the existing reset lifecycle. Metadata is
optional and does not change the native engine interface. Imported circuits
without difficulty metadata display **Unrated**.

## Authoring and validation

From the repository root:

```sh
uv run --script fractal-gas-web/tools/make-racing-scene.py
uv run --script fractal-gas-web/tools/make-racing-scene.py --check
cd fractal-gas-web
npm run test:lab
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-racing-browser.mjs
```

Shapely is pinned for the authoring script only; generated scenes have no new
runtime dependency. The generator validates polygon topology, route clearance,
safe spawn, checkpoint containment, and checkpoint separation before writing.

A deterministic test-only driver applies throttle, steering, and braking through
the native engine, completing two ordered laps on every circuit without contact.
Tests also reject out-of-order progress, restore replay frames, continue restored
runs, compare common vehicle settings, and check active-scene previews. Difficulty
was assigned from corner tightness, clearance, and linked turns, then checked with
these driving trials. Ratings remain qualitative rather than measured human lap
performance. Browser checks cover selection and reset, previews and references,
manual driving, lap readouts, replay, and desktop/tablet layouts.
