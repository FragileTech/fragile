# Original laboratory assets

## World collections

The [Asset Workshop](../asset-gallery.html) now includes **42 world asset types in
each style**. Choose a family in the Asset menu, compare Detailed and Simplified
geometry, and enable **Shared envelope** to inspect the common dimensions.

Each style's `world-high.glb` and `world-low.glb` contain all 42 individually
addressable asset roots, sharing embedded PBR textures. The editable
[`sources/futuristic/world.blend`](sources/futuristic/world.blend) and
[`sources/steampunk/world.blend`](sources/steampunk/world.blend) files contain
both levels, arranged on a grid. `assetModel` extras identify each root; the
manifest uses `world-{lod}.glb#assetModel` to load it from the shared pack.

The [catalog](world-catalog.json) lists every type and its dimensions: six resource
drops, four ore rocks and their clamp, a gravity reactor, recovery dock and beacon,
four checkpoints/arches, ten arena modules, seven racing modules and seven capture
or motion assets. The world pack supersedes the first `dock`, `gate` and `reactor`
exports in the runtime manifest. Those standalone files remain for older consumers.

### Collision and presentation contract

**Switching style never changes collision geometry.** Native scene body hulls,
pickup radii, arena boundaries, holes and checkpoint positions remain authoritative.
The two designs and their LODs carry identical `collisionEnvelope` metadata. These
are fitting envelopes, not replacement physics hulls: gates retain open apertures,
decorations remain nonphysical, and ore is fitted to its existing native hull bounds.

The styles use different geometry: machined split cases, faceted emitters and inset
lights versus boilers, gears, valves, leather straps, lanterns and copper pipework.
Slate shapes and capture assemblies also differ. Small surface detail uses packed
panel, roughness, normal and emissive mineral maps; service fittings remain editable.

The refinement pass adds coil suspension and hydraulic hoses, layered nose parts,
engine radiator vanes, optical fasteners, pressure distributors, conveyor cleats,
and dock heat exchangers. Static fittings merge into existing material batches.
The steampunk kart has narrow, open wire-spoked wheels and a rear seat hoop;
the futuristic kart retains wide tires, its angular cage and autonomous sensor.
Ore uses outward-facing normals, seam-corrected spherical UVs, branching mineral
fractures, and embedded capture sockets. Mineral maps are 1024 pixels in detailed
packs and 512 pixels in simplified packs. Extra vehicle machinery is detailed-only.

Repeated rails, kerbs and pickups are instanced per mesh/material. Detailed geometry
appears above 120 projected pixels and simplified geometry below 90, with hysteresis.
Per-instance frustum culling compacts visible entries into the submitted batch;
off-screen high-detail geometry does not consume vertex work at close zoom.
The asset cache owns shared GPU resources. Authored materials bypass palette remapping.
Towable ore also changes LOD when zooming, preserving its native asymmetric hull origin.
Flat ground and road surfaces retain the exact scene polygons and use the kit's PBR
surface material; modular scenery is fitted around their existing boundaries.

Custom scenes can place any kit piece through their existing environment metadata:

```json
{"environment":{"kind":"arena","assets":[
  {"model":"gantry","position":[12,18],"size":[0.65,4,2.8],"angle":0},
  {"model":"deposit","position":[5,6]}
]}}
```

These optional placements are decorative. Use ordinary native scene bodies and
boundaries when a prop also needs collision behavior. Omit `size` for its shared
catalog dimensions; both styles occupy the same declared placement envelope.

Capture cables/fittings, three thrust stages, collection ribbons, pickup bursts,
rotor airflow and trailing ribbons derive their poses from native state and actions.
Reactor gimbals use simulation time. Pickup bursts use the native respawn countdown;
all these poses can be reconstructed after a replay seek without event history.

### Regenerate and verify world assets

Run `tools/blender/build_world_assets.py` in Blender, or execute
`build_world("futuristic", render=True)` / `build_world("steampunk", render=True)`
through the Blender MCP after loading the script into a namespace. Use a background
Blender authoring process for the complete pack and preview run; rendering a whole
collection can exceed a single bridge request's timeout. User scenes are preserved.

Family reference renders are `previews/{style}/world-{family}.png`.
After rendering, run `python3 tools/blender/make_contact_sheets.py` (Pillow required)
to regenerate the paired vehicle and world overview images.
`world-build.json` records geometry counts, pack sizes and shared dimensions.
Run `npm run test:lab`: the world suite validates every asset, embedded resources,
triangle limits, shared bounds, different structural geometry and deterministic poses.
The browser integration page also checks world loading, native-state preservation,
resource/effect replay, comparisons, failures and context restoration.

## Blender collections

Open the [Vehicle Workshop](../asset-gallery.html) to rotate each model beside its
concept sheet, inspect side/top views, and download its GLB or editable Blender file.
The lab's masthead **Visual style** control switches the entire scene between
**Futuristic** and **Steampunk**. The preference is saved on this device and also
applies to replay and comparison views. Switching preserves simulation state,
camera, selection and diagnostic layers.

Each collection contains `rocket`, `kart`, `drone`, `harvester`, `dock`, `gate`,
and `reactor`. Every asset has `-high.glb` and `-low.glb` exports in
[`futuristic/`](futuristic/) and [`steampunk/`](steampunk/). Sources live under
[`sources/futuristic/`](sources/futuristic/) and
[`sources/steampunk/`](sources/steampunk/). Each `.blend` contains both editable
levels of detail and a studio camera. The original user scene is not overwritten.

The eight vehicle models are original stylized interpretations of the
[concept sheets](../concepts/README.md): distinct hulls and machinery, closed
rocket canopies, an autonomous sensor pod in the futuristic kart, an empty leather
seat in its steampunk counterpart, four ducted drone rotors, and six-wheel ore
harvesters. They use authored geometry and deterministic packed PBR panel maps;
no third-party mesh or texture downloads are needed. Small wear, seams and rivets
are baked into the maps; larger fittings remain editable geometry. The repository's
MIT license applies.

The vehicle fidelity pass is implemented in
[`vehicle_refinement.py`](../../../tools/blender/vehicle_refinement.py):

| Vehicle | Futuristic refinement | Steampunk refinement |
| --- | --- | --- |
| Rocket | Broad shoulder chines, layered engine armor, swept dorsal fin | Round tapered fuselage, arched canopy frames, stepped pressure-pod intakes |
| Kart | Continuous cockpit armor, rounded autonomous sensor, shallow road tread | Rounded bonnet, radiator grille, padded bucket seat and rounded rear hoop |
| Drone | Armor bridges integrated with open rotor ducts, layered avionics and multiple optical lenses | Brass-edged diamond panels, rotor crowns, pressure dome and return manifold |
| Harvester | Articulated intake casing, armored wheel arches, processing conduits and pitched cab roof | Intake cheek plates, conveyor sprockets, cab mullions and irregular ore cargo |

The main silhouette changes appear in both LODs. Service details remain detailed-only,
and a shared 512-pixel metal roughness map (128 in the crowd LOD) adds surface variation
without introducing material batches. Closed-shell normals are repaired before export,
including mirrored armor and Y-axis wheel geometry. The visual assets retain the same
native collision hulls and simulation controls.
Rocket close-up models use dark tinted alpha glazing without a refraction pass;
their crowd LOD keeps opaque glazing. Futuristic drone armor uses the concept's
graphite finish and a lower hull; the steampunk diamond panels form its outer shell.

### Regeneration

Use the installed Blender MCP's `execute_blender_code` tool to run the following,
substituting the absolute checkout path for `SCRIPT`:

```python
import sys
from pathlib import Path

SCRIPT = "/path/to/fragile/fractal-gas-web/tools/blender/build_lab_assets.py"
sys.path.insert(0, str(Path(SCRIPT).parent))
namespace = {"__file__": SCRIPT, "__name__": "lab_assets"}
exec(compile(open(SCRIPT).read(), SCRIPT, "exec"), namespace)
result = namespace["build_asset"]("steampunk", "harvester", render=True)
```

Run `build_asset(style, model, render=True)` for each pair, or run the script with
`blender --background --python tools/blender/build_lab_assets.py` from
`fractal-gas-web/` to rebuild both complete collections. Rendering uses Cycles.
Each invocation owns a new Blender scene and replaces only its own exported files.
Hero, side and top PNGs are written under [`previews/`](previews/).

`npm run build:lab` packages the pinned Three.js loader and retains these Blender
exports. It does **not** require Blender or regenerate the authored collections.
`*-build.json` records source triangle counts, batched mesh counts and file sizes.

### Runtime contract

- GLBs intentionally retain **+X forward / +Z up**, using `export_yup=False` to
  match this lab. Set a Z-up view when importing into tools that default to Y-up.
- Vehicles fit a nominal 1.52-unit envelope in the XY plane, centered at the origin
  and placed above Z=0. Body radius and `visual.scale` apply the runtime scale.
  Collision hulls remain defined by the scene and native engine.
- `motion` extras on nested pivots retain `wheel`, `steer`, `rotor` and `thrust`
  tags. Wheel radius metadata determines rotation speed after scale normalization.
  Poses derive from simulation time and actions; no wall-clock physics is added.
- Detailed models appear above 120 projected CSS pixels; simplified models appear
  below 90 pixels. Between those limits the current level remains. More than 16
  controlled bodies use simplified, instanced geometry. Limits are 50k/3k triangles
  per vehicle (80k/6k for the Harvester), with embedded textures no larger than 2K.
- `visual.color` colors the identification marker, preserving authored materials.
  Custom registered models and JSON kits retain their factories.
- A shared cache owns both collections. Viewport disposal releases instance buffers
  and local materials without disposing cached geometry or textures used elsewhere.
  Failed loading retains the active collection and exposes a Retry control.

### Verification

Run `npm run test:lab` after `npm run build:lab`. The asset suite validates actual
GLB geometry, packed resources, budgets, bounds, animation metadata and replay,
plus loading races, failed transitions, persistence and resource ownership.

Serve the lab and open [`tests/visual-style.html`](../tests/visual-style.html) for
the real-browser integration checks: both styles across six presets, two concurrent
viewports, deterministic replay, 64 mixed vehicles, context recovery, and a missing
GLB followed by retry. It prints frame-time and draw-count measurements; these are
observations on the current browser and machine, not portable performance promises.

The [world validation report](world-validation.json) records the current renderer,
asset and layout checks, including layouts at 390 and 1440 CSS pixels. The earlier
[vehicle report](validation.json) remains available. Re-run
[`tests/style-responsive.html`](../tests/style-responsive.html) to inspect the
masthead selector and horizontal overflow in the lab, workshop and concept library. The
[vehicle render collection](previews/collections.jpg) shows the final hero views;
each vehicle's side and top views sit beside its hero PNG in `previews/{style}/`.
The [vehicle refinement review](VEHICLE_REVIEW.md) lists the final export budgets,
concept changes and fixed-crowd renderer measurements.

The [world render collection](previews/world-collections.jpg) compares all eight
world families in both styles. The [concept library](../concepts/index.html)
contains the sixteen reference sheets and exact generation prompts beside links
to the corresponding implemented assets.

## Legacy procedural exports

These original GLB models were created for this repository from authored geometry
in `../models.js` and `../visuals/vehicles.js`. They use the repository's MIT license. They contain no
third-party images, meshes, fonts or textures.

| File | Asset |
| --- | --- |
| `kestrel-tug.glb` | Finned rocket with raised glass canopy, twin engine bells and layered exhausts |
| `mite-forager.glb` | Electric kart with rubber tires, luminous hubs, open cockpit, roll cage and spoiler |
| `wisp-drone.glb` | Four ducted rotors, survey eye and rear thruster |
| `harvester.glb` | Original procedural steam ore collector |
| `veined-ore.glb` | Faceted ore body with exposed luminous core |
| `recovery-dock.glb` | Circular recovery platform with indexed rim fixtures |
| `flux-gate.glb` | Circular checkpoint field |
| `gravity-reactor.glb` | Suspended reactor with intersecting field rings |

Regenerate these root-level legacy files with `npm run build:lab` from
`fractal-gas-web/`. Their factories support older consumers and the initial loading
view. The selected Blender collection supplies the loaded laboratory presentation.

Three.js is a rendering dependency under its own MIT license, copied into
`../vendor/LICENSE-three.txt` during the build.

Nested parts retain their motion tags in GLB extras. The runtime registry animates
these tags from simulation state; exports contain geometry, materials and tags,
not baked keyframe animation.

## Unloading refinery

The separate `refinery-high.glb` and `refinery-low.glb` assets in each style
implement the [refinery concepts](../concepts/refinery-prompts.json), generated
with built-in imagegen before Blender authoring. Sources are
`sources/{style}/refinery.blend`; hero, side and top previews are
`previews/{style}/refinery-{view}.png`. Regenerate only this asset with:

```bash
blender --background --python tools/blender/build_refinery_assets.py
```

Both models have a 12×12 traversable apron centered at the origin and processing
machinery north of y=6. Runtime scale is refinery radius / 6. The preset positions
machinery beyond the arena boundary. The high/low budgets are 50,000/6,000
triangles, with packed materials and Z-up axes. The workshop offers both styles,
concept sheets, GLB downloads, and editable Blender sources.
