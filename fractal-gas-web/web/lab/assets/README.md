# Original laboratory assets

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

### Regeneration

Use the installed Blender MCP's `execute_blender_code` tool to run the following,
substituting the absolute checkout path for `SCRIPT`:

```python
SCRIPT = "/path/to/fragile/fractal-gas-web/tools/blender/build_lab_assets.py"
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
