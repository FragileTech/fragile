# Complete collection refinement

For the subsequent concept-fidelity update and its stricter performance budgets,
see [Concept fidelity and rendering costs](CONCEPT_REVIEW.md). The measurements
below record the preceding collection pass.

This pass improves all **94 styled assets**: eight vehicles, 84 world pieces and two refineries. Each has detailed and simplified geometry. The runtime uses 24 self-contained GLBs; the editable Blender sources and regeneration scripts are included.

The styles have different construction, with folded alloy armor and sensor equipment for futuristic scenes, and pressure vessels, brass framing, exposed copper plumbing and open claw mechanisms for steampunk scenes. Native collision hulls remain unchanged. Every world pair retains its shared fitting envelope, and refinery machinery leaves the unloading apron open.

## What changed

| Family | Refinement |
| --- | --- |
| Rockets | Canopy shoulders, pod hatches, pressure supply lines and enamel wing inserts |
| Karts | Folded nose armor, wrapped sensor visor, deeper lamp housings, pressure accumulators and bonnet louvers |
| Drones | Broad segmented rotor shrouds, stepped avionics, pressure equipment and optical housings, with clear rotor apertures |
| Harvesters | Chassis equipment bays, cab framing, hopper panels, scanner and bypass plumbing |
| Collectibles and ore | Octagonal trays, crate hardware, capsule chambers, locking lugs, mineral fractures, sockets and outcrops |
| Reactors, docks and checkpoints | Layered governors, service bays, pressure cages and segmented gate assemblies |
| Arena and racing scenery | Reinforced rails, service hatches, road joints, protected lamps, hoists, pit roofs and framed signs |
| Capture and motion effects | Pressure-column tethers, open brass claws, layered flow particles and inexpensive faceted steam billows |
| Refineries | Exposed inclined conveyors, open rear hoppers, distinct processing tanks, control consoles and service rails |

Surface maps use restrained panel seams, fasteners and wear. The refined vehicle forms stay within the previous rendering workload: small hose cross-sections and fitting bevels were simplified while tire profiles, canopies, motion pivots and major circular silhouettes were retained. Detailed vehicles use about 12–30% fewer triangles than their existing limits; crowd LODs use about 6–8% fewer. The rendering-budget fixtures were not relaxed.

These are stylized interpretations of the concept sheets. Faceted minerals and mesh-based flow effects preserve readability and performance; they do not reproduce the concept art's photorealistic surface detail or volumetric smoke.

## Export budgets

| Style | Model | Detailed triangles | Simplified triangles | Detailed batches |
| --- | --- | ---: | ---: | ---: |
| Futuristic | Rocket | 8,022 | 1,338 | 9 |
| Futuristic | Kart | 17,520 | 2,456 | 20 |
| Futuristic | Drone | 12,384 | 1,960 | 14 |
| Futuristic | Harvester | 31,222 | 4,066 | 21 |
| Futuristic | Refinery | 19,736 | 3,596 | 6 |
| Steampunk | Rocket | 9,672 | 1,904 | 10 |
| Steampunk | Kart | 19,124 | 2,610 | 16 |
| Steampunk | Drone | 15,996 | 2,288 | 15 |
| Steampunk | Harvester | 35,064 | 4,318 | 23 |
| Steampunk | Refinery | 17,700 | 3,640 | 8 |

All previous per-vehicle limits pass for triangles, draw batches, materials, image count, decoded texture pixels and download bytes. World assets remain below 50,000 detailed and 6,000 simplified triangles per model. Refineries remain below 50,000 / 6,000.

## Runtime measurements

| View | Style | Draw calls | Submitted triangles | Median render CPU | Median frame interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Overview | Steampunk | 124 | 225,952 | 6.62 ms | 15.72 ms |
| Close | Steampunk | 42 | 37,578 | 3.40 ms | 17.03 ms |
| Overview | Futuristic | 146 | 183,064 | 8.58 ms | 16.59 ms |
| Close | Futuristic | 48 | 44,754 | 3.95 ms | 20.74 ms |

Measurements use the fixed 64-vehicle fixture, with 15 sampled frames after warm-up, after Blender rendering finished. Frame intervals include browser scheduling and host activity; render CPU timing does not measure asynchronous GPU completion. New world equipment adds world draws even when vehicle batches remain unchanged. The overview measured 124 / 146 draws for steampunk / futuristic, versus 122 / 127 before this pass. Preset scene contents were also being edited independently during the pass; the fixed 64-vehicle fixture is the useful before/after comparison.

## Validation and delivery

- Full `npm --prefix fractal-gas-web run test:lab` and `npm --prefix fractal-gas-web run build:lab` passed.
- All 19 vehicle, 43 world and 2 refinery asset checks passed, including the final exports.
- All 372 browser integration checks and 12 desktop/narrow layout checks passed.
- Actual exported geometry changed for all 168 world style/LOD combinations.
- Source compilation, Ruff and whitespace checks passed.

Hero, side and top vehicle/refinery renders and all eight world family previews in both styles were reviewed. Direction symbols were also checked from the front in the live workshop. Final exports, source files and preview images correspond to the reviewed geometry.

[Asset workshop](../asset-gallery.html) · [Vehicles](previews/collections.jpg) · [Vehicle side/top views](previews/vehicle-orthographic.jpg) · [World kit](previews/world-collections.jpg) · [Refineries](previews/refinery-collection.jpg) · [Measurements and export hashes](world-validation.json) · [Regeneration](README.md#regenerate-and-verify-world-assets)
