# Vehicle concept refinement

The eight vehicles have been rebuilt from their concept sheets with revised silhouettes and mechanical assemblies. Native collision hulls, model identifiers, controls and replay remain unchanged.

The main changes are rounded pressure hulls and tinted closed rocket canopies; integrated futuristic drone armor and a steampunk diamond shell; cockpit side armor, a rounded bonnet and bucket seat; and articulated harvester intake housings, wheel arches, conveyor covers and cab framing.

The meshes remain stylized interpretations. Fine surface variation uses packed PBR maps; silhouette details appear in both LODs, while small service fittings use detailed geometry only.

## Export budgets

| Style | Vehicle | Detailed triangles | Simplified triangles | Detailed mesh batches |
| --- | --- | ---: | ---: | ---: |
| Futuristic | Rocket | 11,166 | 1,432 | 9 |
| Futuristic | Kart | 23,620 | 2,636 | 20 |
| Futuristic | Drone | 14,000 | 2,128 | 14 |
| Futuristic | Harvester | 40,954 | 4,356 | 21 |
| Steampunk | Rocket | 13,760 | 2,032 | 10 |
| Steampunk | Kart | 26,672 | 2,810 | 16 |
| Steampunk | Drone | 21,044 | 2,468 | 15 |
| Steampunk | Harvester | 45,628 | 4,616 | 23 |

All vehicles fit the original 1.52-unit authoring envelope before native body scaling. Detailed limits are 50,000 triangles (80,000 for harvesters); simplified limits are 3,000 (6,000 for harvesters). The steampunk harvester’s detailed-only rear fitting was corrected to pass the existing LOD-proportion check.

## Renderer observations

The fixed 64-vehicle fixture retains the previous draw-call counts: 127 for futuristic and 122 for steampunk. Instancing and per-instance visibility culling remain active.

| View | Style | Draw calls | Submitted triangles | Median render submission CPU | Median frame interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Overview | Steampunk | 122 | 216,964 | 9.01 ms | 16.54 ms |
| Close | Steampunk | 41 | 29,162 | 5.46 ms | 29.73 ms |
| Overview | Futuristic | 127 | 182,828 | 9.08 ms | 22.03 ms |
| Close | Futuristic | 45 | 27,638 | 4.30 ms | 21.38 ms |

These observations use 15 sampled frames after warm-up in the local browser, after Blender rendering finished. They include browser scheduling and other host activity; CPU submission timing does not measure asynchronous GPU completion.

## Verification and review

- Full `npm run test:lab` and `npm run build:lab` passed.
- All 19 asset tests passed, including triangle budgets, embedded resources, proportions, motion pivots, outward wheel normals and replay.
- All 372 browser checks and 12 responsive checks passed.
- Hero, side and top renders of both collections were inspected.

Rocket detail uses ordinary tinted alpha glazing, with no refraction pass. The crowd LOD keeps opaque glazing. Shared metal roughness maps add no material batches.

[Open the workshop](../asset-gallery.html) · [Paired vehicle renders](previews/collections.jpg) · [Measurements and export hashes](world-validation.json) · [Regeneration instructions](README.md#regeneration)
