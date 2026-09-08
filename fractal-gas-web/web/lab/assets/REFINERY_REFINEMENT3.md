# Refinery architectural refinement

The third refinement gives both refinery styles a stronger processing sequence through five changes to existing geometry:

- Broader pressure vessels and a lower secondary stage, with gauges, cartridges and connecting pipes carried along the same deformation.
- A suspended conveyor tray with an open intake bay beneath its front section.
- Wider bright carrying frames around the dark conveyor and its mounting feet.
- A taller, outward-tipped receiving bin around the existing irregular cargo.
- A broader control housing with a higher front brow and lower rear roof; attached vents, console and side ribs follow its surface.

[Before/after comparison](previews/refinery-refinement3-comparison.jpg)

The shaping runs after the existing bevels are evaluated, so the new profiles do not increase bevel tessellation. Native roots, transforms, metadata and the open 12 × 12 apron remain unchanged. No materials, textures or images were added.

| Asset | Triangles (unchanged) | GLB bytes before → after | Primitives/materials (unchanged) |
|---|---:|---:|---:|
| Futuristic high | 7,228 | 556,260 → 556,256 | 6 / 6 |
| Futuristic low | 2,848 | 253,996 → 253,996 | 6 / 6 |
| Steampunk high | 9,812 | 688,056 → 688,056 | 7 / 7 |
| Steampunk low | 3,576 | 276,680 → 276,672 | 7 / 7 |

Every GLB retains four embedded 512 × 512 images (1,048,576 pixels total). Validation compares the immediate-before files with the final exports, checks all six asset-cost ceilings, exact root metadata and animation metadata, and loads geometry through the production GLTFLoader to verify bounds and apron clearance. The representative draft and final reference views were visually inspected. The standalone generator retains its background-only guard and publishes matching sources and metadata before rendering previews.

Rebuild from `fractal-gas-web`:

```sh
blender --background --python tools/blender/build_refinery_assets.py
```
