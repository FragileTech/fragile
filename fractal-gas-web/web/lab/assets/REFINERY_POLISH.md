# Refinery polish

Both styles retain their native high/low roots and the open 12 × 12 unloading apron. The pressure domes now meet broader crown fittings, flared tank pedestals and sloped service feet connect to the foundation, and the hopper cheeks carry the conveyor down to its mounting level. One bank of annex roof louvers becomes attached side stiffeners. The cargo uses the same polygons with deterministic irregular placement and orientation.

The existing packed panel maps use the shared vehicle surface finish through an explicit call. A refinery-local albedo adjustment keeps the large burgundy and pale-alloy jackets readable; narrow rubbed edges, darker seams and controlled roughness distinguish painted armor from metal fittings. No shared surface helper was modified.

[Before/after comparison](previews/refinery-polish-comparison.jpg)

| Asset | Triangles before → after | GLB bytes before → after | Primitives/materials |
|---|---:|---:|---:|
| Steampunk high | 9,812 → 9,812 | 1,214,088 → 688,056 | 7 / 7 |
| Steampunk low | 3,576 → 3,576 | 802,336 → 276,680 | 7 / 7 |
| Futuristic high | 7,228 → 7,228 | 1,248,112 → 556,260 | 6 / 6 |
| Futuristic low | 2,848 → 2,848 | 945,456 → 253,996 | 6 / 6 |

Each GLB still embeds four 512 × 512 images (1,048,576 texture pixels). Triangles, primitives, materials, image counts and texture pixels are unchanged; download sizes fall through deterministic, more compressible finish maps. These are measured asset costs, not an FPS benchmark.

Validation compared every final GLB with a snapshot of its immediate predecessor, checked exact native root names/transforms/extras and animation metadata, loaded the geometry through the production GLTFLoader, and checked the collision envelope and all vertices in the drive-through apron. Read-only Blender MCP CLI inspection preceded draft authoring; both draft LODs were visually reviewed. Final sources contain both detail levels, with hero, side and top previews for each style. Each standalone style starts from a clean background scene to preserve its root names.

Rebuild from `fractal-gas-web` with:

```sh
blender --background --python tools/blender/build_refinery_assets.py
```
