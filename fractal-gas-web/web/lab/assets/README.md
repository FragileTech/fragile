# Original laboratory assets

These seven GLB models were created for this repository from authored geometry
in `../models.js` and `../visuals/vehicles.js`. They use the repository's MIT license. They contain no
third-party images, meshes, fonts or textures.

| File | Asset |
| --- | --- |
| `kestrel-tug.glb` | Finned rocket with raised glass canopy, twin engine bells and layered exhausts |
| `mite-forager.glb` | Electric kart with rubber tires, luminous hubs, open cockpit, roll cage and spoiler |
| `wisp-drone.glb` | Four ducted rotors, survey eye and rear thruster |
| `veined-ore.glb` | Faceted ore body with exposed luminous core |
| `recovery-dock.glb` | Circular recovery platform with indexed rim fixtures |
| `flux-gate.glb` | Circular checkpoint field |
| `gravity-reactor.glb` | Suspended reactor with intersecting field rings |

Regenerate with `npm run build:lab` from `fractal-gas-web/`. Physics hulls come
from the scene JSON. The renderer constructs matching model variants directly;
the GLB exports are provided for reuse in external asset tools.

Three.js is a rendering dependency under its own MIT license, copied into
`../vendor/LICENSE-three.txt` during the build.

Nested parts retain their motion tags in GLB extras. The runtime registry animates
these tags from simulation state; exports contain geometry, materials and tags,
not baked keyframe animation.
