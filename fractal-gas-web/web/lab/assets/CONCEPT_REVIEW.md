# Concept fidelity and rendering costs

This pass updates the 94 active styled Lab assets: eight vehicles, 84 world props
and two refineries, each with detailed and simplified exports. The comparison
baseline is the existing working-tree collection at the start of this pass.

## Visual changes

- Vehicles use darker graphite or aged brass hardware, restrained panel grime,
  edge wear and less uniform roughness. Kart tread is shallower, harvester cargo
  sits lower in its tapered hopper, and drone optics have stronger material contrast.
- Minerals use angular fractures and broad dark facets. Crystals reuse the
  existing mineral maps; scattered piles, horizontal capsules and faceted cores
  follow the concept silhouettes more closely.
- World machinery uses broader covers and less rounded fittings. Lamps have
  slimmer heads, road seams are quieter, and both styles retain distinct metals.
- Refineries have inclined service annexes, loaded hoppers, solid processing
  tanks with narrow sight windows, and perimeter guards outside the unloading
  area. Violet crystals and amber gauges use the existing materials.

The original concept sheets are unchanged. These remain stylized game assets;
the reference art's volumetric lighting and microscopic detail are represented
through simpler geometry and packed surfaces.

## Performance contract

`tests/fixtures/lab-render-budgets.json` freezes the preceding 24 runtime packs.
Every pack must stay within its previous triangle, material-batch, material,
embedded-image, decoded-texture-pixel and download-byte counts. The separate
`world-render-budgets.json` limits individual props as well as pack totals.

Vehicle triangles and batches are unchanged; their 16 exports save 3,403,548
bytes. Refinery detailed triangle counts fall from 19,736 to 7,228 in futuristic
style and from 17,700 to 9,812 in steampunk style. The simplified versions also
shrink. Steampunk refinery batches fall from eight to seven; futuristic stays
at six.

Across all 24 runtime packs, downloads fall from 72,176,488 to 53,287,952 bytes
(26.2% less), and triangles fall from 604,006 to 545,166 (9.7% less). The four
world packs alone save 14,120,960 bytes. These totals cover both styles and both
LODs; they are not the amount downloaded for a single scene.

Native collision envelopes, animation pivots, LOD thresholds, instancing and
renderer settings remain unchanged. Export budgets bound resource costs; browser
frame time also depends on the GPU, viewport, scene and host activity.

## Validation

The Lab build and all 160 tests pass. The 19 vehicle/asset, 43 world and 24 rendering
budget checks were also repeated after the final material and source corrections.
The real-browser integration run passes all 372 checks, including style switching,
replay, 64-vehicle instancing, failure recovery and WebGL context restoration.

The browser run used headless Chromium at device scale factor 0.5 while Blender
rendered previews. Its frame timings are not representative of normal interactive
performance. The fixed 64-vehicle overview and close views submit exactly the same
draw counts and triangles as the preceding collection's measurements. No full-scale
before/after frame-rate claim is made; the initial full-scale run timed out.

[Measured export costs, SHA-256 hashes and browser results](concept-validation.json)
record the checks and their limits. The final export checks were repeated after
the last material and source resaves.

## Review images

[Vehicles](previews/collections.jpg), [vehicle side and top views](previews/vehicle-orthographic.jpg),
[world families](previews/world-collections.jpg), and [refineries](previews/refinery-collection.jpg)
are generated from the corresponding editable Blender sources. Open the
[Asset Workshop](../asset-gallery.html) to compare the runtime assets with their
concept sheets and inspect both levels of detail.
