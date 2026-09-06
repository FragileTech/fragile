# Vehicle concept art

Two complete design directions for the Fractal Gas laboratory. Each original PNG
contains a hero view and smaller reference views. Click a sheet to open the full image.

**3D models are available in the [Vehicle Workshop](../asset-gallery.html).**
Rotate each model alongside its concept sheet and download the GLB or editable
Blender source. In the lab, use the masthead's **Visual style** selector to switch
the entire scene between futuristic and steampunk. See the
[asset pipeline and runtime contract](../assets/README.md) for regeneration and LODs.

| Vehicle | Futuristic | Steampunk |
| --- | --- | --- |
| Kestrel rocket / thruster tug | [Rocket](futuristic/rocket.png) | [Rocket](steampunk/rocket.png) |
| Mite electric / racing kart | [Kart](futuristic/kart.png) | [Kart](steampunk/kart.png) |
| Wisp survey drone | [Drone](futuristic/drone.png) | [Drone](steampunk/drone.png) |
| Harvester resource truck | [Harvester](futuristic/harvester.png) | [Harvester](steampunk/harvester.png) |

## Futuristic collection

Graphite and alloy, cyan instrumentation and violet energy details.

![Futuristic rocket](futuristic/rocket.png)
![Futuristic kart](futuristic/kart.png)
![Futuristic drone](futuristic/drone.png)
![Futuristic harvester](futuristic/harvester.png)

## Steampunk collection

Brass, copper plumbing, riveted iron, burgundy enamel, amber glass and boilers.
The Harvester takes its broad industrial resource-collector role from the first
Command & Conquer and translates it into steam machinery.

![Steampunk rocket](steampunk/rocket.png)
![Steampunk kart](steampunk/kart.png)
![Steampunk drone](steampunk/drone.png)
![Steampunk harvester](steampunk/harvester.png)

## Use the Harvester in the lab

Run `npm --prefix fractal-gas-web run build:lab` after building the WebAssembly
engines. Open any preset, choose **Edit scene**, set the placement tool to
**Agent**, select **Harvester · steam ore collector**, and click an empty position
in the world. The catalog definition travels with exported scenes.

The Harvester uses throttle, steering and brake channels with heavier ground
vehicle parameters. Its steampunk model includes six wheels, a collecting drum,
hopper, amber cab, copper pressure vessel and exhaust stack. Wheel, front steering
and intake animation use the shared replay-aware model contract. Resource rewards
come from the scene's existing task; boiler pressure and hopper inventory are
concept details, not additional simulated state.

`build:lab` retains the authored Blender collections and also exports the legacy
`../assets/harvester.glb`. The PNGs are design references; they are not texture maps
or exact engineering orthographic drawings. The Blender models preserve their
characteristic forms with geometry optimized for the interactive laboratory.

## Provenance

Generated with the built-in ImageGen tool. Original PNGs are preserved without
resizing or recompression. The complete prompts are in
[futuristic/prompts.json](futuristic/prompts.json) and
[steampunk/prompts.json](steampunk/prompts.json). Vector and independent-thruster
rockets share the Kestrel design; the racing variant shares the Mite design.
