"""Build the two refinery concepts as editable Blender scenes and standalone GLBs.

The 12 x 12 apron is centered at the origin; machinery lies north of y=6.
Run: blender --background --python tools/blender/build_refinery_assets.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import bpy

SCRIPT = Path(__file__).resolve().with_name("build_lab_assets.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("lab_builder", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class Refinery(lab.Builder):
    def __init__(self, style, low=False):
        super().__init__(style, "refinery", low)
        self.triangle_budget = 6000 if low else 50000

    def build(self):
        self.box("Traversable unloading apron", (0, 0, .025), (12, 12, .05))
        for x in [-4, 0, 4]:
            for y in [-4, 0, 4]:
                self.box("Apron plates", (x, y, .055), (3.85, 3.85, .04), "dark")
            for y in range(-5, 6, 2):
                self.box("Approach markings", (x-1.8, y, .082), (.08, .7, .012), "gold")
        self.box("Machinery foundation", (0, 8.1, .22), (12, 4.1, .44), "trim")
        self.box("Processor hull", (-3.8, 8.1, 1.4), (3.7, 3.6, 2.1), "plate")
        self.box("Receiving hopper bed", (-3.8, 7.6, 2.55), (3.4, 2.6, .22), "dark")
        for x in [-5.5, -2.1]:
            self.box("Hopper side", (x, 7.6, 3), (.16, 2.7, .9), "trim")
        self.box("Hopper rear", (-3.8, 8.95, 3), (3.5, .16, .9), "plate")
        for y in [6.25, 6.55, 6.85, 7.15]:
            self.cyl("Conveyor roller", (-3.8, y, 1.8+(y-6.25)*.6), .17, 3.15, "trim", "x")
        for x in [-4.8,-3.8,-2.8]:
            self.cyl("Resource fragments", (x, 7.7, 2.86), .32, .48, "gold" if self.steam else "energy", r2=.1, segments=6)
        for x in [-.4, 2.3]:
            self.cyl("Tank base", (x, 8.2, .64), 1.13, .4, "dark")
            self.cyl("Processing tank", (x, 8.2, 2.15), .95, 2.7, "plate")
            self.cyl("Tank shoulder", (x, 8.2, 3.65), .95, .35, "trim", r2=.65)
            for z in [1, 2.5, 3.4]:
                self.ring("Pressure band", (x, 8.2, z), .98, .07, "trim")
            self.box("Tank level window", (x, 7.245, 2.2), (.26, .07, 1.9), "energy")
            self.pipe("Transfer pipe", [(x,8.2,3.85),(x,8.2,4.25),(x,9.5,4.25),(x,9.5,.8)], .1, "copper" if self.steam else "trim")
            if self.steam:
                self.cyl("Pressure gauge", (x+.47,7.25,2.7), .2, .1,"dial","y")
            else:
                for dx in [-.72,.72]:
                    self.box("Tank armor rib", (x+dx,7.6,2.1), (.16,.18,2.5),"dark")
        self.box("Control annex", (4.65,8.1,1.15), (2.3,3.3,1.8), "plate")
        for x in [4.1,4.65,5.2]:
            self.box("Control glass", (x,6.43,1.55), (.4,.06,.45), "glass")
        for y in [7.2,7.6,8,8.4,8.8]:
            self.box("Cooling fin", (4.65,y,2.18), (1.8,.13,.16), "dark")
        if self.steam:
            for x in [-5,-2.8]:
                self.cyl("Boiler chimney", (x,9.25,3.3), .26, 2.5, "dark")
                self.ring("Chimney collar", (x,9.25,4.4), .28,.06)
            self.pipe("Apron pump feed", [(-5.8,6.35,.6),(-5.8,7,1.2),(-4.8,7,1.2)], .11)
        else:
            self.box("Angled processor canopy", (-3.8,9.1,3.35), (3.6,.4,.35),"plate")
        for x in [-5.8,5.8]:
            for y in [-5.5,0,5.5]:
                self.box("Apron edge beacon", (x,y,.18), (.22,.3,.3), "trim")
                self.box("Apron light", (x,y,.35), (.17,.22,.05), "energy")
        return self


def build_refinery(style):
    high, low = Refinery(style).build(), Refinery(style, True).build()
    info = {"model":"refinery", "style":style, "apron":[12,12], "machinery_y":[6,10.2]}
    for builder, lod in [(high,"high"),(low,"low")]:
        info[lod] = builder.export(lab.ROOT/style/f"refinery-{lod}.glb")
        lab.studio(builder)
        builder.scene.render.threads_mode = "FIXED"
        builder.scene.render.threads = 4
        builder.scene.cycles.samples = 12
        # Lighting spans the full building rather than vehicle-sized studio bounds.
        for obj in builder.scene.objects:
            if obj.type == "LIGHT":
                obj.location *= 5
                obj.data.energy *= 25
                obj.data.size *= 5
        lab.frame_camera(builder.scene,(3,-4,5))
    bpy.context.window.scene = high.scene
    for view, direction in [("hero",(3,-4,5)),("side",(1,0,.15)),("top",(0,0,1))]:
        lab.frame_camera(high.scene,direction)
        high.scene.render.filepath=str(lab.ROOT/"previews"/style/f"refinery-{view}.png")
        bpy.ops.render.render(write_still=True)
    lab.frame_camera(high.scene,(3,-4,5))
    bpy.data.libraries.write(str(lab.ROOT/"sources"/style/"refinery.blend"), {high.scene,low.scene}, fake_user=True, compress=True)
    (lab.ROOT/style/"refinery-build.json").write_text(json.dumps(info,indent=2)+"\n")
    return info


if __name__ == "__main__":
    for style in ["steampunk","futuristic"]:
        print(build_refinery(style))
