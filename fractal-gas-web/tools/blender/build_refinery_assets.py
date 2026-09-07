"""Build the two refinery concepts as editable Blender scenes and standalone GLBs.

The 12 x 12 apron is centered at the origin; machinery lies north of y=6.
Run: blender --background --python tools/blender/build_refinery_assets.py
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy


SCRIPT = Path(__file__).resolve().with_name("build_lab_assets.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("lab_builder", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class Refinery(lab.Builder):
    def __init__(self, style, low=False):
        super().__init__(style, "refinery", low)
        self.triangle_budget = 3596 if low else 19736
        self.seg = 6 if low else 16
        from vehicle_refinement import precision_panel_finish

        precision_panel_finish(self)
        if not self.steam:
            # Mineral cargo shares the process-light material: restrained emission
            # preserves purple facets instead of clipping the hopper to white.
            shader = self.mats["energy"].node_tree.nodes.get("Principled BSDF")
            shader.inputs["Base Color"].default_value = (0.16, 0.022, 0.38, 1)
            shader.inputs["Emission Color"].default_value = (0.20, 0.025, 0.48, 1)
            shader.inputs["Emission Strength"].default_value = 0.18
            shader.inputs["Roughness"].default_value = 0.32

    def build(self):
        self.box("Traversable unloading apron", (0, 0, 0.025), (12, 12, 0.05))
        for x in [-4, 0, 4]:
            for y in [-4, 0, 4]:
                self.box("Apron plates", (x, y, 0.055), (3.85, 3.85, 0.04), "dark")
            for y in range(-5, 6, 2):
                self.box("Approach markings", (x - 1.8, y, 0.082), (0.08, 0.7, 0.012), "gold")
        self.box("Machinery foundation", (0, 8.1, 0.22), (12, 4.1, 0.44), "trim")
        self.box("Processor hull", (-3.8, 8.1, 1.4), (3.7, 3.6, 2.1), "plate")
        self.box("Receiving hopper bed", (-3.8, 7.6, 2.55), (3.4, 2.6, 0.22), "dark")
        for x in [-5.5, -2.1]:
            self.box("Hopper side", (x, 7.6, 3), (0.16, 2.7, 0.9), "trim")
        self.box("Hopper rear", (-3.8, 8.95, 3), (3.5, 0.16, 0.9), "plate")
        for y in [6.25, 6.55, 6.85, 7.15]:
            self.cyl("Conveyor roller", (-3.8, y, 1.8 + (y - 6.25) * 0.6), 0.17, 3.15, "trim", "x")
        for x in [-4.8, -3.8, -2.8]:
            self.cyl(
                "Resource fragments",
                (x, 7.7, 2.86),
                0.32,
                0.48,
                "gold" if self.steam else "energy",
                r2=0.1,
                segments=6,
            )
        for x in [-0.4, 2.3]:
            self.cyl("Tank base", (x, 8.2, 0.64), 1.13, 0.4, "dark")
            self.cyl("Processing tank", (x, 8.2, 2.15), 0.95, 2.7, "plate")
            self.cyl("Tank shoulder", (x, 8.2, 3.65), 0.95, 0.35, "trim", r2=0.65)
            for z in [1, 2.5, 3.4]:
                self.ring("Pressure band", (x, 8.2, z), 0.98, 0.07, "trim")
            self.box("Tank level window", (x, 7.245, 2.2), (0.26, 0.07, 1.9), "energy")
            self.pipe(
                "Transfer pipe",
                [(x, 8.2, 3.85), (x, 8.2, 4.25), (x, 9.5, 4.25), (x, 9.5, 0.8)],
                0.1,
                "copper" if self.steam else "trim",
            )
            if self.steam:
                self.cyl("Pressure gauge", (x + 0.47, 7.25, 2.7), 0.2, 0.1, "dial", "y")
            else:
                for dx in [-0.72, 0.72]:
                    self.box("Tank armor rib", (x + dx, 7.6, 2.1), (0.16, 0.18, 2.5), "dark")
        self.box("Control annex", (4.65, 8.1, 1.15), (2.3, 3.3, 1.8), "plate")
        for x in [4.1, 4.65, 5.2]:
            self.box("Control glass", (x, 6.43, 1.55), (0.4, 0.06, 0.45), "glass")
        for y in [7.2, 7.6, 8, 8.4, 8.8]:
            self.box("Cooling fin", (4.65, y, 2.18), (1.8, 0.13, 0.16), "dark")
        if self.steam:
            for x in [-5, -2.8]:
                self.cyl("Boiler chimney", (x, 9.25, 3.3), 0.26, 2.5, "dark")
                self.ring("Chimney collar", (x, 9.25, 4.4), 0.28, 0.06)
            self.pipe("Apron pump feed", [(-5.8, 6.35, 0.6), (-5.8, 7, 1.2), (-4.8, 7, 1.2)], 0.11)
        else:
            self.box("Angled processor canopy", (-3.8, 9.1, 3.35), (3.6, 0.4, 0.35), "plate")
        for x in [-5.8, 5.8]:
            for y in [-5.5, 0, 5.5]:
                self.box("Apron edge beacon", (x, y, 0.18), (0.22, 0.3, 0.3), "trim")
                self.box("Apron light", (x, y, 0.35), (0.17, 0.22, 0.05), "energy")
        self.refine()
        self.concept_forms()
        lab.repair_vehicle_normals(self)
        return self

    def concept_forms(self):
        """Recover large concept forms with geometry saved from hidden band detail."""
        self.root["refinementRevision"] = "refinery-concept-3"
        for obj in list(self.root.children_recursive):
            if obj.name.startswith((
                "Control annex",
                "Cooling fin",
                "Annex roof monitor",
                "Roof monitor louver",
                "Control glass",
                "Annex corner pier",
                "Resource fragments",
                "Tank armor rib",
                "Violet process cartridge",
                "Process cartridge socket",
                "Process cooling lamella",
                "Pressure band",
            )):
                bpy.data.objects.remove(obj, do_unlink=True)
            elif obj.name.startswith("Machinery foundation"):
                obj.data.materials.clear()
                obj.data.materials.append(self.mats["dark"])
            elif obj.name.startswith("Processing tank") and not self.steam:
                obj.data.materials.clear()
                obj.data.materials.append(self.mats["plate"])
            elif obj.name.startswith("Sloped hopper cheek"):
                obj.data.materials.clear()
                obj.data.materials.append(self.mats["plate"])
        annex = self.loft(
            "Inclined armored control annex",
            [(6.48, 1.13, 0.42, 1.65), (8.45, 1.13, 0.42, 3.02), (9.72, 1.13, 0.42, 2.82)],
            "plate",
        )
        annex.rotation_euler.z = math.pi / 2
        annex.location.x = 4.65
        for x in [4.14, 5.16]:
            panel = self.box(
                "Sloped annex vent recess", (x, 7.49, 2.42), (0.75, 1.74, 0.04), "dark", bevel=0
            )
            panel.rotation_euler.x = 0.608
            for j in range(3 if self.low else 6):
                y = 6.98 + j * 1.10 / (2 if self.low else 5)
                blade = self.box(
                    "Sloped annex vent blade",
                    (x, y, 2.09 + (y - 6.98) * 0.695),
                    (0.65, 0.12, 0.035),
                    "trim",
                    bevel=0,
                )
                blade.rotation_euler.x = 0.608
        for j in range(9 if self.low else 19):
            self.cyl(
                "Hopper coal" if self.steam else "Hopper violet crystal",
                (-4.88 + (j % 5) * 0.52, 7.72 + (j // 5) * 0.29, 2.87 + 0.08 * (j % 3)),
                0.26,
                0.40 + 0.07 * (j % 3),
                "dark" if self.steam else "energy",
                r2=0.06,
                segments=5,
            )
        for x in [-0.4, 2.3]:
            self.cyl("Tank jacket central band", (x, 8.2, 2.5), 0.98, 0.07, "trim")
            if self.steam:
                self.cyl("Boiler amber sight glass", (x, 7.22, 2.15), 0.20, 0.065, "energy", "y")
                self.ring("Boiler sight brass frame", (x, 7.18, 2.15), 0.22, 0.045, "trim", "y")
            else:
                self.box(
                    "Tank process slot surround",
                    (x, 7.20, 2.2),
                    (0.45, 0.12, 2.02),
                    "dark",
                    bevel=0,
                )
                self.box(
                    "Tank violet sight glass",
                    (x, 7.125, 2.2),
                    (0.14, 0.025, 1.70),
                    "energy",
                    bevel=0,
                )
            self.box("Tank service plinth", (x, 6.94, 0.69), (1.18, 0.67, 0.44), "dark", bevel=0)
        # Edge guards are outside the drive-through interior. Inlaid boundaries
        # stay below the apron clearance of 0.12 used by scene collision tests.
        for x in [-5.77, 5.77]:
            for y in [-4.0, -0.05, 3.9]:
                guard = self.loft(
                    "Apron perimeter armored guard",
                    [
                        (y - 1.70, 0.18, 0.04, 0.17),
                        (y - 1.30, 0.21, 0.04, 0.38),
                        (y + 1.30, 0.21, 0.04, 0.38),
                        (y + 1.70, 0.18, 0.04, 0.17),
                    ],
                    "plate",
                )
                guard.rotation_euler.z = math.pi / 2
                guard.location.x = x
        for x in [-2, 2]:
            self.box("Inlaid loading lane", (x, 0, 0.09), (0.055, 11.5, 0.018), "gold", bevel=0)
        for x in [-4, 0, 4]:
            self.box(
                "Apron loading threshold", (x, -5.65, 0.083), (3.60, 0.30, 0.035), "trim", bevel=0
            )
        if self.steam:
            self.box(
                "Furnace amber fire bed",
                (-3.8, 6.124, 0.91),
                (1.92, 0.018, 0.13),
                "energy",
                bevel=0,
            )
        for obj in self.root.children_recursive:
            for modifier in obj.modifiers:
                if modifier.type == "BEVEL":
                    modifier.segments = 1

    def refine(self):
        """Layer the sheet's industrial mechanisms inside the fixed site envelope."""
        self.root["refinementRevision"] = "refinery-concept-2"
        self.root["collisionEnvelope"] = [12, 16.2, 4.8]
        self.root["physicsSource"] = "scene-defined; traversable 12 x 12 apron"
        # The former solid cuboid occupied the conveyor volume. Replace its
        # upper half with a sloped support, and reserve the level tray for the
        # rear receiving pocket so the belt remains visible from above.
        for obj in list(self.root.children_recursive):
            if obj.name.startswith("Processor hull"):
                bpy.data.objects.remove(obj, do_unlink=True)
            elif obj.name.startswith("Receiving hopper bed"):
                obj.location.y = 8.25
                obj.scale.y = 1.40 / 2.60
            elif obj.name.startswith("Hopper side"):
                obj.location.y = 8.175
                obj.scale.y = 1.55 / 2.70
            elif obj.name.startswith("Resource fragments"):
                obj.location.y = 8.2
        support = self.loft(
            "Conveyor sloped support",
            [
                (6.12, 1.77, 0.35, 1.48),
                (7.58, 1.77, 0.35, 2.35),
                (9.88, 1.77, 0.35, 2.37),
            ],
            "plate",
        )
        support.rotation_euler.z = math.pi / 2
        support.location.x = -3.8
        belt = self.box(
            "Exposed inclined conveyor belt",
            (-3.8, 6.86, 2.16),
            (3.0, 1.50, 0.07),
            "dark",
            bevel=0.01,
        )
        belt.rotation_euler.x = math.atan(0.6)
        # Chamfered intake cheeks and a continuous cleated conveyor bridge the
        # original isolated rollers. All machinery remains north of the apron.
        for x in [-5.42, -2.18]:
            cheek = self.loft(
                "Sloped hopper cheek",
                [
                    (6.10, 0.13, 0.70, 1.80),
                    (6.65, 0.17, 0.85, 2.35),
                    (8.70, 0.17, 1.15, 3.28),
                    (9.03, 0.13, 1.20, 3.34),
                ],
                "trim" if self.steam else "plate",
            )
            cheek.rotation_euler.z = math.pi / 2
            cheek.location.x = x
        for j in range(6 if self.low else 11):
            y = 6.23 + j * 1.22 / (5 if self.low else 10)
            z = 1.79 + (y - 6.25) * 0.6
            bar = self.box(
                "Conveyor cross cleat",
                (-3.8, y, z + 0.08),
                (2.96, 0.11, 0.07),
                "trim",
                bevel=0.015,
            )
            bar.rotation_euler.x = math.atan(0.6)
        self.box("Conveyor motor case", (-5.35, 6.42, 1.19), (0.38, 0.48, 0.42), "dark")
        self.cyl("Conveyor drive cap", (-5.57, 6.42, 1.19), 0.19, 0.08, "trim", "x")
        # Continuous tank silhouette changes: steam has pressure jacket courses;
        # future has exposed violet process cells between silver structural ribs.
        for obj in list(self.root.children_recursive):
            if obj.name.startswith("Processing tank"):
                obj.data.materials.clear()
                obj.data.materials.append(self.mats["plate" if self.steam else "dark"])
        for x in [-0.4, 2.3]:
            for z in [1.05, 3.36]:
                self.cyl("Tank armored collar", (x, 8.2, z), 1.04, 0.18, "trim", segments=12)
                self.cyl(
                    "Tank collar gasket", (x, 8.2, z + 0.11), 1.01, 0.035, "dark", segments=12
                )
            self.cyl("Tank crown boss", (x, 8.2, 3.90), 0.32, 0.20, "dark", segments=12)
            self.cyl("Tank crown pressure cap", (x, 8.2, 4.035), 0.22, 0.07, "trim", segments=12)
            for j in range(6):
                a = j * math.tau / 6
                xx, yy = x + 0.95 * math.cos(a), 8.2 + 0.95 * math.sin(a)
                if self.steam:
                    self.pipe("Tank jacket seam", [(xx, yy, 1.16), (xx, yy, 3.25)], 0.038, "trim")
                else:
                    obj = self.box(
                        "Process cell armor spine", (xx, yy, 2.18), (0.18, 0.14, 2.12), "plate"
                    )
                    obj.rotation_euler.z = a
                    if j in {3, 4, 5}:
                        # Place luminous process cartridges between the ribs.
                        aa = a + math.tau / 12
                        cx, cy = x + 0.955 * math.cos(aa), 8.2 + 0.955 * math.sin(aa)
                        self.cyl(
                            "Violet process cartridge",
                            (cx, cy, 2.16),
                            0.16,
                            1.78,
                            "energy",
                            segments=10,
                        )
                        for z in [1.24, 3.08]:
                            self.cyl(
                                "Process cartridge socket",
                                (cx, cy, z),
                                0.19,
                                0.14,
                                "dark",
                                segments=10,
                            )
            if self.steam:
                self.gauge((x + 0.38, 7.18, 2.72), 0.18)
                self.pipe(
                    "Tank bypass manifold",
                    [
                        (x - 0.5, 7.27, 1.10),
                        (x - 0.73, 7.07, 1.10),
                        (x - 0.73, 7.07, 3.13),
                        (x - 0.48, 7.27, 3.13),
                    ],
                    0.075,
                    "copper",
                )
                self.cyl("Pressure tap", (x, 7.13, 1.47), 0.14, 0.16, "trim", "y")
            elif not self.low:
                for z in [1.39, 1.53, 2.88, 3.02]:
                    self.ring("Process cooling lamella", (x, 8.2, z), 0.98, 0.035, "trim")
        # Factory control annex: roof monitors, a readable front console and
        # physically attached cooling modules replace a plain cuboid.
        self.box("Annex front console", (4.65, 6.37, 0.97), (1.78, 0.22, 0.42), "dark")
        for x in [4.08, 4.65, 5.22]:
            self.box("Console metal bezel", (x, 6.238, 1.01), (0.42, 0.045, 0.23), "trim")
            self.box(
                "Console readout",
                (x, 6.208, 1.01),
                (0.29, 0.02, 0.13),
                "dial" if self.steam else "glass",
                bevel=0.012,
            )
        for x in [3.63, 5.67]:
            self.box("Annex corner pier", (x, 8.08, 1.32), (0.19, 3.28, 2.02), "trim")
        for x in [4.14, 5.16]:
            self.box("Annex roof monitor", (x, 8.1, 2.35), (0.65, 2.10, 0.23), "plate")
            for j in range(3 if self.low else 6):
                y = 7.35 + j * 1.5 / (2 if self.low else 5)
                self.box("Roof monitor louver", (x, y, 2.49), (0.51, 0.09, 0.05), "dark", bevel=0)
        # Handrails, side conduits and maintenance risers frame the rear assembly.
        # The apron remains open; low-detail seams remain entirely below 0.12.
        for x in [-5.80, 5.80]:
            self.pipe(
                "Machinery safety handrail",
                [(x, 6.20, 0.45), (x, 6.20, 1.0), (x, 9.85, 1.0), (x, 9.85, 0.45)],
                0.045,
                "trim",
            )
            self.pipe(
                "Foundation service main",
                [(x, 6.30, 0.57), (x, 9.72, 0.57)],
                0.075,
                "copper" if self.steam else "dark",
            )
        for y in [-3.9, 0.1, 4.1]:
            self.box(
                "Apron recessed expansion joint",
                (0, y, 0.080),
                (11.6, 0.028, 0.012),
                "dark",
                bevel=0,
            )
        if self.steam:
            self.box("Furnace mouth frame", (-3.8, 6.245, 1.05), (2.52, 0.17, 0.75), "trim")
            self.box("Furnace mouth recess", (-3.8, 6.146, 1.05), (2.21, 0.035, 0.51), "dark")
            for x in [-4.6, -4.2, -3.8, -3.4, -3.0]:
                self.box("Furnace grate bar", (x, 6.119, 1.05), (0.063, 0.028, 0.44), "trim")
            for x in [-5, -2.8]:
                self.cyl(
                    "Chimney conical rain crown", (x, 9.25, 4.61), 0.35, 0.19, "trim", r2=0.24
                )
                self.cyl("Chimney soot opening", (x, 9.25, 4.713), 0.20, 0.016, "dark")
            self.pipe(
                "Overhead tank equalizer",
                [(-0.4, 8.2, 4.09), (-0.4, 8.2, 4.43), (2.3, 8.2, 4.43), (2.3, 8.2, 4.09)],
                0.10,
                "copper",
            )
        else:
            self.box(
                "Processor violet scan throat", (-3.8, 6.13, 1.07), (2.65, 0.06, 0.10), "energy"
            )
            for x in [-4.9, -2.7]:
                self.box("Hopper armored fin", (x, 9.00, 3.29), (0.28, 0.28, 0.47), "dark")
            self.pipe("Process bus bridge", [(-0.4, 9.35, 3.43), (2.3, 9.35, 3.43)], 0.12, "trim")


def build_refinery(style, render=True):
    high, low = Refinery(style).build(), Refinery(style, True).build()
    info = {"model": "refinery", "style": style, "apron": [12, 12], "machinery_y": [6, 10.2]}
    for builder, lod in [(high, "high"), (low, "low")]:
        info[lod] = builder.export(lab.ROOT / style / f"refinery-{lod}.glb")
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
        lab.frame_camera(builder.scene, (3, -4, 5))
    bpy.context.window.scene = high.scene
    if render:
        for view, direction in [("hero", (3, -4, 5)), ("side", (1, 0, 0.15)), ("top", (0, 0, 1))]:
            lab.frame_camera(high.scene, direction)
            high.scene.render.filepath = str(
                lab.ROOT / "previews" / style / f"refinery-{view}.png"
            )
            bpy.ops.render.render(write_still=True)
    lab.frame_camera(high.scene, (3, -4, 5))
    bpy.data.libraries.write(
        str(lab.ROOT / "sources" / style / "refinery.blend"),
        {high.scene, low.scene},
        fake_user=True,
        compress=True,
    )
    (lab.ROOT / style / "refinery-build.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    for style in ["steampunk", "futuristic"]:
        print(build_refinery(style))
