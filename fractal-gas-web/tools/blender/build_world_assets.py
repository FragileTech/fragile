"""Reproducible world kit: distinct machinery inside shared collision envelopes.

Run build_world(style, render=True) through Blender MCP, or this script in Blender.
Each GLB contains individually addressable asset roots, sharing packed PBR maps.
The .blend lays the editable roots out on a grid after exporting their local poses.
"""

import importlib.util
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
import numpy as np
from vehicle_refinement import precision_panel_finish
from world_machinery_refinement import machinery_polish_after_fit, refine_world_machinery
from world_scenery_refinement import refine_world_scenery, scenery_polish_after_fit
from world_surface_finish import finish_world_materials, finish_world_panels, pack_rgb


SCRIPT = Path(__file__).with_name("build_lab_assets.py")
spec = importlib.util.spec_from_file_location("lab_builder", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
ROOT = lab.ROOT
TAU = math.tau

# Shared by both styles and both LODs. These are visual envelopes, not new physics.
FAMILIES = {
    "collectible-drops": [
        "drop-crystal",
        "drop-nugget",
        "drop-salvage",
        "drop-capsule",
        "drop-core",
        "drop-pile",
    ],
    "ore-rocks": ["ore-small", "ore-medium", "ore-large", "ore-captured", "capture-clamp"],
    "gravity-well": ["reactor"],
    "recovery-dock": ["dock", "beacon"],
    "checkpoints": ["gate", "gate-pylons", "gate-marker", "finish-arch"],
    "arena-scenery": [
        "rail",
        "corner-inner",
        "corner-outer",
        "bollard",
        "rock-obstacle",
        "floor-tile",
        "lamp",
        "utility-column",
        "deposit",
        "gantry",
    ],
    "racing-scenery": [
        "track-straight",
        "track-bend",
        "kerb",
        "guardrail",
        "start-light",
        "pit-station",
        "direction-board",
    ],
    "capture-effects": [
        "tether-fitting",
        "capture-latch",
        "thrust-plume",
        "intake-swirl",
        "pickup-burst",
        "rotor-airflow",
        "path-trail",
    ],
}
SIZES = {
    **{k: [1, 1, 0.8] for k in FAMILIES["collectible-drops"]},
    **{k: [2, 2, 1.6] for k in FAMILIES["ore-rocks"][:4]},
    "capture-clamp": [0.6, 0.5, 0.35],
    "reactor": [2.4, 2.4, 2.8],
    "dock": [2, 2, 0.38],
    "beacon": [0.45, 0.45, 0.9],
    "gate": [0.45, 2, 2],
    "gate-pylons": [0.5, 2, 1.6],
    "gate-marker": [2, 2, 0.12],
    "finish-arch": [0.6, 2, 1.4],
    "rail": [2, 0.3, 0.75],
    "guardrail": [2, 0.3, 0.5],
    "corner-inner": [0.8, 0.8, 0.75],
    "corner-outer": [0.8, 0.8, 0.75],
    "bollard": [0.45, 0.45, 0.65],
    "rock-obstacle": [2, 2, 1.5],
    "floor-tile": [2, 2, 0.08],
    "lamp": [0.5, 0.5, 2],
    "utility-column": [0.8, 0.8, 1.8],
    "deposit": [1.8, 1.8, 1.4],
    "gantry": [0.65, 2, 1.8],
    "track-straight": [4, 3, 0.12],
    "track-bend": [4, 4, 0.12],
    "kerb": [2, 0.4, 0.12],
    "start-light": [0.5, 0.5, 2],
    "pit-station": [2, 1.5, 1.5],
    "direction-board": [0.5, 1.3, 1.4],
    "tether-fitting": [0.5, 0.4, 0.5],
    "capture-latch": [0.6, 0.28, 0.28],
    "thrust-plume": [1, 0.4, 0.4],
    "intake-swirl": [1, 1, 0.3],
    "pickup-burst": [1, 1, 1],
    "rotor-airflow": [1.5, 1.5, 0.4],
    "path-trail": [2, 0.4, 0.12],
}


def stone_maps(style, n=1024):
    y, x = np.mgrid[0:n, 0:n] / n
    rng = np.random.default_rng(972)
    # Broken mineral fractures and broad planes reproduce the concept swatches.
    # Quantized grain compresses well and avoids shimmering white noise.
    noise = np.repeat(np.repeat(rng.integers(0, 8, (n // 4, n // 4)), 4, 0), 4, 1) / 7
    # Periodic domain warping breaks the smooth cellular outlines into mineral
    # fractures without a seam at the spherical UV join. Track two neighbors
    # incrementally rather than allocating a full cells × pixels volume.
    wx = (x + 0.023 * np.sin(TAU * y * 5) + 0.009 * np.sin(TAU * (x * 11 + y * 7))) % 1
    wy = (y + 0.020 * np.sin(TAU * x * 4) + 0.008 * np.sin(TAU * (y * 13 - x * 3))) % 1
    nearest = np.full_like(x, np.inf)
    second = nearest.copy()
    cell_id = np.zeros(x.shape, dtype=np.int32)
    for index, (px, py) in enumerate(rng.random((48, 2))):
        dx = np.minimum(abs(wx - px), 1 - abs(wx - px))
        dy = np.minimum(abs(wy - py), 1 - abs(wy - py))
        distance = dx * dx + dy * dy
        closer = distance < nearest
        second = np.where(closer, nearest, np.minimum(second, distance))
        cell_id[closer] = index
        nearest = np.minimum(nearest, distance)
    gap = np.sqrt(second) - np.sqrt(nearest)
    vein = np.exp(-gap * 440)
    # Concentrated deposits leave stretches of unlit fracture and rough host rock.
    deposits = 0.22 + 0.78 * np.clip(np.sin(TAU * (x * 3 + y * 2)) * 0.7 + 0.5, 0, 1)
    vein *= deposits
    facet = rng.uniform(0.62, 1.28, 48)[cell_id]
    base = np.array([0.13, 0.078, 0.033] if style == "steampunk" else [0.075, 0.062, 0.115])
    glow = np.array([1, 0.42, 0.045] if style == "steampunk" else [0.48, 0.12, 1])
    grain = facet * (0.87 + noise * 0.18)
    color = base * grain[..., None] + vein[..., None] * glow * 0.8
    height = facet * 0.015 - vein * 0.12
    dy, dx = np.gradient(height)
    normal = np.stack([-dx * 3, -dy * 3, np.ones_like(x)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    maps = {}
    for key, data in [
        ("Base Color", color),
        ("Normal", normal * 0.5 + 0.5),
        ("Roughness", np.repeat((0.61 + noise * 0.10)[..., None], 3, 2)),
        ("Emission Color", vein[..., None] * glow),
    ]:
        image = bpy.data.images.new(f"{style} mineral {key}", width=n, height=n)
        if key not in {"Base Color", "Emission Color"}:
            image.colorspace_settings.name = "Non-Color"
        pack_rgb(image, data, steps=31 if key == "Roughness" else 255)
        maps[key] = image
    return maps


def road_maps(style, n=256):
    """A quiet graphite tread with plated seams in the existing three-map budget."""
    maps = lab.panel_maps(style, n=n)
    y, x = np.mgrid[0:n, 0:n] / n
    edge = np.minimum.reduce([x, y, 1 - x, 1 - y])
    seam = (edge < 0.014) | (abs(x - 0.5) < 0.004)
    border = (edge > 0.026) & (edge < 0.037)
    grain = np.sin(y * 373) * 0.006
    traffic = np.exp(-(((x - 0.24) / 0.11) ** 2)) + np.exp(-(((x - 0.76) / 0.11) ** 2))
    base = np.array([0.085, 0.070, 0.053] if style == "steampunk" else [0.060, 0.070, 0.082])
    rgb = np.broadcast_to(base, (n, n, 3)).copy() * (1 + grain + traffic * 0.065)[..., None]
    rgb[seam] *= 0.32
    rgb[border] *= 1.7
    rough = np.repeat((0.69 + grain - traffic * 0.055)[..., None], 3, 2)
    height = seam * -0.045
    dy, dx = np.gradient(height)
    normals = np.stack([-dx * 4, -dy * 4, np.ones_like(x)], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    for key, data in [("Base Color", rgb), ("Roughness", rough), ("Normal", normals * 0.5 + 0.5)]:
        image = maps[key]
        pack_rgb(image, data, steps=63 if key == "Roughness" else 255)
    return maps


def world_metal_finish(builder):
    """Apply the concept metal swatches without adding textures or materials."""
    # Restrained metal values match the aged brass / graphite concept swatches.
    palette = (
        {"trim": (0.27, 0.18, 0.065), "copper": (0.24, 0.075, 0.027), "gold": (0.34, 0.23, 0.085)}
        if builder.steam
        else {
            "trim": (0.105, 0.135, 0.165),
            "copper": (0.065, 0.038, 0.105),
            "gold": (0.32, 0.24, 0.10),
        }
    )
    for key, color in palette.items():
        material = builder.mats[key]
        shader = material.node_tree.nodes["Principled BSDF"]
        material.diffuse_color = (*color, 1)
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Metallic"].default_value = 0.82
        shader.inputs["Roughness"].default_value = 0.34


class WorldBuilder(lab.Builder):
    def __init__(self, style, low=False):
        super().__init__(style, "world", low)
        precision_panel_finish(self)
        world_metal_finish(self)
        finish_world_panels(self)
        self.pack = self.root
        self.roots = {}
        self.seg = 8 if low else 24
        self.triangle_budget = 100000 if low else 1000000
        maps = stone_maps(style, n=512 if low else 1024)
        self.mats["stone"] = lab.material(
            "Veined natural slate", (0.15, 0.17, 0.21), 0.08, 0.86, emission=1.1, texture=maps
        )
        self.mats["crystal"] = lab.material(
            "Amber mineral" if self.steam else "Violet crystal",
            (0.40, 0.17, 0.015) if self.steam else (0.16, 0.035, 0.36),
            0.35,
            0.18,
            0.65,
            texture={key: maps[key] for key in ["Base Color", "Normal", "Emission Color"]},
        )
        self.mats["road"] = lab.material(
            "Riveted iron road" if self.steam else "Quiet graphite road",
            (0.085, 0.07, 0.06) if self.steam else (0.065, 0.075, 0.09),
            0.35,
            0.78,
            texture=road_maps(style),
        )
        self.mats["window"] = lab.material(
            "Pressure glass" if self.steam else "Field glass",
            (0.8, 0.36, 0.065) if self.steam else (0.055, 0.5, 0.75),
            0.05,
            0.12,
        )
        glass = self.mats["window"]
        glass.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.24
        glass.diffuse_color = (*glass.diffuse_color[:3], 0.24)
        self.mats["vapor"] = lab.material(
            "Steam ribbons" if self.steam else "Field ribbons",
            (0.44, 0.39, 0.31) if self.steam else (0.055, 0.40, 0.68),
            0,
            0.8,
            0.6,
        )
        self.mats["vapor"].node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.18
        finish_world_materials(self)

    def assembly(self, name, pos=(0, 0, 0), parent=None, motion=None):
        return self.empty(name, pos, parent or self.root, motion)

    def polygon_band(
        self, name, radius, width, z, height, mat="trim", axis="z", parent=None, start=0, end=TAU
    ):
        n = max(4, round(self.seg * (end - start) / TAU))
        vertices = []
        for t in range(n + 1):
            a = start + (end - start) * t / n
            for r, h in [
                (radius - width / 2, z - height / 2),
                (radius + width / 2, z - height / 2),
                (radius + width / 2, z + height / 2),
                (radius - width / 2, z + height / 2),
            ]:
                p = [r * math.cos(a), r * math.sin(a), h]
                vertices.append([p[2], p[0], p[1]] if axis == "x" else p)
        faces = [(0, 3, 2, 1), tuple(range(n * 4, n * 4 + 4))]
        for i in range(n):
            faces.extend(
                (i * 4 + j, i * 4 + (j + 1) % 4, (i + 1) * 4 + (j + 1) % 4, (i + 1) * 4 + j)
                for j in range(4)
            )
        return self.mesh(name, vertices, faces, mat, parent)

    def crystal(self, pos, radius, height, seed=1, parent=None):
        rng = np.random.default_rng(seed)
        n = 5 if self.low else 6
        v = [
            (math.cos(i * TAU / n) * radius, math.sin(i * TAU / n) * radius, z)
            for z in [0, height * 0.7]
            for i in range(n)
        ]
        v.append((radius * 0.18, -radius * 0.13, height))
        faces = [tuple(reversed(range(n)))]
        for i in range(n):
            j = (i + 1) % n
            faces.extend([(i, j, j + n, i + n), (i + n, j + n, 2 * n)])
        o = self.mesh("Mineral prism", v, faces, "crystal", parent)
        # A few long fractures on each cut plane remain legible at lab zoom.
        for uv in o.data.uv_layers.active.data:
            uv.uv = uv.uv * 0.40
        o.location = pos
        o.rotation_euler = (rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18), seed * 0.7)
        return o

    def rock(self, radius=1, seed=1, pos=(0, 0, 0), parent=None, smooth=False):
        seed += 13 if self.steam else 0
        # An icosphere distributes facets evenly and avoids the UV-sphere's polar spikes.
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2 if self.low or smooth else 3, radius=1
        )
        o = bpy.context.object
        o.name = "Fractured mineral stone"
        o.parent = parent or self.root
        o.location = pos
        o.data.materials.append(self.mats["crystal" if smooth else "stone"])
        directions = [v.co.normalized().copy() for v in o.data.vertices]
        for vertex, direction in zip(o.data.vertices, directions):
            x, y, z = direction
            r = radius * (
                0.87
                + 0.10 * math.sin(x * 5 + seed) * math.cos(y * 6 + z * 4)
                + 0.05 * math.sin(x * 13 + z * 7 + seed)
            )
            if smooth:
                # Suspended cores are cut geodesic crystals, not rounded boulders.
                vertex.co = (x * radius, y * radius, z * radius + radius * 0.65)
            else:
                vertex.co = (x * r, y * r, max(0, z * r * 0.85 + radius * 0.65))
        o.data.update()
        layer = o.data.uv_layers.new()
        for face in o.data.polygons:
            face.use_smooth = False
            uv = []
            for index in face.vertices:
                direction = directions[index]
                uv.append([
                    math.atan2(direction.y, direction.x) / TAU + 0.5,
                    math.acos(max(-1, min(1, direction.z))) / math.pi,
                ])
            if max(p[0] for p in uv) - min(p[0] for p in uv) > 0.5:
                for point in uv:
                    if point[0] < 0.5:
                        point[0] += 1
            for point in uv:
                if point[1] < 1e-6 or point[1] > 1 - 1e-6:
                    others = [q[0] for q in uv if 1e-6 < q[1] < 1 - 1e-6]
                    if others:
                        point[0] = sum(others) / len(others)
            for index, point in zip(face.loop_indices, uv):
                layer.data[index].uv = point
        return o

    def gear(self, pos, r=0.18, axis="z", parent=None):
        self.cyl("Gear hub", pos, r * 0.7, 0.06, "trim", axis, parent=parent)
        count = 8 if self.low else 14
        for j in range(count):
            a = j * TAU / count
            point = [pos[0] + r * math.cos(a), pos[1] + r * math.sin(a), pos[2]]
            if axis == "x":
                point = [pos[0], pos[1] + r * math.cos(a), pos[2] + r * math.sin(a)]
            o = self.box("Gear tooth", point, (r * 0.25, r * 0.25, 0.075), "trim", parent, bevel=0)
            if axis == "x":
                o.rotation_euler.y = math.pi / 2

    def vessel(self, pos, r=0.2, height=0.7, parent=None):
        parent = parent or self.root
        self.cyl(
            "Pressure vessel" if self.steam else "Field cartridge",
            pos,
            r,
            height,
            "copper" if self.steam else "window",
            parent=parent,
        )
        for dz in [-0.5, 0.5]:
            p = (pos[0], pos[1], pos[2] + height * dz)
            self.cyl(
                "Vessel cap", p, r * 1.15, 0.07, "trim" if self.steam else "plate", parent=parent
            )
        for j in range(4):
            a = j * TAU / 4
            self.pipe(
                "Copper return" if self.steam else "Cartridge strut",
                [
                    (pos[0] + r * math.cos(a), pos[1] + r * math.sin(a), pos[2] - height * 0.5),
                    (pos[0] + r * math.cos(a), pos[1] + r * math.sin(a), pos[2] + height * 0.5),
                ],
                0.018,
                "copper" if self.steam else "trim",
                parent,
            )
        if self.steam:
            self.gear((pos[0], pos[1], pos[2] + height * 0.5 + 0.1), r * 0.6, parent=parent)
        else:
            self.cyl(
                "Energy cartridge interior", pos, r * 0.42, height * 0.9, "energy", parent=parent
            )
        if not self.low:
            for dz in [-0.35, 0.35]:
                self.ring(
                    "Service flange",
                    (pos[0], pos[1], pos[2] + height * dz),
                    r * 1.025,
                    0.012,
                    "trim",
                    parent=parent,
                )
            if self.steam:
                self.pipe(
                    "Pressure bypass loop",
                    [
                        (pos[0] + r, pos[1], pos[2] - height * 0.3),
                        (pos[0] + r * 1.4, pos[1], pos[2] - height * 0.3),
                        (pos[0] + r * 1.4, pos[1], pos[2] + height * 0.3),
                        (pos[0] + r, pos[1], pos[2] + height * 0.3),
                    ],
                    0.014,
                    "copper",
                    parent,
                )
            else:
                for j in range(5):
                    self.ring(
                        "Capacitor cooling lamella",
                        (pos[0], pos[1], pos[2] - height * 0.3 + j * height * 0.045),
                        r * 1.04,
                        0.008,
                        "trim",
                        parent=parent,
                    )

    def pedestal(self, r=0.45):
        self.cyl(
            "Low armored base",
            (0, 0, 0.06),
            r,
            0.12,
            "dark",
            segments=8 if not self.steam else self.seg,
        )
        self.polygon_band(
            "Perimeter retaining band",
            r * 0.94,
            0.05,
            0.13,
            0.08,
            "trim" if self.steam else "plate",
        )
        if self.steam:
            self.pipe(
                "Pressure feed", [(-r, 0, 0.15), (-r, -r * 0.5, 0.15), (0, -r * 0.7, 0.15)], 0.03
            )
        else:
            for s in [-1, 1]:
                self.box(
                    "Recessed light guide",
                    (0, s * r * 0.78, 0.15),
                    (r, 0.02, 0.025),
                    "light",
                    bevel=0,
                )

    def collectible(self, kind):
        self.pedestal()
        if kind in {"drop-crystal", "deposit"}:
            for j in range(7):
                a = j * 2.4
                r = 0.24 if j else 0
                self.crystal(
                    (math.cos(a) * r, math.sin(a) * r, 0.16),
                    0.13 if j else 0.18,
                    0.38 + (j % 3) * 0.14,
                    seed=j,
                )
            if self.steam:
                self.vessel((-0.35, 0.05, 0.3), 0.065, 0.2)
                self.gauge((0.18, -0.39, 0.20), 0.075)
        elif kind == "drop-nugget":
            self.rock(0.36, 4, (0, 0, 0.1))
            self.crystal((0, 0, 0.22), 0.27, 0.45, 4)
            if self.steam:
                for y in [-0.2, 0.2]:
                    self.ring(
                        "Nugget retaining strap", (0, y, 0.34), 0.34, 0.035, "trim", axis="y"
                    )
            else:
                for a in [0, TAU / 3, 2 * TAU / 3]:
                    self.box(
                        "Mineral scanner jaw",
                        (0.32 * math.cos(a), 0.32 * math.sin(a), 0.35),
                        (0.1, 0.12, 0.32),
                        "plate",
                    )
        elif kind == "drop-salvage":
            self.box("Salvage crate", (0, 0, 0.37), (0.65, 0.6, 0.52), "dark")
            if self.steam:
                for y in [-0.18, 0.18]:
                    self.box("Leather salvage belt", (0, y, 0.65), (0.67, 0.09, 0.04), "seat")
                for j in range(3):
                    self.gear((-0.2 + j * 0.18, 0, 0.67), 0.1)
                self.vessel((0.2, 0.23, 0.4), 0.06, 0.36)
            else:
                for s in [-1, 1]:
                    self.box("Armored split lid", (s * 0.175, 0, 0.64), (0.3, 0.65, 0.12), "plate")
                    self.box("Crate latch", (s * 0.2, -0.32, 0.35), (0.1, 0.06, 0.17), "light")
        elif kind == "drop-capsule":
            self.vessel((0, 0, 0.42), 0.25, 0.52)
            if self.steam:
                self.gauge((0, -0.27, 0.4), 0.09)
            self.crystal((0, 0, 0.2), 0.12, 0.4)
        elif kind == "drop-core":
            self.rock(0.33, 6, (0, 0, 0.2), smooth=True)
            if self.steam:
                for axis in ["x", "y", "z"]:
                    self.ring("Core cage", (0, 0, 0.45), 0.37, 0.032, "trim", axis)
                self.gear((0, 0, 0.83), 0.10)
            else:
                for j in range(6):
                    a = j * TAU / 6
                    self.box(
                        "Core facet armor",
                        (0.32 * math.cos(a), 0.32 * math.sin(a), 0.45),
                        (0.16, 0.12, 0.24),
                        "plate",
                    )
                self.polygon_band("Core energy band", 0.35, 0.025, 0.45, 0.06, "energy")
        else:
            for j in range(9):
                a = j * 2.4
                r = 0.26 * math.sqrt(j / 8)
                self.rock(0.10 + j % 3 * 0.025, j, (math.cos(a) * r, math.sin(a) * r, 0.13))
                if j % 3 == 0:
                    self.crystal((math.cos(a) * r, math.sin(a) * r, 0.16), 0.065, 0.14, j)

    def clamp(self):
        if self.steam:
            self.box("Winch chassis", (0, 0, 0.1), (0.42, 0.3, 0.18), "dark")
            self.cyl("Cable drum", (0, 0, 0.2), 0.09, 0.34, "copper", axis="y")
            for y in [-0.2, 0.2]:
                self.ring("Forged anchor eye", (0.13, y, 0.2), 0.08, 0.025, "trim", axis="x")
            self.gear((-0.23, 0, 0.12), 0.09, axis="x")
        else:
            self.box("Magnetic capture body", (0, 0, 0.1), (0.4, 0.24, 0.16), "plate")
            self.box("Capture field", (0.14, 0, 0.13), (0.03, 0.23, 0.08), "energy")
            for s in [-1, 1]:
                self.box("Split capture jaw", (0.2, s * 0.16, 0.12), (0.25, 0.06, 0.18), "trim")
                self.box(
                    "Jaw status lamp",
                    (0.22, s * 0.197, 0.16),
                    (0.1, 0.01, 0.025),
                    "light",
                    bevel=0,
                )

    def ore(self, kind):
        if kind == "capture-clamp":
            return self.clamp()
        seed = {
            "ore-small": 2,
            "ore-medium": 5,
            "ore-large": 8,
            "ore-captured": 5,
            "rock-obstacle": 11,
        }.get(kind, 5)
        self.rock(1, seed)
        # The concept's embedded capture socket reads at normal simulation zoom.
        self.cyl("Recessed ore socket", (0.76, 0, 0.74), 0.22, 0.075, "dark", "x")
        self.ring("Ore docking collar", (0.81, 0, 0.74), 0.205, 0.037, "trim", "x")
        self.cyl("Exposed mineral socket", (0.815, 0, 0.74), 0.145, 0.014, "crystal", "x")
        if not self.low:
            for j in range(6):
                a = j * TAU / 6
                self.cyl(
                    "Socket locking stud",
                    (0.86, 0.205 * math.cos(a), 0.74 + 0.205 * math.sin(a)),
                    0.023,
                    0.035,
                    "gold" if self.steam else "trim",
                    "x",
                    segments=6,
                )
        for j in range(3):
            a = j * 2.4
            self.crystal((0.62 * math.cos(a), 0.62 * math.sin(a), 0.42), 0.10, 0.25, j)
        if kind == "ore-captured":
            previous = self.root
            self.root = self.assembly("Removable capture clamp", (0.88, 0, 0.65))
            self.root.rotation_euler.y = math.pi / 2
            self.clamp()
            self.root = previous
        return None

    def reactor(self):
        self.pedestal(1)
        for j in range(8):
            a = j * TAU / 8
            p = (0.82 * math.cos(a), 0.82 * math.sin(a), 0.25)
            o = self.box(
                "Ring bearing mount", p, (0.32, 0.24, 0.22), "dark" if self.steam else "plate"
            )
            o.rotation_euler.z = a
        center = self.assembly("Suspended mineral core", (0, 0, 1.25), motion="world-spin")
        self.rock(0.42, 9, (0, 0, -0.28), center, True)
        self.cyl("Attraction funnel", (0, 0, 0.54), 0.10, 0.75, "vapor", r2=0.35)
        self.polygon_band("Core energy collector", 0.32, 0.08, 0.22, 0.04, "energy")
        for j, axis in enumerate(["x", "y", "z"]):
            pivot = self.assembly("Gimbal bearing", (0, 0, 1.25), motion="world-gimbal")
            pivot["axisIndex"] = j
            pivot["speed"] = [0.11, -0.16, 0.09][j]
            if self.steam:
                self.ring(
                    "Forged containment hoop",
                    (0, 0, 0),
                    0.82 + j * 0.025,
                    0.045,
                    "trim",
                    axis,
                    pivot,
                )
                for a in range(4):
                    self.cyl(
                        "Hoop hinge",
                        (0, math.cos(a * TAU / 4) * 0.82, math.sin(a * TAU / 4) * 0.82),
                        0.065,
                        0.13,
                        "copper",
                        axis="x",
                        parent=pivot,
                    )
            else:
                hoop = self.polygon_band(
                    "Segmented field conductor",
                    0.82,
                    0.085,
                    0,
                    0.08,
                    "plate",
                    axis="x",
                    parent=pivot,
                )
                if axis == "y":
                    hoop.rotation_euler.z = math.pi / 2
                if axis == "z":
                    hoop.rotation_euler.y = math.pi / 2
                self.ring("Inner energy trace", (0, 0, 0), 0.77, 0.012, "energy", axis, pivot)
            if not self.low:
                for j in range(12):
                    a = j * TAU / 12
                    pos = [0, 0.82 * math.cos(a), 0.82 * math.sin(a)]
                    if axis == "y":
                        pos = [pos[1], 0, pos[2]]
                    if axis == "z":
                        pos = [pos[1], pos[2], 0]
                    self.box(
                        "Indexed gimbal hinge",
                        pos,
                        (0.095, 0.095, 0.095),
                        "copper" if self.steam else "trim",
                        pivot,
                        0.006,
                    )
                    self.cyl(
                        "Gimbal bearing pin",
                        pos,
                        0.028,
                        0.12,
                        "trim" if self.steam else "gold",
                        axis,
                        parent=pivot,
                        segments=8,
                    )
        for s in [-1, 1]:
            if self.steam:
                self.vessel((s * 0.82, 0, 0.63), 0.13, 0.75)
                self.pipe(
                    "Overhead pressure feed",
                    [(s * 0.82, 0, 0.3), (s * 0.82, 0, 1.9), (s * 0.4, 0, 2.18)],
                    0.045,
                )
            else:
                self.box("Faceted support tower", (s * 0.88, 0, 0.8), (0.16, 0.26, 1.25), "dark")
                self.box(
                    "Support luminous inset", (s * 0.90, -0.14, 0.85), (0.065, 0.02, 0.75), "light"
                )
        if self.steam:
            self.cyl("Top pressure governor", (0, 0, 2.2), 0.35, 0.18, "copper")
            self.gauge((0, -0.86, 0.35), 0.15)
        else:
            self.polygon_band("Upper field crown", 0.58, 0.10, 2.18, 0.1, "plate")

    def dock(self):
        self.cyl("Clear recovery deck", (0, 0, 0.03), 0.9, 0.06, "road", segments=32)
        self.polygon_band("Recessed machinery skirt", 0.94, 0.22, 0.075, 0.13, "dark")
        self.polygon_band("Deck lower retaining flange", 1.015, 0.065, 0.018, 0.035, "trim")
        for j in range(12):
            a = j * TAU / 12
            o = self.box(
                "Perimeter recovery machinery",
                (0.94 * math.cos(a), 0.94 * math.sin(a), 0.13),
                (0.19, 0.30, 0.18),
                "dark" if self.steam else "plate",
            )
            o.rotation_euler.z = a
            if not self.low:
                for dz in [0.07, 0.16]:
                    p = (1.045 * math.cos(a), 1.045 * math.sin(a), dz)
                    if self.steam:
                        self.pipe(
                            "Skirt pressure manifold",
                            [
                                (1.02 * math.cos(a - 0.10), 1.02 * math.sin(a - 0.10), dz),
                                p,
                                (1.02 * math.cos(a + 0.10), 1.02 * math.sin(a + 0.10), dz),
                            ],
                            0.016,
                            "copper",
                        )
                    else:
                        panel = self.box(
                            "Dock inset heat exchanger", p, (0.015, 0.19, 0.045), "dark", bevel=0
                        )
                        panel.rotation_euler.z = a
                for k in range(4):
                    b = a + (k - 1.5) * 0.043
                    part = self.box(
                        "Skirt cooling fin",
                        (1.055 * math.cos(b), 1.055 * math.sin(b), 0.115),
                        (0.05, 0.012, 0.14),
                        "trim",
                        bevel=0,
                    )
                    part.rotation_euler.z = b
                if j % 3 == 0:
                    self.cyl(
                        "Deck service reservoir",
                        (0.98 * math.cos(a), 0.98 * math.sin(a), 0.22),
                        0.065,
                        0.11,
                        "copper" if self.steam else "energy",
                    )
            if self.steam:
                self.cyl(
                    "Deck valve",
                    (0.94 * math.cos(a), 0.94 * math.sin(a), 0.25),
                    0.04,
                    0.04,
                    "trim",
                )
            else:
                lamp = self.box(
                    "Guidance strip",
                    (0.87 * math.cos(a), 0.87 * math.sin(a), 0.20),
                    (0.03, 0.18, 0.018),
                    "light",
                    bevel=0,
                )
                lamp.rotation_euler.z = a
        self.polygon_band("Deck perimeter rim", 0.97, 0.045, 0.21, 0.05, "trim")
        self.box("Recessed intake housing", (0.65, 0, 0.06), (0.50, 0.45, 0.12), "dark")
        for j in range(7 if not self.low else 4):
            self.cyl("Intake roller", (0.44 + j * 0.065, 0, 0.11), 0.035, 0.4, "trim", axis="y")
        if self.steam:
            self.vessel((-0.74, 0, 0.29), 0.13, 0.30)
        else:
            self.box("Dock control cabinet", (-0.74, 0, 0.25), (0.23, 0.27, 0.33), "plate")
            self.box("Control glass", (-0.60, 0, 0.3), (0.015, 0.20, 0.12), "glass")
        for j in range(4):
            a = j * TAU / 4 + TAU / 8
            self.cyl(
                "Low deck beacon",
                (0.90 * math.cos(a), 0.90 * math.sin(a), 0.28),
                0.04,
                0.12,
                "light",
            )

    def beacon(self, kind):
        self.pedestal(0.23)
        h = 1.4 if kind in {"lamp", "start-light", "utility-column"} else 0.55
        if kind == "start-light":
            self.cyl("Signal mast", (0, 0, h / 2), 0.045, h, "trim")
            self.box("Start signal housing", (0, 0, h), (0.18, 0.24, 0.58), "dark")
            for j in range(3):
                self.cyl(
                    "Signal lens", (0.1, 0, h - 0.18 + j * 0.18), 0.065, 0.025, "light", axis="x"
                )
        elif self.steam:
            self.vessel((0, 0, h * 0.6), 0.13, h * 0.72)
            self.cyl("Lantern chimney", (0, 0, h * 1.03), 0.11, 0.15, "trim", r2=0.045)
            if kind == "utility-column":
                self.gauge((0, -0.16, 0.6), 0.1)
                self.pipe(
                    "Column return pipe",
                    [(0.18, 0, 0.15), (0.18, 0, h * 0.9), (0, 0, h * 0.9)],
                    0.028,
                )
        else:
            self.box(
                "Angular instrument pedestal", (0, 0, h * 0.35), (0.22, 0.22, h * 0.6), "dark"
            )
            if kind == "beacon":
                self.crystal((0, 0, h * 0.6), 0.14, 0.38, 2)
            else:
                self.cyl("Emitter chamber", (0, 0, h * 0.75), 0.10, h * 0.4, "window")
                self.cyl("Emitter core", (0, 0, h * 0.75), 0.04, h * 0.4, "light")
            self.box("Alloy lamp hood", (0, 0, h * 1.05), (0.3, 0.26, 0.10), "plate")

    def gate(self, kind):
        if kind == "gate-marker":
            self.pedestal(0.98)
            self.polygon_band("Ground checkpoint annulus", 0.8, 0.045, 0.17, 0.025, "light")
            return
        for s in [-1, 1]:
            self.box("Gate foot", (0, s * 0.9, 0.08), (0.38, 0.28, 0.16), "dark")
            if self.steam:
                self.vessel((0, s * 0.9, 0.48), 0.10, 0.60)
                self.gear((0.15, s * 0.9, 0.38), 0.12, axis="x")
            else:
                self.box("Gate angled pedestal", (0, s * 0.9, 0.40), (0.2, 0.2, 0.72), "plate")
                self.box("Gate status spine", (0.115, s * 0.9, 0.44), (0.02, 0.06, 0.50), "light")
        if kind == "gate-pylons":
            return
        if kind in {"finish-arch", "gantry"}:
            for s in [-1, 1]:
                self.box(
                    "Arch upright",
                    (0, s * 0.9, 0.85),
                    (0.18, 0.16, 1.45),
                    "trim" if self.steam else "plate",
                )
            self.box("Overhead crossbeam", (0, 0, 1.58), (0.26, 2, 0.2), "dark")
            if self.steam:
                self.pipe(
                    "Arch steam main",
                    [(0, -0.9, 0.3), (0, -0.9, 1.75), (0, 0.9, 1.75), (0, 0.9, 0.3)],
                    0.035,
                )
                self.gear((0.16, 0, 1.58), 0.16, axis="x")
            else:
                self.box("Arch index light", (0.15, 0, 1.58), (0.015, 1.5, 0.045), "light")
            return
        hoop = self.assembly("Open checkpoint ring", (0, 0, 1.05))
        self.polygon_band(
            "Ring frame",
            0.94,
            0.14,
            0,
            0.15,
            "dark" if self.steam else "plate",
            axis="x",
            parent=hoop,
        )
        for x in [-0.09, 0.09]:
            self.ring("Ring edge rail", (x, 0, 0), 0.94, 0.025, "trim", axis="x", parent=hoop)
        for j in range(8):
            a = j * TAU / 8
            self.cyl(
                "Mechanical ring lock" if self.steam else "Ring emitter",
                (0.11, 0.94 * math.cos(a), 0.94 * math.sin(a)),
                0.075,
                0.08,
                "copper" if self.steam else "glass",
                axis="x",
                parent=hoop,
            )
        if not self.steam:
            self.ring(
                "Checkpoint inner energy", (0, 0, 0), 0.865, 0.014, "energy", axis="x", parent=hoop
            )

    def rail(self, kind):
        self.box("Rail structural base", (0, 0, 0.06), (2, 0.3, 0.12), "dark")
        for x in [-0.9, 0.9]:
            self.box(
                "Rail post", (x, 0, 0.40), (0.15, 0.28, 0.72), "trim" if self.steam else "plate"
            )
        if self.steam:
            self.box("Burgundy rail panel", (0, 0, 0.36), (1.75, 0.08, 0.42), "plate")
            for z in [0.18, 0.65]:
                self.cyl("Continuous copper rail", (0, 0, z), 0.045, 1.9, "copper", axis="x")
            self.gear((0.9, 0, 0.79), 0.07)
        else:
            self.loft(
                "Armored rail wedge",
                [(-0.88, 0.10, 0.13, 0.56), (0, 0.13, 0.13, 0.64), (0.88, 0.10, 0.13, 0.56)],
                "dark",
            )
            self.box("Alloy rail cap", (0, 0, 0.65), (1.8, 0.22, 0.08), "plate")
            for y in [-0.14, 0.14]:
                self.box("Recessed cyan rail", (0, y, 0.35), (1.3, 0.02, 0.035), "light", bevel=0)

    def tile(self, kind):
        if kind == "track-bend":
            self.polygon_band(
                "Curved road module", 2, 1.2, 0, 0.08, "road", start=0, end=math.pi / 2
            )
            for r in [1.4, 2.6]:
                self.polygon_band(
                    "Curved road rim", r, 0.08, 0.04, 0.05, "trim", start=0, end=math.pi / 2
                )
            for j in range(7):
                a = j * math.pi / 12
                if self.steam:
                    self.pipe(
                        "Radial expansion joint",
                        [
                            (1.43 * math.cos(a), 1.43 * math.sin(a), 0.045),
                            (2.57 * math.cos(a), 2.57 * math.sin(a), 0.045),
                        ],
                        0.012,
                        "trim",
                    )
                else:
                    self.box(
                        "Bend luminous index",
                        (2.50 * math.cos(a), 2.50 * math.sin(a), 0.06),
                        (0.07, 0.07, 0.02),
                        "light",
                        bevel=0,
                    )
            return
        sx, sy = (4, 3) if kind == "track-straight" else (2, 0.4) if kind == "kerb" else (2, 2)
        self.box("Shared planar surface", (0, 0, 0), (sx, sy, 0.06), "road", bevel=0)
        if self.steam:
            for x in [-sx / 2 + 0.06, 0, sx / 2 - 0.06]:
                self.box("Riveted road seam", (x, 0, 0.04), (0.035, sy, 0.025), "trim", bevel=0)
            if not self.low:
                for x in [-sx / 2 + 0.1, sx / 2 - 0.1]:
                    for y in [-sy / 2 + 0.1, sy / 2 - 0.1]:
                        self.cyl("Deck bolt", (x, y, 0.065), 0.03, 0.02, "trim", segments=6)
        else:
            for y in [-sy / 2 + 0.07, sy / 2 - 0.07]:
                self.box(
                    "Inset edge guide", (0, y, 0.04), (sx - 0.15, 0.035, 0.02), "light", bevel=0
                )
            for x in [-sx * 0.46, sx * 0.46]:
                self.box("Plate joint", (x, 0, 0.045), (0.025, sy - 0.15, 0.015), "trim", bevel=0)

    def station(self, kind):
        if kind == "direction-board":
            self.cyl("Sign mast", (0, 0, 0.6), 0.06, 1.2, "trim")
            self.box("Direction board", (0, 0, 1.15), (0.15, 1, 0.42), "dark")
            for y in [-0.22, 0.15]:
                self.pipe(
                    "Raised direction chevron",
                    [(0.10, y - 0.09, 1.02), (0.10, y + 0.07, 1.15), (0.10, y - 0.09, 1.28)],
                    0.025,
                    "trim" if self.steam else "light",
                )
            if self.steam:
                self.gear((0.13, 0, 1.48), 0.12, axis="x")
            return
        self.box("Service station foundation", (0, 0, 0.1), (1.8, 1.3, 0.2), "dark")
        if self.steam:
            self.box("Riveted boiler room", (-0.3, 0, 0.65), (0.9, 1, 1), "dark")
            self.boiler((-0.3, 0, 1.22), 0.3, 1.15, axis="y")
            self.vessel((0.5, 0.35, 0.63), 0.18, 0.8)
            self.pipe(
                "Service steam pipe",
                [(0.55, -0.45, 0.1), (0.55, -0.45, 1.1), (-0.3, -0.45, 1.1)],
                0.045,
            )
            self.gauge((-0.3, -0.53, 0.7), 0.14)
        else:
            self.box("Cantilever alloy roof", (0, 0, 1.2), (1.8, 1.3, 0.13), "plate")
            self.box("Service equipment cabinet", (-0.5, 0, 0.66), (0.6, 1.1, 1.0), "dark")
            self.box("Diagnostic screen", (-0.18, -0.1, 0.8), (0.02, 0.65, 0.45), "glass")
            self.box("Service canopy light", (0.4, -0.65, 1.16), (0.65, 0.025, 0.03), "light")
            self.vessel((0.55, 0.25, 0.55), 0.13, 0.7)

    def effects(self, kind):
        if kind in {"tether-fitting", "capture-latch"}:
            self.clamp()
            if kind == "tether-fitting":
                self.pedestal(0.20)
                self.ring("Tether attachment eye", (-0.2, 0, 0.3), 0.09, 0.025, "trim", axis="x")
            return
        if kind == "thrust-plume":
            for j in range(3):
                pivot = self.assembly(f"Thrust stage {j}", motion="effect-stage")
                pivot["stage"] = j
                self.cyl(
                    "Exhaust cone",
                    (-0.2 - j * 0.1, 0, 0),
                    0.1 + j * 0.03,
                    0.4 + j * 0.2,
                    "vapor",
                    axis="x",
                    r2=0.005,
                    parent=pivot,
                )
                self.cyl(
                    "Hot plume core",
                    (-0.12 - j * 0.08, 0, 0),
                    0.055,
                    0.25 + j * 0.12,
                    "energy" if self.steam else "light",
                    axis="x",
                    r2=0.001,
                    parent=pivot,
                )
                if self.steam:
                    for k in range(3):
                        puff = self.rock(
                            0.055 + k * 0.015, k, (-0.2 - k * 0.12, 0, -0.04), pivot, True
                        )
                        puff.data.materials[0] = self.mats["vapor"]
            return
        if kind == "pickup-burst":
            for j in range(12 if not self.low else 6):
                a = j * 2.4
                self.crystal(
                    (math.cos(a) * 0.4, math.sin(a) * 0.4, (j % 3) * 0.2),
                    0.045,
                    0.065 if self.steam else 0.12,
                    j,
                )
            return
        arms = 4 if kind == "rotor-airflow" else 2
        for k in range(arms):
            ox, oy = (
                (math.cos(k * TAU / 4) * 0.4, math.sin(k * TAU / 4) * 0.4)
                if kind == "rotor-airflow"
                else (0, 0)
            )
            steps = 12 if self.low else 32
            vertices = []
            for j in range(steps + 1):
                t = j / steps
                a = t * TAU * 1.4 + k * math.pi
                if kind == "path-trail":
                    x, y, z = -t * 2, math.sin(t * 8 + k) * 0.04, 0
                else:
                    r = 0.1 + t * 0.35
                    x, y, z = ox + math.cos(a) * r, oy + math.sin(a) * r, -t * 0.2
                width = (
                    0.04 + math.sin(t * math.pi) * 0.09
                    if self.steam
                    else 0.018 + math.sin(t * math.pi) * 0.045
                )
                width *= 0.12 + 0.88 * math.sin(t * math.pi)
                wx, wy = (
                    (0, width)
                    if kind == "path-trail"
                    else (math.cos(a) * width, math.sin(a) * width)
                )
                vertices.extend([(x, y, z), (x + wx, y + wy, z + 0.01)])
            self.mesh(
                "Steam wisp" if self.steam else "Energy ribbon",
                vertices,
                [(j * 2, j * 2 + 1, j * 2 + 3, j * 2 + 2) for j in range(steps)],
                "vapor",
            )

    def detail_pass(self):
        if self.low:
            return
        bpy.context.view_layer.update()
        for obj in list(self.root.children_recursive):
            if obj.type != "MESH" or len(obj.data.vertices) != 8:
                continue
            for face in obj.data.polygons:
                if len(face.vertices) != 4 or face.area < 0.035 or face.normal.z < -0.1:
                    continue
                corners = [obj.data.vertices[i].co.copy() for i in face.vertices]
                center = sum(corners, Vector()) / 4
                radius = min(0.016, math.sqrt(face.area) * 0.055)
                normal = obj.matrix_local.to_3x3() @ face.normal
                for corner in corners:
                    point = obj.matrix_local @ (corner.lerp(center, 0.16) + face.normal * 0.006)
                    bolt = self.cyl(
                        "Brass service fastener" if self.steam else "Recessed alloy fastener",
                        point,
                        radius,
                        0.009,
                        "trim",
                        parent=obj.parent,
                        segments=6,
                    )
                    bolt.rotation_mode = "QUATERNION"
                    bolt.rotation_quaternion = normal.to_track_quat("Z", "Y")
                if face.area > 0.22 and face.normal.z > 0.5:
                    inset = [
                        tuple(obj.matrix_local @ (v.lerp(center, 0.20) + face.normal * 0.007))
                        for v in corners
                    ]
                    self.mesh(
                        "Replaceable access plate",
                        inset,
                        [(0, 1, 2, 3)],
                        "plate" if self.steam else "dark",
                        obj.parent,
                    )

    def concept_finish(self, kind):
        """Refine silhouettes using existing vertices, materials and assemblies."""
        if kind == "drop-capsule":
            # The reference capsule lies across its tray with a long visible chamber.
            transform = (
                Matrix.Translation((0, 0, 0.38))
                @ Matrix.Rotation(math.pi / 2, 4, "Y")
                @ Matrix.Translation((0, 0, -0.42))
            )
            fixed = (
                "Low armored base",
                "Perimeter retaining band",
                "Pressure feed",
                "Recessed light guide",
            )
            for obj in self.root.children:
                if obj.type == "MESH" and not obj.name.startswith(fixed):
                    obj.matrix_local = transform @ obj.matrix_local
        elif kind == "drop-pile":
            # Loose ore should read as scattered rubble rather than another machine.
            prefixes = (
                "Octagonal mineral tray",
                "Segmented tray rim",
                "Tray clasp",
                "Recovered ore clasp",
                "Pressure feed",
                "Recessed light guide",
            )
            for obj in list(self.root.children_recursive):
                if obj.type == "MESH" and obj.name.startswith(prefixes):
                    bpy.data.objects.remove(obj, do_unlink=True)
        self.root["surfaceRevision"] = "concept-mineral-planes-3"

    def build(self):
        for family, kinds in FAMILIES.items():
            for kind in kinds:
                self.root = self.empty(f"{kind}-root", parent=self.pack)
                self.root["assetModel"] = kind
                self.root["assetStyle"] = self.style
                self.root["conceptFamily"] = family
                self.root["collisionEnvelope"] = SIZES[kind]
                self.root["physicsSource"] = "scene-defined; identical across styles"
                self.root["authoringAxes"] = "+X forward, +Z up"
                self.roots[kind] = self.root
                if family == "collectible-drops" or kind == "deposit":
                    self.collectible(kind)
                elif family == "ore-rocks" or kind == "rock-obstacle":
                    self.ore(kind)
                elif kind == "reactor":
                    self.reactor()
                elif kind == "dock":
                    self.dock()
                elif kind in {"beacon", "lamp", "bollard", "utility-column", "start-light"}:
                    self.beacon(kind)
                elif family == "checkpoints" or kind == "gantry":
                    self.gate(kind)
                elif kind in {"rail", "guardrail"}:
                    self.rail(kind)
                elif kind.startswith("corner"):
                    main = self.root
                    for j in range(2):
                        self.root = self.assembly("Corner wing", parent=main)
                        self.root.rotation_euler.z = (
                            j * math.pi / 2 * (-1 if kind == "corner-inner" else 1)
                        )
                        self.root.location = (0.6 if j == 0 else 0, 0.6 if j else 0, 0)
                        self.rail("rail")
                    self.root = main
                elif kind in {"floor-tile", "track-straight", "track-bend", "kerb"}:
                    self.tile(kind)
                elif kind in {"pit-station", "direction-board"}:
                    self.station(kind)
                elif family == "capture-effects":
                    self.effects(kind)
                refine_world_machinery(self, kind)
                refine_world_scenery(self, kind)
                self.detail_pass()
                self.concept_finish(kind)
                bpy.context.view_layer.update()
                nodes = list(self.root.children_recursive)
                coords = [
                    o.matrix_world @ Vector(v)
                    for o in nodes
                    if o.type == "MESH"
                    for v in o.bound_box
                ]
                lo = Vector(tuple(min(v[i] for v in coords) for i in range(3)))
                hi = Vector(tuple(max(v[i] for v in coords) for i in range(3)))
                factors = Vector(
                    tuple(SIZES[kind][i] / max(hi[i] - lo[i], 0.001) for i in range(3))
                )
                self.root.scale = factors
                self.root.location = (
                    -(hi.x + lo.x) * factors.x / 2,
                    -(hi.y + lo.y) * factors.y / 2,
                    -lo.z * factors.z + 0.025,
                )
        self.root = self.pack
        lab.repair_vehicle_normals(self)
        machinery_polish_after_fit(self)
        scenery_polish_after_fit(self)
        bpy.context.view_layer.update()
        return self


def render_world(high, preview_families=None):
    """Render family comparisons from either a fresh builder or its saved scene."""
    style = high.style
    for family, kinds in FAMILIES.items():
        if preview_families and family not in preview_families:
            continue
        spacing = 3.2
        for root in high.roots.values():
            for o in root.children_recursive:
                o.hide_render = root.get("assetModel") not in kinds
        # Temporarily frame only the current family, leaving source poses intact.
        visible = [o for o in high.scene.objects if o.type == "MESH" and not o.hide_render]
        old_locations = {r: r.location.copy() for r in high.roots.values()}
        old_scales = {r: r.scale.copy() for r in high.roots.values()}
        for i, kind in enumerate(kinds):
            r = high.roots[kind]
            # Inspection thumbnails give small props equal visual weight; restore
            # their exact scene scale immediately after rendering each family.
            factor = 2.5 / max(SIZES[kind])
            r.scale *= factor
            source_index = list(high.roots).index(kind)
            r.location.x = (r.location.x - (source_index % 7) * 5) * factor + (i % 3) * spacing
            r.location.y = (r.location.y - (source_index // 7) * 5) * factor + (i // 3) * spacing
            r.location.z *= factor
        bpy.context.window.scene = high.scene
        bpy.context.view_layer.update()
        corners = [o.matrix_world @ Vector(v) for o in visible for v in o.bound_box]
        lo = Vector(tuple(min(v[i] for v in corners) for i in range(3)))
        hi = Vector(tuple(max(v[i] for v in corners) for i in range(3)))
        center = (lo + hi) / 2
        camera = high.scene.camera
        camera.location = center + Vector((3, -4, 3.4)).normalized() * 20
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        projected = [camera.matrix_world.inverted() @ v for v in corners]
        camera.data.ortho_scale = (
            max(
                max(v.x for v in projected) - min(v.x for v in projected),
                (max(v.y for v in projected) - min(v.y for v in projected)) * 4 / 3,
            )
            * 1.15
        )
        for light in [o for o in high.scene.objects if o.type == "LIGHT"]:
            light.location += center
            light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        dest = ROOT / "previews" / style / f"world-{family}.png"
        high.scene.render.filepath = str(dest)
        bpy.ops.render.render(write_still=True)
        for light in [o for o in high.scene.objects if o.type == "LIGHT"]:
            light.location -= center
        for r, p in old_locations.items():
            r.location = p
            r.scale = old_scales[r]
    for o in high.scene.objects:
        o.hide_render = False
    lab.frame_camera(high.scene, (3, -4, 5))


def build_world(style, render=False, preview_families=None):
    high = WorldBuilder(style).build()
    low = WorldBuilder(style, True).build()
    info = {"style": style, "families": FAMILIES, "envelopes": SIZES}
    info["high"] = high.export(ROOT / style / "world-high.glb")
    info["low"] = low.export(ROOT / style / "world-low.glb")
    for builder in [high, low]:
        bpy.context.window.scene = builder.scene
        for i, root in enumerate(builder.roots.values()):
            root.location.x += (i % 7) * 5
            root.location.y += (i // 7) * 5
        lab.studio(builder)
        builder.scene.render.threads_mode = "FIXED"
        builder.scene.render.threads = 4
        builder.scene.cycles.samples = 12
        lab.frame_camera(builder.scene, (3, -4, 5))
    source = ROOT / "sources" / style / "world.blend"
    bpy.data.libraries.write(str(source), {high.scene, low.scene}, fake_user=True, compress=True)
    (ROOT / style / "world-build.json").write_text(json.dumps(info, indent=2) + "\n")
    (ROOT / "world-catalog.json").write_text(
        json.dumps({"families": FAMILIES, "envelopes": SIZES}, indent=2) + "\n"
    )
    # Persist runtime/source outputs before the optional, slower visual review.
    if render:
        render_world(high, preview_families)
    return info


if __name__ == "__main__":
    for style in ["futuristic", "steampunk"]:
        print(build_world(style, render=True))
