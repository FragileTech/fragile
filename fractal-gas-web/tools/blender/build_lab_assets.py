"""Original concept-matched lab assets. Run in Blender, including through its MCP.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(SCRIPT).parent))
    exec(compile(open(SCRIPT).read(), SCRIPT, 'exec'))
    build_asset('steampunk', 'rocket')

The current user scene is never cleared. Each build owns a new scene. Exported
GLBs deliberately retain the lab's +X-forward, +Z-up coordinates (export_yup=False).
"""

import itertools
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np
from vehicle_refinement import refine_vehicle, repair_vehicle_normals
from vehicle_surface_finish import finish_vehicle_surfaces


ROOT = Path(__file__).resolve().parents[2] / "web/lab/assets"
TAU = math.tau


def material(name, color, metallic=0.7, rough=0.36, emission=0, texture=None):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    p = mat.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    if emission:
        p.inputs["Emission Color"].default_value = (*color, 1)
        p.inputs["Emission Strength"].default_value = emission
    if texture:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        for field, image in texture.items():
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = image
            if field == "Normal":
                normal = nodes.new("ShaderNodeNormalMap")
                normal.inputs["Strength"].default_value = 0.35
                links.new(tex.outputs["Color"], normal.inputs["Color"])
                links.new(normal.outputs["Normal"], p.inputs["Normal"])
            else:
                links.new(tex.outputs["Color"], p.inputs[field])
    return mat


def panel_maps(style, pale=False, n=512, concept=False):
    """Bake a deterministic original panel/rivet/wear atlas into packed PBR maps."""
    y, x = np.mgrid[0:n, 0:n] / n
    rng = np.random.default_rng(1841)
    noise = rng.random((n, n))
    seam = (x < 0.016) | (x > 0.984) | (y < 0.016) | (y > 0.984)
    height = np.zeros((n, n), dtype=np.float32)
    height[seam] = -0.2
    for u in [0.065, 0.32, 0.68, 0.935]:
        for v in [0.065, 0.935]:
            r = np.sqrt((x - u) ** 2 + (y - v) ** 2)
            height += np.clip(1 - r / 0.018, 0, 1) * 0.28
    base = np.array([0.20, 0.22, 0.25] if style == "futuristic" else [0.16, 0.135, 0.105])
    if pale:
        base = np.array([0.43, 0.49, 0.54] if style == "futuristic" else [0.18, 0.027, 0.024])
    rgb = np.broadcast_to(base, (n, n, 3)).copy()
    rgb *= 0.86 + noise[..., None] * 0.25
    rgb[seam] *= 0.35
    rgb[height > 0.06] = [0.50, 0.53, 0.56] if style == "futuristic" else [0.55, 0.34, 0.12]
    scratches = (noise > 0.997) | ((x < 0.035) & (noise > 0.70))
    rgb[scratches] *= 1.7
    if concept:
        # Panel breaks, fasteners and worn edges live in the existing atlas: no
        # extra geometry, texture slots or runtime shader work at either LOD.
        edge = np.minimum.reduce([x, 1 - x, y, 1 - y])
        cut = np.maximum(np.abs(x - 0.5) - 0.28, np.abs(y - 0.5) - 0.31)
        hatch = (cut < 0) & (np.abs(x - 0.5) + np.abs(y - 0.5) < 0.51)
        border = hatch & ((cut > -0.009) | (np.abs(x - 0.5) + np.abs(y - 0.5) > 0.498))
        rib = np.abs(x - (0.74 + 0.11 * np.sin(y * math.pi))) < 0.006
        seams = seam | border | (rib & ~hatch)
        patina = np.sin(x * 19 + np.sin(y * 11)) * np.sin(y * 23 + x * 7)
        # Neutral alloy and charcoal iron, with restrained burgundy enamel.
        base = np.array(
            ([0.39, 0.42, 0.44] if pale else [0.105, 0.125, 0.145])
            if style == "futuristic"
            else ([0.15, 0.030, 0.022] if pale else [0.085, 0.076, 0.058])
        )
        rgb = np.broadcast_to(base, (n, n, 3)).copy()
        rgb *= (0.96 + patina * 0.13 + (noise - 0.5) * 0.09)[..., None]
        rgb[hatch] *= 0.80
        rgb[seams] *= 0.23
        height[seams] = -0.18
        wear = ((edge < 0.035) | (border & (noise > 0.6))) & (noise > 0.70)
        metal = np.array([0.47, 0.49, 0.50] if style == "futuristic" else [0.36, 0.24, 0.095])
        rgb[wear] = metal * 0.80
        for u in [0.055, 0.945]:
            for v in [0.055, 0.27, 0.50, 0.73, 0.945]:
                r = np.hypot(x - u, y - v)
                collar = r < 0.011
                head = r < 0.006
                rgb[collar] *= 0.35
                rgb[head] = metal
                height[head] = 0.18
        # Long hairline wear reads as worked metal instead of salt-and-pepper noise.
        scratch = (np.abs(np.sin(y * 227 + x * 11)) < 0.018) & (patina > 0.40)
        rgb[scratch & ~seams] = metal * 0.65
    dy, dx = np.gradient(height)
    normal = np.stack([-dx * 18, -dy * 18, np.ones_like(x)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    maps = {}
    for name, data in [
        ("Base Color", rgb),
        ("Roughness", np.repeat((0.36 + noise * 0.18)[..., None], 3, 2)),
        ("Normal", normal * 0.5 + 0.5),
    ]:
        im = bpy.data.images.new(f"{style} / baked panel {name}", width=n, height=n)
        if name != "Base Color":
            im.colorspace_settings.name = "Non-Color"
        rgba = np.concatenate([np.clip(data, 0, 1), np.ones((n, n, 1))], axis=2).astype(np.float32)
        im.pixels.foreach_set(rgba.ravel())
        im.pack()
        maps[name] = im
    return maps


class Builder:
    def __init__(self, style, kind, low=False):
        self.style, self.kind, self.low = style, kind, low
        self.steam = style == "steampunk"
        self.seg = 8 if low else 24
        self.scene = bpy.data.scenes.new(f"Lab / {style} / {kind} / {'low' if low else 'high'}")
        bpy.context.window.scene = self.scene
        self.root = self.empty(f"{kind}-root")
        self.root["assetStyle"] = style
        self.root["assetModel"] = kind
        self.root["authoringAxes"] = "+X forward, +Z up"
        self.mats = {
            "dark": material(
                "Riveted iron" if self.steam else "Ceramic graphite",
                (0.12, 0.105, 0.085) if self.steam else (0.055, 0.072, 0.09),
                texture=panel_maps(
                    style, concept=kind in {"rocket", "kart", "drone", "harvester"}
                ),
            ),
            "plate": material(
                "Burgundy enamel" if self.steam else "Brushed pale alloy",
                (0.18, 0.027, 0.024) if self.steam else (0.48, 0.56, 0.62),
                rough=0.42,
                texture=panel_maps(
                    style, True, concept=kind in {"rocket", "kart", "drone", "harvester"}
                ),
            ),
            "trim": material(
                "Machined brass" if self.steam else "Titanium edges",
                (0.55, 0.32, 0.095) if self.steam else (0.22, 0.29, 0.36),
                rough=0.27,
            ),
            "copper": material(
                "Copper feed lines" if self.steam else "Anodized hardware",
                (0.50, 0.18, 0.07) if self.steam else (0.13, 0.10, 0.19),
            ),
            "rubber": material("Tire rubber", (0.022, 0.026, 0.028), 0.02, 0.82),
            "seat": material(
                "Quilted leather" if self.steam else "Cockpit upholstery",
                (0.10, 0.037, 0.016) if self.steam else (0.025, 0.034, 0.041),
                0,
                0.64,
            ),
            "glass": material(
                "Amber glazing" if self.steam else "Cyan glazing",
                (0.48, 0.22, 0.035) if self.steam else (0.022, 0.25, 0.34),
                0.48,
                0.12,
                0.10,
            ),
            "light": material(
                "Amber lamps" if self.steam else "Cyan instrumentation",
                (1, 0.40, 0.075) if self.steam else (0.055, 0.8, 1),
                0.25,
                0.20,
                2,
            ),
            "energy": material(
                "Furnace glow" if self.steam else "Violet processing",
                (0.8, 0.22, 0.015) if self.steam else (0.42, 0.07, 0.95),
                0.25,
                0.22,
                1.4,
            ),
            "gold": material("Ore gold / optics", (0.62, 0.35, 0.075), 0.7, 0.3),
            "dial": material("Ivory pressure dial", (0.72, 0.64, 0.45), 0, 0.72),
        }

    def empty(self, name, pos=(0, 0, 0), parent=None, motion=None):
        o = bpy.data.objects.new(name, None)
        self.scene.collection.objects.link(o)
        o.location = pos
        o.parent = parent
        if motion:
            o["motion"] = motion
        return o

    def mesh(self, name, verts, faces, mat="dark", parent=None, bevel=0):
        data = bpy.data.meshes.new(name)
        data.from_pydata(verts, [], faces)
        data.update()
        obj = bpy.data.objects.new(name, data)
        self.scene.collection.objects.link(obj)
        obj.parent = parent or self.root
        data.materials.append(self.mats[mat])
        uv = data.uv_layers.new(name="Panel UV")
        for poly in data.polygons:
            axis = max(range(3), key=lambda i: abs(poly.normal[i]))
            axes = [i for i in range(3) if i != axis]
            coords = [data.vertices[data.loops[i].vertex_index].co for i in poly.loop_indices]
            lo = [min(v[a] for v in coords) for a in axes]
            hi = [max(v[a] for v in coords) for a in axes]
            for li, v in zip(poly.loop_indices, coords):
                uv.data[li].uv = [
                    (v[a] - lo[k]) / max(hi[k] - lo[k], 0.001) for k, a in enumerate(axes)
                ]
        if bevel and not self.low:
            mod = obj.modifiers.new("Machined edge", "BEVEL")
            mod.width, mod.segments = bevel, 2
            mod = obj.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
        return obj

    def box(self, name, pos, size, mat="dark", parent=None, bevel=0.035):
        x, y, z = [s / 2 for s in size]
        v = [
            (-x, -y, -z),
            (x, -y, -z),
            (x, y, -z),
            (-x, y, -z),
            (-x, -y, z),
            (x, -y, z),
            (x, y, z),
            (-x, y, z),
        ]
        o = self.mesh(
            name,
            v,
            [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
            mat,
            parent,
            bevel,
        )
        o.location = pos
        return o

    def loft(self, name, sections, mat="dark", parent=None):
        # Each cross section is (x, half-width, lower-z, upper-z).
        verts = []
        for x, w, lo, hi in sections:
            verts.extend([
                (x, -w * 0.72, lo),
                (x, w * 0.72, lo),
                (x, w, lo + (hi - lo) * 0.35),
                (x, w * 0.65, hi),
                (x, -w * 0.65, hi),
                (x, -w, lo + (hi - lo) * 0.35),
            ])
        faces = [tuple(reversed(range(6)))]
        for i in range(len(sections) - 1):
            for j in range(6):
                faces.append((
                    i * 6 + j,
                    i * 6 + (j + 1) % 6,
                    (i + 1) * 6 + (j + 1) % 6,
                    (i + 1) * 6 + j,
                ))
        faces.append(tuple(range((len(sections) - 1) * 6, len(sections) * 6)))
        return self.mesh(name, verts, faces, mat, parent, 0.025)

    def plate(self, name, points, z, thick, mat="plate", parent=None):
        n = len(points)
        vs = [(x, y, h) for h in [z, z + thick] for x, y in points]
        return self.mesh(
            name,
            vs,
            [tuple(reversed(range(n))), tuple(range(n, n * 2))]
            + [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)],
            mat,
            parent,
            0.022,
        )

    def cyl(
        self, name, pos, radius, length, mat="trim", axis="z", r2=None, parent=None, segments=None
    ):
        n = segments or self.seg
        r2 = radius if r2 is None else r2
        verts = []
        for z, r in [(-length / 2, radius), (length / 2, r2)]:
            for i in range(n):
                a = i * TAU / n
                v = [r * math.cos(a), r * math.sin(a), z]
                if axis == "x":
                    v = [v[2], v[0], v[1]]
                if axis == "y":
                    v = [v[0], v[2], v[1]]
                verts.append(v)
        faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))] + [
            (i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)
        ]
        o = self.mesh(name, verts, faces, mat, parent)
        o.location = pos
        for p in o.data.polygons[2:]:
            p.use_smooth = True
        return o

    def ring(self, name, pos, r, tube, mat="trim", axis="z", parent=None):
        n = self.seg
        m = 4 if self.low else 8
        verts = []
        for i in range(n):
            a = i * TAU / n
            for j in range(m):
                b = j * TAU / m
                v = [
                    (r + tube * math.cos(b)) * math.cos(a),
                    (r + tube * math.cos(b)) * math.sin(a),
                    tube * math.sin(b),
                ]
                if axis == "x":
                    v = [v[2], v[0], v[1]]
                if axis == "y":
                    v = [v[0], v[2], v[1]]
                verts.append(v)
        faces = [
            (
                i * m + j,
                ((i + 1) % n) * m + j,
                ((i + 1) % n) * m + (j + 1) % m,
                i * m + (j + 1) % m,
            )
            for i in range(n)
            for j in range(m)
        ]
        o = self.mesh(name, verts, faces, mat, parent)
        o.location = pos
        for p in o.data.polygons:
            p.use_smooth = True
        return o

    def duct(self, name, pos, r, depth):
        n = self.seg
        verts = [
            (radius * math.cos(i * TAU / n), radius * math.sin(i * TAU / n), z)
            for z, radius in [
                (-depth / 2, r),
                (depth / 2, r),
                (-depth / 2, r - 0.075),
                (depth / 2, r - 0.075),
            ]
            for i in range(n)
        ]
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.extend([
                (i, j, j + n, i + n),
                (i + 2 * n, i + 3 * n, j + 3 * n, j + 2 * n),
                (i + n, j + n, j + 3 * n, i + 3 * n),
                (i, j, i + 2 * n, j + 2 * n),
            ])
        obj = self.mesh(name, verts, faces, "plate" if self.steam else "dark")
        obj.location = pos
        return obj

    def pipe(self, name, points, r=0.035, mat="copper", parent=None):
        # Independent straight sections keep these exportable meshes and work at both LODs.
        for a, b in itertools.pairwise(points):
            d = Vector(b) - Vector(a)
            o = self.cyl(
                name,
                (Vector(a) + Vector(b)) / 2,
                r,
                d.length,
                mat,
                parent=parent,
                segments=6 if self.low else 10,
            )
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = d.to_track_quat("Z", "Y")

    def gauge(self, pos, r=0.14, axis="y"):
        if self.low:
            return
        self.cyl("Pressure gauge brass bezel", pos, r, 0.05, "trim", axis)
        p = list(pos)
        p[1 if axis == "y" else 0] -= 0.03
        self.cyl("Ivory gauge face", p, r * 0.82, 0.014, "dial", axis)
        self.box(
            "Pressure needle",
            (p[0], p[1] - 0.01, p[2] + r * 0.18),
            (r * 0.06, 0.015, r * 0.8),
            "dark",
            bevel=0,
        )
        for j in range(9):
            a = j * math.pi / 6 - math.pi / 6
            tick = self.box(
                "Dial index",
                (p[0] + math.cos(a) * r * 0.64, p[1] - 0.012, p[2] + math.sin(a) * r * 0.64),
                (r * 0.12, 0.016, r * 0.035),
                "dark",
                bevel=0,
            )
            tick.rotation_euler.y = -a

    def vents(self, pos, n=6, span=0.5, axis="y"):
        if self.low:
            return
        for i in range(n):
            p = list(pos)
            p[0] += (i / (n - 1) - 0.5) * span
            self.box("Recessed cooling grille", p, (0.026, 0.025, 0.16), "rubber", bevel=0)

    def wheels(self, axles, width, radius, tire_width):
        for axle, x in enumerate(axles):
            for side in [-1, 1]:
                steer = self.empty(
                    f"Steering {axle} {side}",
                    (x, side * width, radius),
                    self.root,
                    "steer" if axle == 0 else None,
                )
                pivot = self.empty(f"Wheel {axle} {side}", parent=steer, motion="wheel")
                pivot["wheelRadius"] = radius
                wire_wheel = self.steam and self.kind == "kart"
                if wire_wheel:
                    self.ring(
                        "Narrow open-center tire",
                        (0, 0, 0),
                        radius * 0.86,
                        radius * 0.14,
                        "rubber",
                        "y",
                        pivot,
                    )
                else:
                    self.cyl(
                        "Rubber tire", (0, 0, 0), radius, tire_width, "rubber", "y", parent=pivot
                    )
                self.ring(
                    "Rounded tire shoulder",
                    (0, side * tire_width * 0.39, 0),
                    radius * 0.83,
                    radius * 0.16,
                    "rubber",
                    "y",
                    pivot,
                )
                self.cyl(
                    "Wheel hub",
                    (0, side * (tire_width / 2 + 0.015), 0),
                    radius * (0.17 if wire_wheel else 0.30),
                    0.08,
                    "trim",
                    "y",
                    parent=pivot,
                )
                self.ring(
                    "Wheel rim",
                    (0, side * (tire_width / 2 + 0.025), 0),
                    radius * (0.75 if wire_wheel else 0.65),
                    radius * 0.055,
                    "trim",
                    "y",
                    pivot,
                )
                count = (
                    8
                    if self.low and wire_wheel
                    else 6
                    if self.low
                    else (16 if wire_wheel else 10 if self.steam else 7)
                )
                for j in range(count):
                    a = j * TAU / count
                    self.pipe(
                        "Radial wheel spoke",
                        [
                            (
                                math.cos(a) * radius * (0.15 if wire_wheel else 0.25),
                                side * tire_width * 0.52,
                                math.sin(a) * radius * (0.15 if wire_wheel else 0.25),
                            ),
                            (
                                math.cos(a) * radius * (0.74 if wire_wheel else 0.63),
                                side * tire_width * 0.52,
                                math.sin(a) * radius * (0.74 if wire_wheel else 0.63),
                            ),
                        ],
                        radius * (0.018 if wire_wheel else 0.035),
                        "trim",
                        pivot,
                    )
                if not self.low and not wire_wheel:
                    for j in range(24):
                        a = j * TAU / 24
                        tread = self.box(
                            "Tire tread block",
                            (math.sin(a) * radius, 0, math.cos(a) * radius),
                            (radius * 0.18, tire_width * 0.96, 0.035),
                            "rubber",
                            pivot,
                            0,
                        )
                        tread.rotation_euler.y = a
                if not self.steam and self.kind != "harvester":
                    self.ring(
                        "Luminous hub index",
                        (0, side * (tire_width * 0.54), 0),
                        radius * 0.70,
                        0.016,
                        "light",
                        "y",
                        pivot,
                    )
                self.pipe(
                    "Wishbone suspension",
                    [
                        (x - 0.22, side * 0.34, radius * 0.9),
                        (x, side * width, radius),
                        (x + 0.22, side * 0.34, radius * 0.7),
                    ],
                    0.045,
                    "trim",
                )

    def boiler(self, pos, r=0.35, length=1.0, axis="x"):
        self.cyl("Copper boiler pressure vessel", pos, r, length, "copper", axis)
        for offset in [-0.44, 0, 0.44]:
            p = list(pos)
            p["xyz".index(axis)] += offset * length
            self.ring("Boiler retaining hoop", p, r + 0.008, 0.035, "trim", axis)
        p = list(pos)
        p[2] += r + 0.14
        self.cyl("Pressure valve", p, 0.07, 0.25, "trim")
        self.ring("Valve hand wheel", (p[0], p[1], p[2] + 0.13), 0.10, 0.025)
        self.gauge((pos[0] + 0.15, pos[1] - r - 0.02, pos[2]), 0.12)

    def rocket(self):
        if self.steam:
            self.loft(
                "Riveted tapered fuselage",
                [
                    (-1.55, 0.38, 0.22, 0.77),
                    (-0.55, 0.48, 0.18, 0.96),
                    (0.65, 0.39, 0.20, 0.82),
                    (1.95, 0.018, 0.30, 0.34),
                ],
            )
            self.loft(
                "Amber segmented canopy",
                [(-0.45, 0.28, 0.72, 1.04), (0.35, 0.29, 0.71, 1.02), (0.87, 0.12, 0.55, 0.78)],
                "glass",
            )
            for x, w, z in [(-0.4, 0.29, 1.04), (0.08, 0.29, 1.04), (0.55, 0.23, 0.94)]:
                self.pipe(
                    "Brass canopy frame",
                    [(x, -w, 0.72), (x, -w * 0.65, z), (x, w * 0.65, z), (x, w, 0.72)],
                    0.025,
                    "trim",
                )
        else:
            self.loft(
                "Arrowhead armored fuselage",
                [
                    (-1.55, 0.44, 0.19, 0.67),
                    (-0.5, 0.56, 0.14, 0.79),
                    (0.55, 0.42, 0.20, 0.62),
                    (2.05, 0.018, 0.28, 0.31),
                ],
                "plate",
            )
            self.loft(
                "Graphite cockpit spine",
                [
                    (-1.4, 0.28, 0.58, 0.83),
                    (-0.15, 0.36, 0.51, 0.87),
                    (0.7, 0.24, 0.40, 0.66),
                    (1.45, 0.08, 0.30, 0.42),
                ],
            )
            self.loft(
                "Closed faceted cyan canopy",
                [(-0.45, 0.22, 0.79, 1.03), (0.23, 0.24, 0.61, 0.96), (0.88, 0.12, 0.48, 0.64)],
                "glass",
            )
            for s in [-1, 1]:
                self.pipe(
                    "Cockpit titanium frame",
                    [
                        (-0.45, s * 0.22, 0.79),
                        (-0.45, s * 0.14, 1.03),
                        (0.23, s * 0.16, 0.96),
                        (0.88, s * 0.08, 0.64),
                    ],
                    0.023,
                    "trim",
                )
                self.pipe(
                    "Nose cyan tracer",
                    [(1.75, s * 0.055, 0.33), (0.65, s * 0.36, 0.40), (-0.3, s * 0.48, 0.43)],
                    0.018,
                    "light",
                )
        for s in [-1, 1]:
            y = s * 0.78
            self.plate(
                "Swept stabilizer",
                [(-0.35, s * 0.28), (-1.5, s * 1.19), (-1.69, s * 1.1), (-1.5, s * 0.24)],
                0.29,
                0.065,
            )
            self.cyl(
                "Independent engine pod",
                (-0.84, y, 0.54),
                0.26,
                1.48,
                "copper" if self.steam else "plate",
                "x",
            )
            for x in [-1.48, -0.86, -0.19]:
                self.ring("Engine pod binding", (x, y, 0.54), 0.27, 0.035, "trim", "x")
            self.cyl("Rear exhaust bell", (-1.74, y, 0.54), 0.31, 0.34, "trim", "x", 0.16)
            self.cyl(
                "Exhaust throat",
                (-1.92, y, 0.54),
                0.24,
                0.018,
                "energy" if self.steam else "light",
                "x",
            )
            self.cyl("Pod forward intake", (-0.06, y, 0.54), 0.19, 0.12, "dark", "x", 0.13)
            self.pipe(
                "Exposed propellant feed",
                [
                    (-1.45, s * 0.4, 0.80),
                    (-0.95, s * 0.47, 0.83),
                    (-0.85, y, 0.82),
                    (-0.22, y, 0.82),
                ],
                0.036,
            )
            if self.steam:
                self.gauge((-0.65, y + s * 0.26, 0.62), 0.13)
            else:
                self.box(
                    "Pod status strip", (-0.65, y + s * 0.257, 0.57), (0.45, 0.018, 0.035), "light"
                )
            self.cyl("Vernier nozzle", (0.15, s * 0.45, 0.16), 0.105, 0.15, "trim", r2=0.07)
            pivot = self.empty("Engine exhaust", (-1.96, y, 0.54), self.root, "thrust")
            self.cyl(
                "Exhaust flame",
                (-0.22, 0, 0),
                0.008,
                0.44,
                "energy" if self.steam else "light",
                "x",
                0.16,
                pivot,
            )
        # Dorsal fin is an extruded vertical polygon.
        fin = self.plate(
            "Dorsal stabilizer", [(-1.6, 0), (-1.48, 0.77), (-0.75, 1.12), (-0.68, 0)], 0, 0.06
        )
        fin.rotation_euler.x = math.pi / 2
        fin.location = (0, 0.03, 0.65)
        self.vents((-0.85, -0.455, 0.58))

    def kart(self):
        self.box("Chassis rails", (-0.15, 0, 0.25), (2.75, 0.80, 0.16))
        self.wheels(
            [0.94, -1.02], 0.72, 0.48 if self.steam else 0.43, 0.14 if self.steam else 0.32
        )
        self.loft(
            "Riveted bonnet" if self.steam else "Armored wedge nose",
            [(0.03, 0.37, 0.34, 0.70), (0.9, 0.31, 0.24, 0.56), (1.55, 0.24, 0.20, 0.32)],
            "dark" if self.steam else "plate",
        )
        self.loft(
            "Bonnet center accent",
            [(0.09, 0.10, 0.691, 0.709), (0.9, 0.075, 0.565, 0.583), (1.48, 0.075, 0.343, 0.361)],
            "plate" if self.steam else "dark",
        )
        self.box("Seat cushion", (-0.48, 0, 0.40), (0.55, 0.51, 0.15), "seat")
        seat = self.box("Seat back", (-0.83, 0, 0.72), (0.12, 0.54, 0.63), "seat")
        seat.rotation_euler.y = -0.18
        for s in [-1, 1]:
            self.pipe(
                "Rear seat hoop" if self.steam else "Roll cage",
                [
                    (-0.67, s * 0.32, 0.40),
                    (-0.84, s * 0.32, 1.13),
                    (-1.00, s * 0.32, 1.20),
                    (-1.12, s * 0.32, 0.40),
                ]
                if self.steam
                else [
                    (0.17, s * 0.40, 0.35),
                    (-0.48, s * 0.42, 1.20),
                    (-0.99, s * 0.42, 1.17),
                    (-1.12, s * 0.42, 0.36),
                ],
                0.037,
                "trim",
            )
            self.pipe(
                "Lower side rail",
                [(1.4, s * 0.43, 0.25), (-1.3, s * 0.44, 0.25)],
                0.045,
                "copper" if self.steam else "trim",
            )
            self.cyl("Headlamp bezel", (1.15, s * 0.35, 0.43), 0.115, 0.09, "trim", "x")
            self.cyl("Headlamp glass", (1.20, s * 0.35, 0.43), 0.084, 0.01, "light", "x")
            if not self.steam:
                self.plate(
                    "Side pod armor",
                    [(-0.7, s * 0.28), (-0.75, s * 0.55), (-0.02, s * 0.52), (0.24, s * 0.33)],
                    0.36,
                    0.13,
                )
                self.pipe(
                    "Side energy trim",
                    [(-0.85, s * 0.50, 0.32), (-0.18, s * 0.49, 0.32), (0.10, s * 0.4, 0.39)],
                    0.018,
                    "light",
                )
        if self.steam:
            self.pipe(
                "Rear hoop crossbar", [(-1.00, -0.32, 1.20), (-1.00, 0.32, 1.20)], 0.032, "trim"
            )
        else:
            self.pipe(
                "Roll cage crossbar", [(-0.48, -0.42, 1.20), (-0.48, 0.42, 1.20)], 0.04, "trim"
            )
        self.pipe("Steering column", [(0.3, 0, 0.40), (-0.04, 0, 0.78)], 0.03, "trim")
        steering = self.ring("Steering wheel", (-0.04, 0, 0.78), 0.17, 0.022, "rubber", "x")
        steering.rotation_euler.y = -0.4
        if self.steam:
            self.boiler((-1.17, 0, 0.76), 0.30, 0.65, "y")
            self.cyl("Boiler smokestack", (-1.22, 0.20, 1.18), 0.09, 0.40, "dark")
            self.ring("Chimney cap", (-1.22, 0.20, 1.38), 0.105, 0.028)
            self.vents((0.60, -0.30, 0.42), 7, 0.65)
            for s in [-1, 1]:
                self.pipe(
                    "Steam drive plumbing",
                    [(-1.1, s * 0.36, 0.62), (-0.91, s * 0.46, 0.37), (0.65, s * 0.40, 0.32)],
                    0.045,
                )
        else:
            self.loft(
                "Autonomous sensor helmet",
                [(-0.59, 0.15, 0.77, 1.02), (-0.28, 0.20, 0.77, 1.09), (-0.10, 0.15, 0.79, 1.0)],
                "plate",
            )
            self.box("Sensor visor", (-0.09, 0, 0.93), (0.035, 0.27, 0.085), "glass")
            self.box("Sensor pixel band", (-0.066, 0, 0.92), (0.018, 0.17, 0.019), "light")
            self.box("Rear battery", (-1.16, 0, 0.54), (0.55, 0.56, 0.28), "dark")
            for s in [-1, 1]:
                self.pipe(
                    "Aero support",
                    [(-1.22, s * 0.3, 0.47), (-1.36, s * 0.36, 1.01)],
                    0.025,
                    "trim",
                )
            self.box("Rear aerofoil", (-1.34, 0, 1.04), (0.30, 1.30, 0.07), "plate")
            self.box("Aerofoil cyan edge", (-1.18, 0, 1.05), (0.018, 1.24, 0.024), "light")
            for s in [-1, 1]:
                self.box("Aerofoil endplate", (-1.34, s * 0.65, 1.09), (0.37, 0.025, 0.20))

    def drone(self):
        reach = 1.12 if self.steam else 0.83
        self.loft(
            "Diamond survey hull",
            [
                (-1.2, 0.16, 0.31, 0.63),
                (-0.45, 0.57, 0.19, 0.88),
                (0.45, 0.59, 0.22, 0.81),
                (1.06, 0.20, 0.33, 0.58),
            ],
            "dark",
        )
        self.plate(
            "Upper armored shell",
            [(-1.06, 0), (-0.42, -0.43), (0.47, -0.41), (0.91, 0), (0.47, 0.41), (-0.42, 0.43)],
            0.68,
            0.10,
            "plate",
        )
        self.cyl("Forward optical housing", (1.04, 0, 0.49), 0.245, 0.22, "trim", "x")
        self.cyl("Amber survey eye", (1.165, 0, 0.49), 0.192, 0.06, "gold", "x", 0.17)
        self.cyl("Optical pupil", (1.2, 0, 0.49), 0.085, 0.013, "light", "x")
        for x in [-0.73, 0.70]:
            for s in [-1, 1]:
                y = s * reach
                r = 0.48 if self.steam else 0.49
                self.pipe(
                    "Rotor suspension arm", [(x * 0.6, s * 0.34, 0.48), (x, y, 0.48)], 0.09, "trim"
                )
                self.ring(
                    "Propeller protective shroud",
                    (x, y, 0.49),
                    r,
                    0.07,
                    "trim" if self.steam else "plate",
                )
                self.ring("Lower rotor safety ring", (x, y, 0.37), r, 0.033, "trim")
                self.duct("Riveted rotor duct", (x, y, 0.43), r, 0.17)
                spin = self.empty("Survey rotor", (x, y, 0.46), self.root, "rotor")
                self.cyl("Rotor motor hub", (0, 0, 0), 0.11, 0.18, "trim", parent=spin)
                for j in range(4):
                    a = j * TAU / 4
                    blade = self.plate(
                        "Twisted propeller blade",
                        [(0.08, -0.045), (0.39, -0.09), (0.43, 0.015), (0.13, 0.05)],
                        0,
                        0.016,
                        "dark",
                        spin,
                    )
                    blade.rotation_euler.z = a
                for j in range(4 if self.low else 8):
                    a = j * TAU / (4 if self.low else 8)
                    px = x + r * math.cos(a)
                    py = y + r * math.sin(a)
                    self.box(
                        "Duct retaining bracket", (px, py, 0.45), (0.045, 0.045, 0.20), "trim"
                    )
                if self.steam:
                    self.pipe(
                        "Rotor copper steam line",
                        [(x * 0.4, s * 0.40, 0.80), (x, s * 0.78, 0.73), (x, y, 0.60)],
                        0.026,
                    )
                    self.cyl("Rotor steam piston", (x, s * 0.67, 0.55), 0.065, 0.24, "copper", "y")
                else:
                    self.box(
                        "Rotor status marker", (x, y + s * r, 0.56), (0.12, 0.02, 0.025), "light"
                    )
        if self.steam:
            self.boiler((-0.22, 0, 1.02), 0.20, 0.42, "z")
            self.gauge((0.15, -0.20, 0.94), 0.15)
            for s in [-1, 1]:
                self.pipe(
                    "Underslung suspension",
                    [(0.73, s * 0.27, 0.30), (0.3, s * 0.43, 0.08), (-0.62, s * 0.32, 0.18)],
                    0.036,
                    "trim",
                )
        else:
            self.cyl("Rear vector jet", (-1.21, 0, 0.49), 0.16, 0.30, "trim", "x")
            flame = self.empty("Rear jet plume", (-1.38, 0, 0.49), self.root, "thrust")
            self.cyl("Vector exhaust", (-0.15, 0, 0), 0.002, 0.3, "light", "x", 0.105, flame)
            self.box("Survey avionics", (-0.25, 0, 0.82), (0.48, 0.30, 0.10), "dark")
            self.box("Avionics status", (0.01, 0, 0.86), (0.028, 0.16, 0.02), "light")

    def harvester(self):
        self.box("Heavy ladder chassis", (-0.05, 0, 0.43), (3.65, 1.58, 0.35))
        self.wheels([0.93, -0.18, -1.27], 0.93, 0.58, 0.40)
        self.box("Forward processing deck", (0.87, 0, 1.03), (1.1, 1.40, 0.48))
        self.box(
            "Raised sensor cab",
            (0.98, -0.05, 1.69),
            (0.82, 1.08, 0.78),
            "dark" if self.steam else "plate",
        )
        self.box("Panoramic front cab glass", (1.401, -0.05, 1.74), (0.018, 0.87, 0.45), "glass")
        for s in [-1, 1]:
            self.box(
                "Side cab glass", (0.98, -0.05 + s * 0.548, 1.74), (0.61, 0.018, 0.45), "glass"
            )
            for x in [0.62, 1.35]:
                self.box(
                    "Cab corner frame", (x, -0.05 + s * 0.56, 1.74), (0.045, 0.045, 0.57), "trim"
                )
            self.cyl("Deck work lamp", (1.40, s * 0.56, 1.25), 0.10, 0.08, "trim", "x")
            self.cyl("Work lamp lens", (1.45, s * 0.56, 1.25), 0.075, 0.02, "light", "x")
        self.box("Cab roof visor", (0.99, -0.05, 2.12), (0.98, 1.23, 0.10), "dark")
        self.box("Cab sensor array", (1.1, -0.05, 2.23), (0.30, 0.32, 0.13), "trim")
        self.cyl("Amber hazard beacon", (0.68, 0.33, 2.23), 0.065, 0.15, "light")
        # Open ore hopper with inclined walls and a visible cargo bed.
        self.box("Ore hopper bed", (-0.80, 0, 1.18), (1.86, 1.38, 0.13))
        for s in [-1, 1]:
            wall = self.box(
                "Flared cargo hopper side", (-0.80, s * 0.78, 1.54), (1.95, 0.13, 0.72), "plate"
            )
            wall.rotation_euler.x = -s * 0.18
            self.pipe(
                "Hopper rim", [(0.20, s * 0.85, 1.91), (-1.83, s * 0.85, 1.91)], 0.042, "trim"
            )
            for x in [-1.65, -1.06, -0.45, 0.10]:
                self.box("Hopper external rib", (x, s * 0.86, 1.51), (0.065, 0.055, 0.69), "trim")
        for x in [-1.82, 0.20]:
            self.box("Hopper end wall", (x, 0, 1.51), (0.13, 1.69, 0.71), "dark")
        rng = np.random.default_rng(203)
        for i in range(8 if self.low else 24):
            x, y = -1.62 + rng.random() * 1.53, (rng.random() - 0.5) * 1.28
            r = 0.12 + rng.random() * 0.13
            self.cyl(
                "Collected mineral",
                (x, y, 1.38 + rng.random() * 0.24),
                r,
                r * 1.7,
                "gold" if self.steam else "energy",
                r2=r * 0.34,
                segments=5,
            )
        # Intake roller rotates independently of the two front steering hubs.
        drum = self.empty("Mineral intake roller", (1.91, 0, 0.56), self.root, "wheel")
        drum["wheelRadius"] = 0.39
        self.cyl("Rotary collector drum", (0, 0, 0), 0.35, 1.67, "trim", "y", parent=drum)
        for j in range(5 if self.low else 9):
            y = -0.72 + j * (1.44 / (4 if self.low else 8))
            self.ring("Collector cutting disc", (0, y, 0), 0.38, 0.036, "dark", "y", drum)
            for k in range(4 if self.low else 7):
                a = k * TAU / (4 if self.low else 7) + j * 0.30
                tooth = self.box(
                    "Intake iron tooth",
                    (math.cos(a) * 0.39, y, math.sin(a) * 0.39),
                    (0.13, 0.09, 0.12),
                    "trim",
                    drum,
                    0,
                )
                tooth.rotation_euler.y = -a
        for s in [-1, 1]:
            self.pipe(
                "Collector hydraulic arm",
                [(1.86, s * 0.94, 0.52), (1.44, s * 0.96, 0.99), (0.47, s * 0.96, 0.94)],
                0.10,
                "dark",
            )
            self.pipe(
                "Inclined conveyor housing",
                [(1.62, s * 0.76, 0.75), (0.35, s * 0.74, 1.42), (-0.27, s * 0.74, 1.6)],
                0.14,
                "dark",
            )
            self.pipe(
                "Conveyor edge rail",
                [(1.64, s * 0.91, 0.80), (0.35, s * 0.9, 1.51), (-0.27, s * 0.9, 1.69)],
                0.033,
                "trim",
            )
            self.box(
                "Side processing chamber",
                (-0.72, s * 0.80, 0.97),
                (1.25, 0.17, 0.22),
                "copper" if self.steam else "energy",
            )
        if self.steam:
            self.boiler((-1.0, 0.15, 2.12), 0.26, 1.28)
            for x in [-1.40, -0.85]:
                self.cyl("Soot black chimney", (x, 0.17, 2.50), 0.07, 0.44, "dark")
            self.gauge((1.22, -0.73, 1.05), 0.13)
        else:
            self.box("Rear processor", (-1.98, 0, 1.24), (0.37, 1.48, 0.92), "plate")
            for s in [-1, 1]:
                for z in [0.99, 1.14, 1.29, 1.44]:
                    self.box(
                        "Processing vent", (-2.175, s * 0.39, z), (0.02, 0.42, 0.055), "energy"
                    )

    def prop(self):
        if self.kind in {"dock", "gate"}:
            self.cyl("Platform foundation", (0, 0, 0.03), 1, 0.06, "dark", segments=32)
            for r in [0.83, 1.0]:
                self.ring(
                    "Indexed platform rim",
                    (0, 0, 0.10),
                    r,
                    0.033,
                    "trim" if self.steam else "light",
                )
            for j in range(8):
                a = j * TAU / 8
                self.box(
                    "Platform machinery",
                    (math.cos(a) * 0.95, math.sin(a) * 0.95, 0.17),
                    (0.15, 0.15, 0.25 if self.kind == "dock" else 0.10),
                    "trim",
                )
                self.cyl(
                    "Platform signal lamp",
                    (math.cos(a) * 0.95, math.sin(a) * 0.95, 0.31),
                    0.036,
                    0.024,
                    "light",
                )
        else:
            self.cyl("Reactor plinth", (0, 0, 0.14), 0.80, 0.28, "dark")
            self.cyl(
                "Reactor core", (0, 0, 0.85), 0.27, 1.10, "copper" if self.steam else "energy"
            )
            for z in [0.34, 0.62, 1.08, 1.40]:
                self.ring(
                    "Reactor confinement hoop",
                    (0, 0, z),
                    0.55,
                    0.043,
                    "trim" if self.steam else "light",
                )
            for j in range(4):
                a = j * TAU / 4
                self.pipe(
                    "Reactor support",
                    [
                        (math.cos(a) * 0.65, math.sin(a) * 0.65, 0.2),
                        (math.cos(a) * 0.65, math.sin(a) * 0.65, 1.35),
                    ],
                    0.045,
                    "trim",
                )

    def build(self):
        if self.kind in {"rocket", "kart", "drone", "harvester"}:
            getattr(self, self.kind)()
        else:
            self.prop()
        bpy.context.view_layer.update()
        if self.kind in {"rocket", "kart", "drone", "harvester"}:
            refine_vehicle(self)
            bpy.context.view_layer.update()
        if not self.low:
            self.concept_details()
            self.surface_details()
        if self.kind in {"rocket", "kart", "drone", "harvester"}:
            repair_vehicle_normals(self)
            finish_vehicle_surfaces(self)
        bpy.context.view_layer.update()
        # All meshes and motion origins share a uniform authoring-to-lab scale.
        coords = [
            o.matrix_world @ Vector(v)
            for o in self.scene.objects
            if o.type == "MESH" and not (o.parent and o.parent.get("motion") == "thrust")
            for v in o.bound_box
        ]
        size = max(max(v[i] for v in coords) - min(v[i] for v in coords) for i in [0, 1])
        factor = 1.52 / size if self.kind in {"rocket", "kart", "drone", "harvester"} else 1
        lo = min(v.z for v in coords)
        center = [(max(v[i] for v in coords) + min(v[i] for v in coords)) / 2 for i in [0, 1]]
        self.root.scale = (factor,) * 3
        self.root.location = (-center[0] * factor, -center[1] * factor, -lo * factor + 0.025)
        self.root["normalizationScale"] = factor
        bpy.context.view_layer.update()
        return self

    def concept_details(self):
        """Static mechanical assemblies batch into existing material draws at export."""
        if self.kind == "rocket":
            for s in [-1, 1]:
                for j in range(7):
                    x = -1.38 + j * 0.17
                    self.box(
                        "Engine radiator vane",
                        (x, s * 0.99, 0.66),
                        (0.055, 0.11, 0.20),
                        "copper" if self.steam else "dark",
                        bevel=0.006,
                    )
                if self.steam:
                    self.pipe(
                        "Paired pressure return",
                        [
                            (-1.45, s * 0.57, 0.84),
                            (-0.92, s * 0.57, 0.91),
                            (-0.3, s * 0.72, 0.85),
                            (-0.12, s * 0.72, 0.68),
                        ],
                        0.022,
                        "copper",
                    )
                    for x in [-1.32, -0.28]:
                        self.cyl(
                            "Pod pressure regulator", (x, s * 0.78, 0.86), 0.06, 0.075, "trim"
                        )
                        self.ring("Regulator handwheel", (x, s * 0.78, 0.91), 0.075, 0.012, "trim")
                else:
                    self.plate(
                        "Layered nose cheek",
                        [(0.2, s * 0.41), (0.65, s * 0.34), (1.52, s * 0.12), (1.2, s * 0.23)],
                        0.41,
                        0.035,
                        "plate",
                    )
                    self.box(
                        "Recessed engine telemetry",
                        (-1.12, s * 1.047, 0.55),
                        (0.23, 0.014, 0.11),
                        "dark",
                    )
                    for j in range(3):
                        self.box(
                            "Telemetry status segment",
                            (-1.19 + j * 0.065, s * 1.06, 0.55),
                            (0.038, 0.009, 0.025),
                            "light",
                            bevel=0,
                        )
        elif self.kind in {"kart", "harvester"}:
            axles = [0.94, -1.02] if self.kind == "kart" else [0.93, -0.18, -1.27]
            width, radius = (
                (0.72, 0.48 if self.steam else 0.43) if self.kind == "kart" else (0.93, 0.58)
            )
            for x in axles:
                for s in [-1, 1]:
                    center = Vector((x - 0.08, s * (width - 0.14), radius + 0.08))
                    self.cyl("Damper piston", center, 0.044, 0.38, "trim")
                    self.cyl(
                        "Suspension reservoir",
                        center + Vector((-0.08, 0, 0.04)),
                        0.055,
                        0.2,
                        "copper" if self.steam else "energy",
                    )
                    points = []
                    for j in range(65):
                        a = j * TAU * 5 / 64
                        points.append(
                            tuple(
                                center
                                + Vector((
                                    0.065 * math.cos(a),
                                    0.065 * math.sin(a),
                                    -0.15 + j * 0.3 / 64,
                                ))
                            )
                        )
                    self.pipe("Coil suspension spring", points, 0.014, "trim")
                    self.pipe(
                        "Upper wishbone",
                        [
                            (x - 0.23, s * 0.32, radius + 0.14),
                            (x, s * width, radius + 0.05),
                            (x + 0.23, s * 0.32, radius + 0.14),
                        ],
                        0.029,
                        "trim",
                    )
            if self.kind == "kart":
                if self.steam:
                    for s in [-1, 1]:
                        self.cyl(
                            "Steam drive cylinder",
                            (-0.82, s * 0.43, 0.55),
                            0.10,
                            0.38,
                            "copper",
                            "x",
                        )
                        for x in [-1.0, -0.85, -0.7]:
                            self.ring(
                                "Drive cylinder flange",
                                (x, s * 0.43, 0.55),
                                0.105,
                                0.014,
                                "trim",
                                "x",
                            )
                    for j in range(4):
                        self.box(
                            "Leather seat padded channel",
                            (-0.48, -0.19 + j * 0.125, 0.484),
                            (0.46, 0.095, 0.024),
                            "seat",
                            bevel=0.02,
                        )
                else:
                    self.box("Nose recessed intake", (1.42, 0, 0.285), (0.12, 0.29, 0.15), "dark")
                    for s in [-1, 1]:
                        self.plate(
                            "Front splitter wing",
                            [(1.1, s * 0.22), (1.56, s * 0.22), (1.6, s * 0.51), (1.25, s * 0.49)],
                            0.20,
                            0.033,
                            "trim",
                        )
                        self.pipe(
                            "Nose inset light guide",
                            [
                                (1.43, s * 0.15, 0.374),
                                (0.84, s * 0.19, 0.58),
                                (0.18, s * 0.225, 0.686),
                            ],
                            0.012,
                            "light",
                        )
                        self.cyl(
                            "Sensor side optical pivot",
                            (-0.32, s * 0.195, 0.93),
                            0.085,
                            0.035,
                            "trim",
                            "y",
                        )
                        self.cyl(
                            "Sensor side lens",
                            (-0.32, s * 0.215, 0.93),
                            0.048,
                            0.008,
                            "glass",
                            "y",
                        )
                        self.box(
                            "Battery heat sink",
                            (-1.12, s * 0.34, 0.59),
                            (0.40, 0.06, 0.20),
                            "trim",
                        )
                        for j in range(7):
                            self.box(
                                "Heat sink fin",
                                (-1.28 + j * 0.05, s * 0.38, 0.61),
                                (0.015, 0.065, 0.18),
                                "dark",
                                bevel=0,
                            )
            else:
                for s in [-1, 1]:
                    for j in range(11):
                        t = j / 10
                        x = 1.5 - t * 1.52
                        z = 0.87 + t * 0.75
                        tread = self.box(
                            "Conveyor transverse cleat",
                            (x, s * 0.75, z),
                            (0.10, 0.22, 0.035),
                            "trim",
                            bevel=0,
                        )
                        tread.rotation_euler.y = 0.46
                    self.pipe(
                        "Hydraulic pressure hose",
                        [(1.55, s * 0.99, 0.72), (0.9, s * 1.0, 1.05), (0.34, s * 0.94, 1.0)],
                        0.022,
                        "copper" if self.steam else "rubber",
                    )
                    for j in range(3):
                        self.box(
                            "Cab access step",
                            (0.4, s * 0.72, 0.57 + j * 0.20),
                            (0.31, 0.25, 0.04),
                            "trim",
                        )
                    for x in [-1.5, -0.9, -0.3]:
                        self.box(
                            "Hopper lower hinge", (x, s * 0.86, 1.20), (0.16, 0.08, 0.10), "trim"
                        )
                self.pipe(
                    "Cab windshield wiper",
                    [(1.418, -0.38, 1.55), (1.425, 0.20, 1.88)],
                    0.012,
                    "rubber",
                )
        elif self.kind == "drone":
            reach = 1.12 if self.steam else 0.83
            for x in [-0.73, 0.70]:
                for s in [-1, 1]:
                    if self.steam:
                        for dx in [-0.07, 0.07]:
                            self.pipe(
                                "Rotor torque linkage",
                                [(x * 0.6 + dx, s * 0.4, 0.46), (x + dx, s * reach, 0.46)],
                                0.025,
                                "copper",
                            )
                        self.cyl("Rotor gearbox", (x, s * 0.74, 0.49), 0.105, 0.18, "trim", "y")
                        for y in [0.66, 0.78]:
                            self.ring(
                                "Gearbox retaining flange",
                                (x, s * y, 0.49),
                                0.115,
                                0.015,
                                "trim",
                                "y",
                            )
                    else:
                        for j in range(6):
                            a = j * TAU / 6
                            self.box(
                                "Duct service fastener",
                                (x + 0.48 * math.cos(a), s * reach + 0.48 * math.sin(a), 0.555),
                                (0.035, 0.035, 0.02),
                                "trim",
                                bevel=0,
                            )
            for s in [-1, 1]:
                for j in range(7):
                    self.box(
                        "Avionics cooling slot",
                        (-0.35 + j * 0.1, s * 0.39, 0.787),
                        (0.045, 0.16, 0.015),
                        "dark",
                        bevel=0,
                    )
                if self.steam:
                    self.cyl(
                        "Upper auxiliary pressure vessel",
                        (-0.25, s * 0.38, 0.88),
                        0.073,
                        0.56,
                        "copper",
                        "x",
                    )
                    for x in [-0.45, -0.15]:
                        self.ring(
                            "Auxiliary vessel strap",
                            (x, s * 0.38, 0.88),
                            0.078,
                            0.012,
                            "trim",
                            "x",
                        )
                    self.pipe(
                        "Pressure distributor",
                        [(-0.42, s * 0.35, 1.0), (-0.42, s * 0.48, 0.95), (0.53, s * 0.44, 0.75)],
                        0.024,
                        "copper",
                    )
            for j in range(12):
                a = j * TAU / 12
                self.cyl(
                    "Optical bezel screw",
                    (1.183, 0.225 * math.cos(a), 0.49 + 0.225 * math.sin(a)),
                    0.016,
                    0.028,
                    "trim",
                    "x",
                    segments=6,
                )
            self.ring("Optical iris surround", (1.23, 0, 0.49), 0.105, 0.014, "trim", "x")

    def surface_details(self):
        """Inset replaceable armor panels, fasteners, and pod cooling hardware."""
        targets = [
            o
            for o in self.scene.objects
            if o.type == "MESH"
            and any(
                k in o.name.lower()
                for k in [
                    "fuselage",
                    "survey hull",
                    "bonnet",
                    "wedge nose",
                    "hopper side",
                    "sensor cab",
                    "processor",
                    "upper armored",
                ]
            )
        ]
        for obj in targets:
            for face in list(obj.data.polygons):
                if len(face.vertices) != 4 or face.area < 0.22:
                    continue
                coords = [obj.data.vertices[i].co.copy() for i in face.vertices]
                n = face.normal.copy()
                if n.z < -0.3:
                    continue
                a, b, c, d = coords
                count = max(1, min(5, round(max((b - a).length, (c - d).length) / 0.44)))
                for j in range(count):
                    u0 = (j + 0.035) / count
                    u1 = (j + 0.965) / count
                    corners = [
                        a.lerp(b, u0).lerp(d.lerp(c, u0), 0.06),
                        a.lerp(b, u1).lerp(d.lerp(c, u1), 0.06),
                        a.lerp(b, u1).lerp(d.lerp(c, u1), 0.94),
                        a.lerp(b, u0).lerp(d.lerp(c, u0), 0.94),
                    ]
                    corners = [obj.matrix_local @ (v + n * 0.018) for v in corners]
                    self.mesh(
                        "Replaceable armor panel",
                        [tuple(v) for v in corners],
                        [(0, 1, 2, 3)],
                        "plate"
                        if (j % 3 == 0 and not self.steam)
                        else ("dark" if self.steam else "trim"),
                        obj.parent,
                    )
                    if self.steam:
                        for v in corners:
                            bolt = self.cyl(
                                "Raised brass panel fastener",
                                v,
                                0.019,
                                0.015,
                                "trim",
                                parent=obj.parent,
                                segments=6,
                            )
                            bolt.rotation_mode = "QUATERNION"
                            bolt.rotation_quaternion = (
                                obj.matrix_local.to_3x3() @ n
                            ).to_track_quat("Z", "Y")
        if self.kind == "rocket":
            for side in [-1, 1]:
                for x in [-1.25, -1.05, -0.85, -0.65, -0.45]:
                    self.box(
                        "Pod service plate",
                        (x, side * 0.78, 0.803),
                        (0.14, 0.16, 0.018),
                        "dark",
                        bevel=0.01,
                    )
                for j in range(10):
                    a = j * TAU / 10
                    self.pipe(
                        "Nozzle cooling rib",
                        [
                            (-1.58, side * 0.78 + math.cos(a) * 0.22, 0.54 + math.sin(a) * 0.22),
                            (-1.84, side * 0.78 + math.cos(a) * 0.30, 0.54 + math.sin(a) * 0.30),
                        ],
                        0.015,
                        "trim",
                    )

    def export(self, destination):
        """Batch evaluated static geometry per material and moving pivot for few draws."""
        bpy.context.window.scene = self.scene
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        batches = {}
        originals = [o for o in self.scene.objects if o.type == "MESH"]
        for obj in originals:
            ev = obj.evaluated_get(deps)
            mesh = ev.to_mesh()
            key = (obj.parent, obj.data.materials[0])
            vs, fs, uvs, smooth = batches.setdefault(key, ([], [], [], []))
            transform = obj.parent.matrix_world.inverted() @ obj.matrix_world
            offset = len(vs)
            vs.extend(tuple(transform @ v.co) for v in mesh.vertices)
            for p in mesh.polygons:
                fs.append(tuple(offset + i for i in p.vertices))
                uvs.append([tuple(mesh.uv_layers.active.data[i].uv) for i in p.loop_indices])
                smooth.append(p.use_smooth)
            ev.to_mesh_clear()
        temp = []
        tris = 0
        for (parent, mat), (vs, fs, uvs, smooth) in batches.items():
            data = bpy.data.meshes.new("Export batch")
            data.from_pydata(vs, [], fs)
            data.update()
            layer = data.uv_layers.new()
            for p, coords, is_smooth in zip(data.polygons, uvs, smooth):
                p.use_smooth = is_smooth
                for li, uv in zip(p.loop_indices, coords):
                    layer.data[li].uv = uv
            data.materials.append(mat)
            obj = bpy.data.objects.new(f"{parent.name} / {mat.name}", data)
            self.scene.collection.objects.link(obj)
            obj.parent = parent
            temp.append(obj)
            data.calc_loop_triangles()
            tris += len(data.loop_triangles)
        budget = (
            (6000 if self.kind == "harvester" else 3000)
            if self.low
            else (80000 if self.kind == "harvester" else 50000)
        )
        budget = getattr(self, "triangle_budget", budget)
        if self.low and tris > budget:
            for obj in temp:
                mod = obj.modifiers.new("Crowd LOD reduction", "DECIMATE")
                mod.ratio = (budget * 0.94) / tris
        bpy.ops.object.select_all(action="DESELECT")
        for obj in [*temp, *[o for o in self.scene.objects if o.type == "EMPTY"]]:
            obj.select_set(True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep the running lab's previous GLB readable until its replacement is complete.
        staging = destination.with_name(f".{destination.stem}.pending.glb")
        bpy.ops.export_scene.gltf(
            filepath=str(staging),
            export_format="GLB",
            use_selection=True,
            use_active_scene=True,
            export_yup=False,
            export_extras=True,
            export_animations=False,
            export_apply=True,
            export_cameras=False,
            export_lights=False,
        )
        staging.replace(destination)
        for obj in temp:
            bpy.data.objects.remove(obj, do_unlink=True)
        return {
            "sourceTriangles": tris,
            "drawMeshes": len(batches),
            "bytes": destination.stat().st_size,
        }


def studio(builder):
    scene = builder.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("Lab studio world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
        (0.20, 0.17, 0.13, 1) if builder.steam else (0.12, 0.16, 0.22, 1)
    )
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    scene.view_settings.view_transform = "AgX"
    for name, pos, power, size, color in [
        ("Key", (2, -3, 4), 400, 4, (1, 0.85, 0.68) if builder.steam else (0.72, 0.88, 1)),
        ("Rim", (-2, 1, 3), 550, 3, (1, 0.55, 0.27) if builder.steam else (0.56, 0.65, 1)),
        ("Fill", (1, 4, 2), 350, 3, (1, 1, 1)),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power
        data.shape = "DISK"
        data.size = size
        data.color = color
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = pos
        obj.rotation_euler = (
            (Vector((0, 0, 0.3)) - obj.location).to_track_quat("-Z", "Y").to_euler()
        )
    data = bpy.data.cameras.new("Asset reference camera")
    camera = bpy.data.objects.new("Asset reference camera", data)
    scene.collection.objects.link(camera)
    data.type = "ORTHO"
    data.ortho_scale = 2.20
    scene.camera = camera
    scene.render.film_transparent = False
    return camera


def frame_camera(scene, direction):
    """Frame all authored geometry with consistent margins in any reference view."""
    bpy.context.window.scene = scene
    bpy.context.view_layer.update()
    corners = [
        o.matrix_world @ Vector(v) for o in scene.objects if o.type == "MESH" for v in o.bound_box
    ]
    lo = Vector(tuple(min(v[i] for v in corners) for i in range(3)))
    hi = Vector(tuple(max(v[i] for v in corners) for i in range(3)))
    center = (lo + hi) / 2
    camera = scene.camera
    camera.location = center + Vector(direction).normalized() * 5
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    projected = [camera.matrix_world.inverted() @ v for v in corners]
    width = max(v.x for v in projected) - min(v.x for v in projected)
    height = max(v.y for v in projected) - min(v.y for v in projected)
    camera.data.ortho_scale = (
        max(width, height * scene.render.resolution_x / scene.render.resolution_y) * 1.18
    )


def build_asset(style, kind, render=False):
    high = Builder(style, kind).build()
    low = Builder(style, kind, True).build()
    info = {
        "high": high.export(ROOT / style / f"{kind}-high.glb"),
        "low": low.export(ROOT / style / f"{kind}-low.glb"),
    }
    bpy.context.window.scene = high.scene
    studio(high)
    preview = ROOT / "previews" / style
    preview.mkdir(parents=True, exist_ok=True)
    views = [("hero", (2.8, -3.4, 2.5)), ("side", (0, -4, 0.9)), ("top", (0, 0, 5))]
    if kind in {"rocket", "kart", "drone", "harvester"}:
        views.append(("rear", (-3, -3, 1.8)))
    for name, pos in views:
        frame_camera(high.scene, pos)
        if render:
            high.scene.render.filepath = str(preview / f"{kind}-{name}.png")
            bpy.ops.render.render(write_still=True)
    frame_camera(high.scene, (2.8, -3.4, 2.5))
    sources = ROOT / "sources" / style
    sources.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(
        str(sources / f"{kind}.blend"), {high.scene, low.scene}, fake_user=True, compress=True
    )
    info["style"] = style
    info["model"] = kind
    (ROOT / style / f"{kind}-build.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    for style in ["futuristic", "steampunk"]:
        for kind in ["rocket", "kart", "drone", "harvester", "dock", "gate", "reactor"]:
            print(
                build_asset(style, kind, render=kind in {"rocket", "kart", "drone", "harvester"})
            )
