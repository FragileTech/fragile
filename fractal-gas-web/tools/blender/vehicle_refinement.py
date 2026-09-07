"""Concept-specific silhouettes and equipment; shared by detailed and crowd builds.

All geometry is authored before the common footprint normalization. Small service
parts use existing materials and batch with the hull, leaving motion pivots intact.
"""

import math

import bmesh
import bpy
import numpy as np


def remove(b, *prefixes):
    for obj in list(b.scene.objects):
        if any(obj.name.startswith(prefix) for prefix in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)


def oval_loft(b, name, sections, mat="dark", segments=None):
    """Closed elliptical sections along X, with proper cylindrical UV continuity."""
    n = segments or (8 if b.low else 20)
    verts = [
        (
            x,
            w * math.cos(j * math.tau / n),
            (lo + hi) / 2 + (hi - lo) / 2 * math.sin(j * math.tau / n),
        )
        for x, w, lo, hi in sections
        for j in range(n)
    ]
    faces = [tuple(reversed(range(n)))]
    for i in range(len(sections) - 1):
        faces.extend(
            (i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j)
            for j in range(n)
        )
    faces.append(tuple(range((len(sections) - 1) * n, len(sections) * n)))
    obj = b.mesh(name, verts, faces, mat)
    for poly in obj.data.polygons[1:-1]:
        poly.use_smooth = True
    return obj


def side_plate(b, name, points, y, thick, mat="plate"):
    obj = b.plate(name, points, 0, thick, mat)
    obj.rotation_euler.x = math.pi / 2
    obj.location.y = y + thick / 2
    return obj


def mechanical_finish(b):
    # A single small roughness map is shared by the exposed metals. It adds no draws.
    n = 128 if b.low else 512
    y, x = np.mgrid[0:n, 0:n] / n
    rng = np.random.default_rng(481)
    fine = rng.random((n, n))
    stain = np.sin(x * 31 + np.sin(y * 13) * 2) * np.sin(y * 27 + x * 5)
    brushed = np.sin(y * 1100 + np.sin(x * 21))
    rough = np.clip(0.42 + stain * 0.09 + brushed * 0.035 + (fine - 0.5) * 0.10, 0.25, 0.65)
    im = bpy.data.images.new(f"{b.style} brushed metal roughness", width=n, height=n)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(
        np.stack([rough, rough, rough, np.ones_like(x)], -1).astype(np.float32).ravel()
    )
    im.pack()
    for key in ["trim", "copper", "gold"]:
        mat = b.mats[key]
        p = mat.node_tree.nodes.get("Principled BSDF")
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = im
        mat.node_tree.links.new(tex.outputs["Color"], p.inputs["Roughness"])
    # The sheets use aged brass/copper, not uniformly bright orange hardware.
    for key, color in (
        [("trim", (0.38, 0.255, 0.095)), ("copper", (0.32, 0.12, 0.052))]
        if b.steam
        else [("trim", (0.19, 0.22, 0.25)), ("copper", (0.075, 0.065, 0.095))]
    ):
        b.mats[key].node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (
            *color,
            1,
        )
    glass = b.mats["glass"].node_tree.nodes.get("Principled BSDF")
    glass.inputs["Metallic"].default_value = 0.15
    glass.inputs["Roughness"].default_value = 0.10
    glass.inputs["Base Color"].default_value = (
        (0.22, 0.085, 0.015, 1) if b.steam else (0.008, 0.055, 0.075, 1)
    )
    glass.inputs["Emission Strength"].default_value = 0
    b.mats["energy"].node_tree.nodes.get("Principled BSDF").inputs[
        "Emission Strength"
    ].default_value = 0.65
    if b.kind == "rocket":
        # Tinted, ordinary alpha glazing avoids a transmission/refraction render pass.
        # Crowds use an opaque approximation, so their instanced path stays opaque.
        glass.inputs["Base Color"].default_value = (
            (0.16, 0.055, 0.012, 1) if b.steam else (0.012, 0.12, 0.20, 1)
        )
        if not b.low:
            glass.inputs["Alpha"].default_value = 0.72
            b.mats["glass"].surface_render_method = "BLENDED"


def rocket(b):
    for s in [-1, 1]:
        b.box(
            "Vernier armored mounting bracket", (0.15, s * 0.39, 0.28), (0.28, 0.20, 0.22), "dark"
        )
    remove(b, "Dorsal stabilizer")
    points = [(-1.60, 0), (-1.58, 0.88), (-1.40, 0.98), (-0.58, 0.03)]
    fin = side_plate(b, "Swept dorsal fin", points, 0, 0.06, "dark")
    fin.location.z = 0.67
    inset = side_plate(
        b,
        "Dorsal inset panel",
        [(-1.51, 0.16), (-1.49, 0.78), (-1.39, 0.85), (-0.78, 0.16)],
        -0.033,
        0.012,
        "plate",
    )
    inset.location.z = 0.67
    if b.steam:
        remove(b, "Riveted tapered fuselage", "Amber segmented canopy", "Brass canopy frame")
        oval_loft(
            b,
            "Riveted round fuselage",
            [
                (-1.55, 0.34, 0.22, 0.82),
                (-0.8, 0.47, 0.15, 0.99),
                (-0.1, 0.46, 0.17, 0.96),
                (0.72, 0.34, 0.22, 0.80),
                (1.50, 0.14, 0.28, 0.52),
                (1.95, 0.018, 0.33, 0.37),
            ],
        )
        oval_loft(
            b,
            "Closed arched amber canopy",
            [
                (-0.5, 0.29, 0.68, 1.04),
                (-0.18, 0.31, 0.66, 1.075),
                (0.32, 0.275, 0.64, 1.01),
                (0.85, 0.125, 0.55, 0.79),
            ],
            "glass",
        )
        for x, w, z, rz in [
            (-0.48, 0.298, 0.85, 0.20),
            (-0.05, 0.307, 0.85, 0.215),
            (0.45, 0.238, 0.80, 0.17),
        ]:
            points = [
                (x, w * math.cos(a * math.pi / 8), z + rz * math.sin(a * math.pi / 8))
                for a in range(9)
            ]
            b.pipe("Arched canopy mullion", points, 0.017, "trim")
        for s in [-1, 1]:
            b.pipe(
                "Canopy sill",
                [
                    (-0.50, s * 0.29, 0.85),
                    (-0.05, s * 0.31, 0.85),
                    (0.45, s * 0.24, 0.80),
                    (0.85, s * 0.12, 0.66),
                ],
                0.019,
                "trim",
            )
            for x in [-1.34, -0.88]:
                band = b.ring("Fuselage pressure seam", (x, 0, 0.56), 0.40, 0.018, "trim", "x")
                band.scale.z = 0.93
            b.cyl(
                "Pod intake stepped flange",
                (0.005, s * 0.78, 0.54),
                0.16,
                0.13,
                "trim",
                "x",
                0.115,
            )
            b.cyl(
                "Intake central spindle", (0.095, s * 0.78, 0.54), 0.09, 0.10, "copper", "x", 0.045
            )
    else:
        # Broad shoulder chines and swept wings are conspicuous in the top reference.
        for s in [-1, 1]:
            b.plate(
                "Arrowhead shoulder armor",
                [
                    (-1.32, s * 0.40),
                    (-0.38, s * 0.57),
                    (0.75, s * 0.36),
                    (1.70, s * 0.10),
                    (0.34, s * 0.56),
                    (-0.78, s * 0.66),
                ],
                0.37,
                0.085,
                "plate",
            )
            b.plate(
                "Swept outer wing armor",
                [(-1.48, s * 0.49), (-0.37, s * 0.42), (-0.40, s * 0.94), (-1.25, s * 1.16)],
                0.30,
                0.065,
                "plate",
            )
            b.cyl(
                "Forward turbine collar", (-0.005, s * 0.78, 0.54), 0.19, 0.14, "trim", "x", 0.13
            )
            b.cyl(
                "Forward cyan turbine", (0.085, s * 0.78, 0.54), 0.115, 0.085, "light", "x", 0.078
            )
        if not b.low:
            for s in [-1, 1]:
                for x in [-1.30, -0.93, -0.55]:
                    b.loft(
                        "Engine segmented armor",
                        [(x - 0.14, 0.24, 0.62, 0.80), (x + 0.14, 0.24, 0.62, 0.80)],
                        "plate",
                    ).location.y = s * 0.78


def kart(b):
    if b.steam:
        remove(
            b,
            "Riveted bonnet",
            "Bonnet center accent",
            "Seat back",
            "Rear seat hoop",
            "Rear hoop crossbar",
        )
        oval_loft(
            b,
            "Rounded riveted bonnet",
            [
                (0.02, 0.38, 0.27, 0.74),
                (0.65, 0.34, 0.23, 0.64),
                (1.30, 0.27, 0.20, 0.50),
                (1.53, 0.25, 0.20, 0.46),
            ],
        )
        b.loft(
            "Burgundy bonnet inlay",
            [
                (0.04, 0.115, 0.715, 0.735),
                (0.65, 0.105, 0.625, 0.645),
                (1.45, 0.065, 0.455, 0.475),
            ],
            "plate",
        )
        # Rounded seat silhouette, with visible padded channels at close range.
        back = b.cyl("Leather bucket seat back", (-0.81, 0, 0.78), 0.31, 0.10, "seat", "x")
        back.scale.z = 1.12
        back.rotation_euler.y = -0.2
        hoop = [
            (-0.96, 0.32 * math.cos(a * math.pi / 10), 0.90 + 0.37 * math.sin(a * math.pi / 10))
            for a in range(11)
        ]
        b.pipe(
            "Rounded brass seat hoop",
            [(-0.93, 0.32, 0.42), *hoop, (-0.93, -0.32, 0.42)],
            0.027,
            "trim",
        )
        b.box(
            "Recessed radiator grille",
            (1.535, 0, 0.335),
            (0.025, 0.38, 0.20),
            "rubber",
            bevel=0.012,
        )
        for y in [-0.18, -0.09, 0, 0.09, 0.18]:
            b.pipe(
                "Brass radiator vertical slat",
                [(1.553, y, 0.235), (1.553, y, 0.425)],
                0.012,
                "trim",
            )
        b.pipe(
            "Rounded grille surround",
            [
                (1.55, -0.21, 0.23),
                (1.56, -0.21, 0.39),
                (1.56, -0.14, 0.46),
                (1.56, 0.14, 0.46),
                (1.56, 0.21, 0.39),
                (1.55, 0.21, 0.23),
                (1.55, -0.21, 0.23),
            ],
            0.023,
            "trim",
        )
        for s in [-1, 1]:
            side_plate(
                b,
                "Open cockpit side sill",
                [
                    (-1.12, 0.28),
                    (-1.05, 0.58),
                    (-0.66, 0.44),
                    (-0.15, 0.40),
                    (0.08, 0.66),
                    (0.16, 0.25),
                ],
                s * 0.38,
                0.07,
                "dark",
            )
        if not b.low:
            for y in [-0.20, -0.10, 0, 0.10, 0.20]:
                b.box(
                    "Vertical leather bolster",
                    (-0.743, y, 0.78),
                    (0.06, 0.075, 0.45),
                    "seat",
                    bevel=0.03,
                )
    else:
        # Continuous armor around the cockpit and rear battery; preserve the sensor pod.
        for s in [-1, 1]:
            side_plate(
                b,
                "Faceted cockpit side armor",
                [
                    (-1.08, 0.30),
                    (-1.08, 0.60),
                    (-0.72, 0.66),
                    (-0.40, 0.48),
                    (0.03, 0.48),
                    (0.27, 0.69),
                    (0.45, 0.51),
                    (0.22, 0.26),
                ],
                s * 0.43,
                0.10,
                "plate",
            )
            side_plate(
                b,
                "Graphite rocker inset",
                [(-0.99, 0.29), (-0.93, 0.44), (-0.45, 0.39), (0.16, 0.36), (0.25, 0.27)],
                s * 0.488,
                0.012,
                "dark",
            )
            b.pipe(
                "Inset rocker cyan rail",
                [
                    (-1.0, s * 0.50, 0.30),
                    (-0.72, s * 0.50, 0.30),
                    (-0.48, s * 0.50, 0.40),
                    (0.12, s * 0.50, 0.40),
                ],
                0.012,
                "light",
            )
        remove(b, "Autonomous sensor helmet")
        oval_loft(
            b,
            "Rounded autonomous sensor shell",
            [
                (-0.61, 0.09, 0.82, 0.99),
                (-0.48, 0.19, 0.76, 1.10),
                (-0.28, 0.21, 0.77, 1.105),
                (-0.10, 0.15, 0.79, 1.01),
            ],
            "plate",
            12 if b.low else 24,
        )
        if not b.low:
            for s in [-1, 1]:
                for x in [-1.27, -1.08]:
                    b.cyl("Rear drive motor casing", (x, s * 0.41, 0.58), 0.105, 0.23, "trim", "y")
    # Road tires have shallow tread grooves, not the harvesters' chunky mining lugs.
    for obj in b.scene.objects:
        if obj.name.startswith("Tire tread block"):
            obj.scale.z = 0.30


def drone(b):
    remove(b, "Diamond survey hull")
    if not b.steam:
        # The drone sheet uses graphite alloy rather than the rocket's pale armor.
        shader = b.mats["plate"].node_tree.nodes.get("Principled BSDF")
        atlas = shader.inputs["Base Color"].links[0].from_node.image
        pixels = np.empty(len(atlas.pixels), dtype=np.float32)
        atlas.pixels.foreach_get(pixels)
        pixels.reshape(-1, 4)[:, :3] *= 0.52
        atlas.pixels.foreach_set(pixels)
        atlas.pack()
        b.loft(
            "Low armored drone hull",
            [
                (-1.2, 0.16, 0.31, 0.57),
                (-0.45, 0.57, 0.29, 0.65),
                (0.45, 0.59, 0.29, 0.65),
                (1.06, 0.20, 0.33, 0.55),
            ],
            "dark",
        )
        for x in [-0.73, 0.70]:
            for s in [-1, 1]:
                # Bridge the inner half of each duct to the central armor without
                # covering its aperture. Curved edge follows the duct's outer rim.
                arc = [
                    (
                        x + 0.49 * math.cos(a * math.pi / 8),
                        s * (0.83 - 0.49 * math.sin(a * math.pi / 8)),
                    )
                    for a in range(9)
                ]
                points = [(x + 0.49, s * 0.27), *arc, (x - 0.49, s * 0.27)]
                b.plate("Integrated rotor shoulder", points, 0.55, 0.07, "plate")
        remove(b, "Upper armored shell")
        b.loft(
            "Layered avionics armor",
            [
                (-1.03, 0.24, 0.58, 0.72),
                (-0.55, 0.56, 0.59, 0.75),
                (0.35, 0.58, 0.57, 0.74),
                (0.93, 0.26, 0.50, 0.64),
            ],
            "plate",
        )
        for s in [-1, 1]:
            for z, yy, r in [(0.64, 0.28, 0.07), (0.34, 0.28, 0.095)]:
                b.cyl("Auxiliary golden optical bezel", (1.075, s * yy, z), r, 0.08, "trim", "x")
                b.cyl("Auxiliary optical lens", (1.12, s * yy, z), r * 0.69, 0.018, "gold", "x")
        lens = oval_loft(
            b,
            "Convex gold survey lens",
            [(1.19, 0.185, 0.305, 0.675), (1.29, 0.15, 0.34, 0.64), (1.33, 0.055, 0.435, 0.545)],
            "gold",
        )
        lens["opticalLens"] = True
        b.cyl("Survey iris aperture", (1.333, 0, 0.49), 0.058, 0.016, "dark", "x")
        b.cyl("Survey iris illuminator", (1.344, 0, 0.49), 0.028, 0.008, "light", "x")
    else:
        b.loft(
            "Diamond pressure hull",
            [(-1.2, 0.07, 0.35, 0.57), (0, 0.61, 0.18, 0.60), (1.08, 0.11, 0.33, 0.57)],
            "dark",
        )
        remove(b, "Upper armored shell")
        # Burgundy diamond facets divided by broad brass seams.
        for s in [-1, 1]:
            vertices = [(-1.06, 0, 0.64), (0, s * 0.60, 0.54), (1.04, 0, 0.60), (0, 0, 0.91)]
            faces = [(0, 3, 1), (1, 3, 2)] if s > 0 else [(0, 1, 3), (1, 2, 3)]
            b.mesh("Diamond burgundy armor facet", vertices, faces, "plate")
            b.pipe("Diamond hull brass perimeter", [*vertices, vertices[0]], 0.018, "trim")
        for x in [-0.73, 0.70]:
            for s in [-1, 1]:
                b.cyl("Rotor bearing crown", (x, s * 1.12, 0.64), 0.10, 0.08, "trim")
                if not b.low:
                    for j in range(8):
                        a = j * math.tau / 8
                        b.cyl(
                            "Duct crown rivet",
                            (x + 0.48 * math.cos(a), s * 1.12 + 0.48 * math.sin(a), 0.566),
                            0.012,
                            0.018,
                            "trim",
                            segments=6,
                        )
        b.cyl("Rounded pressure dome", (-0.22, 0, 1.24), 0.19, 0.08, "trim", r2=0.12)
        b.pipe(
            "Underslung return manifold",
            [(-0.72, -0.34, 0.23), (-0.85, 0, 0.11), (-0.72, 0.34, 0.23)],
            0.032,
            "copper",
        )


def harvester(b):
    # The concept puts the intake inside a heavy articulated casing, with covered
    # wheel arches and a visibly sloping hopper rather than exposed box primitives.
    # Reprofile existing hopper walls and cargo, preserving topology and pivots.
    # The concepts have tapered heavy bins and low, irregular loads.
    for obj in b.scene.objects:
        if obj.name.startswith("Flared cargo hopper side"):
            for vertex in obj.data.vertices:
                if vertex.co.z < 0:
                    vertex.co.x *= 0.83
        elif obj.name.startswith("Collected mineral"):
            obj.scale.z = 0.65
            obj.rotation_euler.x = 0.27 * math.sin(obj.location.x * 17)
            obj.rotation_euler.y = 0.35 * math.cos(obj.location.y * 19)
    for s in [-1, 1]:
        side_plate(
            b,
            "Collector cheek armor",
            [(1.13, 0.36), (1.35, 0.98), (1.69, 1.03), (2.14, 0.73), (2.32, 0.22), (1.85, 0.18)],
            s * 0.90,
            0.13,
            "dark" if b.steam else "plate",
        )
        side_plate(
            b,
            "Collector cheek inset",
            [(1.37, 0.40), (1.50, 0.85), (1.70, 0.87), (2.08, 0.64), (2.16, 0.30), (1.86, 0.27)],
            s * 0.978,
            0.014,
            "plate" if b.steam else "dark",
        )
        b.cyl("Collector side axle cover", (1.91, s * 1.0, 0.56), 0.15, 0.10, "trim", "y")
        for x in [0.93, -0.18, -1.27]:
            side_plate(
                b,
                "Wheel arch armor",
                [
                    (x - 0.52, 0.87),
                    (x - 0.37, 1.18),
                    (x + 0.32, 1.18),
                    (x + 0.52, 0.87),
                    (x + 0.38, 0.82),
                    (x + 0.26, 1.02),
                    (x - 0.26, 1.02),
                    (x - 0.39, 0.82),
                ],
                s * 0.99,
                0.12,
                "dark" if b.steam else "plate",
            )
        for x in [-1.64, -0.10]:
            side_plate(
                b,
                "Angled hopper reinforcement",
                [
                    (x - 0.14, 1.23),
                    (x - 0.22, 1.79),
                    (x - 0.12, 1.96),
                    (x + 0.12, 1.96),
                    (x + 0.22, 1.77),
                    (x + 0.13, 1.23),
                ],
                s * 0.86,
                0.10,
                "dark" if b.steam else "plate",
            )
        b.pipe(
            "Cab hand rail",
            [
                (0.55, s * 0.70, 1.17),
                (0.55, s * 0.70, 1.50),
                (1.20, s * 0.70, 1.50),
                (1.20, s * 0.70, 1.17),
            ],
            0.022,
            "trim",
        )
        # Separate segmented conveyors leave the harvesting drum's pivot untouched.
        side_plate(
            b,
            "Conveyor articulated side plate",
            [(-0.34, 1.46), (0.34, 1.40), (1.50, 0.74), (1.40, 1.01), (0.40, 1.68), (-0.34, 1.66)],
            s * 0.79,
            0.10,
            "dark",
        )
        for x, z in [(0.35, 1.52), (1.36, 0.98)]:
            b.cyl("Conveyor sprocket cover", (x, s * 0.86, z), 0.15, 0.08, "trim", "y")
    # Pitched roof, glass mullions and instrument brow give the cab its own form.
    remove(b, "Cab roof visor", "Cab sensor array")
    b.loft(
        "Faceted cab roof",
        [
            (0.47, 0.63, 2.06, 2.12),
            (0.62, 0.61, 2.07, 2.22),
            (1.36, 0.61, 2.07, 2.22),
            (1.53, 0.61, 2.06, 2.12),
        ],
        "dark" if b.steam else "plate",
    )
    for y in [-0.33, 0.23]:
        b.box("Front glass mullion", (1.417, y, 1.74), (0.03, 0.028, 0.49), "trim")
    b.box("Cab lower instrument brow", (1.43, -0.05, 1.48), (0.12, 1.03, 0.12), "dark")
    b.box("Low roof sensor array", (1.02, -0.05, 2.25), (0.38, 0.30, 0.10), "trim")
    # The steampunk concept carries irregular ore, not gold crystals.
    if b.steam:
        remove(b, "Collected mineral")
        rng = np.random.default_rng(203)
        for i in range(8 if b.low else 28):
            x, y = -1.62 + rng.random() * 1.53, (rng.random() - 0.5) * 1.25
            size = 0.09 + rng.random() * 0.10
            obj = b.cyl(
                "Collected ore nugget",
                (x, y, 1.39 + rng.random() * 0.20),
                size,
                size * 1.2,
                "gold" if i % 3 else "dark",
                r2=size * 0.6,
                segments=5,
            )
            obj.rotation_euler = (rng.random(), rng.random(), rng.random())
    if not b.low:
        for s in [-1, 1]:
            for x in [-1.30, -0.70, -0.10]:
                b.cyl("Processing conduit collar", (x, s * 0.92, 0.97), 0.12, 0.055, "trim", "x")
            if not b.steam:
                b.cyl(
                    "Violet processing conduit",
                    (-0.72, s * 0.90, 0.97),
                    0.073,
                    1.28,
                    "energy",
                    "x",
                )
                for z in [1.25, 1.39, 1.53, 1.67]:
                    b.box(
                        "Rear processor cooling louver",
                        (-1.94, s * 0.75, z),
                        (0.22, 0.045, 0.04),
                        "trim",
                        bevel=0,
                    )


def refine_vehicle(builder):
    mechanical_finish(builder)
    {"rocket": rocket, "kart": kart, "drone": drone, "harvester": harvester}[builder.kind](builder)


def repair_vehicle_normals(builder):
    """Reorient closed shells after mirrored panels and Y-axis primitives."""
    for obj in builder.scene.objects:
        if obj.type != "MESH" or len(obj.data.polygons) < 2:
            continue
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        if all(edge.is_manifold for edge in mesh.edges):
            bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
            mesh.to_mesh(obj.data)
            obj.data.update()
        mesh.free()
