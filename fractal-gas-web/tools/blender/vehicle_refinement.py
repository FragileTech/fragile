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


def precision_panel_finish(b):
    """Repaint existing packed maps; restrained seams survive reduced screen size."""
    n = 512
    y, x = np.mgrid[0:n, 0:n] / n
    rng = np.random.default_rng(947)
    noise = rng.random((n, n)) - 0.5
    edge = np.minimum.reduce([x, 1 - x, y, 1 - y])
    # Straight folded seams replace the previous large decorative curved hatch.
    chamfer = np.minimum(x + y, 2 - x - y)
    seam = (edge < 0.008) | ((chamfer > 0.20) & (chamfer < 0.208))
    rivets = np.zeros_like(x, dtype=bool)
    for u in [0.045, 0.955]:
        for v in [0.07, 0.35, 0.65, 0.93]:
            rivets |= np.hypot(x - u, y - v) < (0.006 if b.steam else 0.004)
    wear = (edge < 0.021) & (noise > 0.34)
    hairline = (np.abs(np.sin(y * 317 + x * 4)) < 0.01) & (noise > 0.40)
    height = np.zeros_like(x)
    height[seam] = -0.09
    height[rivets] = 0.08
    dy, dx = np.gradient(height)
    normals = np.stack([-dx * 12, -dy * 12, np.ones_like(x)], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    for key in ["plate", "dark"]:
        base = np.array(
            ([0.39, 0.43, 0.47] if key == "plate" else [0.075, 0.093, 0.11])
            if not b.steam
            else ([0.12, 0.023, 0.018] if key == "plate" else [0.073, 0.066, 0.050])
        )
        metal = np.array([0.48, 0.50, 0.51] if not b.steam else [0.36, 0.24, 0.105])
        rgb = np.broadcast_to(base, (n, n, 3)).copy()
        rgb *= (1 + noise * 0.045 + np.sin(y * 53) * 0.012)[..., None]
        rgb[seam] *= 0.35
        rgb[rivets | wear] = metal * 0.72
        rgb[hairline & ~seam] = metal * 0.60
        rough = np.repeat((0.37 + noise * 0.045 + (edge < 0.03) * 0.07)[..., None], 3, 2)
        shader = b.mats[key].node_tree.nodes.get("Principled BSDF")
        images = {
            "Base Color": shader.inputs["Base Color"].links[0].from_node.image,
            "Roughness": shader.inputs["Roughness"].links[0].from_node.image,
            "Normal": shader
            .inputs["Normal"]
            .links[0]
            .from_node.inputs["Color"]
            .links[0]
            .from_node.image,
        }
        for field, data in [
            ("Base Color", rgb),
            ("Roughness", rough),
            ("Normal", normals * 0.5 + 0.5),
        ]:
            image = images[field]
            rgba = np.concatenate([data, np.ones((n, n, 1))], -1).astype(np.float32)
            image.pixels.foreach_set(rgba.ravel())
            image.pack()


def annular_panel(b, name, center, radius, width, z, depth, start, end, mat):
    """Closed angular duct armor; six faces per span without torus tessellation."""
    count = 2 if b.low else 6
    vertices = []
    for zz in [z, z + depth]:
        for rr in [radius - width, radius]:
            vertices.extend(
                (
                    center[0] + rr * math.cos(start + (end - start) * i / count),
                    center[1] + rr * math.sin(start + (end - start) * i / count),
                    zz,
                )
                for i in range(count + 1)
            )
    stride = count + 1
    faces = []
    for i in range(count):
        j = i + 1
        faces.extend([
            (i, j, stride + j, stride + i),
            (2 * stride + i, 3 * stride + i, 3 * stride + j, 2 * stride + j),
            (i, 2 * stride + i, 2 * stride + j, j),
            (stride + i, stride + j, 3 * stride + j, 3 * stride + i),
        ])
    faces.extend([
        (0, stride, 3 * stride, 2 * stride),
        (count, 2 * stride + count, 3 * stride + count, stride + count),
    ])
    return b.mesh(name, vertices, faces, mat)


def rocket_equipment(b):
    for s in [-1, 1]:
        if b.steam:
            # Broad enamel wing inserts and the nose's brass service framing.
            b.plate(
                "Inset burgundy pressure wing",
                [(-1.39, s * 0.52), (-0.61, s * 0.52), (-0.48, s * 0.90), (-1.24, s * 1.08)],
                0.341,
                0.025,
                "plate",
            )
            b.pipe(
                "Nose brass service rail",
                [(0.54, s * 0.27, 0.42), (1.45, s * 0.14, 0.38)],
                0.016,
                "trim",
            )
            b.pipe(
                "Lower pressure supply",
                [
                    (-1.20, s * 0.38, 0.31),
                    (-0.63, s * 0.49, 0.30),
                    (0.27, s * 0.43, 0.36),
                    (0.49, s * 0.35, 0.44),
                ],
                0.030,
                "copper",
            )
            if not b.low:
                for x in [0.72, 0.89, 1.06]:
                    side_plate(
                        b,
                        "Nose inset ventilation grille",
                        [(x, 0.37), (x, 0.48), (x + 0.055, 0.465), (x + 0.055, 0.365)],
                        s * (0.31 - (x - 0.72) * 0.18),
                        0.012,
                        "rubber",
                    )
                for x in [-0.94, -0.53, 0.04]:
                    b.cyl(
                        "Pressure line union",
                        (x, s * 0.45, 0.32),
                        0.05,
                        0.075,
                        "trim",
                        "x",
                        segments=8,
                    )
        else:
            b.loft(
                "Raised arrowhead canopy shoulder",
                [
                    (-0.48, 0.105, 0.48, 0.69),
                    (0.24, 0.12, 0.46, 0.64),
                    (0.83, 0.06, 0.41, 0.50),
                    (1.39, 0.01, 0.38, 0.40),
                ],
                "plate",
            ).location.y = s * 0.32
            b.box(
                "Engine recessed collar band",
                (-0.39, s * 0.78, 0.77),
                (0.15, 0.31, 0.035),
                "dark",
                bevel=0,
            )
            if not b.low:
                side_plate(
                    b,
                    "Dorsal violet identification inset",
                    [(-1.48, 1.30), (-1.48, 1.37), (-1.39, 1.45), (-1.39, 1.38)],
                    s * 0.039,
                    0.008,
                    "copper",
                )
                for x in [-1.27, -0.91]:
                    b.box(
                        "Engine armor service hatch",
                        (x, s * 0.78, 0.815),
                        (0.20, 0.19, 0.018),
                        "dark",
                        bevel=0.009,
                    )


def kart_equipment(b):
    if b.steam:
        # A full-height round grille and the cylindrical lamps anchor the front view.
        for s in [-1, 1]:
            b.cyl("Deep brass headlamp housing", (1.10, s * 0.35, 0.43), 0.13, 0.16, "copper", "x")
            b.ring("Headlamp retaining rim", (1.21, s * 0.35, 0.43), 0.096, 0.014, "trim", "x")
            b.cyl("Side pressure accumulator", (-0.63, s * 0.44, 0.49), 0.065, 0.32, "copper")
            b.pipe(
                "Accumulator return pipe",
                [(-0.63, s * 0.44, 0.34), (-0.63, s * 0.50, 0.27), (0.31, s * 0.43, 0.28)],
                0.022,
                "trim",
            )
            if not b.low:
                for j in range(4):
                    x = 0.27 + j * 0.13
                    side_plate(
                        b,
                        "Bonnet recessed louver",
                        [(x, 0.38), (x, 0.47), (x + 0.08, 0.45), (x + 0.08, 0.37)],
                        s * (0.363 - j * 0.014),
                        0.015,
                        "rubber",
                    )
                b.pipe(
                    "Lamp protective crossbar",
                    [(1.222, s * 0.35 - 0.073, 0.43), (1.222, s * 0.35 + 0.073, 0.43)],
                    0.007,
                    "trim",
                )
        b.cyl("Bonnet filler neck", (0.66, 0, 0.66), 0.057, 0.032, "trim", segments=8)
    else:
        for s in [-1, 1]:
            # Separate folded shoulder plates create the concept's wide armored snout.
            b.plate(
                "Nose folded shoulder plate",
                [(0.30, s * 0.26), (0.54, s * 0.44), (1.28, s * 0.31), (1.46, s * 0.21)],
                0.41,
                0.065,
                "plate",
            )
            b.box(
                "Sensor dark wraparound visor",
                (-0.091, s * 0.105, 0.936),
                (0.035, 0.11, 0.116),
                "glass",
                bevel=0.016,
            )
            if not b.low:
                side_plate(
                    b,
                    "Rocker lower cooling aperture",
                    [(-0.70, 0.39), (-0.59, 0.43), (-0.30, 0.43), (-0.37, 0.38)],
                    s * 0.499,
                    0.014,
                    "rubber",
                )
                b.box(
                    "Battery service plate",
                    (-1.14, s * 0.16, 0.697),
                    (0.30, 0.20, 0.025),
                    "plate",
                    bevel=0.008,
                )
        if not b.low:
            for y in [-0.045, 0, 0.045]:
                for z in [0.91, 0.945, 0.98]:
                    b.box(
                        "Autonomous optical pixel",
                        (-0.069, y, z),
                        (0.008, 0.012, 0.012),
                        "light",
                        bevel=0,
                    )


def drone_equipment(b):
    # Replace skinny toroidal rims with broad segmented armor that leaves every
    # aperture open. Motion children remain on their original rotor pivots.
    reach = 1.12 if b.steam else 0.83
    for x in [-0.73, 0.70]:
        for s in [-1, 1]:
            for j in range(4):
                a = j * math.pi / 2 + 0.055
                annular_panel(
                    b,
                    "Segmented pressure shroud" if b.steam else "Faceted duct armor",
                    (x, s * reach),
                    0.535,
                    0.094,
                    0.39,
                    0.17 if b.steam else 0.19,
                    a,
                    a + math.pi / 2 - 0.11,
                    "plate",
                )
            if not b.low:
                for j in [-1, 1]:
                    b.box(
                        "Duct recessed service slot",
                        (x + j * 0.29, s * (reach + 0.36), 0.455),
                        (0.13, 0.045, 0.060),
                        "dark",
                        bevel=0,
                    )
    if b.steam:
        for s in [-1, 1]:
            b.cyl("Diamond side accumulator", (0.24, s * 0.43, 0.61), 0.073, 0.48, "copper", "x")
            if not b.low:
                for x in [0.05, 0.40]:
                    b.cyl(
                        "Accumulator hex union",
                        (x, s * 0.43, 0.61),
                        0.085,
                        0.055,
                        "trim",
                        "x",
                        segments=6,
                    )
        b.ring("Survey lens rolled brass lip", (1.207, 0, 0.49), 0.19, 0.020, "trim", "x")
    else:
        remove(b, "Survey avionics", "Avionics status")
        b.loft(
            "Stepped central flight computer",
            [
                (-0.72, 0.19, 0.715, 0.80),
                (-0.53, 0.26, 0.72, 0.85),
                (0.29, 0.26, 0.72, 0.85),
                (0.52, 0.17, 0.69, 0.78),
            ],
            "dark",
        )
        b.box(
            "Flight computer cyan status", (0.39, 0, 0.797), (0.08, 0.20, 0.021), "light", bevel=0
        )
        b.ring("Gold iris concentric bezel", (1.303, 0, 0.49), 0.119, 0.014, "trim", "x")
        if not b.low:
            for s in [-1, 1]:
                b.plate(
                    "Flight computer service armor",
                    [(-0.43, s * 0.04), (-0.43, s * 0.21), (0.17, s * 0.21), (0.24, s * 0.04)],
                    0.854,
                    0.015,
                    "plate",
                )


def harvester_equipment(b):
    for s in [-1, 1]:
        # Service compartments give the chassis depth beneath the hopper sides.
        side_plate(
            b,
            "Recessed chassis equipment bay",
            [(-1.45, 0.91), (-1.34, 1.15), (-0.43, 1.15), (-0.31, 0.93)],
            s * 0.97,
            0.06,
            "dark",
        )
        b.pipe(
            "Intake crossbeam mounting",
            [(1.42, s * 0.74, 1.05), (1.57, s * 0.74, 1.16)],
            0.075,
            "trim",
        )
        side_plate(
            b,
            "Cab angular lower sill",
            [(0.51, 1.39), (0.60, 1.29), (1.32, 1.29), (1.45, 1.39)],
            s * 0.59,
            0.05,
            "dark" if b.steam else "plate",
        )
        if b.steam:
            b.cyl(
                "Conveyor hydraulic pressure vessel",
                (0.52, s * 0.93, 1.02),
                0.075,
                0.45,
                "copper",
                "x",
            )
            if not b.low:
                b.pipe(
                    "Chassis copper bypass",
                    [
                        (-1.52, s * 0.96, 0.89),
                        (-1.49, s * 0.98, 0.76),
                        (-0.39, s * 0.98, 0.76),
                        (-0.31, s * 0.96, 0.93),
                    ],
                    0.034,
                    "copper",
                )
                b.ring(
                    "Cab oval service porthole", (0.70, s * 0.62, 1.43), 0.083, 0.016, "trim", "y"
                )
        else:
            side_plate(
                b,
                "Hopper graphite inset field",
                [(-1.50, 1.40), (-1.50, 1.77), (-0.26, 1.77), (-0.27, 1.42)],
                s * 0.88,
                0.015,
                "dark",
            )
            b.box(
                "Cab panoramic lower scanner",
                (1.46, -0.05, 1.40),
                (0.08, 0.43, 0.085),
                "dark",
                bevel=0.01,
            )
            b.box(
                "Cab cyan scanner slit",
                (1.505, -0.05, 1.40),
                (0.014, 0.31, 0.021),
                "light",
                bevel=0,
            )
            if not b.low:
                for x in [-1.37, -1.0, -0.63]:
                    b.box(
                        "Conduit protective saddle",
                        (x, s * 1.00, 1.00),
                        (0.055, 0.11, 0.24),
                        "trim",
                        bevel=0,
                    )
        if not b.low:
            for x in [1.49, 1.96]:
                b.cyl(
                    "Intake hydraulic hinge cap",
                    (x, s * 1.061, 0.62),
                    0.065,
                    0.022,
                    "copper" if b.steam else "energy",
                    "y",
                    segments=8,
                )
    b.box(
        "Collector armored crossbeam",
        (1.58, 0, 1.12),
        (0.20, 1.53, 0.15),
        "dark" if b.steam else "plate",
        bevel=0.025,
    )


def concept_materials_and_profiles(b):
    """Refine existing surfaces and profiles without extra meshes or texture memory."""
    n = 512
    y, x = np.mgrid[0:n, 0:n] / n
    edge = np.minimum.reduce([x, 1 - x, y, 1 - y])
    bloom = (np.sin(x * 19 + np.sin(y * 11)) * np.sin(y * 23 - x * 7) + 1) / 2
    streak = (np.sin(x * 163 + np.sin(y * 17) * 0.7) + 1) / 2
    dirt = np.clip((0.07 - edge) / 0.07, 0, 1)
    for key in ["plate", "dark"]:
        shader = b.mats[key].node_tree.nodes.get("Principled BSDF")
        for field in ["Base Color", "Roughness"]:
            im = shader.inputs[field].links[0].from_node.image
            pixels = np.empty(len(im.pixels), dtype=np.float32)
            im.pixels.foreach_get(pixels)
            rgba = pixels.reshape(n, n, 4)
            if field == "Base Color":
                rgba[:, :, :3] *= (0.98 + bloom * 0.025 - dirt * 0.12)[..., None]
                scrape = (edge < 0.014) & (streak > 0.65) & (bloom > 0.35)
                metal = [0.27, 0.18, 0.07] if b.steam else [0.29, 0.32, 0.35]
                rgba[scrape, :3] = metal
                # Quantization removes expensive invisible noise from packed PNGs.
                rgba[:, :, :3] = np.round(rgba[:, :, :3] * 255) / 255
            else:
                rough = 0.40 + bloom * 0.015 + dirt * 0.045 + streak * 0.005
                rgba[:, :, :3] = (np.round(rough * 47) / 47)[..., None]
            im.pixels.foreach_set(rgba.ravel())
            im.pack()
    colors = (
        {"trim": (0.27, 0.18, 0.065), "copper": (0.24, 0.075, 0.027)}
        if b.steam
        else {"trim": (0.105, 0.135, 0.165), "copper": (0.065, 0.038, 0.105)}
    )
    for key, color in colors.items():
        shader = b.mats[key].node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Metallic"].default_value = 0.82
    if b.kind == "kart":
        for obj in b.scene.objects:
            if obj.name.startswith("Tire tread block"):
                # Shallow road tread with rounded shoulders, not mining paddles.
                obj.scale.x *= 0.72
                obj.scale.y *= 0.82
            elif obj.name.startswith("Rubber tire"):
                obj.scale.z *= 0.85
            elif b.steam and obj.name.startswith("Leather bucket seat back"):
                obj.scale.y *= 1.09
    elif b.kind == "rocket":
        for obj in b.scene.objects:
            if obj.name.startswith("Dorsal inset panel"):
                obj.scale.x *= 1.07
    elif b.kind == "harvester":
        for obj in b.scene.objects:
            if obj.name.startswith(("Collected mineral", "Collected ore nugget")):
                if obj.name.startswith("Collected mineral"):
                    obj.scale.z *= 0.76
                # Re-seat the irregular load after reprofiling: preserve the bed
                # contact rather than shrinking each stone about its floating center.
                rotation = obj.rotation_euler.to_matrix()
                bottom = min(
                    sum(rotation[2][axis] * v.co[axis] * obj.scale[axis] for axis in range(3))
                    for v in obj.data.vertices
                )
                obj.location.z = 1.245 - bottom
            elif obj.name.startswith("Flared cargo hopper side"):
                for vertex in obj.data.vertices:
                    if vertex.co.z < 0:
                        vertex.co.x *= 0.94
    elif b.kind == "drone":
        shader = b.mats["gold"].node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (0.48, 0.245, 0.028, 1)
        shader.inputs["Metallic"].default_value = 0.78
        shader.inputs["Roughness"].default_value = 0.19


def gameplay_profiles(b, fittings_only=False):
    """Give broad existing panels distinct profiles without adding export geometry."""
    for obj in b.scene.objects:
        if obj.type != "MESH" or obj.parent != b.root:
            continue
        name = obj.name
        is_frame = name.startswith((
            "Cockpit titanium frame",
            "Arched canopy mullion",
            "Canopy sill",
            "Diamond hull brass perimeter",
        ))
        if is_frame != fittings_only:
            continue
        transform = None
        if b.kind == "rocket":
            if name.startswith(("Closed faceted cyan canopy", "Cockpit titanium frame")):

                def transform(p):
                    return (p.x, p.y * 1.18, 0.55 + (p.z - 0.55) * 1.12)

            elif name.startswith("Closed arched amber canopy"):

                def transform(p):
                    return (p.x, p.y * 1.12, 0.60 + (p.z - 0.60) * 1.15)

            elif name.startswith(("Arched canopy mullion", "Canopy sill")):

                def transform(p):
                    return (p.x, p.y * 1.18, 0.64 + (p.z - 0.60) * 1.15)

            elif name.startswith("Raised arrowhead canopy shoulder"):

                def transform(p):
                    return (p.x, p.y, p.z + max(0, 1 - abs(p.x) / 1.4) * 0.055)

            elif name.startswith("Engine segmented armor"):

                def transform(p):
                    return (p.x, p.y, p.z + 0.055 * max(0, (p.z - 0.62) / 0.18))

            elif name.startswith("Arrowhead shoulder armor"):

                def transform(p):
                    return (p.x, p.y * 1.12, p.z)

        elif b.kind == "drone":
            if name.startswith(("Layered avionics armor", "Low armored drone hull")):

                def transform(p):
                    return (p.x, p.y * (0.87 + 0.11 * abs(p.x)), p.z)

            elif name.startswith(("Diamond burgundy armor facet", "Diamond hull brass perimeter")):

                def transform(p):
                    return (p.x, p.y * 1.08, p.z)

            elif name.startswith("Convex gold survey lens"):

                def transform(p):
                    return (p.x, p.y * 1.12, 0.49 + (p.z - 0.49) * 0.84)

        elif b.kind == "kart":
            if name.startswith((
                "Seat back",
                "Leather bucket seat back",
                "Vertical leather bolster",
                "Seat cushion",
            )):

                def transform(p):
                    return (p.x, p.y * 1.14, p.z)

            elif name.startswith(("Rounded riveted bonnet", "Burgundy bonnet inlay")):

                def transform(p):
                    return (p.x, p.y * (1.08 - 0.08 * max(0, p.x) / 1.53), p.z)

            elif name.startswith("Nose folded shoulder plate"):

                def transform(p):
                    return (p.x, p.y, p.z + 0.12 * max(0, 1 - p.x / 1.46))

            elif name.startswith("Faceted cockpit side armor"):

                def transform(p):
                    return (p.x, p.y * 1.10, p.z + 0.07 * max(0, (p.z - 0.3) / 0.36))

        elif b.kind == "harvester":
            if name.startswith((
                "Raised sensor cab",
                "Panoramic front cab glass",
                "Side cab glass",
                "Cab corner frame",
                "Front glass mullion",
                "Faceted cab roof",
            )):
                # Shared shear keeps glazing, framing and cabin shell attached.
                def transform(p):
                    return (p.x - 0.20 * max(0, p.z - 1.42), p.y, p.z)

            elif name.startswith("Hopper graphite inset field"):

                def transform(p):
                    return (p.x + 0.10 * max(0, p.z - 1.25), p.y, p.z)

        if transform is not None:
            matrix = obj.matrix_local.copy()
            inverse = matrix.inverted()
            for vertex in obj.data.vertices:
                point = matrix @ vertex.co
                point[:] = transform(point)
                vertex.co = inverse @ point
            obj.data.update()


def refine_vehicle(builder):
    mechanical_finish(builder)
    precision_panel_finish(builder)
    {"rocket": rocket, "kart": kart, "drone": drone, "harvester": harvester}[builder.kind](builder)
    {
        "rocket": rocket_equipment,
        "kart": kart_equipment,
        "drone": drone_equipment,
        "harvester": harvester_equipment,
    }[builder.kind](builder)
    concept_materials_and_profiles(builder)
    gameplay_profiles(builder)


def optimize_small_fittings(builder):
    """Spend triangles on armor forms instead of invisible tube cross-sections.

    Keep tire geometry, every motion transform and major circumference sampling.
    The small hoses are smooth-shaded hexagonal tubes; static torus fittings use
    four tube sides while retaining all 24 major-circle samples.
    """
    if builder.kind not in {"rocket", "kart", "drone", "harvester"}:
        return
    # Existing crowd render ceilings remain fixed as detail is redistributed.
    crowd = {
        "futuristic": {"rocket": 1432, "kart": 2636, "drone": 2128, "harvester": 4356},
        "steampunk": {"rocket": 2032, "kart": 2810, "drone": 2468, "harvester": 4616},
    }
    if builder.low:
        builder.triangle_budget = crowd[builder.style][builder.kind]
        return
    for obj in list(builder.scene.objects):
        if obj.type != "MESH":
            continue
        old = obj.data
        material = old.materials[0]
        key = next((k for k, m in builder.mats.items() if m == material), None)
        if key is None or key in {"rubber", "glass"}:
            continue
        vertices, faces, smooth = None, None, True
        # Builder.pipe uses a ten-sided cylinder aligned along its local Z axis.
        # All main pressure vessels, wheel hubs and engine pods use 24 sides.
        if len(old.vertices) == 20 and len(old.polygons) == 12:
            ring = [old.vertices[i].co for i in range(10)]
            radius = math.hypot(ring[0].x, ring[0].y)
            if radius < 0.10:
                zlo, zhi = old.vertices[0].co.z, old.vertices[10].co.z
                vertices = [
                    (radius * math.cos(j * math.tau / 6), radius * math.sin(j * math.tau / 6), z)
                    for z in [zlo, zhi]
                    for j in range(6)
                ]
                faces = [tuple(reversed(range(6))), tuple(range(6, 12))]
                faces += [(i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6) for i in range(6)]
        elif (
            len(old.vertices) == 192
            and len(old.polygons) == 192
            and all(len(poly.vertices) == 4 for poly in old.polygons)
        ):
            # Torus topology from Builder.ring: 24 major sectors × 8 tube sides.
            # Retain cardinal tube samples exactly so its envelope is unchanged.
            vertices = [tuple(old.vertices[i * 8 + j].co) for i in range(24) for j in [0, 2, 4, 6]]
            faces = [
                (
                    i * 4 + j,
                    ((i + 1) % 24) * 4 + j,
                    ((i + 1) % 24) * 4 + (j + 1) % 4,
                    i * 4 + (j + 1) % 4,
                )
                for i in range(24)
                for j in range(4)
            ]
        if vertices is not None:
            temporary = builder.mesh(obj.name + " efficient fitting", vertices, faces, key)
            obj.data = temporary.data
            for poly in obj.data.polygons:
                poly.use_smooth = smooth and (len(faces) != 8 or poly.index > 1)
            bpy.data.objects.remove(temporary, do_unlink=True)
            if old.users == 0:
                bpy.data.meshes.remove(old)
        # One bevel segment is sufficient on sub-pixel service covers. Large
        # bodywork and the road tire profiles retain their original construction.
        if max(obj.dimensions) < 0.8:
            for modifier in obj.modifiers:
                if modifier.type == "BEVEL" and modifier.width <= 0.035:
                    modifier.segments = 1


def recessed_nozzle(b, name, center, radius, depth, axis="x", direction=-1):
    """A real open bell with a recessed throat, batched into existing materials."""
    n = 8 if b.low else 12

    def point(z, r, a):
        p = [r * math.cos(a), r * math.sin(a), z * direction]
        if axis == "x":
            p = [p[2], p[0], p[1]]
        return tuple(center[i] + p[i] for i in range(3))

    sections = [(-depth, radius * 0.52), (0, radius), (0, radius * 0.82)]
    vertices = [point(z, r, i * math.tau / n) for z, r in sections for i in range(n)]
    faces = [
        (k * n + i, k * n + (i + 1) % n, (k + 1) * n + (i + 1) % n, (k + 1) * n + i)
        for k in range(2)
        for i in range(n)
    ]
    shell = b.mesh(name + " rolled lip", vertices, faces, "trim")
    for p in shell.data.polygons:
        p.use_smooth = p.index < n
    vertices = [
        point(z, r, i * math.tau / n)
        for z, r in [(0, radius * 0.82), (-depth * 0.82, radius * 0.23)]
        for i in range(n)
    ]
    interior = b.mesh(
        name + " recessed carbon interior",
        vertices,
        [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)],
        "dark",
    )
    for p in interior.data.polygons:
        p.use_smooth = True
    vertices = [point(-depth * 0.83, radius * 0.23, i * math.tau / n) for i in range(n)]
    b.mesh(
        name + " deep ignition face", vertices, [tuple(range(n))], "energy" if b.steam else "light"
    )


def rear_brake_hardware(b, x, y, z):
    """Visible brake locations; animated illumination lives in the shared runtime batch."""
    # Clear the processor's overlapping armor without extending its native envelope.
    lens_offset = 0.068 if b.kind == "harvester" and not b.steam else 0.059
    for side in [-1, 1]:
        b.box(
            "Rear brake lamp armored pocket",
            (x, side * y, z),
            (0.11, 0.24, 0.13),
            "dark",
            bevel=0.015,
        )
        b.box(
            "Rear brake lamp lens",
            (x - lens_offset, side * y, z),
            (0.012, 0.18, 0.065),
            "light",
            bevel=0,
        )
        socket = b.empty(
            "Rear brake socket L" if side < 0 else "Rear brake socket R",
            (x - 0.075, side * y, z),
            b.root,
        )
        socket["effectSocket"] = "brake"
        socket["outwardAxis"] = "-X"
    # Fixed housing sockets for the shared runtime illumination batch.
    reverse_pos = (
        (-1.532, 0, 0.30)
        if b.kind == "kart"
        else ((-1.89, 0, 1.3) if b.steam else (-2.17, 0, 0.94))
    )
    if b.kind == "kart":
        drive_pos = (-1.475, 0, 0.76) if b.steam else (-1.44, 0, 0.54)
    else:
        drive_pos = (-1.65, 0.15, 2.12) if b.steam else (-2.17, 0, 1.42)
    for effect, pos in [("reverse", reverse_pos), ("drive", drive_pos)]:
        socket = b.empty("Rear " + effect + " socket", pos, b.root)
        socket["effectSocket"] = effect
        socket["outwardAxis"] = "-X"


def mechanical_second_pass(b):
    """Expose functional mechanisms using the existing draw/material palette."""
    if b.kind == "rocket":
        remove(b, "Rear exhaust bell", "Exhaust throat", "Vernier nozzle")
        for side in [-1, 1]:
            recessed_nozzle(b, "Main expansion nozzle", (-1.929, side * 0.78, 0.54), 0.31, 0.34)
            recessed_nozzle(
                b, "Vernier control nozzle", (0.15, side * 0.45, 0.085), 0.105, 0.15, "z"
            )
            b.pipe(
                "Vernier supply coupling",
                [(0.15, side * 0.45, 0.30), (0.15, side * 0.39, 0.38)],
                0.033,
                "trim",
            )
            b.cyl(
                "Nozzle gimbal trunnion",
                (-1.59, side * 1.015, 0.54),
                0.065,
                0.09,
                "dark",
                "y",
                segments=6,
            )
        # Preserve the existing thrust pivots; reduce invisible cone circumference.
        for obj in list(b.scene.objects):
            if obj.name.startswith("Exhaust flame"):
                parent = obj.parent
                bpy.data.objects.remove(obj, do_unlink=True)
                b.cyl(
                    "Tapered exhaust core",
                    (-0.22, 0, 0),
                    0.001,
                    0.44,
                    "energy" if b.steam else "light",
                    "x",
                    0.13,
                    parent,
                    segments=8 if b.low else 12,
                )
    elif b.kind == "kart":
        rear_brake_hardware(b, -1.43, 0.29, 0.41)
        b.cyl("Steering rack housing", (0.47, 0, 0.28), 0.065, 0.53, "dark", "y", segments=8)
        b.pipe("Steering column lower knuckle", [(0.30, 0, 0.40), (0.47, 0, 0.28)], 0.032, "trim")
        b.cyl("Rear drive axle", (-1.02, 0, 0.43), 0.065, 1.25, "trim", "y", segments=8)
        b.cyl(
            "Rear differential casing",
            (-1.02, 0, 0.43),
            0.145,
            0.27,
            "dark",
            "y",
            segments=12 if not b.low else 8,
        )
        for side in [-1, 1]:
            b.pipe(
                "Steering tie rod",
                [(0.47, side * 0.23, 0.28), (0.94, side * 0.59, 0.36)],
                0.024,
                "trim",
            )
            b.cyl(
                "Drive axle coupling",
                (-1.02, side * 0.48, 0.43),
                0.088,
                0.095,
                "dark",
                "y",
                segments=8,
            )
            b.box(
                "Rear brake caliper",
                (-0.87, side * 0.59, 0.56),
                (0.14, 0.075, 0.15),
                "trim",
                bevel=0.01,
            )
            if not b.low:
                b.pipe(
                    "Brake pressure line",
                    [
                        (-1.28, side * 0.36, 0.32),
                        (-1.14, side * 0.55, 0.31),
                        (-0.87, side * 0.59, 0.50),
                    ],
                    0.014,
                    "dark",
                )
    elif b.kind == "drone":
        reach = 1.12 if b.steam else 0.83
        for x in [-0.73, 0.70]:
            for side in [-1, 1]:
                b.cyl(
                    "Rotor lower drive housing",
                    (x, side * reach, 0.34),
                    0.145,
                    0.14,
                    "dark",
                    r2=0.10,
                    segments=8 if b.low else 12,
                )
                b.cyl(
                    "Rotor drive retaining collar",
                    (x, side * reach, 0.38),
                    0.15,
                    0.035,
                    "trim",
                    segments=8 if b.low else 12,
                )
        for obj in b.scene.objects:
            if obj.name.startswith("Rotor motor hub"):
                for vertex in obj.data.vertices:
                    if vertex.co.z > 0:
                        vertex.co.x *= 0.63
                        vertex.co.y *= 0.63
            elif obj.name.startswith("Twisted propeller blade"):
                for vertex in obj.data.vertices:
                    vertex.co.z += (vertex.co.x - 0.08) * vertex.co.y * 1.3
        if not b.steam:
            remove(b, "Rear vector jet")
            recessed_nozzle(b, "Rear vector expansion nozzle", (-1.36, 0, 0.49), 0.16, 0.30)
    elif b.kind == "harvester":
        rear_brake_hardware(b, -1.819 if b.steam else -2.11, 0.60, 1.13 if b.steam else 0.86)
        b.cyl(
            "Protected longitudinal drive shaft",
            (-0.17, 0, 0.54),
            0.085,
            2.42,
            "trim",
            "x",
            segments=8,
        )
        for x in [0.93, -0.18, -1.27]:
            b.cyl(
                "Heavy transverse drive axle", (x, 0, 0.58), 0.115, 1.63, "trim", "y", segments=8
            )
            b.cyl(
                "Heavy differential housing",
                (x, 0, 0.58),
                0.23,
                0.35,
                "dark",
                "y",
                segments=8 if b.low else 12,
            )
            for side in [-1, 1]:
                b.box(
                    "Heavy wheel brake caliper",
                    (x - 0.22, side * 0.73, 0.80),
                    (0.18, 0.11, 0.23),
                    "trim",
                    bevel=0.018,
                )
        for side in [-1, 1]:
            b.pipe(
                "Intake hydraulic ram",
                [(0.36, side * 0.82, 1.21), (1.28, side * 0.86, 0.93)],
                0.055,
                "dark",
            )
            b.pipe(
                "Intake polished piston",
                [(1.28, side * 0.86, 0.93), (1.57, side * 0.90, 0.73)],
                0.032,
                "trim",
            )


def reclaim_mechanical_detail(b):
    """Exchange subpixel circumference samples for the functional geometry above."""
    if b.low:
        current = {
            "futuristic": {"rocket": 1338, "kart": 2456, "drone": 1960, "harvester": 4066},
            "steampunk": {"rocket": 1904, "kart": 2610, "drone": 2288, "harvester": 4314},
        }
        b.triangle_budget = current[b.style][b.kind]
        return
    for obj in list(b.scene.objects):
        if obj.type != "MESH":
            continue
        old = obj.data
        vertices, faces = None, None
        if obj.name.startswith("Coil suspension spring") and len(old.vertices) == 12:
            r = math.hypot(old.vertices[0].co.x, old.vertices[0].co.y)
            vertices = [
                (r * math.cos(j * math.tau / 4), r * math.sin(j * math.tau / 4), z)
                for z in [old.vertices[0].co.z, old.vertices[6].co.z]
                for j in range(4)
            ]
            faces = [(3, 2, 1, 0), (4, 5, 6, 7)] + [
                (j, (j + 1) % 4, (j + 1) % 4 + 4, j + 4) for j in range(4)
            ]
        elif obj.name.startswith(("Propeller protective shroud", "Lower rotor safety ring")):
            # Original24 major-circle samples retain every other sector here.
            if len(old.vertices) == 96:
                vertices = [
                    tuple(old.vertices[i * 4 + j].co) for i in range(0, 24, 2) for j in range(4)
                ]
                faces = [
                    (
                        i * 4 + j,
                        ((i + 1) % 12) * 4 + j,
                        ((i + 1) % 12) * 4 + (j + 1) % 4,
                        i * 4 + (j + 1) % 4,
                    )
                    for i in range(12)
                    for j in range(4)
                ]
        if vertices is not None:
            key = next(k for k, mat in b.mats.items() if mat == old.materials[0])
            temp = b.mesh(obj.name + " efficient mechanics", vertices, faces, key)
            obj.data = temp.data
            for p in obj.data.polygons:
                p.use_smooth = True
            bpy.data.objects.remove(temp, do_unlink=True)
            if old.users == 0:
                bpy.data.meshes.remove(old)


def repair_vehicle_normals(builder):
    """Reorient closed shells after mirrored panels and Y-axis primitives."""
    mechanical_second_pass(builder)
    optimize_small_fittings(builder)
    reclaim_mechanical_detail(builder)
    if builder.kind == "kart" and not builder.steam and not builder.low:
        # Six optical pixels retain the sensor readout; fund the broader armor bevels.
        for obj in list(builder.scene.objects):
            if (
                obj.name.startswith("Autonomous optical pixel")
                and abs(obj.location.z - 0.945) < 0.001
            ):
                bpy.data.objects.remove(obj, do_unlink=True)
    # Cylindrical fitting simplification rebuilds local rings; profile frames afterward.
    gameplay_profiles(builder, fittings_only=True)
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
