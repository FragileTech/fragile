"""Concept-led scenery and effect details, fitted by the shared world builder.

No physics dimensions or motion clocks are authored here. Large silhouette details
exist at both LODs; close-up inserts are omitted from the simplified exports.
"""

import math

import bpy


TAU = math.tau
COVERED_ASSETS = {
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
    "track-straight",
    "track-bend",
    "kerb",
    "guardrail",
    "start-light",
    "pit-station",
    "direction-board",
    "tether-fitting",
    "capture-latch",
    "thrust-plume",
    "intake-swirl",
    "pickup-burst",
    "rotor-airflow",
    "path-trail",
}


def remove(b, *prefixes):
    for obj in list(b.root.children_recursive):
        if obj.type == "MESH" and any(obj.name.startswith(p) for p in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)


def bar(b, name, a, c, width=0.025, mat="trim", parent=None):
    b.pipe(name, [a, c], width, mat, parent)


def rail(b):
    if b.steam:
        for x in [-0.58, 0, 0.58]:
            b.box("Rail panel brass stile", (x, 0, 0.36), (0.035, 0.13, 0.43), "trim", bevel=0)
            b.cyl("Copper main coupling", (x, 0, 0.65), 0.063, 0.075, "trim", "x")
        for x in [-0.9, 0.9]:
            b.cyl("Forged post shoulder", (x, 0, 0.68), 0.15, 0.09, "trim", segments=8)
            b.cyl("Rail post domed cap", (x, 0, 0.75), 0.14, 0.08, "copper", r2=0.08)
        if not b.low:
            for s in [-1, 1]:
                for z in [0.24, 0.5]:
                    b.box(
                        "Inset brass panel border",
                        (0, s * 0.049, z),
                        (1.68, 0.014, 0.018),
                        "trim",
                        bevel=0,
                    )
    else:
        for x in [-0.70, 0.70]:
            o = b.box(
                "Segmented alloy impact shoulder",
                (x, 0, 0.45),
                (0.32, 0.285, 0.37),
                "plate",
                bevel=0.022,
            )
            o.rotation_euler.y = -math.copysign(0.16, x)
            b.box(
                "Impact shoulder gold latch",
                (x, -0.148, 0.31),
                (0.09, 0.018, 0.08),
                "gold",
                bevel=0,
            )
        for x in [-0.9, 0.9]:
            b.box("Post stepped alloy foot", (x, 0, 0.14), (0.24, 0.3, 0.09), "plate", bevel=0.014)
            b.box(
                "Post inset equipment lid", (x, 0, 0.78), (0.12, 0.17, 0.025), "dark", bevel=0.008
            )
        for s in [-1, 1]:
            b.box(
                "Recessed impact absorber",
                (0, s * 0.119, 0.46),
                (1.1, 0.019, 0.15),
                "dark",
                bevel=0.008,
            )


def surfaces(b, kind):
    if kind == "track-bend":
        count = 8 if b.low else 12
        for r in [1.46, 2.54]:
            for j in range(count):
                a0 = (j + 0.06) * math.pi / (2 * count)
                a1 = (j + 0.94) * math.pi / (2 * count)
                b.polygon_band(
                    "Alternating curved kerb block",
                    r,
                    0.12,
                    0.07,
                    0.035,
                    "plate" if j % 2 else "dark",
                    start=a0,
                    end=a1,
                )
            if b.steam:
                b.polygon_band(
                    "Brass kerb retaining lip",
                    r,
                    0.012,
                    0.091,
                    0.015,
                    "trim",
                    start=0,
                    end=math.pi / 2,
                )
        return
    sx, sy = (4, 3) if kind == "track-straight" else (2, 0.4) if kind == "kerb" else (2, 2)
    if kind in {"kerb", "track-straight"}:
        ys = [0] if kind == "kerb" else [-sy / 2 + 0.16, sy / 2 - 0.16]
        for y in ys:
            for j in range(8):
                x = -sx / 2 + (j + 0.5) * sx / 8
                b.box(
                    "Inset alternating kerb slab",
                    (x, y, 0.065),
                    (sx / 8 - 0.025, 0.27, 0.028),
                    "plate" if j % 2 else "dark",
                    bevel=0,
                )
        if kind == "track-straight":
            for x in [-1.2, 0, 1.2]:
                b.box(
                    "Track central expansion seam",
                    (x, 0, 0.038),
                    (0.014, 2.62, 0.008),
                    "trim" if b.steam else "dark",
                    bevel=0,
                )
        return
    # The concept floor has a functional central service hatch and diagonal seams.
    if b.steam:
        b.cyl("Circular iron inspection hatch", (0, 0, 0.056), 0.32, 0.02, "dark", segments=12)
        b.ring("Brass inspection hatch rim", (0, 0, 0.071), 0.30, 0.018, "trim")
        for j in range(-2, 3):
            x = j * 0.085
            length = 2 * math.sqrt(max(0, 0.26**2 - x**2))
            b.box(
                "Inspection hatch grille", (x, 0, 0.073), (0.018, length, 0.016), "trim", bevel=0
            )
        for x in [-0.87, 0.87]:
            for y in [-0.87, 0.87]:
                b.box("Corner deck hinge", (x, y, 0.061), (0.22, 0.22, 0.025), "trim", bevel=0)
                b.box("Hinge iron inset", (x, y, 0.075), (0.15, 0.15, 0.01), "dark", bevel=0)
    else:
        b.box(
            "Recessed square service hatch", (0, 0, 0.049), (0.35, 0.35, 0.02), "dark", bevel=0.006
        )
        for x in [-0.88, 0.88]:
            for y in [-0.88, 0.88]:
                bar(
                    b,
                    "Diagonal segmented deck seam",
                    (math.copysign(0.21, x), math.copysign(0.21, y), 0.048),
                    (x, y, 0.048),
                    0.008,
                    "trim",
                )
                b.box(
                    "Corner recessed tie down", (x, y, 0.054), (0.12, 0.12, 0.021), "dark", bevel=0
                )
        for j in range(-2, 3):
            b.box(
                "Service hatch ventilation slot",
                (j * 0.045, 0, 0.063),
                (0.018, 0.23, 0.008),
                "trim",
                bevel=0,
            )


def column(b, kind):
    h = 0.55 if kind == "bollard" else 1.4
    if kind == "lamp":
        # Replace the short utility cylinder with the reference's slender lamp pole.
        remove(
            b,
            "Pressure",
            "Vessel",
            "Lantern",
            "Emitter",
            "Angular instrument",
            "Alloy lamp",
            "Glass",
            "Containment",
            "Tank",
            "Copper return",
            "Gear",
            "Service flange",
            "Pressure bypass",
            "Cartridge",
            "Field cartridge",
            "Energy cartridge",
            "Capacitor",
        )
        b.cyl("Lamp fluted foot", (0, 0, 0.23), 0.14, 0.24, "dark", segments=8)
        b.cyl("Slender lamp standard", (0, 0, 0.73), 0.048, 0.85, "trim")
        for z in [0.34, 1.08]:
            b.cyl("Lamp standard collar", (0, 0, z), 0.075, 0.06, "copper" if b.steam else "dark")
        if b.steam:
            b.cyl("Hexagonal lantern glass", (0, 0, 1.3), 0.13, 0.30, "glass", segments=6)
            for j in range(6):
                a = j * TAU / 6
                bar(
                    b,
                    "Lantern brass mullion",
                    (math.cos(a) * 0.135, math.sin(a) * 0.135, 1.13),
                    (math.cos(a) * 0.135, math.sin(a) * 0.135, 1.46),
                    0.012,
                )
            b.cyl("Lantern bell roof", (0, 0, 1.5), 0.17, 0.13, "trim", r2=0.065)
            b.cyl("Lantern pointed finial", (0, 0, 1.61), 0.055, 0.12, "trim", r2=0.005)
        else:
            b.box(
                "Directional lamp armored head",
                (0, 0, 1.25),
                (0.15, 0.44, 0.18),
                "plate",
                bevel=0.025,
            )
            for s in [-1, 1]:
                b.box(
                    "Lamp optical gasket",
                    (s * 0.098, 0, 1.25),
                    (0.022, 0.38, 0.14),
                    "dark",
                    bevel=0.009,
                )
                b.box(
                    "Lamp segmented optical face",
                    (s * 0.114, 0, 1.25),
                    (0.014, 0.32, 0.09),
                    "glass",
                    bevel=0.005,
                )
        return
    for j in range(4):
        a = TAU * j / 4
        r = 0.16
        x, y = r * math.cos(a), r * math.sin(a)
        if b.steam:
            bar(b, "Column protective brass rod", (x, y, h * 0.28), (x, y, h * 0.92), 0.016)
        else:
            b.box(
                "Column armor corner pilaster",
                (x, y, h * 0.60),
                (0.065, 0.065, h * 0.65),
                "plate",
                bevel=0.012,
            )
    for z in [h * 0.23, h * 0.93]:
        b.cyl(
            "Column machined perimeter collar",
            (0, 0, z),
            0.205,
            0.09,
            "trim" if b.steam else "dark",
            segments=8,
        )
    if b.steam and kind == "utility-column":
        b.ring("Pressure return valve wheel", (0.245, 0, 0.76), 0.09, 0.016, "plate", "x")
        bar(b, "Valve spindle", (0.12, 0, 0.76), (0.25, 0, 0.76), 0.028)
        for axis in [-1, 1]:
            bar(
                b,
                "Valve wheel spoke",
                (0.25, -0.07, 0.76 + axis * 0.07),
                (0.25, 0.07, 0.76 - axis * 0.07),
                0.01,
            )


def minerals(b, kind):
    if kind == "deposit":
        for j in range(6):
            a = TAU * j / 6
            x, y = 0.40 * math.cos(a), 0.40 * math.sin(a)
            if b.steam:
                b.crystal((x, y, 0.10), 0.07, 0.18 + j % 2 * 0.08, seed=20 + j)
                b.cyl(
                    "Deposit brass assay stake",
                    (x * 1.1, y * 1.1, 0.18),
                    0.035,
                    0.26,
                    "trim",
                    segments=6,
                )
            else:
                o = b.box(
                    "Deposit articulated retaining shoe",
                    (x, y, 0.16),
                    (0.17, 0.15, 0.18),
                    "dark",
                    bevel=0.02,
                )
                o.rotation_euler.z = a
                b.box(
                    "Deposit scanner window",
                    (x * 1.08, y * 1.08, 0.22),
                    (0.08, 0.07, 0.035),
                    "light",
                    bevel=0,
                )
        if b.steam:
            b.polygon_band("Mineral catch tray lip", 0.44, 0.04, 0.13, 0.07, "trim")
        else:
            b.polygon_band("Armored mineral tray shoulder", 0.39, 0.07, 0.14, 0.08, "plate")
    else:
        # Additional fractured outcrops use tiny polyhedra, not expensive rock spheres.
        for j in range(5 if b.low else 8):
            a = j * 2.4
            x, y = math.cos(a) * 0.64, math.sin(a) * 0.64
            r = 0.18 + j % 3 * 0.035
            verts = [
                (x - r, y - r, 0.05),
                (x + r, y - r, 0.05),
                (x + r, y + r, 0.05),
                (x - r, y + r, 0.05),
                (x - r * 0.5, y - r * 0.3, 0.32),
                (x + r * 0.4, y - r * 0.3, 0.4),
                (x + r * 0.2, y + r * 0.5, 0.31),
            ]
            b.mesh(
                "Broken outcrop scree",
                verts,
                [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 6), (3, 0, 4, 6), (4, 5, 6)],
                "stone",
            )
            if j % 2 == 0:
                b.crystal(
                    (x * 0.88, y * 0.88, 0.30), 0.055, 0.24 if not b.steam else 0.17, seed=31 + j
                )


def gantry(b):
    for s in [-1, 1]:
        for x in [-0.11, 0.11]:
            bar(
                b,
                "Gantry column reinforcement",
                (x, s * 0.9, 0.24),
                (x, s * 0.9, 1.4),
                0.025,
                "copper" if b.steam else "dark",
            )
        b.box(
            "Gantry bolted capital",
            (0, s * 0.9, 1.55),
            (0.34, 0.28, 0.20),
            "trim" if b.steam else "plate",
            bevel=0.02,
        )
    if b.steam:
        b.cyl("Overhead winch drum", (0, -0.45, 1.71), 0.09, 0.35, "copper", "y")
        for y in [-0.25, 0.4]:
            bar(b, "Suspended gantry chain core", (0, y, 1.49), (0, y, 0.95), 0.015, "dark")
            if not b.low:
                for j in range(6):
                    b.ring(
                        "Hanging oval chain link",
                        (0, y, 1.02 + j * 0.075),
                        0.035,
                        0.008,
                        "trim",
                        "x" if j % 2 else "y",
                    )
            b.ring("Gantry lifting hook", (0, y, 0.94), 0.055, 0.017, "trim", "x")
    else:
        b.box("Sliding hoist carriage", (0, 0.2, 1.46), (0.24, 0.34, 0.13), "plate", bevel=0.018)
        bar(b, "Retractable hoist shaft", (0, 0.2, 1.44), (0, 0.2, 0.99), 0.025, "dark")
        b.box("Magnetic hoist gripper", (0, 0.2, 0.96), (0.17, 0.20, 0.08), "gold", bevel=0.015)
        for y in [-0.55, 0.55]:
            b.box("Crossbeam sliding rail", (0.14, y, 1.68), (0.015, 0.55, 0.025), "trim", bevel=0)


def station(b, kind):
    if kind == "start-light":
        for j in range(3):
            z = 1.4 - 0.18 + j * 0.18
            b.ring("Signal lens machined bezel", (0.125, 0, z), 0.075, 0.014, "trim", "x")
            b.box(
                "Signal eyebrow visor",
                (0.15, 0, z + 0.073),
                (0.14, 0.18, 0.023),
                "trim" if b.steam else "plate",
                bevel=0.007,
            )
        if b.steam:
            bar(
                b,
                "Signal steam service conduit",
                (-0.11, 0, 0.2),
                (-0.11, 0, 1.65),
                0.018,
                "copper",
            )
            b.cyl("Signal crown", (0, 0, 1.76), 0.06, 0.10, "trim", r2=0.015)
        else:
            b.box(
                "Signal service backplane",
                (-0.105, 0, 1.4),
                (0.05, 0.18, 0.48),
                "plate",
                bevel=0.014,
            )
    elif kind == "direction-board":
        for z in [0.94, 1.36]:
            b.box(
                "Sign frame horizontal surround",
                (0.01, 0, z),
                (0.17, 1.07, 0.045),
                "trim" if b.steam else "plate",
                bevel=0.007,
            )
        for y in [-0.51, 0.51]:
            b.box(
                "Sign frame corner cap",
                (0.01, y, 1.15),
                (0.17, 0.06, 0.46),
                "trim" if b.steam else "plate",
                bevel=0.012,
            )
        if b.steam:
            bar(b, "Sign scroll bracket", (0, -0.22, 0.95), (0, 0, 0.70), 0.025)
            bar(b, "Sign scroll bracket", (0, 0.22, 0.95), (0, 0, 0.70), 0.025)
        else:
            b.box(
                "Sign solar equipment pack",
                (-0.13, 0, 1.13),
                (0.07, 0.6, 0.25),
                "dark",
                bevel=0.015,
            )
    else:
        if b.steam:
            # A curved burgundy shelter roof makes this read as the illustrated pit.
            steps = 4 if b.low else 8
            verts = []
            for j in range(steps + 1):
                y = -0.62 + 1.24 * j / steps
                z = 1.2 + 0.23 * math.cos(y / 0.62 * math.pi / 2)
                verts.extend([(-0.85, y, z), (0.85, y, z)])
            b.mesh(
                "Vaulted burgundy pit shelter",
                verts,
                [(j * 2, j * 2 + 1, j * 2 + 3, j * 2 + 2) for j in range(steps)],
                "plate",
            )
            for x in [-0.78, 0.78]:
                points = [
                    (
                        x,
                        -0.62 + 1.24 * j / steps,
                        1.21 + 0.23 * math.cos((-0.62 + 1.24 * j / steps) / 0.62 * math.pi / 2),
                    )
                    for j in range(steps + 1)
                ]
                b.pipe("Pit roof brass rib", points, 0.018, "trim")
                for y in [-0.55, 0.55]:
                    b.cyl("Pit canopy support", (x, y, 0.68), 0.035, 1.04, "trim")
        else:
            for y in [-0.54, 0.54]:
                b.box(
                    "Pit roof raised equipment rail",
                    (0, y, 1.30),
                    (1.55, 0.1, 0.10),
                    "dark",
                    bevel=0.014,
                )
            for j in range(4):
                b.box(
                    "Pit diagnostics status row",
                    (-0.16, -0.12, 0.65 + j * 0.08),
                    (0.025, 0.4, 0.014),
                    "light",
                    bevel=0,
                )
        for j in range(3):
            b.box(
                "Pit approach threshold strip",
                (0.4 + j * 0.16, -0.48, 0.212),
                (0.07, 0.25, 0.012),
                "trim",
                bevel=0,
            )


def steam_fitting(b, kind):
    remove(b, "Winch", "Cable drum", "Forged anchor", "Gear", "Tether attachment")
    if kind == "tether-fitting":
        b.cyl("Burgundy tether pressure column", (0, 0, 0.31), 0.11, 0.31, "plate")
        for z, radius in [(0.19, 0.145), (0.43, 0.13), (0.47, 0.115)]:
            b.cyl("Pressure column brass collar", (0, 0, z), radius, 0.045, "trim")
        b.cyl("Pressure column iron crown", (0, 0, 0.50), 0.09, 0.035, "dark")
        b.cyl("Tether spherical joint socket", (0.11, 0, 0.33), 0.075, 0.075, "dark", "x")
        b.ring("Tether joint brass bezel", (0.155, 0, 0.33), 0.072, 0.014, "trim", "x")
        b.cyl("Tether mechanical coupling", (0.20, 0, 0.33), 0.045, 0.075, "copper", "x")
        b.ring("Forged rear lifting handle", (-0.125, 0, 0.34), 0.082, 0.018, "trim", "y")
        for y in [-0.07, 0.07]:
            bar(
                b,
                "Tether column reinforcement",
                (-0.075, y, 0.2),
                (-0.075, y, 0.43),
                0.012,
                "trim",
            )
    else:
        b.cyl("Latch burgundy pressure coupling", (-0.105, 0, 0.12), 0.062, 0.25, "plate", "x")
        for x in [-0.23, -0.16, -0.03]:
            b.cyl("Latch brass pipe ferrule", (x, 0, 0.12), 0.075, 0.032, "trim", "x")
        b.cyl("Open claw rotary gearbox", (0.045, 0, 0.12), 0.09, 0.09, "dark", "z")
        b.cyl("Claw brass gearbox face", (0.045, 0, 0.17), 0.07, 0.02, "trim", "z")
        points = [
            (0.025, 0.03),
            (0.11, 0.115),
            (0.23, 0.14),
            (0.29, 0.075),
            (0.26, 0.045),
            (0.22, 0.09),
            (0.13, 0.075),
            (0.075, 0.005),
        ]
        for side in [-1, 1]:
            b.plate(
                "Articulated forged capture jaw",
                [(x, side * y) for x, y in points],
                0.12,
                0.045,
                "trim",
            )
            b.cyl("Claw jaw hinge pin", (0.082, side * 0.055, 0.15), 0.025, 0.025, "copper")
            b.cyl("Claw jaw dark hinge socket", (0.082, side * 0.055, 0.167), 0.012, 0.009, "dark")
        bar(b, "Claw manual engagement lever", (0.01, 0, 0.17), (-0.04, 0, 0.24), 0.012, "trim")


def fittings(b, kind):
    if b.steam:
        steam_fitting(b, kind)
        return
    if kind == "tether-fitting":
        remove(
            b,
            "Winch",
            "Cable drum",
            "Forged anchor",
            "Gear",
            "Magnetic capture",
            "Capture field",
            "Split capture",
            "Jaw status",
            "Tether attachment",
        )
        b.loft(
            "Tether tapered pedestal",
            [(-0.16, 0.17, 0.1, 0.18), (-0.05, 0.10, 0.1, 0.45), (0.1, 0.075, 0.1, 0.39)],
            "dark",
        )
        b.cyl(
            "Tether swivel bearing",
            (0.06, 0, 0.39),
            0.145,
            0.12,
            "copper" if b.steam else "plate",
            "x",
            segments=8,
        )
        b.cyl("Tether keyed spindle", (0.15, 0, 0.39), 0.065, 0.10, "trim", "x")
        b.ring("Tether bearing rim", (0.13, 0, 0.39), 0.125, 0.018, "trim", "x")
        if b.steam:
            b.gear((-0.015, 0, 0.39), 0.12, "x")
        else:
            for y in [-0.11, 0.11]:
                b.box(
                    "Tether stand optical index",
                    (0, y, 0.3),
                    (0.08, 0.015, 0.11),
                    "light",
                    bevel=0,
                )
    else:
        remove(
            b,
            "Winch",
            "Cable drum",
            "Forged anchor",
            "Gear",
            "Magnetic capture",
            "Capture field",
            "Split capture",
            "Jaw status",
        )
        b.box(
            "Latch glass pressure chamber", (0, 0, 0.12), (0.26, 0.18, 0.17), "glass", bevel=0.025
        )
        for x in [-0.18, 0.18]:
            b.cyl(
                "Latch cable gland",
                (x, 0, 0.12),
                0.075,
                0.13,
                "copper" if b.steam else "trim",
                "x",
                segments=8,
            )
            b.box(
                "Latch chamber end cage",
                (x * 0.75, 0, 0.12),
                (0.035, 0.23, 0.24),
                "trim",
                bevel=0.015,
            )
        for y in [-0.105, 0.105]:
            for z in [0.02, 0.22]:
                bar(b, "Latch chamber protective rail", (-0.14, y, z), (0.14, y, z), 0.014, "trim")
        if b.steam:
            b.ring("Latch pressure valve", (0, 0, 0.26), 0.055, 0.012, "plate")
        else:
            b.box(
                "Latch field excitation core",
                (0, 0, 0.12),
                (0.12, 0.08, 0.07),
                "energy",
                bevel=0.01,
            )


def shard(b, name, pos, size, mat, parent=None):
    x, y, z = pos
    sx, sy, sz = size
    b.mesh(
        name,
        [
            (x - sx, y, z),
            (x + sx, y, z),
            (x, y - sy, z),
            (x, y + sy, z),
            (x, y, z - sz),
            (x, y, z + sz),
        ],
        [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)],
        mat,
        parent,
    )


def effects(b, kind):
    if kind == "thrust-plume":
        # Previously every steam puff was a subdivided mineral sphere. Eight-face
        # billows preserve a layered plume at a fraction of that triangle cost.
        remove(b, "Fractured mineral stone")
        stages = [o for o in b.root.children_recursive if o.get("motion") == "effect-stage"]
        for pivot in stages:
            j = int(pivot.get("stage", 0))
            for k in range(3 if b.low else 5):
                t = k / 5
                a = k * 2.4 + j
                if b.steam:
                    shard(
                        b,
                        "Steam billow",
                        (-0.16 - t * 0.56, math.cos(a) * 0.035, math.sin(a) * 0.04),
                        (0.14, 0.065 + t * 0.07, 0.065 + t * 0.07),
                        "vapor",
                        pivot,
                    )
                else:
                    shard(
                        b,
                        "Plume shock diamond",
                        (-0.18 - t * 0.47, 0, 0),
                        (0.05, 0.028 * (1 - t * 0.5), 0.028 * (1 - t * 0.5)),
                        "light",
                        pivot,
                    )
        return
    count = 8 if b.low else 16
    if kind == "pickup-burst":
        for j in range(count):
            a = j * 2.4
            r = 0.22 + (j % 3) * 0.09
            shard(
                b,
                "Pickup radiant splinter",
                (r * math.cos(a), r * math.sin(a), 0.15 + (j % 4) * 0.13),
                (0.024, 0.019, 0.035 if b.steam else 0.065),
                "trim" if b.steam else "light",
            )
        return
    # Discrete tapered flecks add depth to the existing smooth ribbon meshes.
    arms = 4 if kind == "rotor-airflow" else 1
    for arm in range(arms):
        ox, oy = (
            (math.cos(arm * TAU / 4) * 0.4, math.sin(arm * TAU / 4) * 0.4) if arms == 4 else (0, 0)
        )
        for j in range(4 if b.low else 8):
            t = (j + 0.5) / 8
            a = t * TAU * 1.4 + arm
            if kind == "path-trail":
                p = (-t * 2, math.sin(j * 2.4) * 0.10, 0.02 + 0.015 * (j % 2))
                size = (0.035 if b.steam else 0.06, 0.018, 0.012)
            else:
                r = 0.14 + t * 0.3
                p = (ox + math.cos(a) * r, oy + math.sin(a) * r, -t * 0.2)
                size = (0.028, 0.025, 0.019)
            shard(
                b,
                "Drifting steam eddy" if b.steam else "Field flow glint",
                p,
                size,
                "vapor" if b.steam else "light",
            )


def scenery_profiles(b, kind):
    """Resculpt existing silhouettes, retaining each mesh's exact local bounds.

    The world builder computes normalization from bounds. Anchoring all three
    extrema keeps native envelopes and motion pivots independent of this finish.
    No extra mesh, vertex, material slot or modifier is introduced here.
    """
    roof_names = (
        "Cantilever alloy roof",
        "Directional lamp armored head",
        "Gantry bolted capital",
        "Signal service backplane",
        "Segmented alloy impact shoulder",
        "Column armor corner pilaster",
        "Magnetic hoist gripper",
        "Lamp fluted foot",
        "Column machined perimeter collar",
        "Pressure column brass collar",
        "Rail post domed cap",
    )
    panel_names = (
        "Armored rail wedge",
        "Burgundy rail panel",
        "Alloy rail cap",
        "Gantry column reinforcement",
        "Recessed impact absorber",
        "Service equipment cabinet",
        "Riveted boiler room",
        "Tether tapered pedestal",
        "Latch chamber end cage",
        "Latch glass pressure chamber",
        "Sign frame horizontal surround",
        "Direction board",
    )
    fleck_names = (
        "Pickup radiant splinter",
        "Field flow glint",
        "Drifting steam eddy",
        "Plume shock diamond",
        "Steam billow",
    )
    for obj in list(b.root.children_recursive):
        if obj.type != "MESH" or not obj.data.vertices:
            continue
        name = obj.name
        if (
            b.steam
            and not b.low
            and kind == "direction-board"
            and name.startswith(("Direction board", "Sign frame horizontal surround"))
        ):
            # Their bevels and generated face fasteners define the high-LOD
            # envelope in X; preserve those boundaries and native fitting scale.
            continue
        mode = None
        if name.startswith(roof_names):
            mode = "shoulder"
        elif name.startswith(panel_names):
            mode = "frame"
        elif name.startswith("Inset alternating kerb slab"):
            mode = "kerb"
        elif name.startswith(("Energy ribbon", "Steam wisp")):
            mode = "ribbon"
        elif name.startswith(fleck_names):
            mode = "fleck"
        elif kind in {"deposit", "rock-obstacle"} and name.startswith((
            "Fractured mineral stone",
            "Broken outcrop scree",
            "Crystal",
        )):
            mode = "mineral"
        elif name.startswith(("Articulated forged capture jaw", "Vaulted burgundy pit shelter")):
            mode = "fold"
        if mode is None:
            continue
        vertices = obj.data.vertices
        lo = [min(v.co[i] for v in vertices) for i in range(3)]
        hi = [max(v.co[i] for v in vertices) for i in range(3)]
        span = [max(hi[i] - lo[i], 1e-8) for i in range(3)]
        center = [(a + z) / 2 for a, z in zip(lo, hi)]
        for vertex in vertices:
            v = vertex.co
            x, y, z = [(v[i] - lo[i]) / span[i] for i in range(3)]
            if mode == "shoulder":
                # Narrow upper armor exposes a strong sloped shoulder highlight.
                v.x = center[0] + (v.x - center[0]) * (1 - 0.20 * z)
                v.y = center[1] + (v.y - center[1]) * (1 - 0.16 * z)
            elif mode == "frame":
                # Recess the lower face so broad structures gain a planted lip.
                v.y = center[1] + (v.y - center[1]) * (0.80 + 0.20 * z)
                v.x = center[0] + (v.x - center[0]) * (0.94 + 0.06 * z)
            elif mode == "kerb":
                v.y = center[1] + (v.y - center[1]) * (1 - 0.24 * z)
            elif mode == "ribbon":
                # Pinching intermediate ribbon edges breaks the uniform tape look.
                v.y += span[1] * 0.06 * math.sin(x * math.pi * 4) * y * (1 - y)
            elif mode == "fleck":
                v.x -= span[0] * 0.22 * (1 - abs(2 * x - 1))
            elif mode == "mineral":
                v.x += span[0] * 0.18 * x * (1 - x) * (2 * z - 1)
                v.y -= span[1] * 0.14 * y * (1 - y) * (2 * x - 1)
            elif mode == "fold":
                v.z += span[2] * 0.10 * z * (1 - z) * (2 * y - 1)
        # Retain authored bounds even when a tilted or irregular face held an
        # extremum. The transform stays inside the preceding geometry envelope.
        for axis in range(3):
            new_lo = min(v.co[axis] for v in vertices)
            new_hi = max(v.co[axis] for v in vertices)
            if new_hi - new_lo > 1e-8:
                factor = (hi[axis] - lo[axis]) / (new_hi - new_lo)
                for vertex in vertices:
                    vertex.co[axis] = lo[axis] + (vertex.co[axis] - new_lo) * factor
        obj.data.update()


def refine_world_scenery(b, kind):
    """Refine an unnormalized asset root; return whether the ID was handled."""
    if kind not in COVERED_ASSETS:
        return False
    if kind in {"rail", "guardrail"}:
        rail(b)
    elif kind.startswith("corner"):
        original = b.root
        for wing in list(original.children):
            if wing.type == "EMPTY" and wing.name.startswith("Corner wing"):
                b.root = wing
                rail(b)
        b.root = original
    elif kind in {"floor-tile", "track-straight", "track-bend", "kerb"}:
        surfaces(b, kind)
    elif kind in {"lamp", "utility-column", "bollard"}:
        column(b, kind)
    elif kind in {"deposit", "rock-obstacle"}:
        minerals(b, kind)
    elif kind == "gantry":
        gantry(b)
    elif kind in {"start-light", "pit-station", "direction-board"}:
        station(b, kind)
    elif kind in {"tether-fitting", "capture-latch"}:
        fittings(b, kind)
    else:
        effects(b, kind)
    scenery_profiles(b, kind)
    b.root["sceneryRefinement"] = "concept-assemblies-v3"
    return True


def scenery_polish_after_fit(b):
    """Finish fixed scenery after fitting, without changing its runtime contract.

    Bake only modifiers already evaluated by export, then retain their polygon
    topology and local bounds. Frame details created earlier stay attached, and
    no normalization, motion transform, socket or material batch is introduced.
    """
    bpy.context.window.scene = b.scene
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    changed = []
    roofs = (
        "Cantilever alloy roof",
        "Vaulted burgundy pit shelter",
        "Pit roof brass rib",
        "Directional lamp armored head",
        "Lantern bell roof",
        "Gantry bolted capital",
        "Signal eyebrow visor",
        "Alloy rail cap",
        "Sign frame horizontal surround",
    )
    panels = (
        "Burgundy rail panel",
        "Armored rail wedge",
        "Segmented alloy impact shoulder",
        "Rail post",
        "Column armor corner pilaster",
        "Tether tapered pedestal",
        "Service equipment cabinet",
        "Riveted boiler room",
        "Direction board",
        "Start signal housing",
        "Latch chamber end cage",
        "Post stepped alloy foot",
    )
    collars = (
        "Tether swivel bearing",
        "Tether keyed spindle",
        "Tether mechanical coupling",
        "Latch brass pipe ferrule",
        "Latch cable gland",
        "Pressure column brass collar",
        "Forged post shoulder",
        "Column machined perimeter collar",
        "Lamp fluted foot",
    )
    for kind, root in b.roots.items():
        if kind not in COVERED_ASSETS:
            continue
        meshes = [o for o in root.children_recursive if o.type == "MESH"]
        parent_materials = {}
        for obj in meshes:
            parent_materials.setdefault(obj.parent, set()).update(obj.data.materials)
        for obj in meshes:
            name = obj.name
            mode = (
                "roof"
                if name.startswith(roofs)
                else "panel"
                if name.startswith(panels)
                else "collar"
                if name.startswith(collars)
                else "jaw"
                if name.startswith("Articulated forged capture jaw")
                else "kerb"
                if name.startswith((
                    "Inset alternating kerb slab",
                    "Alternating curved kerb block",
                ))
                else "ribbon"
                if name.startswith(("Energy ribbon", "Steam wisp"))
                else None
            )
            if mode is None:
                continue
            # Export already evaluates these bevels. Baking freezes their cost
            # before a shape change can affect bevel clamps or tessellation.
            if obj.modifiers:
                mesh = bpy.data.meshes.new_from_object(
                    obj.evaluated_get(deps), preserve_all_data_layers=True, depsgraph=deps
                )
                obj.modifiers.clear()
                obj.data = mesh
            elif obj.data.users > 1:
                obj.data = obj.data.copy()
            mesh = obj.data
            # Keep evaluated polygons intact: splitting a flat-shaded quad here
            # gives its deformed triangles different normals and duplicates
            # exported vertices. Export triangulates each polygon at the same
            # n-2 triangle cost while sharing the polygon normal.
            vertices = mesh.vertices
            lo = [min(v.co[i] for v in vertices) for i in range(3)]
            hi = [max(v.co[i] for v in vertices) for i in range(3)]
            span = [max(hi[i] - lo[i], 1e-9) for i in range(3)]
            center = [(a + z) / 2 for a, z in zip(lo, hi)]
            if mode == "ribbon":
                # Paired ribbon vertices share a spine; taper only its outer
                # edge, leaving flow trajectories and the near-nozzle join fixed.
                pairs = len(vertices) // 2
                for j in range(pairs):
                    spine = vertices[2 * j].co.copy()
                    edge = vertices[2 * j + 1].co
                    t = j / max(1, pairs - 1)
                    width = 0.22 + 0.78 * math.sin(math.pi * t) ** 0.65
                    width *= 0.82 + 0.18 * math.cos(t * math.pi * 5) ** 2
                    edge[:] = spine + (edge - spine) * width
            else:
                for vertex in vertices:
                    v = vertex.co
                    x, y, z = [(v[i] - lo[i]) / span[i] for i in range(3)]
                    if mode == "roof":
                        # Broad lower eaves support a narrower raised crown.
                        crown = max(0, (z - 0.24) / 0.76)
                        v.x = center[0] + (v.x - center[0]) * (1 - 0.18 * crown)
                        v.y = center[1] + (v.y - center[1]) * (1 - 0.26 * crown)
                    elif mode == "panel":
                        axis = 0 if kind in {"direction-board", "start-light"} else 1
                        v[axis] = center[axis] + (v[axis] - center[axis]) * (0.62 + 0.38 * z)
                    elif mode == "collar":
                        # Keep each ferrule's mating end broad and its opposite
                        # face smaller, revealing a readable machined shoulder.
                        v.x = center[0] + (v.x - center[0]) * (1 - 0.20 * z)
                        v.y = center[1] + (v.y - center[1]) * (1 - 0.20 * z)
                    elif mode == "jaw":
                        v.y = center[1] + (v.y - center[1]) * (0.76 + 0.24 * x)
                        v.x += span[0] * 0.10 * x * (1 - x) * (2 * y - 1)
                    elif mode == "kerb":
                        v.y = center[1] + (v.y - center[1]) * (1 - 0.28 * z)
            for axis in range(3):
                lower = min(v.co[axis] for v in vertices)
                upper = max(v.co[axis] for v in vertices)
                if upper - lower > 1e-9:
                    factor = (hi[axis] - lo[axis]) / (upper - lower)
                    for vertex in vertices:
                        vertex.co[axis] = lo[axis] + (vertex.co[axis] - lower) * factor
            # These are complete, pre-existing batches, never additional slots.
            target = None
            if name.startswith(("Latch chamber end cage", "Tether keyed spindle")):
                target = b.mats["dark"]
            elif not b.steam and name.startswith(("Rail post", "Column armor corner pilaster")):
                target = b.mats["trim"]
            if target is not None and target in parent_materials[obj.parent]:
                mesh.materials[0] = target
            mesh.update()
            changed.append((kind, obj.name))
    bpy.context.view_layer.update()
    return changed
