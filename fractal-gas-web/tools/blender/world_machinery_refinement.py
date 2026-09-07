"""Concept-specific resource and machinery construction before envelope fitting.

Additions share the pack's existing materials and moving parents. Small fasteners
remain in the established panel atlas; the low LOD keeps only readable assemblies.
"""

import math

import bpy


TAU = math.tau
COVERAGE = (
    "drop-crystal",
    "drop-nugget",
    "drop-salvage",
    "drop-capsule",
    "drop-core",
    "drop-pile",
    "ore-small",
    "ore-medium",
    "ore-large",
    "ore-captured",
    "capture-clamp",
    "reactor",
    "dock",
    "beacon",
    "gate",
    "gate-pylons",
    "gate-marker",
    "finish-arch",
)


def remove(b, *prefixes):
    for obj in list(b.root.children_recursive):
        if obj.name.startswith(prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)


def radial_box(b, name, radius, angle, z, size, mat="plate", parent=None):
    obj = b.box(
        name,
        (radius * math.cos(angle), radius * math.sin(angle), z),
        size,
        mat,
        parent,
        bevel=0.008,
    )
    obj.rotation_euler.z = angle
    return obj


def strut(b, name, points, radius=0.025, mat="trim", parent=None):
    # Cylinder ends are deliberately buried in the neighboring housing.
    b.pipe(name, points, radius, mat, parent)


def resources(b, kind):
    if kind in {"drop-crystal", "drop-pile"}:
        # The concept uses a shallow octagonal tray and broken host rock, not a
        # smooth circular pedestal under freestanding prisms.
        remove(b, "Low armored base", "Perimeter retaining band")
        b.cyl("Octagonal mineral tray", (0, 0, 0.07), 0.46, 0.14, "dark", segments=8)
        for j in range(8):
            a = TAU * j / 8
            radial_box(
                b,
                "Segmented tray rim",
                0.425,
                a,
                0.14,
                (0.075, 0.29, 0.075),
                "trim" if b.steam else "plate",
            )
            if j % 2 == 0:
                radial_box(
                    b,
                    "Tray clasp",
                    0.433,
                    a,
                    0.18,
                    (0.07, 0.07, 0.055),
                    "copper" if b.steam else "gold",
                )
        if kind == "drop-crystal":
            for j in range(5 if b.low else 11):
                a = j * 2.4
                # Cheap seven-sided chunks avoid another high-resolution stone
                # sphere per fragment while preserving angular host-rock rubble.
                o = b.cyl(
                    "Broken host-rock shard",
                    (0.30 * math.cos(a), 0.30 * math.sin(a), 0.20),
                    0.065 + j % 3 * 0.014,
                    0.14 + j % 2 * 0.05,
                    "stone",
                    r2=0.025,
                    segments=5,
                )
                o.rotation_euler = (0.24 * math.sin(a), 0.3 * math.cos(a), a)
        else:
            for j in range(3):
                radial_box(
                    b, "Recovered ore clasp", 0.28, j * 2.4, 0.24, (0.12, 0.10, 0.07), "trim"
                )
    elif kind == "drop-nugget":
        for j in range(5):
            a = j * TAU / 5
            radial_box(
                b,
                "Cut nugget armor facets",
                0.28,
                a,
                0.42,
                (0.08, 0.18, 0.25),
                "trim" if b.steam else "plate",
            )
            radial_box(b, "Facet fastening shoe", 0.322, a, 0.31, (0.028, 0.08, 0.07), "gold")
    elif kind == "drop-salvage":
        for x in [-0.305, 0.305]:
            for y in [-0.28, 0.28]:
                b.box("Crate corner shoe", (x, y, 0.38), (0.10, 0.10, 0.51), "trim", bevel=0.018)
        for x in [-0.20, 0.20]:
            b.box(
                "Crate front inset",
                (x, -0.306, 0.39),
                (0.17, 0.024, 0.28),
                "seat" if b.steam else "plate",
                bevel=0.015,
            )
            b.box("Buckle backing", (x, -0.329, 0.40), (0.085, 0.028, 0.10), "trim")
            b.box("Buckle inset", (x, -0.347, 0.40), (0.043, 0.012, 0.049), "dark", bevel=0)
        if not b.low:
            for y in [-0.20, 0.20]:
                strut(
                    b,
                    "Recessed carry handle",
                    [(-0.12, y, 0.67), (-0.12, y, 0.73), (0.12, y, 0.73), (0.12, y, 0.67)],
                    0.018,
                )
    elif kind == "drop-capsule":
        # A visible mineral chamber matches the capsule cutaway. At small sizes
        # opaque tinted glazing avoids transparent sorting and overdraw.
        for obj in b.root.children_recursive:
            if obj.type == "MESH" and obj.name.startswith(("Pressure vessel", "Field cartridge")):
                obj.data.materials.clear()
                obj.data.materials.append(b.mats["glass" if b.low else "window"])
        for z in [0.17, 0.67]:
            b.cyl(
                "Capsule stepped armor collar",
                (0, 0, z),
                0.29,
                0.09,
                "trim" if b.steam else "plate",
                r2=0.25,
                segments=8,
            )
            b.cyl(
                "Capsule terminal insert",
                (0, 0, z + (0.051 if z > 0.4 else -0.051)),
                0.17,
                0.018,
                "dark",
                segments=8,
            )
        b.cyl("Capsule terminal contact", (0, 0, 0.738), 0.085, 0.045, "gold", segments=8)
        for j in range(4):
            a = j * TAU / 4 + TAU / 8
            radial_box(
                b,
                "Capsule longitudinal guard",
                0.245,
                a,
                0.42,
                (0.055, 0.058, 0.45),
                "copper" if b.steam else "dark",
            )
            radial_box(b, "Capsule locking cleat", 0.272, a, 0.65, (0.06, 0.085, 0.065), "gold")
        if b.steam:
            strut(
                b,
                "Capsule side siphon",
                [(0.25, 0, 0.23), (0.32, 0, 0.23), (0.32, 0, 0.58), (0.25, 0, 0.58)],
                0.024,
                "copper",
            )
        else:
            for z in [0.25, 0.58]:
                b.polygon_band("Cap inset light annulus", 0.25, 0.018, z, 0.025, "light")
    elif kind == "drop-core":
        # Nested armor belts leave the glowing mineral face exposed.
        for j in range(6):
            a = j * TAU / 6
            radial_box(b, "Core radial locking lug", 0.355, a, 0.46, (0.10, 0.13, 0.11), "trim")
            radial_box(
                b,
                "Core indexing insert",
                0.412,
                a,
                0.46,
                (0.022, 0.07, 0.055),
                "copper" if b.steam else "light",
            )
        for z in [0.23, 0.68]:
            b.cyl("Core polar socket", (0, 0, z), 0.15, 0.07, "dark", segments=8)
            b.cyl("Core socket contact", (0, 0, z + 0.04), 0.08, 0.035, "gold", segments=8)


def ores(b, kind):
    if kind == "capture-clamp":
        for y in [-0.10, 0.10]:
            b.box("Clamp cheek housing", (-0.09, y, 0.14), (0.24, 0.05, 0.20), "trim")
        if b.steam:
            for y in [-0.12, 0.12]:
                b.cyl("Winch drum cheek", (0, y, 0.20), 0.12, 0.035, "dark", "y", segments=10)
            b.box("Winch ratchet pawl", (-0.17, -0.07, 0.245), (0.15, 0.05, 0.05), "trim")
        else:
            for s in [-1, 1]:
                b.box("Replaceable jaw pad", (0.22, s * 0.11, 0.12), (0.18, 0.026, 0.10), "dark")
                b.cyl("Capture ram", (-0.015, s * 0.14, 0.20), 0.03, 0.20, "gold", "x", segments=8)
        return
    # Grounded insertion socket and fractured mineral seams retain a natural
    # outline; a restrained mechanical saddle makes the tow interface readable.
    b.cyl("Ore socket backing shoulder", (0.77, 0, 0.74), 0.27, 0.10, "dark", "x", segments=8)
    b.cyl("Ore receiver inset", (0.842, 0, 0.74), 0.105, 0.035, "dark", "x", segments=8)
    for j in range(3):
        a = j * TAU / 3
        p = (0.865, 0.185 * math.cos(a), 0.74 + 0.185 * math.sin(a))
        b.cyl(
            "Bayonet locking receiver",
            p,
            0.043,
            0.045,
            "gold" if b.steam else "trim",
            "x",
            segments=6,
        )
    if b.steam:
        for s in [-1, 1]:
            strut(
                b,
                "Ore anchor retaining bar",
                [(0.81, s * 0.18, 0.63), (0.64, s * 0.32, 0.40), (0.40, s * 0.42, 0.26)],
                0.027,
                "trim",
            )
        if not b.low:
            strut(
                b,
                "Capture pressure signal",
                [(0.83, 0, 0.95), (0.63, 0.12, 1.12), (0.22, 0.20, 1.19)],
                0.022,
                "copper",
            )
    else:
        for s in [-1, 1]:
            b.box("Embedded capture locator", (0.82, s * 0.28, 0.72), (0.06, 0.07, 0.12), "light")
    # Broken outcrops differ by ore size in the sheets; don't cover every rock
    # in an identical three-prism crown.
    remove(b, "Mineral prism")
    count = {"ore-small": 2, "ore-medium": 3, "ore-large": 5, "ore-captured": 3}[kind]
    for j in range(count):
        a = j * 2.4 + 0.6
        o = b.crystal(
            (0.58 * math.cos(a), 0.58 * math.sin(a), 0.52 + j % 2 * 0.16),
            0.085 + (j % 2) * 0.025,
            0.19 + (j % 3) * 0.045,
            seed=17 + j,
        )
        o.rotation_euler.y += 0.35 * math.cos(a)


def reactor(b):
    # Strengthen the stacked governor and broad machinery plinth in the sheets.
    for z, r in [(0.16, 0.98), (0.24, 0.88)]:
        b.cyl("Reactor stepped foundation", (0, 0, z), r, 0.08, "dark", segments=16)
    for j in range(8):
        a = j * TAU / 8
        radial_box(
            b,
            "Reactor service bay",
            0.90,
            a,
            0.30,
            (0.19, 0.34, 0.14),
            "plate",
        )
        if not b.low:
            radial_box(
                b,
                "Service bay inset",
                0.98,
                a,
                0.30,
                (0.013, 0.22, 0.065),
                "dark",
            )
    if b.steam:
        for z, r, mat in [
            (2.11, 0.34, "dark"),
            (2.19, 0.38, "trim"),
            (2.29, 0.31, "plate"),
            (2.35, 0.25, "trim"),
        ]:
            b.cyl("Governor stepped casing", (0, 0, z), r, 0.085, mat)
        for x in [-0.13, 0.13]:
            b.cyl("Governor safety valve", (x, 0, 2.44), 0.042, 0.13, "copper", segments=8)
        b.gauge((0, -0.30, 2.27), 0.085)
        for j in range(4):
            a = j * TAU / 4 + TAU / 8
            strut(
                b,
                "Plinth steam distribution",
                [
                    (0.8 * math.cos(a), 0.8 * math.sin(a), 0.20),
                    (0.97 * math.cos(a), 0.97 * math.sin(a), 0.20),
                    (0.97 * math.cos(a), 0.97 * math.sin(a), 0.38),
                ],
                0.023,
                "copper",
            )
    else:
        for j in range(6):
            a = j * TAU / 6
            radial_box(b, "Crown field cartridge", 0.49, a, 2.14, (0.20, 0.13, 0.17), "dark")
            radial_box(b, "Crown violet aperture", 0.52, a, 2.24, (0.14, 0.055, 0.017), "energy")
        b.cyl("Upper suspension boss", (0, 0, 2.03), 0.13, 0.22, "trim", segments=8)
    # Detailed-only concentric tracks are attached to the existing deterministic
    # gimbal pivots, never a second unrelated animation system.
    if not b.low:
        for pivot in list(b.root.children_recursive):
            if pivot.get("motion") != "world-gimbal":
                continue
            axis = "xyz"[pivot["axisIndex"]]
            b.ring("Inset gimbal bearing track", (0, 0, 0), 0.85, 0.013, "dark", axis, pivot)


def dock(b):
    for j in range(6):
        a = j * TAU / 6
        radial_box(
            b, "Dock replaceable deck quadrant", 0.82, a, 0.076, (0.24, 0.40, 0.023), "plate"
        )
        radial_box(b, "Dock service bay surround", 0.97, a, 0.19, (0.14, 0.20, 0.12), "trim")
        radial_box(
            b,
            "Dock recessed service window",
            1.048,
            a,
            0.19,
            (0.014, 0.13, 0.064),
            "dark" if b.steam else "energy",
        )
    # Nested line markings and radial plate joints leave the entire recovery
    # surface visually traversable.
    b.polygon_band(
        "Recovery center index", 0.20, 0.009, 0.066, 0.005, "trim" if b.steam else "light"
    )
    for y in [-0.24, 0.24]:
        b.box(
            "Intake side cheek",
            (0.64, y, 0.13),
            (0.48, 0.047, 0.15),
            "trim" if b.steam else "plate",
        )
    if not b.low:
        for j in range(8):
            a = j * TAU / 8
            radial_box(b, "Deck expansion seam", 0.53, a, 0.065, (0.58, 0.007, 0.004), "dark")
    if b.steam:
        for z in [0.22, 0.30]:
            strut(
                b,
                "Dock pump feed",
                [(-0.75, 0, z), (-0.74, -0.27, z), (-0.45, -0.48, z)],
                0.026,
                "copper",
            )
    else:
        b.box("Dock control hood", (-0.72, 0, 0.415), (0.25, 0.29, 0.045), "dark")


def checkpoints(b, kind):
    if kind == "beacon":
        if b.steam:
            for j in range(4):
                a = j * TAU / 4
                radial_box(
                    b, "Lantern protective upright", 0.14, a, 0.37, (0.028, 0.032, 0.33), "dark"
                )
            b.cyl("Lantern rain cap", (0, 0, 0.62), 0.17, 0.055, "plate", r2=0.11, segments=8)
            b.cyl("Lantern foot flange", (0, 0, 0.16), 0.18, 0.04, "trim")
        else:
            for j in range(3):
                a = j * TAU / 3
                radial_box(
                    b, "Beacon crystal cradle", 0.13, a, 0.39, (0.055, 0.058, 0.21), "plate"
                )
            b.cyl("Beacon emitter collar", (0, 0, 0.32), 0.15, 0.04, "trim", segments=8)
        return
    if kind == "gate-marker":
        for j in range(8):
            a = j * TAU / 8
            radial_box(
                b,
                "Checkpoint ground index",
                0.82,
                a,
                0.155,
                (0.17, 0.11, 0.055),
                "trim" if b.steam else "plate",
            )
            radial_box(
                b,
                "Index directional inset",
                0.78,
                a,
                0.19,
                (0.065, 0.035, 0.012),
                "gold" if b.steam else "light",
            )
        return
    for s in [-1, 1]:
        b.box("Checkpoint foot plinth", (0, s * 0.9, 0.06), (0.39, 0.29, 0.08), "trim")
        if b.steam:
            strut(
                b,
                "Checkpoint pressure return",
                [
                    (-0.08, s * 0.84, 0.18),
                    (-0.13, s * 0.76, 0.25),
                    (-0.13, s * 0.76, 0.66),
                    (-0.08, s * 0.84, 0.70),
                ],
                0.022,
                "copper",
            )
        else:
            b.box("Checkpoint heel gusset", (-0.10, s * 0.9, 0.23), (0.09, 0.22, 0.29), "dark")
        if kind == "gate-pylons":
            b.cyl(
                "Pylon terminal turret",
                (0, s * 0.9, 0.78),
                0.12,
                0.10,
                "trim",
                r2=0.09,
                segments=8,
            )
            b.cyl("Pylon terminal lens", (0, s * 0.9, 0.847), 0.066, 0.042, "energy", segments=8)
    if kind == "gate":
        # Raised segmented covers visibly replace the featureless flat ring.
        for j in range(8):
            a = j * TAU / 8
            parent = next(o for o in b.root.children if o.name.startswith("Open checkpoint ring"))
            b.polygon_band(
                "Gate articulated armor sector",
                0.954,
                0.115,
                0.109,
                0.044,
                "dark" if b.steam else "plate",
                axis="x",
                parent=parent,
                start=a + 0.095,
                end=a + TAU / 8 - 0.095,
            )
            if b.steam and not b.low:
                b.cyl(
                    "Ring regulator dial",
                    (0.155, 0.94 * math.cos(a), 0.94 * math.sin(a)),
                    0.037,
                    0.024,
                    "dial",
                    "x",
                    parent=parent,
                    segments=10,
                )
    elif kind == "finish-arch":
        for s in [-1, 1]:
            strut(b, "Arch knee brace", [(0, s * 0.90, 1.28), (0, s * 0.61, 1.56)], 0.055, "trim")
        for j in range(5):
            b.box(
                "Finish beam cassette",
                (0.147, -0.6 + j * 0.3, 1.59),
                (0.038, 0.22, 0.11),
                "trim" if b.steam else "plate",
            )
            if not b.steam:
                b.box(
                    "Finish indexed lens",
                    (0.170, -0.6 + j * 0.3, 1.59),
                    (0.012, 0.12, 0.035),
                    "light",
                )


def refine_world_machinery(builder, kind):
    if kind not in COVERAGE:
        return
    builder.root["refinementRevision"] = "machinery-concept-3"
    if kind.startswith("drop-"):
        resources(builder, kind)
    elif kind.startswith("ore-") or kind == "capture-clamp":
        ores(builder, kind)
    elif kind == "reactor":
        reactor(builder)
    elif kind == "dock":
        dock(builder)
    else:
        checkpoints(builder, kind)
