# /// script
# requires-python = ">=3.10"
# dependencies = ["shapely==2.1.2"]
# ///
"""Generate the kart library: uv run --script tools/make-racing-scene.py.

Only this authoring tool needs Shapely. The browser consumes standalone JSON.
Use --check to validate geometry and detect stale generated scenes without writing.
"""

import argparse
import copy
import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity

ROOT = Path(__file__).resolve().parents[1] / "web/lab"


def ring(radius):
    """Counter-clockwise rounded rectangle, sharing corner centres at all widths."""
    points = []
    for x, y, start in [(58, 22, -90), (58, 34, 0), (22, 34, 90), (22, 22, 180)]:
        for i in range(17):
            angle = math.radians(start + i * 90 / 16)
            points.append([
                round(x + radius * math.cos(angle), 5),
                round(y + radius * math.sin(angle), 5),
            ])
    return points


def arc(x, y, angle):
    angle = math.radians(angle)
    return [round(x + 12 * math.cos(angle), 5), round(y + 12 * math.sin(angle), 5)]


def racing_scene():
    checkpoints = [
        [40, 10],
        [54, 10],
        arc(58, 22, -60),
        arc(58, 22, -30),
        [70, 28],
        arc(58, 34, 30),
        arc(58, 34, 60),
        [54, 46],
        [40, 46],
        [26, 46],
        arc(22, 34, 120),
        arc(22, 34, 150),
        [10, 28],
        arc(22, 22, 210),
        arc(22, 22, 240),
        [28, 10],
    ]
    types = json.loads((ROOT / "agent-catalog.json").read_text())
    types["racing_kart"] = {
        "extends": "kart",
        "label": "Mite R · circuit kart",
        "physics": {
            "radius": 0.85,
            "thrust": 12,
            "drag": 0.65,
            "angular_drag": 5,
            "restitution": 0.05,
            "actuator": {
                "kind": "kart",
                "wheelbase": 1.1,
                "steering_limit": 0.55,
                "lateral_grip": 18,
                "brake_deceleration": 22,
            },
        },
        "visual": {"model": "kart", "color": "#c69bd9", "scale": 1.2},
    }
    return {
        "version": 1,
        "name": "Violet Circuit · kart racing",
        "task": "navigation",
        "description": "Drive the Mite R around the circuit. Reach all 16 checkpoints in order to complete a lap. W/S drive, A/D steer, Space brakes.",
        "size": [80, 56],
        "physics": {
            "dt": 1 / 60,
            "substeps": 4,
            "solver_iterations": 8,
            "lethal_walls": False,
            "lethal_bodies": False,
        },
        "collision_penalty": 4,
        "progress_reward": 2,
        "gate_reward": 12,
        "boundary": ring(16),
        "holes": [ring(8)],
        "agent_types": types,
        "bodies": [{"agent_type": "racing_kart", "position": [25, 10], "angle": 0}],
        "gates": [{"position": p, "radius": 3.4} for p in checkpoints],
        "evaluation": {"metric": "gates", "target": len(checkpoints)},
        "presentation": {
            "task_label": "Kart time trial",
            "score": {"metric": "gates", "divisor": len(checkpoints), "label": "Laps completed"},
            "progress": {"metric": "gates", "cycle": len(checkpoints), "label": "Checkpoint"},
        },
        "environment": {
            "kind": "circuit",
            "centerline": ring(12),
            "width": 8,
            "start": {"position": [28, 10], "angle": 0},
            "sponsor": {"position": [40, 28], "size": 11},
        },
    }


def smooth(points, closed=True):
    """Sample a restrained interpolating cubic without changing traced anchor points."""
    result = []
    n = len(points)
    for i in range(n if closed else n - 1):
        a, b = points[i], points[(i + 1) % n]
        before = points[(i - 1) % n] if closed or i else a
        after = points[(i + 2) % n] if closed or i + 2 < n else b
        steps = max(2, math.ceil(math.dist(a, b) / 4))
        for j in range(steps):
            t = j / steps
            h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
            h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
            result.append([
                h00 * a[k] + h10 * 0.35 * (b[k] - before[k])
                + h01 * b[k] + h11 * 0.35 * (after[k] - a[k])
                for k in range(2)
            ])
    return result


def replace_section(points, patch):
    start, end = points.index(patch["from"]), points.index(patch["to"])
    return points[:start + 1] + patch["points"] + points[end:]


def trace_scene(spec, base):
    scale, height, padding = spec["scale"], spec["frame"][1], 5

    def world(p):
        return [round(p[0] * scale + padding, 5),
                round((height - p[1]) * scale + padding, 5)]

    route = LineString([world(p) for p in smooth(spec["route"])])
    route = LineString([*route.coords, route.coords[0]])
    if "outer" in spec:
        holes = [[world(p) for p in smooth(spec["inner"])]]
        holes += [[world(p) for p in smooth(r)] for r in spec.get("extra_holes", [])]
        for obstacle in spec.get("obstacles", []):
            rx, ry = obstacle.get("radii", [obstacle.get("radius")] * 2)
            x, y = obstacle["position"]
            holes.append([world([x + rx * math.cos(i * math.tau / 24),
                                 y + ry * math.sin(i * math.tau / 24)]) for i in range(24)])
        road = Polygon([world(p) for p in smooth(spec["outer"])], holes)
    else:
        road = route.buffer(spec["width"] * scale / 2, quad_segs=8)
    if not road.is_valid or road.geom_type != "Polygon":
        raise ValueError(f"{spec['id']}: invalid trace: {explain_validity(road)}")
    road = orient(road, sign=1)
    # Small simplification removes redundant collinear samples and keeps native
    # collision work bounded. It never repairs invalid hand traces silently.
    road = road.simplify(0.035, preserve_topology=True)
    route = route.simplify(0.025)
    points = list(route.coords)[:-1]
    count = max(16, math.ceil(route.length / 6))
    spacing = route.length / count
    checkpoints = []
    for i in range(1, count + 1):
        p = route.interpolate((i * spacing) % route.length)
        clearance = p.distance(road.boundary)
        radius = min(2.5, clearance - 1.0, spacing * 0.32)
        if not road.contains(p) or radius < 0.65:
            raise ValueError(f"{spec['id']}: checkpoint {i} lacks clearance at {p}")
        checkpoints.append({"position": list(p.coords[0]), "radius": radius})
    start = route.interpolate(0)
    spawn = route.interpolate(route.length - 3)
    tangent = route.interpolate(0.5)
    angle = math.atan2(tangent.y - start.y, tangent.x - start.x)
    spawn_forward = route.interpolate(route.length - 2.5)
    spawn_angle = math.atan2(spawn_forward.y - spawn.y, spawn_forward.x - spawn.x)
    scene = copy.deepcopy(base)
    scene.update({
        "name": f"{spec['name']} · kart racing",
        "description": spec["description"],
        "size": [spec["frame"][0] * scale + 2 * padding, height * scale + 2 * padding],
        "boundary": list(road.exterior.coords)[:-1],
        "holes": [list(h.coords)[:-1] for h in road.interiors],
        "bodies": [{"agent_type": "racing_kart", "position": list(spawn.coords[0]),
                    "angle": spawn_angle}],
        "gates": checkpoints,
        "evaluation": {"metric": "gates", "target": count},
        "environment": {"kind": "circuit", "centerline": points,
                        "width": spec["width"] * scale,
                        "start": {"position": list(start.coords[0]), "angle": angle}},
        "circuit": {
            "id": spec["id"], "name": spec["name"], "difficulty": spec["difficulty"],
            "description": spec["description"],
            "direction": "counterclockwise" if Polygon(points).exterior.is_ccw else "clockwise",
            "scale_note": "Video reconstruction; distances are simulation units, not surveyed track dimensions.",
            "uncertainty": spec["uncertainty"],
            "sources": [{"url": f"https://www.youtube.com/watch?v={s['video']}&t={s['seconds']}s",
                         "title": s["title"], "seconds": s["seconds"]} for s in spec["sources"]],
        },
    })
    scene["presentation"]["score"]["divisor"] = count
    scene["presentation"]["progress"]["cycle"] = count
    # Quantize once, including checkpoints and angles, for stable generated files.
    def rounded(value):
        if isinstance(value, float):
            return round(value, 5)
        if isinstance(value, (list, tuple)):
            return [rounded(v) for v in value]
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        return value
    result = rounded(scene)
    result["physics"] = copy.deepcopy(base["physics"])
    return result


def validate_scene(scene):
    road = Polygon(scene["boundary"], scene["holes"])
    route = LineString([*scene["environment"]["centerline"], scene["environment"]["centerline"][0]])
    if not road.is_valid:
        raise ValueError(explain_validity(road))
    if not road.buffer(-1.0).contains(route):
        outside = route.difference(road.buffer(-1.0))
        raise ValueError(f"{scene['name']}: route lacks kart clearance: {outside.wkt[:500]}")
    if not road.buffer(-1.0).contains(Point(scene["bodies"][0]["position"])):
        raise ValueError(f"{scene['name']}: unsafe spawn")
    gates = scene["gates"]
    for i, gate in enumerate(gates):
        disk = Point(gate["position"]).buffer(gate["radius"])
        if not road.contains(disk):
            raise ValueError(f"{scene['name']}: checkpoint {i} crosses a wall")
        for j, other in enumerate(gates[i + 1:], i + 1):
            # Violet's established adjacent gate disks overlap slightly. Preserve
            # that scene's physics and progression for existing recordings.
            if scene["circuit"]["id"] == "racing" and (j - i in (1, len(gates) - 1)):
                continue
            if math.dist(gate["position"], other["position"]) <= gate["radius"] + other["radius"]:
                raise ValueError(f"{scene['name']}: overlapping checkpoints")
    print(f"{scene['name']}: {len(gates)} checkpoints, {len(scene['holes'])} infield/obstacle holes, "
          f"minimum route clearance {route.distance(road.boundary):.2f}")


def library():
    violet = racing_scene()
    violet["circuit"] = {
        "id": "racing", "name": "Violet Circuit", "difficulty": "Easy",
        "description": "The original lab oval. Wide, consistent corners for learning the kart controls.",
        "direction": "counterclockwise", "sources": [],
    }
    yield "racing", violet
    specs = json.loads(Path(__file__).with_name("kart-circuits.json").read_text())["circuits"]
    resolved = {}
    for spec in specs:
        if "base" in spec:
            spec = {**copy.deepcopy(resolved[spec["base"]]), **spec}
            for key in ["outer", "inner", "route"]:
                if f"{key}_replace" in spec:
                    spec[key] = replace_section(spec[key], spec[f"{key}_replace"])
        resolved[spec["id"]] = spec
        yield spec["id"], trace_scene(spec, violet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scenes = list(library())
    # Validate the entire library before replacing any output.
    for _, scene in scenes:
        validate_scene(scene)
    for scene_id, scene in scenes:
        path = ROOT / f"scenarios/{scene_id}.json"
        content = json.dumps(scene, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale generated circuit: {path}")
        else:
            path.write_text(content)
