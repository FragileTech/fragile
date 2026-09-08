"""Author the five original, editable control-laboratory presets."""

import json
import math
from pathlib import Path


DEST = Path(__file__).resolve().parents[1] / "web/lab/scenarios"
DEST.mkdir(parents=True, exist_ok=True)


def ship(x, y, angle=0, **kwargs):
    return dict(
        position=[x, y],
        angle=angle,
        radius=0.65,
        mass=1,
        controlled=True,
        thrust=16,
        torque=3,
        angular_drag=3,
        drag=0.2,
        **kwargs,
    )


def rock(x, y, radius=1.2, mass=3):
    return {
        "position": [x, y],
        "mass": mass,
        "cargo": True,
        "drag": 0.08,
        "vertices": [
            [
                round(radius * math.cos(i * math.tau / 7), 5),
                round(radius * math.sin(i * math.tau / 7), 5),
            ]
            for i in range(7)
        ],
    }


def save(key, name, description, task, lethal_walls=True, **data):
    scene = dict(
        version=1,
        name=name,
        description=description,
        task=task,
        size=[64, 44],
        physics={
            "dt": 1 / 60,
            "substeps": 4,
            "solver_iterations": 8,
            "lethal_walls": lethal_walls,
            "lethal_bodies": False,
        },
        **data,
    )
    scene["agent_types"] = json.loads((DEST.parent / "agent-catalog.json").read_text())
    for body in scene["bodies"]:
        if body.get("controlled"):
            body.setdefault("agent_type", "kart" if task == "forage" else "rocket")
    (DEST / f"{key}.json").write_text(json.dumps(scene, indent=2) + "\n")


cave = [
    [2, 8],
    [10, 2],
    [29, 2],
    [33, 6],
    [53, 3],
    [62, 12],
    [60, 28],
    [53, 41],
    [36, 41],
    [30, 37],
    [12, 41],
    [3, 32],
    [5, 21],
]
holes = [[[27, 17], [33, 15], [38, 20], [36, 26], [29, 28], [25, 23]]]
save(
    "harvest",
    "Asteroid harvesting",
    "Recover the ore. Every branch is a possible future.",
    "harvest",
    boundary=cave,
    holes=holes,
    bodies=[
        ship(16, 16, 0.45),
        dict(rock(19, 19), respawn=True),
        dict(rock(45, 31, 1.6, 5), respawn=True),
        dict(rock(47, 12, 1.1, 2), respawn=True),
        dict(rock(16, 31, 0.9, 2), respawn=True),
        dict(rock(41, 9, 1.4, 4), respawn=True),
    ],
    bases=[{"position": [12, 11], "radius": 3}],
    gravity=[{"position": [46, 22], "strength": 28, "softening": 3}],
    tethers=[{"a": 0, "b": -1, "automatic": True, "hook_range": 2.8, "rest_length": 2.5}],
)
# Four-metre grid, excluding the central pillar with 1.5 metres of clearance.
ants_positions = [
    [x, y]
    for y in range(4, 41, 4)
    for x in range(4, 61, 4)
    if math.hypot(max(29 - x, 0, x - 33), max(17 - y, 0, y - 28)) >= 1.5
]
save(
    "ants",
    "Ants & drops",
    "5 harvesters. Fill 5-drop tanks and unload at the refinery over 2 simulation seconds. Drops return after 3 seconds.",
    "forage",
    lethal_walls=False,
    bodies=[
        {
            "agent_type": "harvester",
            "position": ants_positions[int((i + 0.5) * len(ants_positions) / 5)],
            "angle": (i % 6) - 3,
        }
        for i in range(5)
    ],
    boundary=[[2, 2], [62, 2], [62, 42], [2, 42]],
    holes=[[[29, 17], [33, 17], [33, 28], [29, 28]]],
    pickups=[
        {"position": [6 + (i * 13) % 52, 5 + (i * 17) % 33], "radius": 0.4} for i in range(24)
    ],
    respawn_seconds=3,
    cargo={"capacity": 5, "unload_seconds": 2},
    refineries=[{"position": [12, 36], "radius": 6}],
)
save(
    "tandem",
    "Tandem flight",
    "Guide a coordinated pair through the checkpoint loop.",
    "tandem",
    boundary=cave,
    holes=holes,
    bodies=[ship(12, 12), ship(12, 16)],
    formation_distance=4,
    gates=[
        {"position": p, "radius": 3.5}
        for p in [[23, 10], [45, 10], [53, 22], [44, 34], [20, 34], [12, 21]]
    ],
)
save(
    "mining",
    "Collaborative mining",
    "One liftable rock at a time. Team up to haul it faster; delivery spawns the next.",
    "harvest",
    boundary=cave,
    holes=holes,
    bodies=[
        ship(19, 10),
        ship(19, 15),
        dict(rock(22, 12.5, 1.4, 0.24), drag=0.8, respawn=True),
    ],
    bases=[{"position": [12, 12], "radius": 3}],
    gravity=[{"position": [47, 22], "strength": 18, "softening": 4}],
    tethers=[
        {
            "a": i,
            "b": 2,
            "automatic": True,
            "hook_range": 3.5,
            "rest_length": 3.5,
            "stiffness": 35,
            "damping": 8,
        }
        for i in range(2)
    ],
)
save(
    "rocket",
    "Mining rocket · thinking graphs",
    "Read the search. Explore risk, cloning and ancestry.",
    "harvest",
    boundary=cave,
    holes=holes,
    bodies=[ship(20, 12, 0.4), rock(23, 14, 1.2, 4), rock(47, 30, 1.3, 4)],
    bases=[{"position": [12, 11], "radius": 3}],
    gravity=[{"position": [46, 22], "strength": 45, "softening": 3}],
    tethers=[{"a": 0, "b": 1, "automatic": True, "hook_range": 3, "rest_length": 3.6}],
)
