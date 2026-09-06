"""Author an editable kart circuit using the existing native checkpoint task."""

import json
import math
from pathlib import Path


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


if __name__ == "__main__":
    (ROOT / "scenarios/racing.json").write_text(json.dumps(racing_scene(), indent=2) + "\n")
