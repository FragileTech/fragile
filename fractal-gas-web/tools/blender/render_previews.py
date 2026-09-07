"""Refresh reference views from saved editable sources without rebuilding meshes."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", required=True, choices=["futuristic", "steampunk"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rocket", "kart", "drone", "harvester", "world", "refinery"],
        default=["rocket", "kart", "drone", "harvester", "world", "refinery"],
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if not bpy.app.background:
        message = "Run preview batches with blender --background"
        raise RuntimeError(message)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_lab_assets as lab
    from build_world_assets import FAMILIES, render_world

    for kind in args.models:
        print(f"Rendering {args.style}/{kind}", flush=True)
        bpy.ops.wm.open_mainfile(filepath=str(lab.ROOT / "sources" / args.style / f"{kind}.blend"))
        scene = bpy.data.scenes[f"Lab / {args.style} / {kind} / high"]
        bpy.context.window.scene = scene
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 4
        if kind == "world":
            roots = {
                kind: next(o for o in scene.objects if o.get("assetModel") == kind)
                for kinds in FAMILIES.values()
                for kind in kinds
            }
            render_world(SimpleNamespace(scene=scene, style=args.style, roots=roots))
        else:
            views = (
                [("hero", (3, -4, 5)), ("side", (1, 0, 0.15)), ("top", (0, 0, 1))]
                if kind == "refinery"
                else [("hero", (2.8, -3.4, 2.5)), ("side", (0, -4, 0.9)), ("top", (0, 0, 5))]
            )
            for view, direction in views:
                lab.frame_camera(scene, direction)
                scene.render.filepath = str(
                    lab.ROOT / "previews" / args.style / f"{kind}-{view}.png"
                )
                bpy.ops.render.render(write_still=True)
    print("Preview render complete", flush=True)


if __name__ == "__main__":
    main()
