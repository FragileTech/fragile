"""Rebuild the complete authored kit in a dedicated background Blender process.

blender --background --threads 4 --python tools/blender/regenerate.py -- --render
Use --style and --models to rebuild independent parts of the collection.
"""

import argparse
from pathlib import Path
import sys

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["futuristic", "steampunk"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rocket", "kart", "drone", "harvester", "world", "refinery"],
        default=["rocket", "kart", "drone", "harvester", "world", "refinery"],
    )
    parser.add_argument("--render", action="store_true", help="Refresh reference renders too")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    if not bpy.app.background:
        message = "Run this batch command with blender --background"
        raise RuntimeError(message)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_lab_assets import build_asset
    from build_refinery_assets import build_refinery
    from build_world_assets import build_world

    initial_scene = bpy.context.window.scene
    for style in [args.style] if args.style else ["futuristic", "steampunk"]:
        for kind in args.models:
            existing = set(bpy.data.scenes)
            print(f"Building {style}/{kind}", flush=True)
            if kind == "world":
                info = build_world(style, render=args.render)
            elif kind == "refinery":
                info = build_refinery(style, render=args.render)
            else:
                info = build_asset(style, kind, render=args.render)
            print(info, flush=True)
            bpy.context.window.scene = initial_scene
            for scene in set(bpy.data.scenes) - existing:
                bpy.data.scenes.remove(scene)
            bpy.data.orphans_purge(do_recursive=True)
    print("Collection rebuild complete", flush=True)


if __name__ == "__main__":
    main()
