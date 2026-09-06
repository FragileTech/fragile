"""Assemble the final Blender reference renders; run with Python and Pillow."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2] / "web/lab/assets"


def contact_sheet(destination, rows, filename, width=800):
    height = width * 3 // 4
    heading = 40
    image = Image.new("RGB", (width * 2, (height + heading) * len(rows)), "#131b23")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for row, name in enumerate(rows):
        for column, style in enumerate(["futuristic", "steampunk"]):
            path = ROOT / "previews" / style / filename(name)
            with Image.open(path) as source:
                tile = source.convert("RGB")
                tile.thumbnail((width, height), Image.Resampling.LANCZOS)
                x = column * width + (width - tile.width) // 2
                y = row * (height + heading) + heading + (height - tile.height) // 2
                image.paste(tile, (x, y))
            draw.text(
                (column * width + 18, row * (height + heading) + 11),
                f"{style} / {name.replace('-', ' ')}".upper(),
                fill="#dee6ee",
                font=font,
            )
    image.save(ROOT / "previews" / destination, quality=92)


if __name__ == "__main__":
    contact_sheet(
        "collections.jpg",
        ["rocket", "kart", "drone", "harvester"],
        lambda name: f"{name}-hero.png",
    )
    families = json.loads((ROOT / "world-catalog.json").read_text())["families"]
    contact_sheet("world-collections.jpg", list(families), lambda name: f"world-{name}.png", 600)
