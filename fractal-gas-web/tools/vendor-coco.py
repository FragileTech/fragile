"""Reproduce the pinned COCO amalgamation from a verified upstream checkout."""

import hashlib
import json
from pathlib import Path
import re
import sys


source = Path(sys.argv[1]) / "src"
destination = Path(__file__).resolve().parents[1] / "third_party/coco"
manifest = json.loads((destination / "upstream-files.json").read_text())
for name, expected in manifest["files"].items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Upstream checksum mismatch: {name}")
seen = set()
output = []


def expand(path):
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    output.append(f"\n/* COCO upstream: {path.name} */\n")
    for line in path.read_text().splitlines(keepends=True):
        match = re.match(r'#include "(.*)"', line)
        if match:
            expand(path.parent / match[1])
        else:
            output.append(line)


for entry in ("coco_random.c", "coco_suite.c", "coco_observer.c", "coco_archive.c"):
    expand(source / entry)
output.append('\nconst char *coco_version = "2.8.2";\n')
expand(source / "coco_runtime_c.c")
(destination / "coco.c").write_text(
    "/* Generated from COCO v2.8.2; see README.md. Do not edit benchmark formulas. */\n"
    + "".join(output)
)
