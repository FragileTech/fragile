"""Recover the proof baseline and index its formal claims without editing lectures."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


BASELINE = "b416d3a98112b360e6ae7f6655fb47497916cfcb"
PREFIX = "docs/source/3_fractal_gas/convergence_program/"
REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "algorithmic-gas" / "proof-recovery"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def main() -> None:
    paths = git("ls-tree", "-r", "--name-only", BASELINE, PREFIX).decode().splitlines()
    selected = [
        path
        for path in paths
        if path.endswith(".md")
        and (
            path[len(PREFIX) :].startswith("proofs/")
            or int(Path(path).name.split("_", 1)[0]) <= 15
        )
    ]
    # This separate reference contains an alternative joint-law LSI argument.
    selected.append(
        "docs/source/3_fractal_gas/appendices/references_do_not_cite/"
        "15_geometric_gas_lsi_proof.md"
    )
    files = []
    claims = []
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(selected):
            data = git("show", f"{BASELINE}:{path}")
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
            files.append(
                {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
            lines = data.decode().splitlines()
            title = ""
            for line_number, line in enumerate(lines, 1):
                directive = re.match(r"\s*:+\{prf:([^}]+)\}\s*(.*)", line)
                if directive:
                    title = directive.group(2)
                label = re.match(r"\s*:label:\s*(\S+)", line)
                if label:
                    claims.append(
                        {"path": path, "line": line_number, "label": label.group(1), "title": title}
                    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    compressed = gzip.compress(archive.getvalue(), mtime=0)
    (OUTPUT / "pre-september-5.tar.gz").write_bytes(compressed)
    manifest = {
        "baseline_commit": BASELINE,
        "archive_sha256": hashlib.sha256(compressed).hexdigest(),
        "files": files,
        "claims": claims,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Recovered {len(files)} files and indexed {len(claims)} labels.")


if __name__ == "__main__":
    main()
