"""Assemble the documentation portal, Theory book, and Lab guide."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
from urllib.parse import unquote, urlsplit

from fragile_redirects import redirect_html


STALE_LECTURE_LINK = re.compile(
    r"(?:href|src)=[\"'](?:https://fragiletech\.github\.io/fragile)?"
    r"/docs/(?!theory/|lab/)[^\"']*[\"']"
)


class _LocalLinkParser(HTMLParser):
    """Collect navigation and asset URLs from one generated HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute = "src" if tag in {"img", "script"} else "href"
        if tag not in {"a", "link", "img", "script"}:
            return
        values = dict(attrs)
        if values.get(attribute):
            self.links.append(values[attribute] or "")


def _require_site(path: Path, label: str) -> None:
    if not (path / "index.html").is_file():
        raise FileNotFoundError(f"{label} site has no index.html: {path}")
    if not (path / "searchindex.js").is_file():
        raise FileNotFoundError(f"{label} site has no searchindex.js: {path}")


def stale_lecture_links(site: Path) -> list[str]:
    """Return generated Theory pages containing old root-level documentation URLs."""
    stale = []
    for page in site.rglob("*.html"):
        if STALE_LECTURE_LINK.search(page.read_text(encoding="utf-8")):
            stale.append(page.relative_to(site).as_posix())
    return sorted(stale)


def broken_internal_links(site: Path) -> list[str]:
    """Return generated links whose local file target is absent from the bundle."""
    site = site.resolve()
    broken = []
    for page in site.rglob("*.html"):
        parser = _LocalLinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for raw_link in parser.links:
            if "{{" in raw_link or "{%" in raw_link:
                continue
            link = urlsplit(raw_link)
            if link.scheme or link.netloc or not link.path:
                continue
            path = unquote(link.path)
            if path.startswith("/docs/"):
                target = site / path.removeprefix("/docs/")
            elif path.startswith("/"):
                continue
            else:
                target = (page.parent / path).resolve()
                if target != site and site not in target.parents:
                    continue
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                broken.append(f"{page.relative_to(site).as_posix()}: {raw_link}")
    return sorted(set(broken))


def _write_redirect(output: Path, old: PurePosixPath, target: PurePosixPath) -> None:
    page = output.joinpath(*old.parts)
    page.parent.mkdir(parents=True, exist_ok=True)
    relative = posixpath.relpath(target.as_posix(), old.parent.as_posix() or ".")
    page.write_text(redirect_html(relative), encoding="utf-8")


def _write_legacy_redirects(output: Path, theory: Path, lab: Path) -> None:
    for page in theory.rglob("*.html"):
        relative = PurePosixPath(page.relative_to(theory).as_posix())
        if relative == PurePosixPath("index.html"):
            continue
        _write_redirect(output, relative, PurePosixPath("theory") / relative)

    for page in lab.glob("source/project/*.html"):
        relative = PurePosixPath(page.relative_to(lab).as_posix())
        _write_redirect(output, relative, PurePosixPath("lab") / relative)


def assemble_docs(theory: Path, lab: Path, portal: Path, output: Path) -> None:
    """Create one deployable documentation tree from two Jupyter Book builds."""
    theory = theory.resolve()
    lab = lab.resolve()
    portal = portal.resolve()
    output = output.resolve()
    _require_site(theory, "Theory")
    _require_site(lab, "Lab")
    if not (portal / "index.html").is_file():
        raise FileNotFoundError(f"Documentation portal has no index.html: {portal}")
    if output in {Path("/"), theory, lab, portal}:
        raise ValueError(f"Unsafe documentation output path: {output}")

    stale = stale_lecture_links(theory)
    if stale:
        joined = ", ".join(stale[:10])
        raise ValueError(f"Theory contains stale root-level /docs/ links: {joined}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(theory, output / "theory")
    shutil.copytree(lab, output / "lab")

    for item in portal.iterdir():
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    for asset in ("logo.png", "favicon.png"):
        shutil.copy2(portal.parent / asset, output / asset)

    prompt_source = theory / "_static/prompts"
    if prompt_source.is_dir():
        shutil.copytree(prompt_source, output / "_static/prompts", dirs_exist_ok=True)

    _write_legacy_redirects(output, theory, lab)
    broken = broken_internal_links(output)
    if broken:
        joined = ", ".join(broken[:10])
        raise ValueError(f"Documentation bundle contains broken local links: {joined}")


def main() -> None:
    docs = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory", type=Path, default=docs / "_build/theory-site/_build/html"
    )
    parser.add_argument("--lab", type=Path, default=docs / "_build/lab-site/_build/html")
    parser.add_argument("--portal", type=Path, default=docs / "portal")
    parser.add_argument("--output", type=Path, default=docs / "_build/html")
    args = parser.parse_args()
    assemble_docs(args.theory, args.lab, args.portal, args.output)
    print(f"Assembled documentation portal: {args.output.resolve()}")


if __name__ == "__main__":
    main()
