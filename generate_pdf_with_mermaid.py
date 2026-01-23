#!/usr/bin/env python3
"""
Generate PDF from markdown with rendered Mermaid diagrams.

This script implements a multi-stage pipeline:
1. Creates a temporary staging copy of the markdown file
2. Extracts Mermaid diagrams and converts them to PNG images (via SVG intermediate)
3. Replaces diagram code blocks with image references in the staging copy
4. Compiles the modified copy to PDF using pandoc
5. Cleans up all temporary files

The original source markdown file is never modified.

Pipeline: Mermaid → SVG (mmdc) → PNG (Chrome headless) → PDF (pandoc)

Note: Chrome headless is required because mmdc generates SVG files with text embedded
in foreignObject+HTML, which only tools with full HTML/CSS rendering support can
properly convert to PNG. Neither rsvg-convert nor Inkscape can render this correctly.
Chrome is automatically installed with mmdc via puppeteer.
"""

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


@dataclass
class MermaidDiagram:
    """Represents an extracted Mermaid diagram."""

    content: str
    start_pos: int
    end_pos: int
    index: int


@dataclass
class Config:
    """Configuration for PDF generation."""

    source_file: Path = Path("single_agent_tldr.md")
    output_pdf: Path = Path("single_agent_tldr.pdf")
    temp_dir: Path | None = None
    pandoc_engine: str = "xelatex"
    pandoc_args: list[str] = field(
        default_factory=lambda: ["-V", "geometry:margin=1in", "-V", "fontsize=11pt"]
    )
    mermaid_theme: str = "dark"
    keep_temp_files: bool = False
    diagram_png_width: int = 3600  # PNG resolution for diagrams

    def __post_init__(self):
        if self.temp_dir is None:
            self.temp_dir = Path("/tmp") / f"tldr_build_{int(time.time())}"


def setup_logging(debug: bool = False) -> None:
    """Configure logging output."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def check_dependencies() -> bool:
    """
    Verify that required external tools are available.

    Returns:
        True if all dependencies are available, False otherwise.
    """
    missing = []

    # Check for mmdc (Mermaid CLI) - also installs Chrome via puppeteer
    if shutil.which("mmdc") is None:
        missing.append("mmdc - Install with: npm install -g @mermaid-js/mermaid-cli")

    # Check for pandoc
    if shutil.which("pandoc") is None:
        missing.append("pandoc - Install with: apt-get install pandoc")

    # Check for Chrome (installed by puppeteer with mmdc)
    if shutil.which("mmdc") is not None:
        chrome_path = find_chrome_executable()
        if chrome_path is None:
            missing.append("Chrome - Should be installed automatically with mmdc via puppeteer")

    if missing:
        logging.error("Missing required dependencies:")
        for dep in missing:
            logging.error(f"  ✗ {dep}")
        return False

    logging.debug("✓ All dependencies available")
    return True


def extract_mermaid_diagrams(markdown_content: str) -> list[MermaidDiagram]:
    """
    Extract Mermaid diagram code blocks from markdown content.

    Args:
        markdown_content: The full markdown content as a string.

    Returns:
        List of MermaidDiagram objects with content and position information.
    """
    pattern = r"^```mermaid\n(.*?)\n^```"
    matches = list(re.finditer(pattern, markdown_content, re.MULTILINE | re.DOTALL))

    diagrams = []
    for idx, match in enumerate(matches):
        diagrams.append(
            MermaidDiagram(
                content=match.group(1),
                start_pos=match.start(),
                end_pos=match.end(),
                index=idx,
            )
        )

    logging.debug(f"Extracted {len(diagrams)} Mermaid diagrams")
    return diagrams


def setup_puppeteer_config(temp_dir: Path) -> Path | None:
    """
    Create puppeteer configuration file for mmdc.

    Args:
        temp_dir: Directory to store the config file.

    Returns:
        Path to the puppeteer config file, or None if setup failed.
    """
    import json

    # Find chrome executable installed by puppeteer
    chrome_path = Path.home() / ".cache/puppeteer/chrome"
    if chrome_path.exists():
        # Find the latest chrome version
        chrome_versions = sorted(chrome_path.glob("linux-*/chrome-linux64/chrome"))
        if chrome_versions:
            executable_path = str(chrome_versions[-1])
            config = {
                "executablePath": executable_path,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                ],
            }
        else:
            config = {
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                ]
            }
    else:
        config = {
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ]
        }

    try:
        config_file = temp_dir / "puppeteer-config.json"
        config_file.write_text(json.dumps(config, indent=2))
        logging.debug(f"Created puppeteer config: {config_file}")
        return config_file
    except Exception as e:
        logging.error(f"Failed to create puppeteer config: {e}")
        return None


def convert_diagram_to_svg(
    diagram: MermaidDiagram,
    temp_dir: Path,
    puppeteer_config: Path | None = None,
    theme: str = "dark",
    timeout: int = 30,
) -> Path | None:
    """
    Convert a Mermaid diagram to SVG format.

    Args:
        diagram: The MermaidDiagram to convert.
        temp_dir: Directory to store temporary files.
        puppeteer_config: Path to puppeteer configuration file.
        theme: Mermaid theme to use ('dark', 'default', etc.).
        timeout: Timeout in seconds for the conversion process.

    Returns:
        Path to the generated SVG file, or None if conversion failed.
    """
    mmd_file = temp_dir / f"diagram_{diagram.index}.mmd"
    svg_file = temp_dir / f"diagram_{diagram.index}.svg"

    # Write diagram content to .mmd file
    try:
        mmd_file.write_text(diagram.content)
    except Exception as e:
        logging.error(f"Failed to write diagram {diagram.index}: {e}")
        return None

    # Run mmdc to convert to SVG
    cmd = [
        "mmdc",
        "-i",
        str(mmd_file),
        "-o",
        str(svg_file),
        "-b",
        "transparent",
    ]

    # Add puppeteer config if available
    if puppeteer_config and puppeteer_config.exists():
        cmd.extend(["--puppeteerConfigFile", str(puppeteer_config)])

    try:
        logging.debug(f"Converting diagram {diagram.index}...")
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logging.error(f"mmdc failed for diagram {diagram.index}: {result.stderr}")
            return None

        # Validate SVG file
        if not svg_file.exists():
            logging.error(f"SVG file not created for diagram {diagram.index}")
            return None

        if svg_file.stat().st_size < 100:
            logging.error(
                f"SVG file too small for diagram {diagram.index} ({svg_file.stat().st_size} bytes)"
            )
            return None

        logging.debug(f"✓ Converted diagram {diagram.index} ({svg_file.stat().st_size} bytes)")
        return svg_file

    except subprocess.TimeoutExpired:
        logging.error(f"Timeout converting diagram {diagram.index}")
        return None
    except Exception as e:
        logging.error(f"Error converting diagram {diagram.index}: {e}")
        return None


def find_chrome_executable() -> Path | None:
    """Find Chrome executable installed by puppeteer."""
    chrome_path = Path.home() / ".cache/puppeteer/chrome"
    if chrome_path.exists():
        chrome_versions = sorted(chrome_path.glob("linux-*/chrome-linux64/chrome"))
        if chrome_versions:
            return chrome_versions[-1]
    return None


def convert_svg_to_png(
    svg_file: Path,
    png_file: Path,
    width: int = 3600,
    timeout: int = 30,
) -> Path | None:
    """
    Convert SVG to PNG using Chrome headless.

    Since the SVG contains foreignObject with HTML content that neither
    rsvg-convert nor Inkscape can render properly, we use Chrome headless
    which has full HTML/CSS rendering support.

    Args:
        svg_file: Path to source SVG file.
        png_file: Path to output PNG file.
        width: Output width in pixels (default 3600 for high quality).
        timeout: Timeout in seconds.

    Returns:
        Path to PNG file, or None if conversion failed.
    """
    chrome_path = find_chrome_executable()
    if not chrome_path:
        logging.error("Chrome executable not found")
        return None

    # Read SVG content
    try:
        svg_content = svg_file.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read SVG: {e}")
        return None

    # Create HTML wrapper
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            background: transparent;
        }}
        svg {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
{svg_content}
</body>
</html>
"""

    # Write HTML to temporary file
    html_file = svg_file.parent / f"{svg_file.stem}_wrapper.html"
    try:
        html_file.write_text(html_content)
    except Exception as e:
        logging.error(f"Failed to write HTML wrapper: {e}")
        return None

    # Use Chrome headless to screenshot
    cmd = [
        str(chrome_path),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        f"--screenshot={png_file}",
        f"--window-size={width},2000",
        f"file://{html_file.absolute()}",
    ]

    try:
        logging.debug(f"Converting {svg_file.name} to PNG using Chrome headless...")
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )

        # Clean up HTML wrapper
        html_file.unlink(missing_ok=True)

        if result.returncode != 0:
            logging.error(f"Chrome headless failed: {result.stderr}")
            return None

        if not png_file.exists():
            logging.error("PNG file not created")
            return None

        if png_file.stat().st_size < 1000:
            logging.error(f"PNG file too small ({png_file.stat().st_size} bytes)")
            return None

        logging.debug(f"✓ Converted to PNG ({png_file.stat().st_size} bytes)")
        return png_file

    except subprocess.TimeoutExpired:
        html_file.unlink(missing_ok=True)
        logging.error("Timeout converting to PNG")
        return None
    except Exception as e:
        html_file.unlink(missing_ok=True)
        logging.error(f"Error converting to PNG: {e}")
        return None


def convert_diagram_to_image(
    diagram: MermaidDiagram,
    temp_dir: Path,
    puppeteer_config: Path | None = None,
    theme: str = "dark",
    png_width: int = 3600,
    timeout: int = 30,
) -> Path | None:
    """
    Convert Mermaid diagram to PNG via SVG intermediate.

    Pipeline: Mermaid (.mmd) → SVG (mmdc) → PNG (Chrome headless)

    Args:
        diagram: The MermaidDiagram to convert.
        temp_dir: Directory for temporary files.
        puppeteer_config: Puppeteer config for mmdc (SVG generation).
        theme: Mermaid theme.
        png_width: PNG width in pixels.
        timeout: Timeout per conversion step.

    Returns:
        Path to PNG file, or None if failed.
    """
    # Step 1: Generate SVG using mmdc
    svg_file = convert_diagram_to_svg(
        diagram, temp_dir, puppeteer_config, theme, timeout
    )

    if svg_file is None:
        return None

    # Step 2: Convert SVG to PNG using Chrome headless
    png_file = temp_dir / f"diagram_{diagram.index}.png"
    return convert_svg_to_png(svg_file, png_file, width=png_width, timeout=timeout)


def replace_diagrams_with_images(
    markdown_content: str,
    diagrams: list[MermaidDiagram],
    image_files: list[Path],
) -> str:
    """
    Replace Mermaid code blocks with image references.

    Args:
        markdown_content: Original markdown content.
        diagrams: List of extracted diagrams.
        image_files: List of paths to generated image files (PNG or SVG).

    Returns:
        Modified markdown content with diagrams replaced by image references.
    """
    # Sort by position (reverse order) to avoid invalidating positions
    replacements = sorted(
        zip(diagrams, image_files),
        key=lambda x: x[0].start_pos,
        reverse=True,
    )

    modified = markdown_content
    for diagram, image_file in replacements:
        # Create image reference (use just the filename for relative path)
        image_ref = f"![Diagram {diagram.index}]({image_file.name})"

        # Replace the code block
        modified = modified[: diagram.start_pos] + image_ref + modified[diagram.end_pos :]

    logging.debug(f"Replaced {len(diagrams)} code blocks with image references")
    return modified


def run_pandoc(
    input_file: Path,
    output_file: Path,
    config: Config,
) -> bool:
    """
    Run pandoc to generate PDF from markdown.

    Args:
        input_file: Path to the input markdown file.
        output_file: Path to the output PDF file (absolute).
        config: Configuration object with pandoc settings.

    Returns:
        True if successful, False otherwise.
    """
    cmd = [
        "pandoc",
        input_file.name,  # Use relative path since we set cwd
        "-o",
        str(output_file.absolute()),  # Use absolute path for output
        "--pdf-engine",
        config.pandoc_engine,
        *config.pandoc_args,
    ]

    try:
        logging.debug("Running pandoc...")
        result = subprocess.run(
            cmd,
            cwd=str(input_file.parent),  # Set working directory to temp dir
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logging.error(f"Pandoc failed: {result.stderr}")
            return False

        if not output_file.exists():
            logging.error("PDF file not created")
            return False

        logging.debug(f"✓ PDF generated ({output_file.stat().st_size} bytes)")
        return True

    except Exception as e:
        logging.error(f"Error running pandoc: {e}")
        return False


def generate_pdf_with_mermaid(config: Config) -> bool:
    """
    Main pipeline to generate PDF with rendered Mermaid diagrams.

    Args:
        config: Configuration object.

    Returns:
        True if successful, False otherwise.
    """
    temp_dir = config.temp_dir

    try:
        # Create temp directory
        logging.debug(f"Creating temp directory: {temp_dir}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Read source markdown
        if not config.source_file.exists():
            logging.error(f"Source file not found: {config.source_file}")
            return False

        logging.debug(f"Reading source: {config.source_file}")
        markdown_content = config.source_file.read_text()

        # Extract diagrams
        diagrams = extract_mermaid_diagrams(markdown_content)

        if not diagrams:
            logging.warning("No Mermaid diagrams found. Generating PDF with original content...")
            # Copy source to temp and run pandoc normally
            staging_file = temp_dir / config.source_file.name
            staging_file.write_text(markdown_content)
            return run_pandoc(staging_file, config.output_pdf, config)

        logging.info(f"Found {len(diagrams)} Mermaid diagrams")

        # Setup puppeteer configuration
        puppeteer_config = setup_puppeteer_config(temp_dir)

        # Convert each diagram to PNG (via SVG intermediate)
        image_files = []
        for diagram in diagrams:
            logging.info(f"Converting diagram {diagram.index + 1}/{len(diagrams)}...")
            png_file = convert_diagram_to_image(
                diagram, temp_dir, puppeteer_config, config.mermaid_theme, config.diagram_png_width
            )
            if png_file is None:
                logging.error(f"Failed to convert diagram {diagram.index}. Aborting.")
                return False
            image_files.append(png_file)

        logging.info("✓ All diagrams converted")

        # Replace diagrams in markdown
        modified_markdown = replace_diagrams_with_images(markdown_content, diagrams, image_files)

        # Write staging file
        staging_file = temp_dir / config.source_file.name
        staging_file.write_text(modified_markdown)
        logging.debug(f"Wrote staging file: {staging_file}")

        # Run pandoc
        logging.info("Generating PDF...")
        success = run_pandoc(staging_file, config.output_pdf, config)

        if success:
            logging.info(f"✓ PDF generated: {config.output_pdf}")

        return success

    finally:
        # Cleanup temp directory
        if not config.keep_temp_files and temp_dir.exists():
            logging.debug(f"Cleaning up: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif config.keep_temp_files:
            logging.info(f"Temp files retained: {temp_dir}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate PDF from markdown with rendered Mermaid diagrams"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (keep temp files, verbose output)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("single_agent_tldr.md"),
        help="Source markdown file (default: single_agent_tldr.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("single_agent_tldr.pdf"),
        help="Output PDF file (default: single_agent_tldr.pdf)",
    )
    parser.add_argument(
        "--theme",
        default="dark",
        help="Mermaid theme (default: dark)",
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Configure
    config = Config(
        source_file=args.source,
        output_pdf=args.output,
        mermaid_theme=args.theme,
        keep_temp_files=args.debug,
    )

    # Generate PDF
    success = generate_pdf_with_mermaid(config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
