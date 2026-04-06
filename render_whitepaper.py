"""Convert the whitepaper markdown to PDF with math, images, and styled tables."""

import base64
import hashlib
import io
import re
from pathlib import Path

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from weasyprint import HTML

SRC = Path(__file__).parent / "fragile_investor_whitepaper.md"
OUT = Path(__file__).parent / "fragile_investor_whitepaper.pdf"
MATH_CACHE = Path(__file__).parent / ".math_cache"
MATH_CACHE.mkdir(exist_ok=True)

# ── Math rendering via matplotlib ──────────────────────────────────

def render_latex_to_png(latex: str, fontsize: int = 14, dpi: int = 150) -> str:
    """Render LaTeX string to a base64-encoded PNG data URI."""
    cache_key = hashlib.md5(f"{latex}_{fontsize}_{dpi}".encode()).hexdigest()
    cache_path = MATH_CACHE / f"{cache_key}.png"

    if not cache_path.exists():
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0)
        ax.set_axis_off()

        # Wrap in display math
        text = ax.text(
            0, 0, f"${latex}$",
            fontsize=fontsize,
            ha='left', va='bottom',
            transform=ax.transAxes,
        )

        # Render to get bounding box
        fig.canvas.draw()
        bbox = text.get_window_extent(fig.canvas.get_renderer())
        bbox = bbox.expanded(1.15, 1.3)

        fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)
        ax.set_position([0, 0, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, transparent=True,
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        cache_path.write_bytes(buf.getvalue())

    b64 = base64.b64encode(cache_path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def latex_block_to_img(latex: str) -> str:
    """Convert a block LaTeX equation to an <img> tag."""
    try:
        uri = render_latex_to_png(latex, fontsize=10, dpi=150)
        return (
            f'<div style="text-align:center; margin:10px 0;">'
            f'<img src="{uri}" style="max-width:95%; height:auto;" /></div>'
        )
    except Exception as e:
        print(f"  WARN: block eq failed: {str(e)[:80]}")
        # Fallback: show as styled code
        return (
            f'<div style="background:#f8f8f8; border-left:3px solid #4285f4; '
            f'padding:8px 12px; margin:8px 0; font-family:monospace; '
            f'font-size:11px; overflow-x:auto; white-space:pre-wrap;">{latex}</div>'
        )


def latex_inline_to_img(latex: str) -> str:
    """Convert inline LaTeX to an <img> tag."""
    try:
        uri = render_latex_to_png(latex, fontsize=9, dpi=130)
        return f'<img src="{uri}" style="height:1.1em; vertical-align:middle;" />'
    except Exception:
        return f'<span style="font-family:serif; font-style:italic;">{latex}</span>'


# ── Mermaid → HTML architecture diagram ───────────────────────────

def _build_architecture_html() -> str:
    """Build a professional HTML/CSS architecture diagram. Single-row layout, no gaps."""
    b = 'display:inline-block; padding:5px 12px; border-radius:5px; font-size:8.5pt; font-weight:500;'
    blue = f'{b} background:#dbe8fd; border:1px solid #4285f4; color:#1a3a6b;'
    green = f'{b} background:#d4edda; border:1px solid #34a853; color:#1a4d2e;'
    red = f'{b} background:#fce8e6; border:1px solid #ea4335; color:#6b1a1a;'
    da = 'color:#888; font-size:11px; margin:2px 0; line-height:1;'  # down arrow
    lbl = 'font-size:7.5pt; color:#999; letter-spacing:0.5px; margin:0 0 3px;'

    return f'''
<div style="border:1px solid #d0d7de; border-radius:10px; padding:14px 10px 10px;
            margin:16px 0; background:#fafbfc; page-break-inside:avoid;">
<table style="width:100%; border:none; border-collapse:collapse; margin:0;">
<tr>

<!-- LEFT: Frozen Base Model (single cell, no row breaks) -->
<td style="width:30%; vertical-align:top; padding:0 0 0 0; border:none;">
  <div style="background:#e8f0fe; border:2px solid #4285f4; border-radius:8px;
              padding:10px 8px; text-align:center;">
    <p style="font-weight:700; color:#4285f4; margin:0 0 8px; font-size:9.5pt;">
      Frozen Base Model</p>
    <div style="{blue}">Observations (images + language)</div>
    <p style="{da}">&#8595;</p>
    <div style="{blue}">Foundation Model</div>
    <p style="{da}">&#8595;</p>
    <div style="{blue}">Feature vector</div>
    <!-- spacer to push Action down to align with World Model -->
    <div style="height:100px;"></div>
    <div style="{blue}">Action output</div>
  </div>
</td>

<!-- MIDDLE: arrows (single cell, positioned with top padding) -->
<td style="width:6%; vertical-align:top; text-align:center; border:none; padding:0 2px;">
  <!-- spacer to align "features" arrow with Feature vector row -->
  <div style="height:133px;"></div>
  <span style="color:#888; font-size:13px;">&#10132;</span><br>
  <span style="font-size:6.5pt; color:#aaa;">features</span>
  <!-- spacer to align "action" arrow with Action output row -->
  <div style="height:91px;"></div>
  <span style="color:#888; font-size:13px;">&#10132;</span><br>
  <span style="font-size:6.5pt; color:#aaa;">action</span>
</td>

<!-- RIGHT: Geometric Shell (single cell, no row breaks) -->
<td style="width:64%; vertical-align:top; padding:0 0 0 0; border:none;">
  <div style="background:#fef7e0; border:2px solid #f9ab00; border-radius:8px;
              padding:10px 8px; text-align:center;">
    <p style="font-weight:700; color:#c98600; margin:0 0 6px; font-size:9.5pt;">
      Geometric Shell (learnable)</p>

    <!-- Encoder sub-box -->
    <div style="background:#e6f4ea; border:1.5px solid #34a853; border-radius:6px;
                padding:6px 6px 8px; margin-bottom:6px;">
      <p style="{lbl}">GEOMETRIC ENCODER</p>
      <div style="{green}">Project to Poincar&eacute; ball</div>
      <p style="{da}">&#8595;</p>
      <div style="{green}">Distance-based chart routing</div>
      <p style="{da}">&#8595;</p>
      <div style="{green}">Per-chart VQ codebook</div>
      <p style="{da}">&#8595;</p>
      <div style="{green}">Split: (chart, geometry, nuisance, texture)</div>
    </div>

    <!-- World Model sub-box -->
    <div style="background:#fce8e6; border:1.5px solid #ea4335; border-radius:6px;
                padding:6px 6px 8px;">
      <p style="{lbl}">GEOMETRIC WORLD MODEL</p>
      <div style="{red}">Hamiltonian integrator + chart transitions</div>
      <p style="{da}">&#8595;</p>
      <div style="{red}">Predicted next state</div>
      <p style="{da}">&#8595;</p>
      <div style="{red}">Decode to feature space</div>
    </div>
  </div>
</td>

</tr>

<!-- Feedback row -->
<tr>
<td colspan="3" style="text-align:center; padding-top:8px; border:none;">
  <div style="background:#f0f0f0; border:1px solid #ccc; border-radius:14px;
              display:inline-block; padding:4px 16px; font-size:8pt; color:#555;">
    &#8634;&ensp;Decoded state feeds back as next observation to the frozen base model
  </div>
</td>
</tr>
</table>
</div>'''


# ── Read and process markdown ──────────────────────────────────────

md_text = SRC.read_text()

# Replace mermaid code blocks with HTML architecture diagram
md_text = re.sub(r'```mermaid\n.*?```', lambda m: _build_architecture_html(), md_text, flags=re.DOTALL)

# Save block math $$ ... $$ as placeholders
math_blocks = []
def save_block_math(m):
    idx = len(math_blocks)
    math_blocks.append(m.group(1).strip())
    return f'MATHBLOCK{idx}ENDMATH'

md_text = re.sub(r'\$\$(.*?)\$\$', save_block_math, md_text, flags=re.DOTALL)

# For inline math, we use a non-HTML-looking placeholder that markdown won't mangle
# The key insight: the placeholder must look like a regular word so markdown keeps it
# inline within the paragraph, not split into its own <p> block.
inline_maths = []
def save_inline_math(m):
    idx = len(inline_maths)
    inline_maths.append(m.group(1))
    # Use a word-like placeholder that won't be split by markdown
    return f'IQMATH{idx}QI'

md_text = re.sub(r'\$([^\$\n]+?)\$', save_inline_math, md_text)

# Convert markdown to HTML
html_body = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'toc'],
)

# Render block math as images
print(f"Rendering {len(math_blocks)} block equations...")
for i, eq in enumerate(math_blocks):
    html_body = html_body.replace(
        f'MATHBLOCK{i}ENDMATH',
        latex_block_to_img(eq)
    )

# Render inline math as images — these should already be inside <p> tags
print(f"Rendering {len(inline_maths)} inline equations...")
for i, eq in enumerate(inline_maths):
    html_body = html_body.replace(
        f'IQMATH{i}QI',
        latex_inline_to_img(eq)
    )

# Fix image paths to absolute
img_dir = SRC.parent
html_body = html_body.replace(
    'src="vla_latent_space_by_timestep.png"',
    f'src="file://{img_dir}/vla_latent_space_by_timestep.png"'
)
html_body = html_body.replace(
    'src="timestep_per_code.png"',
    f'src="file://{img_dir}/timestep_per_code.png"'
)

CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {
        content: "Fragile — Confidential";
        font-size: 9px;
        color: #999;
    }
    @bottom-right {
        content: counter(page);
        font-size: 9px;
        color: #999;
    }
}
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: 100%;
}
h1 {
    font-size: 22pt;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 4px;
    color: #1a1a1a;
}
h2 {
    font-size: 16pt;
    font-weight: 600;
    margin-top: 28px;
    margin-bottom: 12px;
    color: #1a1a1a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
    page-break-after: avoid;
}
h3 {
    font-size: 13pt;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 8px;
    color: #333;
    page-break-after: avoid;
}
p {
    margin: 8px 0;
    text-align: justify;
}
strong {
    font-weight: 600;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th {
    background: #f0f4f8;
    font-weight: 600;
    text-align: left;
    padding: 8px 10px;
    border: 1px solid #d0d7de;
}
td {
    padding: 6px 10px;
    border: 1px solid #d0d7de;
    vertical-align: top;
}
tr:nth-child(even) {
    background: #f9fbfc;
}
ul, ol {
    margin: 8px 0;
    padding-left: 24px;
}
li {
    margin: 4px 0;
}
code {
    background: #f0f4f8;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
    font-family: 'SF Mono', 'Consolas', monospace;
}
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 24px 0;
}
img {
    max-width: 100%;
    height: auto;
}
img[src^="file://"], img[alt] {
    display: block;
    margin: 16px auto;
}
img[src^="data:"] {
    display: inline;
    margin: 0;
    vertical-align: middle;
}
p img[src^="data:"] {
    display: inline !important;
    margin: 0 !important;
}
div img[src^="data:"] {
    display: block;
    margin: 4px auto;
    max-width: 95%;
}
blockquote {
    border-left: 3px solid #4285f4;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f8f9fa;
    color: #555;
}
"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

# Write HTML for debugging
(SRC.parent / "fragile_investor_whitepaper.html").write_text(full_html)

# Convert to PDF
HTML(string=full_html, base_url=str(img_dir)).write_pdf(str(OUT))
print(f"PDF written to {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
