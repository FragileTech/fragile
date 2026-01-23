# Single Agent TLDR - Viewing Options

The `single_agent_tldr.md` file contains detailed technical documentation with Mermaid architecture diagrams.

## Viewing Options

### 1. PDF Version (make tldr)
```bash
make tldr
```
- **Output**: `single_agent_tldr.pdf`
- **Diagrams**: Shown as code blocks (not rendered)
- **Math**: Fully rendered LaTeX equations
- **Best for**: Printing, offline reading, formal documentation

### 2. HTML Version (make tldr-html)
```bash
make tldr-html
```
- **Output**: `single_agent_tldr.html`
- **Diagrams**: Fully rendered with Mermaid (dark theme)
- **Math**: Rendered with MathJax
- **Best for**: Interactive viewing in a browser, seeing the architecture diagrams

### 3. Markdown Viewers
View `single_agent_tldr.md` directly in:
- **GitHub**: Automatic Mermaid rendering
- **VS Code**: With Markdown Preview Mermaid Support extension
- **Obsidian, Typora**: Native Mermaid support

## Architecture Diagrams

The document includes 4 detailed Mermaid diagrams showing the TopoEncoder implementation:

1. **CovariantChartRouter**: Chart routing mechanism with Wilson-line transport
2. **Full TopoEncoder**: Complete encoder-decoder architecture
3. **Decoder Detail**: Focused view of the inverse atlas
4. **Experiment Wiring**: Training losses and optional components

**Note**: Due to the complexity of these diagrams (60-180 lines each with extensive styling), they take significant time to render programmatically. The HTML version pre-loads the Mermaid library for browser-based rendering, which is much faster than server-side rendering.

## Troubleshooting

If the HTML diagrams don't render:
1. Ensure you have an internet connection (Mermaid.js loads from CDN)
2. Use a modern browser (Chrome, Firefox, Safari, Edge)
3. Allow JavaScript execution

If the PDF generation fails:
1. Ensure `pandoc` and `xelatex` are installed
2. Run `make tldr` to see specific errors
