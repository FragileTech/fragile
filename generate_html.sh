#!/bin/bash
# Generate HTML version with rendered Mermaid diagrams

echo "Generating HTML version with rendered Mermaid diagrams..."

# Create temporary CSS file
cat > /tmp/tldr_style.css << 'EOF'
body {
    max-width: 900px;
    margin: 40px auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    padding: 0 20px;
    background: #0d1117;
    color: #e6edf3;
}
pre {
    background: #161b22;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    border: 1px solid #30363d;
}
code {
    background: #161b22;
    padding: 2px 6px;
    border-radius: 3px;
    color: #ff7b72;
}
h1, h2, h3 {
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
}
EOF

# Create temporary header file
cat > /tmp/tldr_header.html << 'EOF'
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'dark' });
</script>
EOF

pandoc single_agent_tldr.md -o single_agent_tldr.html \
    --standalone \
    --metadata title="Single Agent Architecture as Field Theory: Technical TLDR" \
    --css=/tmp/tldr_style.css \
    --include-in-header=/tmp/tldr_header.html \
    --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

if [ $? -eq 0 ]; then
    echo "✓ HTML generated: single_agent_tldr.html"
    echo "  Open this file in a browser to view rendered Mermaid diagrams"
    # Cleanup
    rm -f /tmp/tldr_style.css /tmp/tldr_header.html
else
    echo "✗ Failed to generate HTML"
    rm -f /tmp/tldr_style.css /tmp/tldr_header.html
    exit 1
fi
