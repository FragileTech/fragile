#!/bin/bash
# Compile the Navier-Stokes paper to PDF

echo "Compiling Navier-Stokes paper..."
echo "================================"
echo ""

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "ERROR: pandoc is not installed"
    echo "Install with: sudo apt-get install pandoc (Ubuntu/Debian)"
    echo "         or: brew install pandoc (macOS)"
    exit 1
fi

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex is not installed"
    echo "Install with: sudo apt-get install texlive-latex-base texlive-latex-extra (Ubuntu/Debian)"
    echo "         or: brew install --cask mactex (macOS)"
    exit 1
fi

# Compile
echo "Running pandoc..."
pandoc ns_draft.md \
  -H preamble.tex \
  -o ns_draft.pdf \
  --pdf-engine=pdflatex \
  --number-sections \
  --toc

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS! PDF generated: ns_draft.pdf"
    ls -lh ns_draft.pdf
    echo ""
    echo "Note: Citations will show as [@key] format since pandoc-citeproc is not installed."
    echo "To get proper citations, install pandoc-citeproc or upgrade to pandoc 3.x"
else
    echo ""
    echo "✗ COMPILATION FAILED"
    echo "Check the error messages above"
    exit 1
fi
