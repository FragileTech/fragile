# Navier-Stokes Global Regularity Paper

## Files

- `ns_draft.md` - Main paper (MyST/Jupyter Book format with pandoc-compatible YAML frontmatter)
- `references.bib` - BibTeX bibliography with 13 references
- `ns_draft_original_backup.md` - Original version before MyST conversion
- `ns_draft_myst.md` - Intermediate MyST version (kept for reference)

## Compilation Instructions

### Quick Start: Use the Compilation Script

```bash
cd docs/source/navier_stokes
./compile.sh
```

This will generate `ns_draft.pdf` with professional formatting.

### Option 1: Compile with Pandoc (Manual)

**IMPORTANT**: The document uses a separate `preamble.tex` file for LaTeX packages and theorem definitions to avoid escaping issues with older pandoc versions.

```bash
cd docs/source/navier_stokes

# Basic compilation (WORKING - TESTED ✓)
pandoc ns_draft.md \
  -H preamble.tex \
  -o ns_draft.pdf \
  --pdf-engine=pdflatex \
  --number-sections \
  --toc

# With citations (requires pandoc 3.x or pandoc-citeproc)
# If you have pandoc 3.x:
pandoc ns_draft.md \
  -H preamble.tex \
  -o ns_draft.pdf \
  --bibliography=references.bib \
  --citeproc \
  --pdf-engine=pdflatex \
  --number-sections \
  --toc

# If you have pandoc 2.x with pandoc-citeproc:
pandoc ns_draft.md \
  -H preamble.tex \
  -o ns_draft.pdf \
  --bibliography=references.bib \
  --filter pandoc-citeproc \
  --pdf-engine=pdflatex \
  --number-sections \
  --toc
```

**Note**: The `-H preamble.tex` flag is required to include the LaTeX preamble with theorem environments and math packages.

### Option 2: Compile with Jupyter Book

```bash
# From repository root
jupyter-book build docs/source/

# The output will be in docs/source/_build/html/
```

### Option 3: Generate LaTeX Source

```bash
# Generate standalone LaTeX file for further editing
pandoc ns_draft.md \
  -o ns_draft.tex \
  --bibliography=references.bib \
  --citeproc \
  --number-sections \
  --standalone

# Then compile with pdflatex
pdflatex ns_draft.tex
bibtex ns_draft
pdflatex ns_draft.tex
pdflatex ns_draft.tex
```

## Document Structure

### Conversion Summary

The document has been professionally formatted with:

- **YAML frontmatter**: Pandoc-compatible metadata, geometry, LaTeX packages
- **105 mathematical environments converted to MyST directives**:
  - 21 Theorems (`:::{prf:theorem}`)
  - 16 Lemmas (`:::{prf:lemma}`)
  - 19 Definitions (`:::{prf:definition}`)
  - 3 Propositions (`:::{prf:proposition}`)
  - 3 Corollaries (`:::{prf:corollary}`)
  - 10 Remarks (`:::{prf:remark}`)
  - 32 Proofs (`:::{prf:proof}`)
  - 1 Assumption (`:::{prf:assumption}`)
- **13 references** converted to BibTeX format with citation keys
- **Section labels** for cross-referencing
- **Automatic section numbering** (§ symbols removed)
- **Abstract** moved to YAML frontmatter
- **Professional typography**: 11pt Palatino, 1-inch margins, two-sided layout

### Content Organization

1. Introduction (§1)
2. Mathematical Preliminaries (§2)
3. The Nonlinear Depletion Inequality (§3)
4. Axial Pressure Defocusing and Singular Integral Control (§4)
5. The Helical Stability Interval (§5)
6. High-Swirl Rigidity and Pseudospectral Shielding (§6)
7. The Partition of the Singular Phase Space (§7)
8. Exclusion of Residual Singular Scenarios (§8)
9. Type II Blow-Up and Mass-Flux Capacity (§9)
10. Virial Rigidity and the Exclusion of Stationary Profiles (§10)
11. The Variational Exclusion of High-Twist Filaments and Global Regularity (§11)
12. The Exhaustive Classification and Structure Theorem (§12)
13. References
14. Appendix A: Proof of Quantitative Stability
15. Appendix B: Proof of Fractal Separation
16. Appendix C: Structural Robustness and Closure

## Customization

### Modify Author Information

Edit the YAML frontmatter in `ns_draft.md`:

```yaml
author:
  - name: "Your Name"
    affiliation: "Your Institution"
    email: "your.email@institution.edu"
```

### Add Co-Authors

```yaml
author:
  - name: "First Author"
    affiliation: "Institution A"
  - name: "Second Author"
    affiliation: "Institution B"
```

### Change Fonts

In YAML frontmatter, modify:

```yaml
fontfamily: mathpazo  # Palatino-style (default)
# Or use:
# fontfamily: libertine  # Linux Libertine
# fontfamily: mathptmx   # Times Roman
# fontfamily: lmodern    # Latin Modern
```

### Adjust Margins

```yaml
geometry:
  - margin=1.5in     # Wider margins
  - top=1in
  - bottom=1.25in
  - left=1.5in
  - right=1.5in
```

## Troubleshooting

### Missing Dependencies

If pandoc compilation fails, ensure you have:

```bash
# Ubuntu/Debian
sudo apt-get install pandoc pandoc-citeproc texlive-full

# macOS
brew install pandoc pandoc-citeproc
brew install --cask mactex

# Or use tinytex (lighter weight)
tlmgr install collection-fontsrecommended
```

### Citation Issues

If citations don't appear:
- Verify `references.bib` is in the same directory
- Use `--citeproc` flag (or `--filter pandoc-citeproc` for older pandoc)
- Check BibTeX keys match those in the document

### LaTeX Errors

If you encounter LaTeX package errors, install missing packages:

```bash
tlmgr install <package-name>
# Or install the full collection
tlmgr install collection-latexextra
```

## Next Steps

1. **Review the compiled PDF** for formatting and correctness
2. **Update author information** in the YAML frontmatter
3. **Add institutional affiliation details** and contact information
4. **Verify all theorem statements** and proofs are correctly formatted
5. **Check cross-references** are working (if using Jupyter Book)
6. **Add any additional references** to `references.bib` as needed

## Notes

- The document uses MyST markdown syntax compatible with both Pandoc and Jupyter Book
- Theorem numbering is automatic (by section)
- Proof QED symbols are automatically added by the proof environment
- All citations use `[@citekey]` syntax with pandoc-citeproc
- Math is rendered with standard LaTeX syntax (`$...$` and `$$...$$`)
