.PHONY: build docs serve clean install venv pdf pdf-deps svg-to-pdf

# Use local venv if it exists, otherwise use system python
VENV_DIR := .venv
PYTHON := $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,python)
JUPYTER_BOOK := $(if $(wildcard $(VENV_DIR)/bin/jupyter-book),$(VENV_DIR)/bin/jupyter-book,jupyter-book)

# SVG images directory
SVG_DIR := svg_images
SVG_FILES := $(wildcard $(SVG_DIR)/*.svg)
PDF_FILES := $(SVG_FILES:.svg=.pdf)

# Build the Jupyter Book
build: venv
	$(JUPYTER_BOOK) build .

# Alias for build
docs: build

# Serve the built site
serve: build
	$(PYTHON) -m http.server -d _build/html 8000

# Convert SVG files to PDF for LaTeX compatibility
svg-to-pdf: $(PDF_FILES)

$(SVG_DIR)/%.pdf: $(SVG_DIR)/%.svg
	@echo "Converting $< to $@"
	@rsvg-convert -f pdf -o $@ $<

# Build PDF output (requires LaTeX and librsvg2-bin)
# Install deps on Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-extra latexmk librsvg2-bin
pdf: venv svg-to-pdf
	$(JUPYTER_BOOK) build . --builder pdflatex

# Install PDF build dependencies (Ubuntu/Debian)
pdf-deps:
	@echo "Installing PDF build dependencies..."
	sudo apt-get update
	sudo apt-get install -y texlive-xetex texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra latexmk librsvg2-bin

# Clean build artifacts
clean:
	rm -rf _build .jupyter_cache
	rm -f $(PDF_FILES)

# Create virtual environment with dependencies
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv $(VENV_DIR); \
		$(VENV_DIR)/bin/pip install --upgrade pip; \
		$(VENV_DIR)/bin/pip install "jupyter-book>=1.0.0,<2.0.0" sphinx-tippy sphinx-proof sphinxcontrib-bibtex sphinxcontrib-mermaid; \
	fi

# Force reinstall dependencies
install:
	rm -rf $(VENV_DIR)
	$(MAKE) venv
