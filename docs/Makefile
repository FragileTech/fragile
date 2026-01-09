.PHONY: build docs serve clean install

# Build the Jupyter Book
build:
	jupyter-book build .

# Build docs without serving
docs:
	jupyter-book build .

# Serve the built site (requires python http.server or similar)
serve: build
	python -m http.server -d _build/html 8000

# Build PDF output
pdf:
	jupyter-book build . --builder pdflatex

# Clean build artifacts
clean:
	rm -rf _build

# Install dependencies
install:
	pip install -r requirements.txt
