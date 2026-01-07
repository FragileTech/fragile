.PHONY: build serve clean install

# Build the Jupyter Book
build:
	npx myst build

# Serve the built site on localhost:3000
serve:
	npx myst start

# Serve on custom port
serve-port:
	npx myst start --port 8000

# Build PDF output
pdf:
	npx myst build --pdf

# Clean build artifacts
clean:
	rm -rf _build

# Install dependencies
install:
	npm install
