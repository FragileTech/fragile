.PHONY: style check test docs serve tldr tldr-html tldr-debug tldr-fallback check-tldr-deps prompt claude mlflow videogames web robots physics physics-code latex control-native control-web control-lab control-test

CONTROL_PORT ?= 8080
# Let callers select CMake and resolve its executable through the shell.
CMAKE ?= $(shell command -v cmake)
control-native:
	uv run python -m fragile.fractalai.control.build

# Activate the Emscripten SDK first. Both browser variants use the same sources.
control-web:
	emcmake "$(CMAKE)" -S fractal-gas-web -B fractal-gas-web/build-control-wasm -DFG_CONTROL_ONLY=ON -DCMAKE_BUILD_TYPE=Release
	"$(CMAKE)" --build fractal-gas-web/build-control-wasm --parallel 4
	emcmake "$(CMAKE)" -S fractal-gas-web -B fractal-gas-web/build-control-threaded -DFG_CONTROL_ONLY=ON -DFG_CONTROL_THREADS=ON -DCMAKE_BUILD_TYPE=Release
	"$(CMAKE)" --build fractal-gas-web/build-control-threaded --parallel 4
	npm --prefix fractal-gas-web ci --ignore-scripts
	npm --prefix fractal-gas-web run build:lab

control-lab:
	uv run python fractal-gas-web/tools/serve-control.py --port $(CONTROL_PORT)

control-test: control-native
	ctest --test-dir fractal-gas-web/build-control-native --output-on-failure
	uv run pytest tests/fractalai/test_control_engine.py

style:
	uv run ruff check --fix-only --unsafe-fixes .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --diff .

test:
	PYGLET_HEADLESS=1 uv run pytest tests/

tldr:
	@echo "Generating PDF with rendered Mermaid diagrams..."
	@python3 generate_pdf_with_mermaid.py
	@echo "✓ PDF generated: single_agent_tldr.pdf"

tldr-debug:
	@echo "Generating PDF (debug mode - keeping temp files)..."
	@python3 generate_pdf_with_mermaid.py --debug
	@echo "✓ PDF generated: single_agent_tldr.pdf"
	@echo "  Temp files retained in /tmp/tldr_build_*/"

tldr-fallback:
	@echo "Generating PDF without Mermaid rendering (fallback)..."
	@pandoc single_agent_tldr.md -o single_agent_tldr.pdf \
		--pdf-engine=xelatex \
		-V geometry:margin=1in \
		-V fontsize=11pt
	@echo "✓ PDF generated: single_agent_tldr.pdf"

check-tldr-deps:
	@echo "Checking dependencies for PDF generation..."
	@which mmdc > /dev/null 2>&1 || (echo "✗ mmdc not found. Install: npm install -g @mermaid-js/mermaid-cli" && exit 1)
	@which pandoc > /dev/null 2>&1 || (echo "✗ pandoc not found. Install: apt-get install pandoc" && exit 1)
	@echo "✓ All dependencies available"

tldr-html:
	@chmod +x generate_html.sh
	@bash generate_html.sh

prompt:
	@echo "Collecting prf directives into prompts/..."
	@python3 docs/collect_prf_directives.py --include-proofs --include-file-headings --out-dir prompts
	@echo "Preparing prompt downloads for docs..."
	@python3 docs/build_prompt_downloads.py
	@echo "✓ Prompts generated in prompts/"

# Build the Jupyter Book from the repository root.
docs:
	$(MAKE) prompt
	uv run --with-requirements docs/requirements.txt jupyter-book build docs/

# Build the Jupyter Book and serve it at http://localhost:$(DOCS_PORT)/.
DOCS_PORT ?= 8000
serve: docs
	uv run --with-requirements docs/requirements.txt python3 -m http.server $(DOCS_PORT) --directory docs/_build/html

mlflow:
	uv run mlflow server --host 127.0.0.1 --port 5000

videogames:
	uv run fragile videogames $(ARGS)

# Serve the wasm fractal-gas swarm demo with the COOP/COEP headers wasm
# threads need. Open http://localhost:$(WEB_PORT)/web/ once it is up.
WEB_PORT ?= 8000
web:
	@echo "Serving fractal gas web demo at http://localhost:$(WEB_PORT)/web/"
	cd fractal-gas-web && python3 serve.py $(WEB_PORT)

robots:
	uv run fragile robots $(ARGS)

physics:
	uv run fragile physics $(ARGS)

physics-code:
	@echo "# Physics Code" > physics_code.md
	@echo "" >> physics_code.md
	@find src/fragile/physics -name '*.py' | sort | while read f; do \
		echo "## $$f" >> physics_code.md; \
		echo "" >> physics_code.md; \
		echo '```python' >> physics_code.md; \
		cat "$$f" >> physics_code.md; \
		echo "" >> physics_code.md; \
		echo '```' >> physics_code.md; \
		echo "" >> physics_code.md; \
	done
	@echo "✓ physics_code.md generated"

latex:
	@cd docs/source/4_ymmg && latexmk -pdf -interaction=nonstopmode *.tex

claude:
	CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude --dangerously-skip-permissions
