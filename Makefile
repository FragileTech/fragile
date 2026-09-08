.PHONY: style check test docs docs-theory docs-lab docs-serve serve tldr tldr-html tldr-debug tldr-fallback check-tldr-deps prompt claude mlflow videogames web robots physics physics-code latex control-native control-setup control-web control-web-build control-lab control-test


CONTROL_PORT ?= 8080
CONTROL_BUILD_JOBS ?= 4
# Let callers select CMake and resolve its executable through the shell.
CMAKE ?= $(shell command -v cmake)
control-native:
	uv run python -m fragile.fractalai.control.build

# Reuse an active SDK or install/activate the pinned SDK in .cache/emsdk/.
control-setup:
	bash fractal-gas-web/tools/with-emsdk.sh emcc --version

control-web:
	bash fractal-gas-web/tools/with-emsdk.sh $(MAKE) control-web-build

# Internal target: invoked with the SDK environment inherited by every recipe.
control-web-build:
	emcmake "$(CMAKE)" -S fractal-gas-web -B fractal-gas-web/build-control-wasm -DFG_CONTROL_ONLY=ON -DCMAKE_BUILD_TYPE=Release
	"$(CMAKE)" --build fractal-gas-web/build-control-wasm --parallel $(CONTROL_BUILD_JOBS)
	emcmake "$(CMAKE)" -S fractal-gas-web -B fractal-gas-web/build-control-threaded -DFG_CONTROL_ONLY=ON -DFG_CONTROL_THREADS=ON -DCMAKE_BUILD_TYPE=Release
	"$(CMAKE)" --build fractal-gas-web/build-control-threaded --parallel $(CONTROL_BUILD_JOBS)
	npm --prefix fractal-gas-web ci --ignore-scripts
	npm --prefix fractal-gas-web run build:lab

control-lab:
	uv run --no-project python fractal-gas-web/tools/serve-control.py --port $(CONTROL_PORT)

control-test: control-native
	ctest --test-dir fractal-gas-web/build-control-native --output-on-failure
	uv run pytest tests/fractalai/test_control_engine.py

OPTIMIZATION_PORT ?= 8081
optimization-native:
	"$(CMAKE)" -S fractal-gas-web -B fractal-gas-web/build-optimization-native -DFG_OPTIMIZATION_ONLY=ON -DCMAKE_BUILD_TYPE=Release
	"$(CMAKE)" --build fractal-gas-web/build-optimization-native --parallel $(CONTROL_BUILD_JOBS)

optimization-web:
	bash fractal-gas-web/tools/with-emsdk.sh $(MAKE) optimization-web-build

optimization-web-build:
	emcmake "$(CMAKE)" -S fractal-gas-web -B fractal-gas-web/build-optimization-wasm -DFG_OPTIMIZATION_ONLY=ON -DCMAKE_BUILD_TYPE=Release
	"$(CMAKE)" --build fractal-gas-web/build-optimization-wasm --parallel $(CONTROL_BUILD_JOBS)
	npm --prefix fractal-gas-web ci --ignore-scripts
	npm --prefix fractal-gas-web run build:optimization

optimization-lab:
	uv run --no-project python fractal-gas-web/tools/serve-control.py --port $(OPTIMIZATION_PORT)

optimization-test: optimization-native optimization-web
	ctest --test-dir fractal-gas-web/build-optimization-native --output-on-failure
	uv run pytest tests/test_benchmarks.py tests/fractalai/test_optimization_engine.py
	npm --prefix fractal-gas-web run test:optimization

.PHONY: optimization-native optimization-web optimization-web-build optimization-lab optimization-test

style:
	uv run ruff check --fix-only --unsafe-fixes .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --diff .

test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYGLET_HEADLESS=1 uv run pytest tests/

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

# Build the two-volume lectures without the independent laboratory guide.
docs-theory:
	$(MAKE) prompt
	rm -rf docs/_build/theory-site
	uv run --no-project --with-requirements docs/requirements.txt jupyter-book build docs/ --config $(abspath docs/_config.yml) --toc $(abspath docs/_toc.yml) --path-output docs/_build/theory-site

# Build the independent control-laboratory guide.
docs-lab:
	rm -rf docs/_build/lab-site
	uv run --no-project --with-requirements docs/requirements.txt jupyter-book build docs/ --config $(abspath docs/_config_lab.yml) --toc $(abspath docs/_toc_lab.yml) --path-output docs/_build/lab-site

# Assemble the portal and both independently searchable documentation sites.
docs: docs-theory docs-lab
	uv run --no-project python docs/assemble_docs.py

# Build all documentation and serve the portal at http://localhost:$(DOCS_PORT)/.
DOCS_PORT ?= 8000
serve: docs
	$(MAKE) docs-serve

# Preview the existing build immediately, without rebuilding the books.
docs-serve:
	uv run --no-project python fractal-gas-web/tools/serve-control.py --docs --port $(DOCS_PORT)

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
