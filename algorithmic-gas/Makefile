.PHONY: check rust-test rust-targets web web-test browser-test docs docs-check serve

PORT ?= 8770

rust-test:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	cargo test --workspace --locked

rust-targets:
	cargo check -p algorithmic-gas-benchmarks --features wgpu --locked
	cargo check -p algorithmic-gas-benchmarks --features cuda --locked
	cargo check -p algorithmic-gas-wasm --target wasm32-unknown-unknown --features webgpu --locked

web:
	npm ci --ignore-scripts
	npm run build

web-test: web
	npm test

browser-test:
	LECTURE_BASE_URL=http://127.0.0.1:$(PORT) npm run test:browser

docs-check:
	uv run --no-project --with-requirements docs/requirements.txt pytest -q tests/docs

docs:
	uv run --no-project python docs/build_prompt_downloads.py
	uv run --no-project --with-requirements docs/requirements.txt jupyter-book build docs/ --config $(CURDIR)/docs/_config.yml --toc $(CURDIR)/docs/_toc.yml --path-output docs/_build/site

serve:
	python3 -m http.server --directory web $(PORT)

check: rust-test docs-check
