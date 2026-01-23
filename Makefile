.PHONY: style check test tldr tldr-html

style:
	uv run ruff check --fix-only --unsafe-fixes .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --diff .

test:
	uv run pytest tests/

tldr:
	@echo "Generating PDF... Note: Mermaid diagrams will show as code."
	@echo "For rendered diagrams, run 'make tldr-html' or view single_agent_tldr.md in GitHub/VS Code"
	pandoc single_agent_tldr.md -o single_agent_tldr.pdf \
		--pdf-engine=xelatex \
		-V geometry:margin=1in \
		-V fontsize=11pt
	@echo "✓ PDF generated: single_agent_tldr.pdf"

tldr-html:
	@chmod +x generate_html.sh
	@bash generate_html.sh
