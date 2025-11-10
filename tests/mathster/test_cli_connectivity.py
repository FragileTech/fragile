from pathlib import Path

from click.testing import CliRunner

from mathster.cli import cli


def test_connectivity_cli_accepts_md_path(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_parse(path: Path) -> None:
        calls["parse_path"] = path

    def fake_generate(document_id: str, preprocess_dir: Path | None) -> str:
        calls["document_id"] = document_id
        calls["preprocess_dir"] = preprocess_dir
        return "REPORT CONTENT"

    monkeypatch.setattr(
        "mathster.reports.document_connectivity_report.generate_document_connectivity_report",
        fake_generate,
    )
    monkeypatch.setattr("mathster.cli._run_parsing_stage", fake_parse)
    registry_calls: dict[str, object] = {}

    def fake_process(doc_path: Path, force: bool, verbose: bool) -> None:
        registry_calls["path"] = doc_path
        registry_calls["force"] = force
        registry_calls["verbose"] = verbose

    monkeypatch.setattr(
        "mathster.registry.directives_stage.process_document",
        fake_process,
    )

    markdown = tmp_path / "docs" / "source" / "07_mean_field.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["connectivity", str(markdown)])

    assert result.exit_code == 0
    assert "REPORT CONTENT" in result.output
    assert calls["document_id"] == "07_mean_field"
    assert calls["preprocess_dir"] is None
    assert calls["parse_path"] == markdown
    expected_workspace = markdown.parent / markdown.stem
    assert registry_calls["path"] == expected_workspace
    assert registry_calls["force"] is True
    assert registry_calls["verbose"] is False


def test_connectivity_cli_preprocess_override(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_parse(path: Path) -> None:
        calls["parse_path"] = path

    def fake_generate(document_id: str, preprocess_dir: Path | None) -> str:
        calls["document_id"] = document_id
        calls["preprocess_dir"] = preprocess_dir
        return "ANOTHER REPORT"

    monkeypatch.setattr(
        "mathster.reports.document_connectivity_report.generate_document_connectivity_report",
        fake_generate,
    )
    monkeypatch.setattr("mathster.cli._run_parsing_stage", fake_parse)
    registry_calls: dict[str, object] = {}

    def fake_process(doc_path: Path, force: bool, verbose: bool) -> None:
        registry_calls["path"] = doc_path
        registry_calls["force"] = force
        registry_calls["verbose"] = verbose

    monkeypatch.setattr(
        "mathster.registry.directives_stage.process_document",
        fake_process,
    )

    doc_file = tmp_path / "docs" / "source" / "01_fragile_gas_framework.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("", encoding="utf-8")

    def fake_locate(value: str | Path) -> Path | None:
        assert value == "01_fragile_gas_framework"
        return doc_file

    monkeypatch.setattr("mathster.cli._locate_markdown_file", fake_locate)

    preprocess_dir = tmp_path / "registry" / "preprocess"
    preprocess_dir.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "connectivity",
            "01_fragile_gas_framework",
            "--preprocess-dir",
            str(preprocess_dir),
        ],
    )

    assert result.exit_code == 0
    assert "ANOTHER REPORT" in result.output
    assert calls["document_id"] == "01_fragile_gas_framework"
    assert calls["preprocess_dir"] == preprocess_dir
    assert calls["parse_path"] == doc_file
    expected_workspace = doc_file.parent / doc_file.stem
    assert registry_calls["path"] == expected_workspace
    assert registry_calls["force"] is True
    assert registry_calls["verbose"] is False
