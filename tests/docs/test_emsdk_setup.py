"""Check SDK discovery without downloading a compiler in unit tests."""

import os
from pathlib import Path
import shlex
import subprocess


WRAPPER = Path(__file__).resolve().parents[2] / "fractal-gas-web/tools/with-emsdk.sh"


def compiler(directory, message):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("emcmake", "emcc"):
        path = directory / name
        path.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(message) + "\n")
        path.chmod(0o755)


def run_wrapper(bin_dir, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith("EMSDK")}
    env["PATH"] = str(bin_dir) + ":/usr/bin:/bin"
    env.update(overrides)
    return subprocess.run(
        ["bash", str(WRAPPER), "emcc", "--version"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_active_toolchain_is_reused_without_sdk_download(tmp_path):
    compiler(tmp_path / "bin", "Active toolchain")
    result = run_wrapper(tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Active toolchain"


def test_explicit_sdk_activates_in_paths_with_spaces_and_overrides_path(tmp_path):
    compiler(tmp_path / "bin", "Wrong toolchain")
    sdk = tmp_path / "custom sdk"
    compiler(sdk / "upstream/emscripten", "Requested SDK")
    (sdk / "upstream/emscripten/emcmake.py").touch()
    (sdk / ".emscripten").touch()
    (sdk / "emsdk_env.sh").write_text(
        "export PATH=" + shlex.quote(str(sdk / "upstream/emscripten")) + ':"$PATH"\n'
    )
    result = run_wrapper(tmp_path / "bin", EMSDK_DIR=str(sdk))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Requested SDK"


def test_invalid_explicit_sdk_fails_with_recovery_instructions(tmp_path):
    compiler(tmp_path / "bin", "Unused toolchain")
    bad = tmp_path / "incomplete"
    bad.mkdir()
    result = run_wrapper(tmp_path / "bin", EMSDK_DIR=str(bad))
    assert result.returncode != 0
    assert "not an SDK" in result.stderr
    assert "EMSDK_DIR" in result.stderr
