"""Build the dependency-free native control library with CMake."""

import subprocess

from fragile.fractalai.control._paths import repository_root


def main() -> None:
    """Compile the library and native tests without emulator dependencies."""
    source = repository_root() / "fractal-gas-web"
    build = source / "build-control-native"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DFG_CONTROL_ONLY=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    print(f"Built control engine in {build / 'control'}")


if __name__ == "__main__":
    main()
