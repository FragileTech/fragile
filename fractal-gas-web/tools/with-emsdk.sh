#!/usr/bin/env bash
# Activate an existing SDK, or install the CI version in the repository cache.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 command [arguments...]" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
sdk_version=${EMSDK_VERSION:-6.0.8}

# An explicit SDK directory takes precedence over the current shell. Otherwise
# preserve an activated/system toolchain (including the one provided by CI).
if [[ -z ${EMSDK_DIR:-} ]] && command -v emcmake >/dev/null && command -v emcc >/dev/null; then
    exec "$@"
fi

sdk_dir=${EMSDK_DIR:-${EMSDK:-}}
if [[ -z $sdk_dir ]]; then
    for candidate in "$repo_root/.cache/emsdk/$sdk_version" "$HOME/emsdk" "$HOME/.emsdk" /opt/emsdk; do
        if [[ -f $candidate/emsdk_env.sh ]]; then
            sdk_dir=$candidate
            break
        fi
    done
fi
sdk_dir=${sdk_dir:-$repo_root/.cache/emsdk/$sdk_version}

if [[ ! -f $sdk_dir/emsdk_env.sh ]]; then
    if [[ -e $sdk_dir ]]; then
        echo "EMSDK directory exists but is not an SDK: $sdk_dir" >&2
        echo "Set EMSDK_DIR to an SDK directory, or remove the incomplete cache and retry." >&2
        exit 1
    fi
    command -v git >/dev/null || { echo "Install git to download Emscripten." >&2; exit 1; }
    echo "Installing Emscripten $sdk_version into $sdk_dir (first build only)." >&2
    mkdir -p -- "$(dirname -- "$sdk_dir")"
    git clone --depth 1 --branch "$sdk_version" https://github.com/emscripten-core/emsdk.git "$sdk_dir"
fi

if [[ ! -f $sdk_dir/.emscripten || ! -f $sdk_dir/upstream/emscripten/emcmake.py ]]; then
    "$sdk_dir/emsdk" install "$sdk_version"
    "$sdk_dir/emsdk" activate "$sdk_version"
fi

# SDK scripts are maintained upstream and need not support nounset.
export EMSDK_QUIET=${EMSDK_QUIET:-1}
set +u
source "$sdk_dir/emsdk_env.sh"
set -u
command -v emcmake >/dev/null && command -v emcc >/dev/null || {
    echo "SDK activation failed. Check EMSDK_DIR=$sdk_dir." >&2
    exit 1
}
exec "$@"
