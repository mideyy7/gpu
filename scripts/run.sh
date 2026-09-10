#!/usr/bin/env bash
# Wrapper that makes the Vulkan loader find the Homebrew-installed validation
# layer's shared library at runtime on macOS. On a Linux box with the LunarG
# SDK on the default search path this indirection is unnecessary -- just run
# ./build/profiler directly.
set -euo pipefail

BIN="$(dirname "$0")/../build/profiler"
if [[ ! -x "$BIN" ]]; then
    echo "build first: cmake -S . -B build && cmake --build build" >&2
    exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
    BREW_LIB="$(brew --prefix 2>/dev/null)/lib"
    export DYLD_LIBRARY_PATH="${BREW_LIB}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

exec "$BIN" "$@"
