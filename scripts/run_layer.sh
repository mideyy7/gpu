#!/usr/bin/env bash
# Run the profiler binary (or any Vulkan app) with VK_LAYER_gcp_profiler
# active. The layer's aggregated report prints to stderr on device destroy.
#
#   scripts/run_layer.sh --report --particles 2048 --steps 64 --batch 1
#   scripts/run_layer.sh --report --particles 2048 --steps 64 --batch 32
#
# To profile a different app, set GCP_APP to its path:
#   GCP_APP=/path/to/other_vulkan_app scripts/run_layer.sh <its args...>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
APP="${GCP_APP:-$BUILD/profiler}"

if [[ ! -f "$BUILD/VkLayer_gcp_profiler.json" ]]; then
    echo "build first: cmake -S . -B build && cmake --build build" >&2
    exit 1
fi

# VK_ADD_LAYER_PATH is additive -- it keeps the SDK's default search path
# (so the validation layer is still found) and just adds our build dir.
export VK_ADD_LAYER_PATH="$BUILD${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH}"
export VK_INSTANCE_LAYERS="VK_LAYER_gcp_profiler${VK_INSTANCE_LAYERS:+:$VK_INSTANCE_LAYERS}"

if [[ "$(uname)" == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH="$(brew --prefix 2>/dev/null)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

exec "$APP" "$@"
