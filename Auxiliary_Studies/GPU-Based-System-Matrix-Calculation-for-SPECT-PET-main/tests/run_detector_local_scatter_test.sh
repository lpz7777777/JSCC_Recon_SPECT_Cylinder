#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

g++ -std=c++17 -O2 -I"$ROOT_DIR" \
    "$ROOT_DIR/tests/detector_local_scatter_test.cpp" \
    -o "$BUILD_DIR/detector_local_scatter_test"

exec "$BUILD_DIR/detector_local_scatter_test"
