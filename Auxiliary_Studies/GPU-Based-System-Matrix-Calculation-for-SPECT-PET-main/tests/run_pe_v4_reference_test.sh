#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

g++ -std=c++17 -O2 -I"$ROOT_DIR" \
    "$ROOT_DIR/tests/pe_v4_reference_test.cpp" \
    -o "$BUILD_DIR/pe_v4_reference_test"

exec "$BUILD_DIR/pe_v4_reference_test"
