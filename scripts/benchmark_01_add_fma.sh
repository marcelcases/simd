#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Marcel Cases Freixenet
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${2:-results/01_add_fma.csv}"

exec "$root/scripts/benchmark.sh" 01_add_fma \
    --size "${1:-16777216}" \
    --warmups 3 \
    --iterations 10 \
    --samples 9 \
    --output "$output"
