#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Marcel Cases Freixenet
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${3:-results/01_add.csv}"

exec "$root/scripts/benchmark.sh" 01_add \
    --size "${1:-16777216}" \
    --repetitions "${2:-10}" \
    --output "$output"
