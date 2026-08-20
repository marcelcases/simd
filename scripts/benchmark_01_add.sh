#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# MN5: load the compiler that provides both g++ and its runtime library.
if type module >/dev/null 2>&1 && [[ -d /apps/GPP/GCC/14.1.0_binutils241 ]]; then
    module load gcc/14.1.0_binutils241
fi

size="${1:-16777216}"
repetitions="${2:-10}"
output="${3:-results/01_add.csv}"

mkdir -p "$(dirname "$output")"
make build/01_add_bench
./build/01_add_bench \
    --size "$size" \
    --repetitions "$repetitions" \
    --output "$output"

echo "Wrote $output"
