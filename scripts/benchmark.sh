#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: benchmark.sh <exercise> [benchmark options]" >&2
    exit 1
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# MN5: load the compiler and its runtime library when available.
if type module >/dev/null 2>&1 && [[ -d /apps/GPP/GCC/14.1.0_binutils241 ]]; then
    module load gcc/14.1.0_binutils241
fi

exercise="$1"
shift
case "$exercise" in
    01_add|02_sum|03_clamp|04_count|05_softmax|06_fma|07_filter|08_conv1d) ;;
    *)
        echo "Unknown exercise: $exercise" >&2
        exit 1
        ;;
esac

for ((i = 1; i <= $#; ++i)); do
    if [[ "${!i}" == "--output" ]] && (( i < $# )); then
        next=$((i + 1))
        mkdir -p "$(dirname "${!next}")"
    fi
done

make "build/${exercise}_bench"
"./build/${exercise}_bench" "$@"
