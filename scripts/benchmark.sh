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
    module load gcc/14.1.0_binutils241 >&2
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

output=""
for ((i = 1; i <= $#; ++i)); do
    if [[ "${!i}" == "--output" ]]; then
        if (( i == $# )); then
            echo "--output requires a file" >&2
            exit 1
        fi
        next=$((i + 1))
        output="${!next}"
    fi
done

if [[ -n "$output" ]]; then
    mkdir -p "$(dirname "$output")"
fi

make --no-print-directory -s \
    "build/${exercise}_scalar" "build/${exercise}_simd"

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/simd-benchmark.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT

scalar_output="$temporary_directory/scalar.csv"
simd_output="$temporary_directory/simd.csv"

"./build/${exercise}_scalar" "$@" --output "$scalar_output"
"./build/${exercise}_simd" "$@" --output "$simd_output"

write_combined_output() {
    head -n 1 "$scalar_output"
    tail -n +2 "$scalar_output"
    tail -n +2 "$simd_output"
}

if [[ -n "$output" ]]; then
    write_combined_output > "$output"
else
    write_combined_output
fi
