#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Marcel Cases Freixenet
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# MN5: load the compiler and its runtime library when available.
if type module >/dev/null 2>&1 && [[ -d /apps/GPP/GCC/14.1.0_binutils241 ]]; then
    module load gcc/14.1.0_binutils241 >&2
fi

completed_exercises=(
    01_add_fma
    02_reduction_dot
    03_clamp
    04_count
    05_softmax
)

selection="${1:-all}"
if (( $# > 0 )); then
    shift
fi
case "$selection" in
    all)
        exercises=("${completed_exercises[@]}")
        ;;
    01_add_fma|02_reduction_dot|03_clamp|04_count|05_softmax)
        exercises=("$selection")
        ;;
    *)
        echo "Unknown exercise: $selection" >&2
        echo "Usage: benchmark.sh [all|exercise] [benchmark options]" >&2
        exit 1
        ;;
esac

output=""
arguments=()
while (( $# > 0 )); do
    if [[ "$1" == "--output" ]]; then
        if (( $# < 2 )); then
            echo "--output requires a file" >&2
            exit 1
        fi
        output="$2"
        shift 2
    else
        arguments+=("$1")
        shift
    fi
done

has_option() {
    local expected="$1"
    local argument
    for argument in "${arguments[@]}"; do
        [[ "$argument" == "$expected" ]] && return 0
    done
    return 1
}

has_option --warmups || arguments+=(--warmups 3)
has_option --iterations || arguments+=(--iterations 10)
has_option --samples || arguments+=(--samples 9)

if [[ "$selection" == "all" && -z "$output" ]]; then
    output="results/benchmark.csv"
fi
if [[ -n "$output" ]]; then
    mkdir -p "$(dirname "$output")"
fi

targets=()
for exercise in "${exercises[@]}"; do
    targets+=("build/${exercise}_scalar" "build/${exercise}_simd")
done
make --no-print-directory -s "${targets[@]}"

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/simd-benchmark.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT
combined_output="$temporary_directory/benchmark.csv"

first=true
for exercise in "${exercises[@]}"; do
    scalar_output="$temporary_directory/${exercise}-scalar.csv"
    simd_output="$temporary_directory/${exercise}-simd.csv"

    "./build/${exercise}_scalar" "${arguments[@]}" --output "$scalar_output"
    "./build/${exercise}_simd" "${arguments[@]}" --output "$simd_output"

    if $first; then
        head -n 1 "$scalar_output" > "$combined_output"
        first=false
    fi
    tail -n +2 "$scalar_output" >> "$combined_output"
    tail -n +2 "$simd_output" >> "$combined_output"
done

if [[ -n "$output" ]]; then
    cp "$combined_output" "$output"
    echo "$output"
else
    cat "$combined_output"
fi
