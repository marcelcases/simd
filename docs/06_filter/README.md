# 6. Horizontal Image Blur

[Scalar source](../../src/scalar/06_filter.cpp) | [SIMD source](../../src/simd/06_filter.cpp)

Smooths each row of a grayscale image independently. The horizontal window is
**hardcoded to three pixels**: the left neighbour, the current pixel, and the
right neighbour. Its size is not configurable.

```text
output[row, column] = (left + center + right) / 3
```

The first and last pixels average the two available pixels instead. There is
no vertical averaging.

## Used in

- Simple image smoothing.
- Learning stencil operations with overlapping input windows.

## Kernel workflow

| Step | Scalar | SIMD |
|---|---|---|
| Select a row | Point to its input and output pixels | Same |
| Left edge | Average the first two pixels | Same scalar operation |
| Interior | Average three pixels for one output | Load left, center, and right vectors; average lane-wise |
| Finish | Average the last two pixels | Process the scalar tail, then the right edge |

The image is logically a 2D matrix stored in a flat array of floats, row by row.
The driver owns the arrays; the kernel receives pointers and dimensions:

```cpp
void blur_horizontal(const float* input, float* output,
                     int width, int height) noexcept;
```

`source = input + row * width` selects the start of a row. Then
`source[column]` is equivalent to `input[row * width + column]`.
Input and output must be valid, non-overlapping buffers, with `width >= 2` and
`height >= 1`. The driver validates the dimensions.

## SIMD notes

- Each lane produces one output pixel; SIMD width does not change the
  three-pixel window.
- Three shifted loads align each lane's left, center, and right neighbours.
- `scale` contains `1.f / 3.f` in every lane.
- `element_aligned` permits the shifted addresses without requiring full
  vector alignment; the experimental SIMD API requires this argument.
- The vector loop stops before a right-neighbour load would cross the row.
  A scalar tail handles remaining interior pixels.
- Neighbouring outputs reuse input pixels through caches; three loads in the
  source do not imply three independent main-memory reads per output.

## Benchmark

Use a 1920 × 1080 image and the common median methodology:
`9 × (3 untimed warm-ups + 10 timed calls)`. Allocation, initialization, and
correctness validation are outside timing. Input is read-only, so repeated
calls need no reset. This is warmed execution, not a cold-cache experiment.

<details>
<summary>Benchmark command</summary>

```bash
scripts/benchmark.sh 06_filter --width 1920 --height 1080 \
    --output results/06_filter.csv
```

</details>

### Results

Median milliseconds per image; speedup is scalar median divided by SIMD median.

| Target | Scalar ms | SIMD ms | Speedup |
|---|---:|---:|---:|
| MN5 x86-64, GCC 14.1 | 1.47928 | 0.698366 | 2.12× |
| MN5 x86-64, `icpx` 2025.2 | 0.839835 | 0.703099 | 1.19× |
| Banana Pi F3, GCC 16.2, `VL=4` | 23.8633 | 14.6927 | 1.62× |
| Banana Pi F3, GCC 16.2, `VL=8` | 23.8633 | 11.4468 | 2.08× |

RISC-V execution used `bananaf3-1`, job `327015`, with
`-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl`. Scalar auto-vectorization was
disabled. Fixed-size widths do not change the hardware VLEN of 256 bits.
Both SIMD algorithm functions contain RVV instructions in the final binaries.
All correctness checks passed; maximum absolute difference was `5.96e-8`.
Generated CSV files remain under ignored `results/`.

MN5 execution used one core pinned with `srun --cpu-bind=cores` on exclusive
node `gs24r3b62`, job `46835591`. Builds used `-O3 -march=native` and AVX-512;
scalar auto-vectorization was disabled. Final SIMD binaries contain packed
floating-point vector arithmetic. The image may fit in cache, so these results
are not a measurement of sustained main-memory bandwidth.

All target timings are consolidated in `results/benchmark-filter.csv`, including
image dimensions, compiler, implementation, and timing parameters.
