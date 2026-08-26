# SIMD Expression and Portability

This project is a compact, benchmark-driven study of explicit SIMD in modern C++.

## TL;DR

- Eight progressively more demanding scalar/SIMD algorithms.
- Explicit load–compute–store loops with safe scalar tails.
- Reductions, masks, FMA, sliding windows, softmax, and convolution.
- Independent correctness checks and isolated executables.
- Measured SIMD gains from negligible to about 10×, depending on the bottleneck.
- Final-binary inspection confirms the generated AVX-512 instructions.

## Project structure

| Example | Description |
|---|---|
| [1. Element-wise array addition](docs/01_add/README.md) | Adds arrays element by element; introduces SIMD loops. |
| [2. Sum reduction](docs/02_sum/README.md) | Sums lanes, then horizontally reduces the accumulator. |
| [3. Upper-bound clamp](docs/03_clamp/README.md) | Clamps values using comparisons and conditional masks. |
| [4. Count above threshold](docs/04_count/README.md) | Counts threshold matches with masks and popcount. |
| [5. Numerically stable softmax](docs/05_softmax/README.md) | Computes stable softmax with vector reductions. |
| [6. FMA and dot product](docs/06_fma/README.md) | Contrasts memory-bound FMA with compute-bound dot product. |
| [7. Horizontal image blur](docs/07_filter/README.md) | Blurs rows using overlapping loads and scalar borders. |
| [8. 1D mathematical convolution](docs/08_conv1d/README.md) | Convolves with reversed kernels and vectorized outputs. |

## Key results and performance

Results are from an Intel Xeon Platinum 8480+ on one exclusive MN5 node and one
pinned CPU core. Scalar targets disable compiler vectorization; SIMD targets
use explicit `std::experimental::simd` with normal optimization.

The 1D kernels used 16,777,216 elements, softmax used 4,194,304 elements, and
blur used an 8192 × 4096 image. Values are median speedups from three trials;
speedup means scalar time divided by SIMD time.

| Kernel | GCC | `icpx` |
|---|---:|---:|
| Element-wise addition | 1.17× | 1.54× |
| Sum reduction | 5.14× | 5.15× |
| Upper-bound clamp | 7.85× | 10.29× |
| Count above threshold | 4.91× | 4.19× |
| Softmax | 1.64× | 4.43× |
| Memory-bound FMA | 1.02× | 0.99× |
| Dot product | 1.62× | 4.03× |
| Horizontal blur | 1.32× | 0.94× |
| 1D convolution | 2.65× | 3.00× |

Reductions, masks, dot products, and convolution benefit most. Addition, memory
FMA, and blur are limited mainly by memory traffic.

The normal `icpx` softmax build also auto-vectorizes the scalar exponential
loop through Intel SVML. With compiler auto-vectorization disabled, its softmax
speedup was approximately 1.44×. The softmax benchmark uses a smaller input
because the current float normalization accumulation loses validation accuracy
at much larger sizes.

## Build

### Environment

The current build targets x86-64 Linux on MareNostrum 5:

- Intel Xeon Platinum 8480+ with AVX-512;
- GCC 14.1.0 or Intel `icpx` 2025.2;
- C++2b, `-O3`, and `-march=native`;
- `native_simd<float>::size()` is typically 16 on this CPU.

Use a clean module environment when switching compilers. Both builds produce
the same executable names.

### GCC

```bash
module purge
module load gcc/14.1.0_binutils241
make clean
make drivers

./build/01_add_scalar --size 16777216 --repetitions 10
./build/01_add_simd --size 16777216 --repetitions 10
```

### Intel `icpx`

```bash
module purge
module load intel/2025.2
make clean
make CXX=icpx drivers

./build/01_add_scalar --size 16777216 --repetitions 10
./build/01_add_simd --size 16777216 --repetitions 10
```

Build subsets or run all default drivers with:

```bash
make scalar
make simd
make run
```

A driver can write a combined scalar/SIMD CSV:

```bash
scripts/benchmark.sh 02_sum \
    --size 16777216 \
    --repetitions 10 \
    --output results/02_sum.csv
```

### Inspect generated instructions

Inspect the final executable after linking:

```bash
objdump -d -C build/01_add_simd | grep -E 'vaddps|vmov'
objdump -d -C build/03_clamp_simd | grep -E 'vcmpps|vblend|vmov'
objdump -d -C build/06_fma_simd | grep -E 'vfmadd|vmov'
```

## Conclusion

### What This Project Demonstrates

- SIMD processes several values per instruction, not the whole input at once.
- Explicit SIMD is built from vector loads, lane-wise operations, stores, and a
  scalar tail.
- Reductions require partial lane accumulators and horizontal reduction.
- Compiler choice and generated instructions affect measured performance.
- Memory bandwidth can dominate even when SIMD computation is available.
- Correctness validation, benchmarking, and binary inspection must be done
  together.
