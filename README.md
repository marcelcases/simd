# SIMD Expression and Portability

> Modern C++ SIMD Programming

## Intro

This project is a compact, benchmark-driven study of explicit SIMD in modern
C++. It compares straightforward scalar algorithms with implementations using
`std::experimental::simd` on x86-64 CPUs.

### TL;DR

- Eight progressively more demanding scalar/SIMD algorithms.
- Explicit load–compute–store loops with safe scalar tails.
- Reductions, masks, FMA, sliding windows, softmax, and convolution.
- Independent correctness checks and isolated executables.
- Measured SIMD gains from negligible to about 10×, depending on the bottleneck.
- Final-binary inspection confirms the generated AVX-512 instructions.

## Project structure

```text
include/simd_examples/   Public algorithm interfaces
src/scalar/              Scalar algorithms
src/simd/                Explicit std::simd algorithms
src/simd_common.h        Shared SIMD type aliases
drivers/                 Input generation, timing, validation, and main()
scripts/                 Benchmark launchers
build/                   Ignored executables
results/                 Ignored benchmark CSV files
```

Algorithms contain computation only. Drivers own input generation, timing,
validation, and output.

### Exercises

| Exercise | Description |
|---|---|
| [01_add](src/scalar/01_add.cpp) · [SIMD](src/simd/01_add.cpp) | Adds two arrays element by element and introduces the basic load–operate–store loop. |
| [02_sum](src/scalar/02_sum.cpp) · [SIMD](src/simd/02_sum.cpp) | Sums an array with SIMD lane accumulators followed by a horizontal reduction. |
| [03_clamp](src/scalar/03_clamp.cpp) · [SIMD](src/simd/03_clamp.cpp) | Replaces values above an upper bound using comparison masks and conditional updates. |
| [04_count](src/scalar/04_count.cpp) · [SIMD](src/simd/04_count.cpp) | Counts values above a threshold with a SIMD mask and `popcount`. |
| [05_softmax](src/scalar/05_softmax.cpp) · [SIMD](src/simd/05_softmax.cpp) | Computes numerically stable softmax using maximum and sum reductions, then vector normalization. |
| [06_fma](src/scalar/06_fma.cpp) · [SIMD](src/simd/06_fma.cpp) | Compares a memory-bound FMA kernel with a compute-bound dot product. |
| [07_filter](src/scalar/07_filter.cpp) · [SIMD](src/simd/07_filter.cpp) | Applies a horizontal image blur with overlapping loads and scalar image borders. |
| [08_conv1d](src/scalar/08_conv1d.cpp) · [SIMD](src/simd/08_conv1d.cpp) | Computes valid mathematical convolution with a reversed kernel and vectorized output blocks. |

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

## License

See [LICENSE](LICENSE).
