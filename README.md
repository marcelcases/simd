# Learning SIMD on x86 with C++

A small, runnable set of scalar and explicit SIMD algorithms for learning how
SIMD works on x86-64 CPUs.

The project uses the GCC and Intel compiler implementations of
`std::experimental::simd`. Every exercise has a scalar implementation, an
explicit SIMD implementation, an isolated executable, and a driver that
generates input, measures time, and checks correctness.

## TL;DR

```bash
# GCC on MareNostrum 5
module purge
module load gcc/14.1.0_binutils241
make clean && make
make run

# Intel compiler on MareNostrum 5
module purge
module load intel/2025.2
make clean && make CXX=icpx
make run CXX=icpx
```

## Scope

- CPU: Intel Xeon Platinum 8480+ with AVX-512;
- GCC: 14.1.0;
- Intel compiler: `icpx` 2025.2;
- build: C++2b, `-O3`, `-march=native`.

`native_simd<float>::size()` is typically 16 on this CPU: one SIMD vector
contains 16 single-precision values.

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

## Learning path

| # | Kernel | SIMD concept | Source |
|---|---|---|---|
| 1 | Element-wise addition | Load, operate, store, tail | [scalar](src/scalar/01_add.cpp) · [SIMD](src/simd/01_add.cpp) |
| 2 | Sum reduction | SIMD accumulator and `reduce` | [scalar](src/scalar/02_sum.cpp) · [SIMD](src/simd/02_sum.cpp) |
| 3 | Upper-bound clamp | Comparison masks and `where` | [scalar](src/scalar/03_clamp.cpp) · [SIMD](src/simd/03_clamp.cpp) |
| 4 | Count above threshold | Masks and `popcount` | [scalar](src/scalar/04_count.cpp) · [SIMD](src/simd/04_count.cpp) |
| 5 | Stable softmax | `hmax`, `reduce`, vector normalization | [scalar](src/scalar/05_softmax.cpp) · [SIMD](src/simd/05_softmax.cpp) |
| 6 | FMA and dot product | Memory-bound versus compute-bound work | [scalar](src/scalar/06_fma.cpp) · [SIMD](src/simd/06_fma.cpp) |
| 7 | Horizontal image blur | Sliding windows and scalar borders | [scalar](src/scalar/07_filter.cpp) · [SIMD](src/simd/07_filter.cpp) |
| 8 | 1D convolution | Reversed kernel and vectorized outputs | [scalar](src/scalar/08_conv1d.cpp) · [SIMD](src/simd/08_conv1d.cpp) |

## The SIMD loop

A typical implementation processes complete vector-sized blocks and then a
small scalar tail:

```cpp
using V = simd_examples::native_simd<float>;
constexpr std::size_t width = V::size();

std::size_t i = 0;
for (; i + width <= size; i += width) {
    V values;
    values.copy_from(input + i, stdx::element_aligned);
    values = operation(values);
    values.copy_to(output + i, stdx::element_aligned);
}

for (; i < size; ++i) {
    output[i] = scalar_operation(input[i]);
}
```

`copy_from` loads consecutive values into SIMD lanes. The operation then acts
on all lanes, and `copy_to` stores the result. The scalar tail handles at most
`width - 1` values that do not fill a complete vector.

Useful operations in the examples:

```cpp
stdx::reduce(v);       // sum all lanes
stdx::hmax(v);         // maximum across lanes
stdx::where(mask, v);  // conditional lanes
```

## Advanced examples

- **Softmax** computes
  `exp(x_i - max(x)) / sum(exp(x_j - max(x)))`. Maximum and sum use SIMD
  reductions; normalization is vectorized. The current source keeps exact
  `std::exp` scalar because `std::experimental::simd` has no portable vector
  exponential.
- **FMA** compares a memory-bound element-wise FMA with a compute-bound dot
  product. The dot product keeps partial sums in SIMD lanes.
- **Blur** loads three overlapping horizontal windows. Interior pixels use
  SIMD; image borders use scalar code.
- **Convolution** computes several output positions at once with a reversed
  mathematical kernel and a scalar output tail.

## Build and run

Use a clean module environment when switching compilers because both builds
use the same executable names.

```bash
# GCC
module purge
module load gcc/14.1.0_binutils241
make clean && make drivers
./build/01_add_scalar --size 16777216 --repetitions 10
./build/01_add_simd --size 16777216 --repetitions 10
```

```bash
# Intel compiler
module purge
module load intel/2025.2
make clean && make CXX=icpx drivers
./build/01_add_scalar --size 16777216 --repetitions 10
./build/01_add_simd --size 16777216 --repetitions 10
```

Build subsets with:

```bash
make scalar
make simd
make run
```

The scalar targets disable compiler vectorization so they provide a useful
baseline for studying explicit SIMD.

## Benchmarking

The launcher runs both implementations of one exercise and writes one CSV:

```bash
scripts/benchmark.sh 02_sum \
    --size 16777216 \
    --repetitions 10 \
    --output results/02_sum.csv
```

Drivers use deterministic inputs, independent scalar references, and a volatile
sink to keep the computation observable. Timed regions exclude input setup.

## Current x86 results

Median of three trials; each trial used the best of five repetitions on one
exclusive MN5 node and one pinned CPU core. The 1D size was 16,777,216 elements,
softmax used 4,194,304 elements, and blur used an 8192 × 4096 image.

| Kernel | GCC speedup | `icpx` speedup |
|---|---:|---:|
| Add | **1.17×** | **1.54×** |
| Sum reduction | **5.14×** | **5.15×** |
| Upper clamp | **7.85×** | **10.29×** |
| Count above | **4.91×** | **4.19×** |
| Softmax | **1.64×** | **4.43×** |
| Memory FMA | **1.02×** | **0.99×** |
| Dot product | **1.62×** | **4.03×** |
| Horizontal blur | **1.32×** | **0.94×** |
| 1D convolution | **2.65×** | **3.00×** |

Reductions, masks, and convolution benefit most. Add, memory FMA, and blur are
limited mainly by memory traffic.

The normal `icpx` softmax SIMD build also auto-vectorizes the scalar exponential
loop through Intel SVML. With compiler auto-vectorization disabled, its softmax
speedup was approximately 1.44×.

## Inspecting generated instructions

Inspect the final executable after linking:

```bash
objdump -d -C build/01_add_simd | grep -E 'vaddps|vmov'
objdump -d -C build/03_clamp_simd | grep -E 'vcmpps|vblend|vmov'
objdump -d -C build/06_fma_simd | grep -E 'vfmadd|vmov'
```

Typical AVX-512 instructions include `vmovups` for loads and stores, `vaddps`
for packed addition, `vcmpps` for comparisons, `vfmadd` for FMA, and `vmaxps`
for packed maximum.

## Takeaways

- SIMD operates on `width` values per instruction, not on the whole input at once.
- Reductions need lane accumulators followed by a horizontal reduction.
- Scalar tails are simple and safe for incomplete blocks.
- Wider vectors do not guarantee speedup when memory traffic is the bottleneck.
- Timings should be combined with correctness checks and final-binary inspection.

## License

See [LICENSE](LICENSE).
