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

## Reference

- [C++ experimental SIMD](https://en.cppreference.com/cpp/experimental/simd)

## Key results and performance

The 1D kernels used 16,777,216 elements and softmax used 4,194,304 elements.
Values are median speedups from three trials;
speedup means scalar time divided by SIMD time. The softmax benchmark uses a
smaller input because the current float normalization accumulation loses
validation accuracy at much larger sizes.

### x86_64

Results are from an Intel Xeon Platinum 8480+ on one exclusive MN5 node and one
pinned CPU core. Scalar targets disable compiler vectorization; SIMD targets
use explicit `std::experimental::simd` with normal optimization.

| Kernel | GCC | `icpx` |
|---|---:|---:|
| Element-wise addition | 1.17× | 1.54× |
| Sum reduction | 5.14× | 5.15× |
| Upper-bound clamp | 7.85× | 10.29× |
| Count above threshold | 4.91× | 4.19× |
| Softmax | 1.64× | 4.43× |
| Memory-bound FMA | 1.02× | 0.99× |
| Dot product | 1.62× | 4.03× |
| Horizontal blur | TBD | TBD |
| 1D convolution | TBD | TBD |

Among exercises 1–6, reductions, masks, and dot products benefit most. Addition
and memory FMA are limited mainly by memory traffic.

The normal `icpx` softmax build also auto-vectorizes the scalar exponential
loop through Intel SVML. With compiler auto-vectorization disabled, its softmax
speedup was approximately 1.44×.

### RISC-V

RISC-V binaries were cross-compiled with conda-forge GCC 16.2 and executed on a
Banana Pi F3 through the `bananaf3` queue. The target provides RVV 1.0 with a
256-bit VLEN (`vlenb_bytes=32`). GCC/libstdc++ reports one lane for
`native_simd<float>` on this target, so the comparison uses fixed-size SIMD
widths of four and eight lanes.

| Kernel | `VL=4` speedup | `VL=8` speedup |
|---|---:|---:|
| Element-wise addition | 1.43× | 1.43× |
| Sum reduction | 1.88× | 4.80× |
| Upper-bound clamp | 4.29× | 2.94× |
| Count above threshold | 1.33× | 2.04× |
| Softmax | 1.23× | 1.22× |
| Memory-bound FMA | 1.48× | 1.55× |
| Dot product | 1.30× | 1.94× |
| Horizontal blur | TBD | TBD |
| 1D convolution | TBD | TBD |

The `VL=4` and `VL=8` values select software vector widths; they do not change
the hardware VLEN. The `count_above` SIMD function contained no RVV
instructions in the final binaries, so its measured gain came from scalar
unrolling rather than genuine vector execution.

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

<details>
<summary>GCC build and run commands</summary>

```bash
module purge
module load gcc/14.1.0_binutils241
make clean
make drivers

./build/01_add_scalar --size 16777216 --repetitions 10
./build/01_add_simd --size 16777216 --repetitions 10
```

</details>

### Intel `icpx`

<details>
<summary>Intel <code>icpx</code> build and run commands</summary>

```bash
module purge
module load intel/2025.2
make clean
make CXX=icpx drivers

./build/01_add_scalar --size 16777216 --repetitions 10
./build/01_add_simd --size 16777216 --repetitions 10
```

</details>

Build subsets or run all default drivers with:

<details>
<summary>Make targets</summary>

```bash
make scalar
make simd
make run
```

</details>

A driver can write a combined scalar/SIMD CSV:

<details>
<summary>Benchmark command</summary>

```bash
scripts/benchmark.sh 02_sum \
    --size 16777216 \
    --repetitions 10 \
    --output results/02_sum.csv
```

</details>

### Inspect generated instructions

Inspect the final executable after linking:

<details>
<summary>Inspection commands</summary>

```bash
objdump -d -C build/01_add_simd | grep -E 'vaddps|vmov'
objdump -d -C build/03_clamp_simd | grep -E 'vcmpps|vblend|vmov'
objdump -d -C build/06_fma_simd | grep -E 'vfmadd|vmov'
```

</details>

## Conclusion

- SIMD processes several values per instruction, not the whole input at once.
- Explicit SIMD is built from vector loads, lane-wise operations, stores, and a
  scalar tail.
- Reductions require partial lane accumulators and horizontal reduction.
- Compiler choice and generated instructions affect measured performance.
- Memory bandwidth can dominate even when SIMD computation is available.
- Correctness validation, benchmarking, and binary inspection must be done
  together.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
