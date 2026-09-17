# SIMD Expression and Portability

This project is a compact, benchmark-driven study of explicit SIMD in modern C++.

## TL;DR

- Seven progressively more demanding scalar/SIMD exercises.
- Explicit load–compute–store loops with safe scalar tails.
- Reductions, masks, FMA, sliding windows, softmax, and convolution.
- Independent correctness checks and isolated executables.
- Measured SIMD gains from negligible to about 10×, depending on the bottleneck.
- Final-binary inspection confirms the generated AVX-512 instructions.

## Project structure

| Example | Description |
|---|---|
| [1. Addition and fused multiply-add (FMA)](docs/01_add_fma/README.md) | Element-wise addition and multiply-add with vector loads and stores. |
| [2. Reduction and dot product](docs/02_reduction_dot/README.md) | Accumulates sums and products in lanes, then reduces to a scalar. |
| [3. Upper-bound clamp](docs/03_clamp/README.md) | Clamps values using comparisons and conditional masks. |
| [4. Count above threshold](docs/04_count/README.md) | Counts threshold matches with masks and popcount. |
| [5. Numerically stable softmax](docs/05_softmax/README.md) | Computes stable softmax with vector reductions. |
| [6. Horizontal image blur (TODO)](docs/06_filter/README.md) | Blurs rows using overlapping loads and scalar borders. |
| [7. 1D mathematical convolution (TODO)](docs/07_conv1d/README.md) | Convolves with reversed kernels and vectorized outputs. |

## Key results and performance

Speedup means scalar time divided by SIMD time. Exercises 1–4 use 16,777,216
elements; softmax uses 4,194,304 elements to avoid validation loss from float
normalization accumulation at larger sizes.

### x86_64

Results are from an Intel Xeon Platinum 8480+ on one exclusive MN5 node and one
pinned CPU core. Scalar targets disable compiler vectorization; SIMD targets
use explicit `std::experimental::simd` with normal optimization.

| Kernel | GCC | `icpx` |
|---|---:|---:|
| Element-wise addition | 1.56× | 1.41× |
| Memory-bound FMA | 1.03× | 1.01× |
| Sum reduction | 5.11× | 5.32× |
| Dot product | 1.75× | 4.45× |
| Upper-bound clamp | 6.79× | 10.29× |
| Count above threshold | 5.14× | 4.19× |
| Softmax | 1.57× | 2.35× |
| Horizontal blur | TBD | TBD |
| 1D convolution | TBD | TBD |

Among exercises 1–5, reductions, masks, and dot products benefit most. Addition
and memory FMA are limited mainly by memory traffic.

The normal `icpx` SIMD softmax build also auto-vectorizes the scalar
exponential loop through Intel SVML, so its speedup is not solely from the
explicit SIMD phases.

### RISC-V

RISC-V binaries were cross-compiled with conda-forge GCC 16.2 and executed on a
Banana Pi F3 through the `bananaf3` queue. The target provides RVV 1.0 with a
256-bit VLEN (`vlenb_bytes=32`). GCC/libstdc++ reports one lane for
`native_simd<float>` on this target, so the comparison uses fixed-size SIMD
widths of four and eight lanes.

| Kernel | `VL=4` speedup | `VL=8` speedup |
|---|---:|---:|
| Element-wise addition | 1.64× | 1.66× |
| Memory-bound FMA | 1.27× | 1.53× |
| Sum reduction | 1.85× | 4.78× |
| Dot product | 1.30× | 2.14× |
| Upper-bound clamp | 3.85× | 2.61× |
| Count above threshold | 1.29× | 1.98× |
| Softmax | 1.23× | 1.21× |
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

./build/01_add_fma_scalar --size 16777216 --repetitions 10
./build/01_add_fma_simd --size 16777216 --repetitions 10
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

./build/01_add_fma_scalar --size 16777216 --repetitions 10
./build/01_add_fma_simd --size 16777216 --repetitions 10
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

Run all completed exercises with the default methodology and write one unified
scalar/SIMD CSV:

<details>
<summary>Benchmark commands</summary>

```bash
scripts/benchmark.sh
# results/benchmark.csv
```

The default is three warm-ups, ten inner calls, and nine outer samples. Run one
exercise or override any value when needed:

```bash
scripts/benchmark.sh 02_reduction_dot \
    --size 16777216 \
    --warmups 3 \
    --iterations 10 \
    --samples 9 \
    --output results/02_reduction_dot.csv
```

</details>

### Inspect generated instructions

Inspect the final executable after linking:

<details>
<summary>Inspection commands</summary>

```bash
objdump -d -C build/01_add_fma_simd | grep -E 'vaddps|vmov'
objdump -d -C build/03_clamp_simd | grep -E 'vcmpps|vblend|vmov'
objdump -d -C build/01_add_fma_simd | grep -E 'vfmadd|vmov'
```

</details>

## Benchmark methodology

All exercises use three untimed warm-ups and `9 × 10` measured kernel calls:
nine outer samples with ten inner calls each. For sample `s`, the time per call
is `t_s = elapsed_s / 10`; the reported time is `median(t_1, ..., t_9)`, and
speedup is `median_scalar / median_SIMD`. CSV output also includes the minimum
and maximum sample times.

Nine outer samples are used because an odd sample count has a unique median:
the fifth sorted observation. With ten samples, the median would require
averaging observations five and six.

Allocation, input generation, setup, and correctness checks are outside timed
regions. Mutable inputs are restored between samples. Clamp and softmax use
preinitialized buffers so every inner call receives the original input.

## Conclusion

- SIMD processes several values per instruction, not the whole input at once.
- Explicit SIMD is built from vector loads, lane-wise operations, stores, and a
  scalar tail.
- Reductions require partial lane accumulators and horizontal reduction.
- Compiler choice and generated instructions affect measured performance.
- Memory bandwidth can dominate even when SIMD computation is available.
- Correctness validation, benchmarking, and binary inspection must be done
  together.

## Reference

- [C++ experimental SIMD](https://en.cppreference.com/cpp/experimental/simd)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
