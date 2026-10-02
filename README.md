# SIMD Expression and Portability

This project is a compact, benchmark-driven study of explicit SIMD in modern C++.

## TL;DR

- Seven progressively more demanding scalar/SIMD exercises.
- Explicit load–compute–store loops with safe scalar tails.
- Reductions, masks, FMA, sliding windows, softmax, and convolution.
- Independent correctness checks and isolated executables.
- Measured SIMD gains from negligible to about 25×, depending on the kernel and system.
- Final-binary inspection confirms the generated AVX-512 instructions.

## Project structure

| Exercise | Description |
|---|---|
| [1. Addition and fused multiply-add (FMA)](docs/01_add_fma/README.md) | Element-wise addition and multiply-add with vector loads and stores. |
| [2. Reduction and dot product](docs/02_reduction_dot/README.md) | Accumulates sums and products in lanes, then reduces to a scalar. |
| [3. Upper-bound clamp](docs/03_clamp/README.md) | Clamps values using comparisons and conditional masks. |
| [4. Count above threshold](docs/04_count/README.md) | Counts threshold matches with masks and popcount. |
| [5. Numerically stable softmax](docs/05_softmax/README.md) | Computes stable softmax with vector reductions. |
| [6. Horizontal image blur](docs/06_filter/README.md) | Blurs rows using overlapping loads and scalar borders. |
| [7. 1D mathematical convolution](docs/07_conv1d/README.md) | Convolves with reversed kernels and vectorized outputs. |

Exercises 1–4 are basic; exercises 5–7 are advanced.
The advanced exercises include numerical examples in their READMEs.

## Key results and performance

Speedup means scalar time divided by SIMD time. Exercises 1–4 use 16,777,216
elements; softmax uses 4,194,304 elements to avoid validation loss from float
normalization accumulation at larger sizes. Horizontal blur uses a
1920 × 1080 grayscale image (2,073,600 pixels). Convolution uses 1,048,576
floats uniformly distributed between `-1` and `1` (seed `42`) and the fixed
kernel `[0.25, 0.5, 0.125]`, producing 1,048,574 outputs without padding.

### x86_64 — MareNostrum 5 GPP (BSC)

All seven exercises were tested on an Intel Xeon Platinum 8480+ on MN5,
using GCC 14.1 and `icpx` 2025.2 on one pinned core of an exclusive node.
Scalar builds disable auto-vectorization; SIMD builds use
`std::experimental::simd` with normal optimization.

| Kernel | GCC | `icpx` |
|---|---:|---:|
| Element-wise addition | 1.62× | 1.41× |
| Memory-bound FMA | 1.04× | 1.01× |
| Sum reduction | 5.37× | 5.32× |
| Dot product | 1.78× | 4.45× |
| Upper-bound clamp | 6.81× | 10.29× |
| Count above threshold | 4.85× | 4.19× |
| Softmax | 1.57× | 2.35× |
| Horizontal blur | 2.12× | 1.19× |
| 1D convolution | 5.89× | 4.03× |

Reductions, masks, and convolution show substantial gains. Addition and
memory FMA are limited mainly by memory traffic.

The normal `icpx` SIMD softmax build also auto-vectorizes the scalar
exponential loop through Intel SVML, so its speedup is not solely from the
explicit SIMD phases.

### RISC-V — Banana Pi F3 (BSC)

All seven exercises were cross-compiled with conda-forge GCC 16.2 and tested
on a Banana Pi F3 through the `bananaf3` queue of BSC's Heterogeneous Computer
Architectures (HCA) infrastructure. The board supports RVV 1.0 with a 256-bit
hardware VLEN. GCC/libstdc++ reports one lane for `native_simd<float>`, so
these tests use fixed-size SIMD widths of four and eight lanes.

| Kernel | `VL=4` speedup | `VL=8` speedup |
|---|---:|---:|
| Element-wise addition | 1.65× | 1.68× |
| Memory-bound FMA | 1.43× | 1.75× |
| Sum reduction | 1.84× | 4.51× |
| Dot product | 1.29× | 2.11× |
| Upper-bound clamp | 3.83× | 2.61× |
| Count above threshold | 1.29× | 1.98× |
| Softmax | 1.22× | 1.21× |
| Horizontal blur | 1.62× | 2.08× |
| 1D convolution | 1.13× | 2.15× |

The `VL=4` and `VL=8` values select software vector widths; they do not change
the hardware VLEN. The `count_above` SIMD function contained no RVV
instructions in the final binaries, so its measured gain came from scalar
unrolling rather than genuine vector execution. Both SIMD widths of horizontal
blur and convolution contain RVV instructions.

### AArch64 — Apple M1 Pro

All seven exercises were measured on this specific MacBook Pro
(MacBookPro18,3): Apple M1 Pro with eight performance and two efficiency cores,
16 GB RAM, macOS 27.0.1, and native Homebrew GCC 15.2.0 with libstdc++.
C++23 builds use `-O3 -mcpu=apple-m1`; scalar builds disable compiler
vectorization, while SIMD builds retain normal optimization. The verified
`native_simd<float>` width is four lanes (128-bit NEON).

| Kernel | GCC speedup |
|---|---:|
| Element-wise addition | 2.60× |
| Memory-bound FMA | 1.90× |
| Sum reduction | 4.00× |
| Dot product | 3.99× |
| Upper-bound clamp | 20.00× |
| Count above threshold | 24.51× |
| Softmax | 1.55× |
| Horizontal blur | 4.37× |
| 1D convolution | 4.02× |

These use the input instances above and the nine-sample methodology below,
on AC power with Low Power Mode disabled. Independent-reference checks passed
at benchmark sizes and on small/tail cases. Final-binary inspection confirmed
NEON inside all nine kernels; softmax's exponential loop remains scalar.
The large clamp/count gains also reflect replacing branch-heavy scalar loops
with vector masks, not just the four-lane width.

macOS scheduling was uncontrolled: execution was not pinned to a performance
core. No thermal/performance warnings were reported, but scheduling and thermal
variability remain possible. All samples were retained; the largest sample
maximum/minimum ratio was 1.12. These results describe this system, not AArch64
processors in general.

## Benchmark methodology

Exercises 1–7 use `9 × (3 untimed warm-ups + 10 timed inner calls)`.
Warm-ups run before every outer sample, outside the timed region.
For sample `s`, the time per call is `t_s = elapsed_s / 10`;
the reported time is `median(t_1, ..., t_9)`, and
speedup is `median_scalar / median_SIMD`. CSV output also includes the minimum
and maximum sample times.

The x86-64 results for exercises 1–5 used three initial warm-ups rather than
warm-ups before every sample; those entries have not yet been refreshed.

Nine outer samples are used because an odd sample count has a unique median:
the fifth sorted observation. With ten samples, the median would require
averaging observations five and six.

Allocation, input generation, setup, and correctness checks are outside timed
regions. Mutable inputs are restored before each warm-up and timed batch.
Clamp and softmax use preinitialized buffers so every timed inner call receives
the original input. This measures warmed execution, not guaranteed cold-cache
access; a streaming-memory experiment would require a separate sliding-window
input design.


## Build

### Requirements

- GNU Make and a C++23 compiler (`-std=c++2b` in the Makefile is equivalent).
- libstdc++ with `<experimental/simd>`; the project does not yet use C++26
  `<simd>`. On macOS, use native Homebrew GCC rather than Apple Clang/libc++.
- A supported target and appropriate compiler flags. The Makefile's native
  defaults target MN5's AVX-512 CPU on Linux and Apple M1 on arm64 macOS;
  they are not generic defaults for every Linux or ARM host.

Scalar targets disable compiler vectorization; SIMD targets retain normal
optimization. Use separate build directories for different toolchains, as
below. When changing the compiler or flags within one directory, use `make -B`
to force rebuilding: Make does not track changes to command-line flags.

### MareNostrum 5: GCC and Intel `icpx`

Use a clean module environment for each compiler. These commands use the
versions recorded in the x86-64 results.

<details>
<summary>MN5 build commands</summary>

```bash
# GCC
module purge
module load gcc/14.1.0_binutils241
make CXX=g++ BUILD_DIR=build/gcc drivers

# Intel icpx
module purge
module load intel/2025.2
make CXX=icpx BUILD_DIR=build/icpx drivers
```

</details>

### Apple silicon: Homebrew GCC

Run natively as `arm64`, not under Rosetta. The measured compiler is GCC 15.2;
replace `g++-15` with the installed versioned Homebrew executable if needed.
The Makefile uses `-mcpu=apple-m1` on arm64 macOS.

<details>
<summary>Apple silicon build commands</summary>

```bash
uname -m  # arm64
g++-15 --version
make CXX=g++-15 BUILD_DIR=build/m1-gcc drivers
```

</details>

### RISC-V: cross-compilation on MN5

Use the conda-forge `hpcbook` toolchain on MN5, then execute the binaries on
an allocated HCA board. The unqualified `g++` builds for x86-64; select the
RISC-V-prefixed compiler explicitly.

<details>
<summary>RISC-V build commands</summary>

```bash
module purge
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate hpcbook
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH

make BUILD_DIR=build/riscv \
    RISCV_CXX=riscv64-conda-linux-gnu-g++ \
    RISCV_CXXFLAGS='-std=c++23 -O3 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl -static -fno-math-errno -fno-trapping-math -Wall -Wextra -Idrivers -Iinclude -Isrc' \
    riscv
```

</details>

These targets use the source's `native_simd` alias, which reports one lane on
this toolchain. They do **not** reproduce the fixed-size `VL=4` and `VL=8`
benchmark variants; those used separate builds with temporary SIMD aliases.
RVV flags alone do not guarantee vector execution.

### Run and benchmark

Drivers own input generation, reference checks, timing, and CSV output. For
example, run the GCC addition/FMA executables with:

```bash
./build/gcc/01_add_fma_scalar --size 16777216
./build/gcc/01_add_fma_simd --size 16777216
```

`make scalar` and `make simd` build subsets; `make run` runs all drivers using
the default `build/` directory. Run substantial MN5 workloads on allocated
compute nodes, not login nodes.

The unified script also builds and runs from `build/`, independently of the
separate directories above. It runs all seven exercises with the default
methodology and writes one scalar/SIMD CSV:

<details>
<summary>Benchmark commands</summary>

```bash
scripts/benchmark.sh
# results/benchmark.csv
```

The default is nine outer samples, each with three untimed warm-ups and ten
timed inner calls. Run one exercise or override any value when needed:

```bash
scripts/benchmark.sh 02_reduction_dot \
    --size 16777216 \
    --warmups 3 \
    --iterations 10 \
    --samples 9 \
    --output results/02_reduction_dot.csv
```

</details>

The script uses the Makefile's platform-default compiler. To select another
compiler and avoid reusing stale binaries, pass Make overrides through
`MAKEFLAGS`, with the appropriate compiler environment already loaded:

```bash
MAKEFLAGS='-B CXX=icpx' scripts/benchmark.sh
```

### Inspect generated instructions

Inspect the final executable after linking:

<details>
<summary>Inspection commands</summary>

```bash
# MN5: AVX-512
objdump -d -C build/gcc/01_add_fma_simd

# macOS: NEON
otool -tvV build/m1-gcc/01_add_fma_simd

# Cross-compiled RISC-V
riscv64-conda-linux-gnu-objdump -d -C build/riscv/01_add_fma_simd.riscv
```

</details>

Inspect the kernel functions themselves, not just instructions elsewhere in
the binary. Verify the selected SIMD lane count as well as the generated ISA.

## Conclusion

- SIMD processes several values per instruction, not the whole input at once.
- Explicit SIMD is built from vector loads, lane-wise operations, stores, and a
  scalar tail.
- Reductions require partial lane accumulators and horizontal reduction.
- Compiler choice and generated instructions affect measured performance.
- Memory bandwidth can dominate even when SIMD computation is available.
- Correctness validation, benchmarking, and binary inspection must be done
  together.

## References and further reading

- [C++ experimental SIMD](https://en.cppreference.com/cpp/experimental/simd)
- [BSC HCA: nodes and queues](https://repo.hca.bsc.es/gitlab/epi-public/risc-v-software-development-vehicles/-/wikis/HCA-Nodes-and-Queues)
- [Platform-independent SIMD in Go](https://go.dev/blog/simd-experiment)
- [GCC auto-vectorization](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)
- [Arm NEON intrinsics reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Intel ISPC Performance Guide](https://ispc.github.io/perfguide.html)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
