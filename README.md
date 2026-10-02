# SIMD Expression and Portability

This project is a compact, benchmark-driven study of explicit SIMD in modern C++.

## TL;DR

- **Learn SIMD:** seven compact scalar/SIMD exercises, from array arithmetic
  to softmax and convolution.
- **See the expression:** vector-width groups, loads, lane-wise operations,
  masks, stores, safe scalar tails, and lane accumulators with horizontal
  reductions.
- **Explore portability:** the same C++ kernels tested on x86-64, RISC-V,
  and AArch64.
- **Understand performance:** gains from negligible to about 25×; compiler
  choice, masks, and memory traffic can matter as much as vector width.
- **Verify the evidence:** independent correctness checks, isolated benchmarks,
  and final-binary inspection—not just SIMD labels in the source.

## Exercises

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

Scalar builds disable compiler vectorization; SIMD builds use
`std::experimental::simd` with normal optimization. Speedups compare these
implementations on each system, not the systems' absolute performance.

### x86_64 — MareNostrum 5 GPP (BSC)

Intel Xeon Platinum 8480+, GCC 14.1 and `icpx` 2025.2, with one pinned core
on an exclusive MN5 node.

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

Addition and memory FMA are likely memory-traffic-limited. `icpx` also
vectorizes softmax's scalar exponential loop through Intel SVML, so its
speedup is not solely from explicit SIMD.

### RISC-V — Banana Pi F3 (BSC)

Cross-compiled with conda-forge GCC 16.2 and run through the `bananaf3` queue
of BSC's Heterogeneous Computer Architectures (HCA) infrastructure. The board
supports RVV 1.0 with a 256-bit hardware VLEN. This toolchain reports one lane
for `native_simd<float>`, so the tests use fixed-size widths of four and eight.

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
unrolling rather than genuine vector execution.

### AArch64 — Apple M1 Pro

Apple M1 Pro with eight performance and two efficiency cores, macOS 27.0.1,
and native Homebrew GCC 15.2.0 with libstdc++. The verified
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

Measured on AC power with Low Power Mode disabled, without performance-core
pinning or controlled macOS scheduling. These results describe this M1 Pro
system, not AArch64 processors in general.

Softmax's exponential loop remains scalar. The large clamp/count gains also
reflect replacing branch-heavy scalar decisions with vector masks, not just
the four-lane width.

## Benchmark methodology

Current benchmark defaults are `9 × (3 untimed warm-ups + 10 timed inner calls)`.
Warm-ups run before every outer sample, outside the timed region.
For sample `s`, the time per call is `t_s = elapsed_s / 10`;
the reported time is `median(t_1, ..., t_9)`, and
speedup is `median_scalar / median_SIMD`. CSV output also includes the minimum
and maximum sample times.

The x86-64 results for exercises 1–5 used three initial warm-ups rather than
warm-ups before every sample; those entries have not yet been refreshed.

Nine outer samples are used because an odd sample count has a unique median:
the fifth sorted observation. Use odd sample counts when overriding the default;
the current helper does not average the middle pair for even counts.

Allocation, input generation, setup, and correctness checks are outside timed
regions. Mutable inputs are restored before each warm-up and timed batch.
Clamp and softmax use preinitialized buffers so every timed inner call receives
the original input. This measures warmed execution, not guaranteed cold-cache
access; a streaming-memory experiment would require a separate sliding-window
input design.


## Build and run benchmarks

### Requirements

Run commands from the repository root. Use GNU Make and a C++23 compiler
(`-std=c++2b` in the Makefile is equivalent), with libstdc++ providing
`<experimental/simd>`. The project does not yet use C++26 `<simd>`.

Native Makefile defaults target MN5's AVX-512 CPU on Linux and Apple M1 on
arm64 macOS; they are not generic defaults for every Linux or ARM host.

### Select a native compiler

Use a clean compiler environment and choose one setup below. The `compiler`
variable is used by the build and benchmark commands that follow.

<details>
<summary>MareNostrum 5: GCC</summary>

```bash
module purge
module load gcc/14.1.0_binutils241
compiler=g++
```

</details>

<details>
<summary>MareNostrum 5: Intel icpx</summary>

```bash
module purge
module load intel/2025.2
compiler=icpx
```

</details>

<details>
<summary>Apple silicon: Homebrew GCC</summary>

Run natively as `arm64`, not under Rosetta. Use Homebrew GCC, not Apple
Clang/libc++; replace `g++-15` with the installed versioned executable if needed.

```bash
uname -m  # arm64
compiler=g++-15
```

</details>

### Build, validate, and measure

On MN5, run benchmarks on allocated compute nodes, not login nodes. Check
the selected compiler, then build and benchmark all seven exercises:

```bash
"$compiler" --version
MAKEFLAGS="-B CXX=$compiler" scripts/benchmark.sh
# results/benchmark.csv
```

The script builds both implementations into `build/`, checks their results,
and writes one CSV using the methodology above. `-B` forces rebuilding when
switching compilers or flags; Make does not track those changes. `MAKEFLAGS`
passes the selected compiler to the script's Make invocation.

Run one exercise or override its instance and timing options:

```bash
MAKEFLAGS="-B CXX=$compiler" scripts/benchmark.sh 02_reduction_dot \
    --size 16777216 \
    --warmups 3 \
    --iterations 10 \
    --samples 9 \
    --output results/02_reduction_dot.csv
```

Use distinct `--output` paths to retain results from different compilers.
To build without measuring, or run executables individually:

```bash
make -B CXX="$compiler" drivers
./build/01_add_fma_scalar --size 16777216
./build/01_add_fma_simd --size 16777216
```

Replace the `drivers` target with `scalar` or `simd` to build only one
implementation. Drivers own generation, reference checks, timing, and CSV output.

### RISC-V: cross-build and run

Use the conda-forge `hpcbook` toolchain on MN5. Select the RISC-V-prefixed
compiler explicitly; the unqualified `g++` builds for x86-64.

<details>
<summary>Cross-compilation commands</summary>

```bash
module purge
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate hpcbook
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH

make -B BUILD_DIR=build/riscv \
    RISCV_CXX=riscv64-conda-linux-gnu-g++ \
    RISCV_CXXFLAGS='-std=c++23 -O3 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl -static -fno-math-errno -fno-trapping-math -Wall -Wextra -Idrivers -Iinclude -Isrc' \
    riscv
```

</details>

Stage `build/riscv/` under a temporary directory on HCA. On an allocated
Banana Pi F3, run from that staging directory, for example:

```bash
mkdir -p results
./build/riscv/01_add_fma_scalar.riscv --size 16777216 --output results/riscv-add-scalar.csv
./build/riscv/01_add_fma_simd.riscv --size 16777216 --output results/riscv-add-simd.csv
```

Copy the CSVs back to the repository's `results/` before removing the staging
directory.

These targets use the source's `native_simd` alias, which reports one lane on
this toolchain. They do **not** reproduce the fixed-size `VL=4` and `VL=8`
benchmark variants; those used separate builds with temporary SIMD aliases.
RVV flags alone do not guarantee vector execution. The native benchmark script
cannot execute RISC-V binaries on the x86-64 cross-compilation host.

### Inspect generated instructions

Inspect the final executable after linking:

<details>
<summary>Inspection commands</summary>

```bash
# MN5: AVX-512
objdump -d -C build/01_add_fma_simd

# macOS: NEON
otool -tvV build/01_add_fma_simd

# Cross-compiled RISC-V
riscv64-conda-linux-gnu-objdump -d -C build/riscv/01_add_fma_simd.riscv
```

</details>

Inspect the kernel functions themselves, not just instructions elsewhere in
the binary. Verify the selected SIMD lane count as well as the generated ISA.

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
