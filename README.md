# Modern C++ SIMD Programming with `std::simd`

A practical guide to writing portable, high-performance SIMD code using the C++26 standard library.

## Table of Contents

- [Introduction](#introduction)
- [What is SIMD?](#what-is-simd)
- [The std::simd Library](#the-stdsimd-library)
- [Building and Running Examples](#building-and-running-examples)
- [Understanding Vector Instructions](#understanding-vector-instructions-low-level-details)
- [Basic Examples](#basic-examples)
  - [Example 1: Vector Add](#example-1-vector-add)
  - [Example 2: Sum Reduction](#example-2-sum-reduction)
  - [Example 3: Clamp with Masks](#example-3-clamp-with-masks)
  - [Example 4: Count with Popcount](#example-4-count-with-popcount)
- [Advanced Examples](#advanced-examples)
  - [Example 5: Numerically Stable Softmax](#example-5-numerically-stable-softmax)
  - [Example 6: FMA - Memory vs Compute Bound](#example-6-fma---memory-vs-compute-bound)
  - [Example 7: Image Processing (Horizontal Blur)](#example-7-image-processing-horizontal-blur)
  - [Example 8: When SIMD Makes Things Worse](#example-8-when-simd-makes-things-worse)
- [Compiler Comparison: GCC vs Intel](#compiler-comparison-gcc-vs-intel)
- [Key Takeaways](#key-takeaways)
- [Appendix: Quick Reference](#appendix-quick-reference)

## Additional Resources

- **[ASSEMBLY_EXAMPLES.md](ASSEMBLY_EXAMPLES.md)** - Annotated assembly code with detailed explanations
- **[RISCV_STATUS.md](RISCV_STATUS.md)** - RISC-V vectorization status and limitations

---

## Introduction

SIMD (Single Instruction, Multiple Data) is a parallelization technique that processes multiple data elements with a single CPU instruction. Modern CPUs support SIMD through extensions like:

- **x86**: SSE (128-bit), AVX (256-bit), AVX-512 (512-bit)
- **ARM**: NEON (128-bit), SVE (scalable)
- **RISC-V**: RVV (RISC-V Vector)

Historically, writing SIMD code required:
1. [**Intrinsics**](https://en.wikipedia.org/wiki/Intrinsic_function): Low-level, architecture-specific functions (e.g., `_mm256_add_ps`)
2. [**Autovectorization**](https://en.wikipedia.org/wiki/Automatic_vectorization): Relying on the compiler to convert scalar code to SIMD

The C++26 standard introduces `std::simd`, a portable middle layer that gives you explicit control without writing architecture-specific code.

---

## What is SIMD?

Consider adding two arrays of floats:

```cpp
// Scalar (one element at a time)
for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
}
```

With SIMD, you process multiple elements simultaneously:

```
Register: [a0, a1, a2, a3, a4, a5, a6, a7]  (AVX2 = 256 bits = 8 floats)
Register: [b0, b1, b2, b3, b4, b5, b6, b7]
           +
Result:   [c0, c1, c2, c3, c4, c5, c6, c7]
```

One instruction does 8 additions instead of 1!

---

## The std::simd Library

### Key Types

```cpp
#include <experimental/simd>  // GCC/Clang implementation

namespace stdx = std::experimental;

// Native SIMD type - automatically uses best width for your CPU
template<class T>
using native_simd = stdx::native_simd<T>;

// SIMD mask for conditional operations
template<class T>
using native_mask = stdx::simd_mask<T, stdx::simd_abi::native<T>>;
```

As of February 2026, the standardized C++26 `<simd>` header (with types in namespace `std::datapar`) is not yet fully available in released mainstream compilers. The feature is in progress, but compiler support remains partial/experimental.

### Basic Operations

```cpp
using V = native_simd<float>;
constexpr size_t W = V::size();  // 16 on AVX-512, 8 on AVX2, 4 on NEON

V a, b;
a.copy_from(ptr, stdx::element_aligned);  // Load
b = a + a;                                 // Element-wise add
b.copy_to(ptr, stdx::element_aligned);    // Store
```

### Available Operations

- **Arithmetic**: `+`, `-`, `*`, `/`, unary `-`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=` (returns mask)
- **Logical**: `&`, `|`, `^`, `~`
- **FMA**: `stdx::fma(a, b, c)` = a*b + c
- **Reduce**: `stdx::reduce(v)` = sum all elements
- **Horizontal**: `stdx::hmin`, `stdx::hmax`
- **Masked**: `stdx::where(mask, value)`

---

## Building and Running Examples

### Quick Start

```bash
make          # Build all examples
make run      # Build and run all examples
./01_add      # Run a specific example
```

### Platform-Specific Setup

#### MareNostrum 5 (BSC)

```bash
module purge
module load gcc/13.2.0

# For GCC compilation:
make

# For Intel compiler (recommended for better performance):
module load intel/2025.2
icpx -std=c++2b -O3 -march=native -fiopenmp-simd example.cpp -o example
```

#### macOS (Apple Silicon / Intel)

```bash
# Install Homebrew GCC (required - Apple Clang doesn't support std::simd)
brew install gcc
make
```

#### Ubuntu/Debian

```bash
sudo apt install g++-13
make
```

#### RISC-V Emulation

**⚠️ Important:** GCC 13's `std::experimental::simd` does not yet support RISC-V Vector extensions (RVV). The code compiles and runs but falls back to scalar operations. See **[RISCV_STATUS.md](RISCV_STATUS.md)** for details.

Quick start on Ubuntu:
```bash
# Install QEMU and RISC-V toolchain
sudo apt install -y qemu-user-static gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Build RISC-V binaries
make riscv

# Run examples (currently uses scalar fallback, SIMD width = 1)
make run-riscv-128
```

### Compiler Flags

The Makefile automatically detects your platform and uses appropriate flags:

| Platform | Compiler | Key Flags |
|----------|----------|-----------|
| Linux (AVX-512) | `g++` | `-march=native -mavx512f -mavx512vl` |
| macOS (ARM) | `g++-15` | `-mcpu=apple-m1` |
| macOS (Intel) | `g++-15` | `-march=native` |
| RISC-V (emulated) | `riscv64-linux-gnu-g++` | `-march=rv64gcv -static` |

Common flags for all platforms:
- `-std=c++2b`: C++26 working draft
- `-O3`: Maximum optimization
- `-fno-math-errno -fno-trapping-math`: Allow aggressive FP optimizations

### Benchmarking SIMD Code

When benchmarking SIMD code, there's a subtle but critical issue: modern compilers are very good at optimizing away "dead code." If the compiler can prove that the result of a computation doesn't affect observable program behavior, it may simply eliminate the computation entirely.

In our first attempts, the benchmark timing showed impossible results—16 million elements processed in just 17 nanoseconds. This is physically impossible (memory access alone takes longer), which means the compiler was optimizing away the computation because the result wasn't being used in a way the compiler considered "observable."

**The Solution**: We use a global volatile variable to store the result of each benchmarked function:

```cpp
// Global counter - prevents optimization of function calls
static volatile std::size_t g_benchmark_sink = 0;

template<class F>
double bench_ms(F&& f, int iters = 10) {
    // ...
    g_benchmark_sink = f();  // Store result in global - can't be optimized away
    // ...
}
```

By storing the result in a global volatile variable, we force the compiler to actually execute the function because:
1. The global might be read later (observable behavior)
2. The compiler can't prove it doesn't matter

This is why all our scalar functions include a pragma to disable auto-vectorization AND we use a global sink in the benchmark. Both are necessary for accurate comparisons.

### Measured Results

Results vary significantly by compiler. The table below shows GCC 13.2.0 results. For Intel compiler results (2-4x faster), see [Compiler Comparison](#compiler-comparison-gcc-vs-intel).

| Example | MN5 Scalar | MN5 SIMD | MN5 Speedup | M1 Pro Scalar | M1 Pro SIMD | M1 Pro Speedup |
|---------|------------|----------|-------------|---------------|-------------|----------------|
| 01_add | 20.6 ms | 17.5 ms | **1.2x** | 22.8 ms | 18.5 ms | **1.2x** |
| 02_sum | 11.3 ms | 2.1 ms | **4.9x** | 15.8 ms | 3.9 ms | **4.0x** |
| 03_clamp | 2.3 ms | 0.7 ms | **3.5x** | 5.2 ms | 1.9 ms | **2.7x** |
| 04_count | 11.3 ms | 2.2 ms | **5.2x** | 73.1 ms | 3.0 ms | **24.5x** |
| 05_softmax | 5.4 ms | 1.1 ms | **5.1x** | 4.1 ms | 1.6 ms | **2.7x** |
| 06_fma (mem) | 24.5 ms | 24.0 ms | **1.0x** | - | - | - |
| 06_fma (compute) | 23.8 ms | 10.9 ms | **2.2x** | - | - | - |
| 07_filter | 0.49 ms | 0.21 ms | **2.3x** | 0.38 ms | 0.49 ms | **0.8x** |

**Platform Details:**
- **MN5 (GCC)**: Intel Xeon 8480+ (Sapphire Rapids), AVX-512, 512-bit vectors (16 floats), GCC 13.2.0
- **M1 Pro**: Apple M1 Pro, ARM NEON, 128-bit vectors (4 floats)

**Key Observations:**
- **Best speedups**: sum, count, softmax - compute-bound reductions benefit most from SIMD
- **Modest speedups**: add - memory-bandwidth limited on both platforms
- **Platform differences**: count shows 24x on M1 Pro vs 5x on MN5 (scalar baseline differs)
- **Memory vs compute**: Example 6 demonstrates the dramatic difference (1x vs 2.2x on GCC, 2.4x vs 8.3x on Intel)

### SIMD Instruction Verification

Understanding what instructions your CPU actually executes is crucial for performance debugging. The same `std::simd` code compiles to completely different assembly on different architectures.

#### How to Generate and Inspect Assembly

**ARM NEON (macOS/Apple Silicon):**
```bash
g++-15 -std=c++2b -O3 -mcpu=apple-m1 -S 01_add.cpp -o 01_add.s
grep -E 'fadd\s+v[0-9]+\.4s' 01_add.s   # Look for NEON vector ops
```

**x86 AVX-512 (Linux/Intel):**
```bash
g++ -std=c++2b -O3 -march=native -S 01_add.cpp -o 01_add.s
grep -E 'vaddps|vmulps' 01_add.s        # Look for AVX vector ops
```

**RISC-V RVV (emulated):**
```bash
make verify-riscv  # Generates 01_add.s and checks for RVV instructions
grep -E 'vle32\.v|vfadd\.vv' 01_add.s
```

#### Verified Instructions by Architecture

| Example | ARM NEON (M1 Pro) | x86 AVX-512 (MN5) | RISC-V RVV |
|---------|-------------------|-------------------|------------|
| 01_add | `fadd v30.4s, v31.4s, v30.4s` | `vaddps zmm` | `fadd.s` ⚠️ |
| 02_sum | `fadd v31.4s, v31.4s, v4.4s` | `vaddps zmm` | `fadd.s` ⚠️ |
| 03_clamp | `fcmgt v3.4s, v30.4s, v31.4s` | `vcmpps` + `vblendps` | `flt.s` ⚠️ |
| 04_count | `fcmgt` + `addp v31.2s` | `vcmpps` + `vpopcntd` | `flt.s` ⚠️ |
| 05_softmax | `fmaxnm v31.4s, v31.4s, v4.4s` | `vmaxps zmm` | `fmax.s` ⚠️ |
| 06_fma | `fmla v31.4s, v27.4s, v30.4s` | `vfmadd231ps zmm` | `fmadd.s` ⚠️ |
| 07_filter | `fadd` + `fdiv v17.4s` | `vaddps` + `vdivps` | `fadd.s` ⚠️ |

⚠️ **Note**: RISC-V currently uses scalar instructions due to lack of `std::simd` support for RVV. See [RISCV_STATUS.md](RISCV_STATUS.md).

---

## Understanding Vector Instructions (Low-Level Details)

This section explains what's actually happening at the CPU instruction level when you use `std::simd`. Each architecture has different instruction sets, register layouts, and naming conventions.

### Architecture Comparison Table

| Feature | ARM NEON | x86 AVX-512 | RISC-V RVV |
|---------|----------|-------------|------------|
| **Register width** | 128-bit | 512-bit | Variable (VLEN) |
| **Floats per register** | 4 | 16 | VLEN/32 |
| **Register names** | `v0`-`v31` | `zmm0`-`zmm31` | `v0`-`v31` |
| **Load/Store** | `ld1`/`st1` | `vmovups`/`vmovaps` | `vle32.v`/`vse32.v` |
| **Add** | `fadd` | `vaddps` | `vfadd.vv` |
| **Multiply** | `fmul` | `vmulps` | `vfmul.vv` |
| **FMA** | `fmla` | `vfmadd231ps` | `vfmacc.vv` |

### ARM NEON Instructions (128-bit, 4 floats)

NEON uses a clear syntax: `instruction destination, source1, source2`

**Register naming:**
- `v0` to `v31` = 128-bit vector registers
- `.4s` suffix = interpret as 4 single-precision floats
- `.2d` suffix = interpret as 2 double-precision floats

**Example from `01_add.cpp` (vector addition):**
```assembly
# Load 4 floats from memory into v30
ld1 {v30.4s}, [x1]          # x1 = pointer to array

# Load 4 floats from memory into v31  
ld1 {v31.4s}, [x2]          # x2 = pointer to second array

# Add all 4 lanes simultaneously: v30[i] = v31[i] + v30[i]
fadd v30.4s, v31.4s, v30.4s

# Store 4 floats back to memory
st1 {v30.4s}, [x0]          # x0 = pointer to destination
```

**Visual representation:**
```
Register v31: [a0, a1, a2, a3]  (4 floats loaded from memory)
Register v30: [b0, b1, b2, b3]  (4 floats loaded from memory)
              +  +  +  +        (single fadd instruction)
Result v30:   [c0, c1, c2, c3]  (4 results computed in parallel)
```

**Common NEON instructions:**
- `ld1 {v.4s}, [x]` - Load 4 floats from address in register x
- `st1 {v.4s}, [x]` - Store 4 floats to address in register x
- `fadd v.4s, v1.4s, v2.4s` - Vector add: v = v1 + v2
- `fmul v.4s, v1.4s, v2.4s` - Vector multiply: v = v1 * v2
- `fmla v.4s, v1.4s, v2.4s` - Fused multiply-add: v = v + (v1 * v2)
- `fcmgt v.4s, v1.4s, v2.4s` - Compare greater: v = (v1 > v2) ? all-1s : all-0s
- `fmax v.4s, v1.4s, v2.4s` - Element-wise maximum
- `fdiv v.4s, v1.4s, v2.4s` - Element-wise division

**Example from `06_fma.cpp` (FMA operation):**
```assembly
# Fused multiply-add: v31 = v31 + (v27 * v30)
fmla v31.4s, v27.4s, v30.4s
```
This is equivalent to 4 operations done in one instruction:
```
v31[0] += v27[0] * v30[0]
v31[1] += v27[1] * v30[1]
v31[2] += v27[2] * v30[2]
v31[3] += v27[3] * v30[3]
```

### x86 AVX-512 Instructions (512-bit, 16 floats)

AVX-512 has more complex syntax with Intel's legacy from SSE/AVX. The `v` prefix indicates vector instructions.

**Register naming:**
- `zmm0` to `zmm31` = 512-bit vector registers (AVX-512)
- `ymm0` to `ymm31` = 256-bit vector registers (AVX/AVX2, lower 256 bits of zmm)
- `xmm0` to `xmm31` = 128-bit vector registers (SSE, lower 128 bits of zmm)

**Example from `01_add.cpp` (vector addition):**
```assembly
# Load 16 floats (64 bytes) from memory into zmm1
vmovups zmm1, zmmword ptr [rsi + 4*rax]    # rsi = base address, rax = index

# Load 16 floats into zmm0
vmovups zmm0, zmmword ptr [rdi + 4*rax]    # rdi = base address

# Add all 16 lanes simultaneously
vaddps zmm0, zmm1, zmm0                     # zmm0 = zmm1 + zmm0

# Store 16 floats back to memory
vmovups zmmword ptr [rdi + 4*rax], zmm0
```

**Visual representation:**
```
Register zmm1: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]
Register zmm0: [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15]
               +   +   +   +   +   +   +   +   +   +   +    +    +    +    +    +
Result zmm0:   [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
```

**Common AVX-512 instructions:**
- `vmovups zmm, [addr]` - Unaligned load (16 floats)
- `vmovaps zmm, [addr]` - Aligned load (faster, requires 64-byte alignment)
- `vmovups [addr], zmm` - Unaligned store
- `vaddps zmm0, zmm1, zmm2` - Vector add: zmm0 = zmm1 + zmm2
- `vmulps zmm0, zmm1, zmm2` - Vector multiply: zmm0 = zmm1 * zmm2
- `vfmadd231ps zmm0, zmm1, zmm2` - FMA: zmm0 = zmm0 + (zmm1 * zmm2)
- `vcmpps k0, zmm1, zmm2, imm8` - Compare, result in mask register k0
- `vblendmps zmm0 {k1}, zmm1, zmm2` - Conditional blend using mask k1
- `vmaxps zmm0, zmm1, zmm2` - Element-wise maximum
- `vpopcntd zmm0, zmm1` - Population count (count 1-bits)

**Mask registers (AVX-512 feature):**
AVX-512 has dedicated mask registers `k0` to `k7` for predication:
```assembly
# Compare: create mask where zmm1 > zmm2
vcmpps k1, zmm1, zmm2, 30              # 30 = greater-than predicate

# Conditionally add: only update lanes where k1 is true
vaddps zmm0 {k1}, zmm3, zmm4           # zmm0[i] = (k1[i] ? zmm3[i]+zmm4[i] : zmm0[i])
```

**Example from `06_fma.cpp` (FMA operation):**
```assembly
# Fused multiply-add: zmm0 = zmm0 + (zmm1 * zmm2)
vfmadd231ps zmm0, zmm1, zmm2
```
This is equivalent to 16 operations in one instruction!

### RISC-V Vector (RVV) Instructions (Variable VLEN)

**⚠️ Important**: GCC 13's `std::simd` doesn't support RVV yet, so these instructions are NOT currently generated. However, here's what WOULD be generated when support is added:

**Register naming:**
- `v0` to `v31` = vector registers (width determined by VLEN parameter)
- VLEN can be 128, 256, 512, or 1024 bits (configurable)
- Each instruction can operate on different SEW (selected element width): 8, 16, 32, or 64 bits

**Unique RVV feature: Dynamic vector length**
Unlike NEON (fixed 128-bit) and AVX-512 (fixed 512-bit), RISC-V vectors have runtime-configurable length:
```assembly
# Set vector length: how many 32-bit elements fit in a vector register?
vsetvli t0, a0, e32, m1    # t0 = min(a0, VLEN/32), e32 = 32-bit elements, m1 = 1 register
```

**Example of what `01_add.cpp` should generate (not currently working):**
```assembly
# Set vector type: 32-bit float elements
vsetvli t0, a2, e32, m1       # t0 = actual vector length used

# Load vector of floats from memory
vle32.v v1, (a1)              # a1 = pointer to array

# Load second vector
vle32.v v2, (a0)              # a0 = pointer to second array

# Add vectors element-wise
vfadd.vv v3, v1, v2           # v3[i] = v1[i] + v2[i]

# Store result back to memory
vse32.v v3, (a0)              # Store to destination
```

**What we actually get (scalar fallback):**
```assembly
# Current GCC 13 output - no vectorization!
flw fa5, 0(a0)                # Load single float
flw fa4, 0(a1)                # Load single float
fadd.s fa5, fa5, fa4          # Scalar add (only 1 float!)
fsw fa5, 0(a0)                # Store single float
```

**Common RVV instructions (when supported):**
- `vsetvli rd, rs1, vtypei` - Configure vector length and element type
- `vle32.v vd, (rs1)` - Load vector of 32-bit elements
- `vse32.v vs3, (rs1)` - Store vector of 32-bit elements
- `vfadd.vv vd, vs2, vs1` - Vector + vector addition
- `vfmul.vv vd, vs2, vs1` - Vector × vector multiplication
- `vfmacc.vv vd, vs1, vs2` - FMA: vd = vd + (vs1 * vs2)
- `vmfgt.vf vd, vs2, fs1` - Compare vector > scalar, result is mask
- `vfmax.vv vd, vs2, vs1` - Element-wise maximum

**RVV naming conventions:**
- `.vv` = vector-vector operation (both operands are vectors)
- `.vf` = vector-scalar operation (one vector, one scalar/float register)
- `.vi` = vector-immediate operation (vector and immediate constant)
- `e32` = 32-bit elements
- `m1` = use 1 vector register (can be m2, m4, m8 for wider operations)

### Performance Comparison: Why Vector Width Matters

**Same operation (`c[i] = a[i] + b[i]`) on different architectures:**

| Architecture | Registers | Width | Floats/Instruction | Instructions for 16 floats |
|--------------|-----------|-------|-------------------|---------------------------|
| **Scalar** | Floating-point | 32-bit | 1 | 16 |
| **ARM NEON** | v0-v31 | 128-bit | 4 | 4 |
| **x86 AVX-512** | zmm0-zmm31 | 512-bit | 16 | 1 |
| **RISC-V (VLEN=128)** | v0-v31 | 128-bit | 4 | 4 |
| **RISC-V (VLEN=512)** | v0-v31 | 512-bit | 16 | 1 |

This is why AVX-512 can achieve higher theoretical speedups than NEON—it processes 4x more data per instruction!

### Debugging Tips

**1. Verify vectorization is happening:**
```bash
# Look for vector instructions in assembly
grep -E 'fadd.*\.4s|vaddps|vfadd\.vv' example.s
```

**2. Check register usage:**
```bash
# ARM: Should see v0-v31, NOT s0-s31 (scalar)
grep -E '\sv[0-9]+\.' example.s

# x86: Should see zmm/ymm, NOT xmm (smaller) or scalar
grep -E 'zmm|ymm' example.s
```

**3. Count vector operations:**
```bash
# How many vector adds?
grep -c 'vaddps' example.s        # x86
grep -c 'fadd.*\.4s' example.s    # ARM
```

**4. Look for loop unrolling:**
Modern compilers often unroll loops to hide latency:
```assembly
# Unrolled loop processing 4 vectors (16 floats) per iteration on NEON
ld1 {v0.4s}, [x1], #16
ld1 {v1.4s}, [x1], #16
ld1 {v2.4s}, [x1], #16
ld1 {v3.4s}, [x1], #16
fadd v0.4s, v0.4s, v4.4s
fadd v1.4s, v1.4s, v5.4s
fadd v2.4s, v2.4s, v6.4s
fadd v3.4s, v3.4s, v7.4s
```

**5. Memory alignment matters:**
```bash
# Aligned loads are faster
vmovaps zmm0, [addr]     # Fast: requires 64-byte alignment
vmovups zmm0, [addr]     # Slower: works with any alignment
```

### More Detail

For complete annotated assembly examples showing exactly what each instruction does, see **[ASSEMBLY_EXAMPLES.md](ASSEMBLY_EXAMPLES.md)**. This document includes:
- Line-by-line assembly annotations for all three architectures
- Visual representations of what happens in each register
- Performance comparisons (instructions per 64 floats)
- Real assembly output from compiling the examples

---

# Basic Examples

## Example 1: Vector Add

**File**: [`01_add.cpp`](01_add.cpp)

Every SIMD operation follows the same three-step pattern: load data from memory into a SIMD register, perform the computation on all elements simultaneously, then store the results back to memory. This example shows this pattern applied to array addition. The key insight here is understanding how the loop processes `W` elements (the SIMD width) per iteration—on AVX-512 that's 16 floats, on AVX2 it's 8, and on ARM NEON it's 4. The scalar version uses a pragma to disable auto-vectorization so you can see the true cost of element-by-element processing versus the SIMD version.

Note that this particular operation shows modest speedup (~1.2x) because it's memory-bandwidth limited—the CPU can compute faster than it can fetch data from memory. Additionally, modern compilers often auto-vectorize such simple operations anyway, which is why explicit SIMD is most valuable for more complex patterns.

### Key Concepts

1. **SIMD width**: `V::size()` gives elements per register (16 for AVX-512)
2. **Alignment**: `element_aligned` is minimum; `vector_aligned` is faster
3. **Tail handling**: Process remaining elements with a simple loop

### Code

First, the scalar version:

```cpp
// Scalar (non-vectorized) - one element at a time
// The pragma prevents the compiler from auto-vectorizing this function
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void add_scalar(float* __restrict dst, const float* __restrict src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}
#pragma GCC pop_options
```

Now the SIMD version:

```cpp
void add_simd(float* dst, const float* src, size_t n) {
    using V = native_simd<float>;
    constexpr size_t W = V::size();
    
    size_t i = 0;
    
    // Main loop: W elements per iteration
    for (; i + W <= n; i += W) {
        V a, b;
        a.copy_from(dst + i, stdx::element_aligned);
        b.copy_from(src + i, stdx::element_aligned);
        a += b;  // Single instruction!
        a.copy_to(dst + i, stdx::element_aligned);
    }
    
    // Tail: remaining elements
    for (; i < n; ++i) dst[i] += src[i];
}
```

**Note**: Speedup is modest because this is memory-bandwidth limited, not compute-limited. The compiler also auto-vectorizes this simple case.

### When to Use Explicit SIMD

- Complex tail handling
- Specific alignment requirements  
- When compiler autovectorization fails

---

## Example 2: Sum Reduction

**File**: [`02_sum.cpp`](02_sum.cpp)

This example demonstrates horizontal reduction—converting a SIMD register containing multiple values into a single scalar result by summing all elements. The pattern is: accumulate partial sums in a SIMD register (each lane gets the sum of W elements), then use `stdx::reduce()` to horizontally sum all lanes. This is where explicit SIMD really shines, as compilers struggle to auto-vectorize reductions due to the loop-carried dependency (each iteration depends on the previous sum). You'll see significant speedup (~5x) here compared to simple operations like add, because this is compute-bound rather than memory-bound.

An important teaching point: floating-point addition is not associative, meaning `(a+b)+c` can differ from `a+(b+c)` due to rounding. The SIMD version adds elements in a different order than the scalar version, so results may differ slightly—this is normal and usually acceptable for numerical work, but it's important to be aware of.

### Key Concepts

1. **Horizontal reduction**: `stdx::reduce(acc)` sums all lanes
2. **Accumulator pattern**: Keep partial sums in SIMD registers
3. **Floating-point gotcha**: Results may differ due to non-associativity

### Code

First, the scalar version:

```cpp
// Scalar sum: simple loop
float sum_scalar(const float* a, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += a[i];
    return s;
}
```

Now the SIMD version:

```cpp
float sum_simd(const float* a, size_t n) {
    using V = native_simd<float>;
    constexpr size_t W = V::size();
    
    V acc = 0.f;  // Initialize to zeros
    size_t i = 0;
    
    // Accumulate W elements at a time
    for (; i + W <= n; i += W) {
        V v;
        v.copy_from(a + i, stdx::element_aligned);
        acc += v;
    }
    
    // Horizontal reduction: sum all lanes
    float s = stdx::reduce(acc);
    
    // Tail
    for (; i < n; ++i) s += a[i];
    return s;
}
```

### Important: Floating-Point Non-Associativity

```
(a + b) + c ≠ a + (b + c) in floating-point
```

The SIMD version adds elements in a different order than scalar, causing different rounding errors. This is why the output shows "NO" for sums matching.

**Solutions**:
- Accept small differences for numerical work
- Use Kahan summation for exact results
- Use `-ffast-math` if precision loss is acceptable

---

## Example 3: Clamp with Masks

**File**: [`03_clamp.cpp`](03_clamp.cpp)

Masked operations allow you to apply computations conditionally to individual elements within a SIMD register, without using branches. When you compare two SIMD vectors (e.g., `v > vhi`), you get a mask where each lane is true or false. The `stdx::where(mask, value)` construct then acts as a conditional update: elements where the mask is true get the new value, while others remain unchanged. On CPUs with AVX-512, this compiles to specialized blend instructions that are very efficient. However, this example also serves as an important lesson: not all operations benefit from explicit SIMD. The clamp operation is so simple that the overhead of creating masks and applying conditional updates can exceed the benefit—you may actually see the SIMD version run slower than the scalar version!

### Key Concepts

1. **Mask creation**: Comparisons return a mask type
2. **Conditional update**: `stdx::where(mask, value)`
3. **No branches**: Better for branch prediction

### Code

First, the scalar version:

```cpp
// Scalar clamp: simple loop with if
void clamp_scalar(float* a, size_t n, float hi) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] > hi) a[i] = hi;
    }
}
```

Now the SIMD version:

```cpp
void clamp_simd(float* a, size_t n, float hi) {
    using V = native_simd<float>;
    using M = native_mask<float>;
    constexpr size_t W = V::size();
    
    V vhi(hi);
    
    for (size_t i = 0; i + W <= n; i += W) {
        V v;
        v.copy_from(a + i, stdx::element_aligned);
        
        M mask = v > vhi;  // Element-wise comparison
        
        stdx::where(mask, v) = vhi;  // Conditional update
        
        v.copy_to(a + i, stdx::element_aligned);
    }
}
```

**Why Speedup is Modest**: Not all operations benefit from SIMD. Simple conditional updates may be slower due to the overhead of masked operations.

---

## Example 4: Count with Popcount

**File**: [`04_count.cpp`](04_count.cpp)

This example demonstrates how to efficiently count elements that satisfy a condition using popcount. The approach is elegant: create a mask by comparing each element to a threshold (producing a SIMD mask where true/false becomes 1/0), then use `stdx::popcount()` to count all the 1-bits in a single operation. On modern CPUs with AVX-512, this uses the dedicated `VPOPCNTD` instruction. This is far more efficient than a scalar loop with branches, which suffers from branch misprediction penalties. Popcount is foundational for many algorithms including histograms, threshold filtering, and diversity counting in parallel reductions.

### Key Concepts

1. **Popcount**: `stdx::popcount(mask)` counts true bits
2. **Efficient**: Uses single instruction on most CPUs

### Code

First, the scalar version:

```cpp
// Scalar count: simple loop
size_t count_scalar(const float* a, size_t n, float thr) {
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] > thr) ++cnt;
    }
    return cnt;
}
```

Now the SIMD version:

```cpp
size_t count_simd(const float* a, size_t n, float thr) {
    using V = native_simd<float>;
    using M = native_mask<float>;
    constexpr size_t W = V::size();
    
    size_t total = 0;
    V vthr(thr);
    
    for (size_t i = 0; i + W <= n; i += W) {
        V v;
        v.copy_from(a + i, stdx::element_aligned);
        
        M mask = v > vthr;
        total += stdx::popcount(mask);
    }
    return total;
}
```

---

# Advanced Examples

### At a Glance (What It Is / What It Is Not)

- **Example 5 (`05_softmax.cpp`)**: stable softmax with SIMD passes; **not** a full ML inference pipeline.
- **Example 6 (`06_fma.cpp`)**: FMA microbenchmark comparing memory-bound and compute-bound kernels; **not** convolution.
- **Example 7 (`07_filter.cpp`)**: horizontal image blur with sliding window; **not** a full 2D stencil optimization study.
- **Example 8 (`08_conv1d.cpp`)**: 1D small-kernel convolution showing when explicit SIMD can lose; **not** a general-purpose convolution library.

## Example 5: Numerically Stable Softmax

**File**: [`05_softmax.cpp`](05_softmax.cpp)

Softmax is a fundamental operation in machine learning, but naive implementation suffers from numerical overflow. This example implements the stable version: `softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))`. By subtracting the maximum value before exponentiation, we ensure all arguments to exp() are non-positive, preventing overflow. The implementation uses three passes: first find the maximum, then compute exp() and sum, then normalize. This demonstrates several advanced SIMD concepts including horizontal reductions (`stdx::hmax()` for finding the maximum across all lanes), polynomial approximation for functions not provided by the standard library, and chaining multiple passes over the same data. The polynomial uses Horner's method for efficient evaluation.

### Softmax Formula

```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

Subtracting max prevents overflow in exp().

### Code

First, the scalar version:

```cpp
// Scalar softmax
void softmax_scalar(float* x, size_t n) {
    // Pass 1: find max
    float maxv = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < n; ++i) maxv = std::max(maxv, x[i]);
    
    // Pass 2: exp(x - max) and sum
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float z = std::exp(x[i] - maxv);
        x[i] = z;
        sum += z;
    }
    
    // Pass 3: normalize
    for (size_t i = 0; i < n; ++i) x[i] /= sum;
}
```

Now the SIMD version:

```cpp
// Polynomial approximation for exp(z) where z <= 0
template<class V>
V exp_poly(V z) {
    const V c1(1.f), c2(1.f), c3(.5f), c4(1.f/6.f);
    return c1 + z * (c2 + z * (c3 + z * c4));
}

void softmax_simd(float* x, size_t n) {
    using V = native_simd<float>;
    constexpr size_t W = V::size();
    
    // Pass 1: Find max
    float maxv = find_max_simd(x, n);
    V vmax(maxv);
    
    // Pass 2: exp(x - max) and sum
    V vsum = 0.f;
    for (size_t i = 0; i + W <= n; i += W) {
        V v; v.copy_from(x + i, stdx::element_aligned);
        v -= vmax;
        v = exp_poly(v);
        v.copy_to(x + i, stdx::element_aligned);
        vsum += v;
    }
    float sum = stdx::reduce(vsum);
    
    // Pass 3: Normalize
    V vdiv(sum);
    for (size_t i = 0; i + W <= n; i += W) {
        V v; v.copy_from(x + i, stdx::element_aligned);
        v /= vdiv;
        v.copy_to(x + i, stdx::element_aligned);
    }
}
```

---

## Example 6: FMA - Memory vs Compute Bound

**File**: [`06_fma.cpp`](06_fma.cpp)

Fused Multiply-Add (FMA) computes `a*b + c` in a single CPU instruction rather than two separate operations (multiply then add). This provides two benefits: higher precision because there's only one rounding instead of two, and higher throughput since it's a single instruction.

**This example teaches the most important lesson about SIMD performance**: speedup depends on whether your operation is **memory-bound** or **compute-bound**.

- **Memory-bound**: If you spend most time loading/storing data, wider SIMD registers don't help much
- **Compute-bound**: If you spend most time computing, SIMD can provide significant speedup

We demonstrate both cases side-by-side so you can see the difference.

### Key Concepts

1. **FMA**: `stdx::fma(a, b, c)` = a*b + c with single rounding
2. **Memory-bound operations** show ~1x speedup (bottleneck is memory bandwidth)
3. **Compute-bound operations** show 4-8x speedup (bottleneck is compute)

### Code

**Memory-bound** (y = a*b + c): 3 loads + 1 store per FMA → memory bottleneck

```cpp
void fma_membound_simd(const float* a, const float* b, const float* c, 
                        float* y, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    for (std::size_t i = 0; i + W <= n; i += W) {
        V va, vb, vc;
        va.copy_from(&a[i], stdx::element_aligned);
        vb.copy_from(&b[i], stdx::element_aligned);
        vc.copy_from(&c[i], stdx::element_aligned);
        V r = va * vb + vc;  // Compiler generates FMA
        r.copy_to(&y[i], stdx::element_aligned);
    }
}
```

**Compute-bound** (dot product): 2 loads per iteration, accumulate in registers

```cpp
float dot_simd(const float* a, const float* b, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    V acc = 0.f;
    for (std::size_t i = 0; i + W <= n; i += W) {
        V va, vb;
        va.copy_from(&a[i], stdx::element_aligned);
        vb.copy_from(&b[i], stdx::element_aligned);
        acc = acc + va * vb;  // FMA into accumulator
    }
    return stdx::reduce(acc);
}
```

**Key Lesson**: The same operation (FMA) shows 2.4x speedup when memory-bound, but 8.3x when compute-bound.

---

## Example 7: Image Processing (Horizontal Blur)

**File**: [`07_filter.cpp`](07_filter.cpp)

This example demonstrates a common image processing pattern: sliding window operations. We apply a simple horizontal blur where each output pixel is the average of itself and its two horizontal neighbors: `out[x] = (in[x-1] + in[x] + in[x+1]) / 3`.

The key insight here is **why speedup is modest**: to compute each output, we need to load 3 overlapping regions of the input. These overlapping loads mean we're loading more data than strictly necessary, making the operation memory-bound rather than compute-bound.

### Key Concepts

1. **Border handling**: Edge pixels use fewer neighbors (handled with scalar code)
2. **Overlapping loads**: 3 loads per SIMD output limits speedup
3. **Memory alignment**: 64-byte aligned for AVX-512

### Code

```cpp
void blur_horizontal_simd(const Image& in, Image& out) {
    using V = native_simd<float>;
    constexpr int W = V::size();
    const V inv3(1.f / 3.f);
    
    for (int y = 0; y < in.h; ++y) {
        const float* src = &in.at(y, 0);
        float* dst = &out.at(y, 0);
        
        // Left edge (scalar)
        dst[0] = (src[0] + src[1]) * 0.5f;
        
        // Main SIMD body
        int x = 1;
        for (; x + W < in.w; x += W) {
            V left, center, right;
            left.copy_from(src + x - 1, stdx::element_aligned);    // [x-1, x, x+1, ...]
            center.copy_from(src + x, stdx::element_aligned);      // [x, x+1, x+2, ...]
            right.copy_from(src + x + 1, stdx::element_aligned);   // [x+1, x+2, x+3, ...]
            
            V result = (left + center + right) * inv3;
            result.copy_to(dst + x, stdx::element_aligned);
        }
        
        // Right edge (scalar)
        dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}
```

**Why Speedup is Modest**: Each output requires 3 overlapping loads:

```
left:   [x-1, x,   x+1, x+2, ...]
center: [x,   x+1, x+2, x+3, ...]
right:  [x+1, x+2, x+3, x+4, ...]
```

Memory bandwidth becomes the bottleneck, not compute. For better performance, consider:
- **Separable filters**: Process rows and columns separately
- **Tiling**: Process small tiles that fit in cache

---

## Example 8: When SIMD Makes Things Worse

**File**: [`08_conv1d.cpp`](08_conv1d.cpp)

This example uses a 1D convolution with a small kernel to show that explicit SIMD can be slower than scalar code that the compiler auto-vectorizes well.

### Key Concepts

1. Small kernels can have too little work per output for explicit SIMD setup costs.
2. Overlapping memory accesses can make the kernel memory-bound.
3. Auto-vectorized scalar code can outperform manual SIMD in simple cases.

**Key Lesson**: benchmark before and after SIMD changes; do not assume explicit SIMD is always faster.

---

## Compiler Comparison: GCC vs Intel

We tested these examples with both GCC 13.2.0 and Intel oneAPI DPC++/C++ Compiler 2025.2. The results reveal important differences in how compilers handle `std::simd`.

### Benchmark Results (MareNostrum 5, Intel Xeon 8480+)

| Example | GCC Speedup | Intel Speedup | Notes |
|---------|-------------|---------------|-------|
| 01_add | 1.2x | 3.8x | Memory-bound |
| 02_sum | 4.9x | 20.6x | Reduction |
| 03_clamp | 3.5x | 9.1x | Masked operations |
| 04_count | 5.2x | 50.2x | Popcount |
| 05_softmax | 5.1x | 15.8x | Multi-pass algorithm |
| 06_fma (memory) | 1.0x | 2.4x | Memory-bound FMA |
| 06_fma (compute) | 2.2x | 8.3x | Compute-bound FMA |
| 07_filter | 2.3x | 6.4x | Image processing |

**Key finding**: Intel compiler generates significantly better code for `std::simd`, often 2-4x faster than GCC!

### Critical Issue: `stdx::fma()` on GCC

We discovered that **GCC's `stdx::fma()` generates terrible code**—it scalarizes the FMA operation:

```assembly
# GCC output for stdx::fma(va, vb, acc):
vfmadd132ss xmm0, xmm1, dword ptr [...]   # SCALAR fma (processes 1 float!)
vfmadd132ss xmm0, xmm1, dword ptr [...]   # 16 of these for AVX-512...
```

**Intel generates the expected vector instruction:**
```assembly
# Intel output for stdx::fma(va, vb, acc):
vfmadd231ps zmm0, zmm1, zmm2              # VECTOR fma (processes 16 floats!)
```

**Workaround for GCC**: Use `acc + va * vb` instead of `stdx::fma(va, vb, acc)`:

```cpp
// Cross-compiler solution:
#if defined(__INTEL_LLVM_COMPILER)
    acc = stdx::fma(va, vb, acc);  // Intel: uses vfmadd231ps
#else
    acc = acc + va * vb;           // GCC: compiler recognizes and uses vfmadd
#endif
```

### Disabling Auto-Vectorization

To get fair scalar baselines, we need to disable auto-vectorization. **This requires different pragmas per compiler:**

```cpp
// Cross-compiler pragma pattern
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void my_scalar_function(...) { ... }
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void my_scalar_function(...) { ... }
#pragma GCC pop_options
#endif
```

- **GCC**: `#pragma GCC optimize("no-tree-vectorize")` works
- **Intel/Clang**: Requires `__attribute__((optnone))` to truly disable vectorization

### Using the Intel Compiler on MareNostrum 5

```bash
module purge
module load intel/2025.2
module load gcc/13.2.0

# Compile with Intel
icpx -std=c++2b -O3 -march=native -fiopenmp-simd 01_add.cpp -o 01_add_intel
```

### Recommendations

1. **For best performance**: Use Intel compiler when available
2. **For portability**: Write code that works with both (use the `#if` pattern above)
3. **Always benchmark**: Compiler behavior varies significantly
4. **Check assembly**: Use `objdump -d` or `-S` flag to verify vectorization

---

## Key Takeaways

### When to Use std::simd

**Use explicit SIMD when**:
- Complex tail handling is needed
- Operations are compute-intensive (reductions, math functions)
- Specific alignment guarantees are required
- Compiler autovectorization fails

**Skip explicit SIMD when**:
- Simple operations that compiler auto-vectorizes well
- Memory-bandwidth limited (not compute-limited)
- Overhead exceeds benefits (small arrays, simple ops)

### Portability

One code base works across:
- **x86**: SSE, AVX, AVX-512
- **ARM**: NEON, SVE
- **RISC-V**: RVV

No `#ifdef` needed!

### Best Practices

1. **Profile first**: Use `perf` or similar tools
2. **Align data**: Use `posix_memalign` or aligned allocators
3. **Handle tails**: Process remaining elements with simple loop
4. **Test numerical correctness**: Floating-point may differ

## Appendix: Quick Reference

| Operation | Function |
|-----------|----------|
| Load | `v.copy_from(ptr, alignment)` |
| Store | `v.copy_to(ptr, alignment)` |
| Add | `a + b` |
| Multiply | `a * b` |
| FMA | `stdx::fma(a, b, c)` |
| Reduce sum | `stdx::reduce(v)` |
| Reduce max | `stdx::hmax(v)` |
| Reduce min | `stdx::hmin(v)` |
| Popcount | `stdx::popcount(mask)` |
| Conditional | `stdx::where(mask, v)` |
| Element count | `V::size()` |
