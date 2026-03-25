# Modern C++ SIMD Programming with `std::simd`

A practical guide to writing portable, high-performance SIMD code using the C++26 standard library.

## Table of Contents

- [Introduction](#introduction)
- [What is SIMD?](#what-is-simd)
- [The std::simd Library](#the-stdsimd-library)
- [Building and Running Examples](#building-and-running-examples)
- [Basic Examples](#basic-examples)
  - [Example 1: Vector Add](#example-1-vector-add)
  - [Example 2: Sum Reduction](#example-2-sum-reduction)
  - [Example 3: Clamp with Masks](#example-3-clamp-with-masks)
  - [Example 4: Count with Popcount](#example-4-count-with-popcount)
- [Advanced Examples](#advanced-examples)
  - [Example 5: Numerically Stable Softmax](#example-5-numerically-stable-softmax)
  - [Example 6: 1D Convolution with FMA](#example-6-1d-convolution-with-fma)
  - [Example 7: 3x3 Image Filter](#example-7-3x3-image-filter)
- [Key Takeaways](#key-takeaways)
- [Appendix: Quick Reference](#appendix-quick-reference)

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
module load intel/2024.2
module load gcc/13.2.0
make
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

For testing on RISC-V with vector extensions (RVV), see **[RISCV_SETUP.md](RISCV_SETUP.md)** for detailed instructions on:
- QEMU user-mode emulation (quickest setup)
- Cross-compilation with Docker
- Spike ISA simulator (cycle-accurate)

Quick start with Docker:
```bash
# Install QEMU
brew install qemu  # macOS

# Compile in Docker container
docker run -it -v $(pwd):/work riscv64/ubuntu:22.04 bash
apt-get update && apt-get install -y g++-12
g++-12 -std=c++2b -O3 -march=rv64gcv -static 01_add.cpp -o 01_add.riscv

# Run with QEMU (outside container)
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./01_add.riscv
```

### Compiler Flags

The Makefile automatically detects your platform and uses appropriate flags:

| Platform | Compiler | Key Flags |
|----------|----------|-----------|
| Linux (AVX-512) | `g++` | `-march=native -mavx512f -mavx512vl` |
| macOS (ARM) | `g++-15` | `-mcpu=apple-m1` |
| macOS (Intel) | `g++-15` | `-march=native` |

Common flags for all platforms:
- `-std=c++2b`: C++26 working draft
- `-O3`: Maximum optimization
- `-fno-math-errno -fno-trapping-math`: Allow aggressive FP optimizations

### Important: Benchmarking SIMD Code

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

| Example | MN5 Scalar | MN5 SIMD | MN5 Speedup | M1 Pro Scalar | M1 Pro SIMD | M1 Pro Speedup |
|---------|------------|----------|-------------|---------------|-------------|----------------|
| 01_add | 20.6 ms | 17.5 ms | **1.2x** | 22.8 ms | 18.5 ms | **1.2x** |
| 02_sum | 11.3 ms | 2.1 ms | **5.3x** | 15.8 ms | 3.9 ms | **4.0x** |
| 03_clamp | 2.3 ms | 3.0 ms | **0.8x** | 5.2 ms | 1.9 ms | **2.7x** |
| 04_count | 11.3 ms | 2.3 ms | **5.0x** | 73.1 ms | 3.0 ms | **24.5x** |
| 05_softmax | 5.4 ms | 1.3 ms | **4.2x** | 4.1 ms | 1.6 ms | **2.7x** |
| 06_conv | 0.34 ms | 1.9 ms | **0.2x** | 0.15 ms | 0.14 ms | **1.1x** |
| 07_filter | 0.67 ms | 0.64 ms | **1.0x** | 0.38 ms | 0.49 ms | **0.8x** |

**Platform Details:**
- **MN5**: Intel Xeon 8480+ (Sapphire Rapids), AVX-512, 512-bit vectors (16 floats)
- **M1 Pro**: Apple M1 Pro, ARM NEON, 128-bit vectors (4 floats)

**Key Observations:**
- **Best speedups**: sum, count, softmax - compute-bound reductions benefit most from SIMD
- **Modest speedups**: add - memory-bandwidth limited on both platforms
- **Platform differences**: count shows 24x on M1 Pro vs 5x on MN5 (scalar baseline differs)
- **Overhead matters**: conv/filter show overhead can dominate for simple kernels

### SIMD Instruction Verification

To confirm SIMD is enabled, inspect the generated assembly. On macOS/ARM use `otool -tv` or generate assembly with `-S`:

```bash
g++-15 -std=c++2b -O3 -mcpu=apple-m1 -S 01_add.cpp -o 01_add.s
grep -E 'fadd\s+v[0-9]+\.4s' 01_add.s   # Look for NEON vector ops
```

On Linux/x86 use `objdump -d` or generate assembly:

```bash
g++ -std=c++2b -O3 -march=native -S 01_add.cpp -o 01_add.s
grep -E 'vaddps|vmulps' 01_add.s        # Look for AVX vector ops
```

#### Verified Instructions

| Example | ARM NEON (M1 Pro) | x86 AVX-512 (MN5) |
|---------|-------------------|-------------------|
| 01_add | `fadd v30.4s, v31.4s, v30.4s` | `vaddps zmm` |
| 02_sum | `fadd v31.4s, v31.4s, v4.4s` | `vaddps zmm` |
| 03_clamp | `fcmgt v3.4s, v30.4s, v31.4s` | `vcmpps` + `vblendps` |
| 04_count | `fcmgt` + `addp v31.2s` | `vcmpps` + `vpopcntd` |
| 05_softmax | `fmaxnm v31.4s, v31.4s, v4.4s` | `vmaxps zmm` |
| 06_conv | `fmla v31.4s, v27.4s, v30.4s` | `vfmadd231ps zmm` |
| 07_filter | `fadd` + `fdiv v17.4s` | `vaddps` + `vdivps` |

**Key ARM NEON patterns:**
- `.4s` suffix = 4 single-precision floats (128-bit register)
- `fadd/fmul` = vector arithmetic
- `fmla` = fused multiply-add (a*b + c in one instruction)
- `fcmgt` = compare greater than (returns mask)
- `bsl` = bitwise select (for masked operations)

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

### Expected Results

```
Scalar time: ~20 ms
SIMD time:   ~17 ms  
Speedup:     ~1.2x
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

### Expected Results

```
Scalar time: ~11 ms
SIMD time:   ~2 ms
Speedup:     ~5x
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

### Expected Results

```
Scalar time: ~2.3 ms
SIMD time:   ~3.0 ms
Speedup:     ~0.77x (slower!)
```

**Teaching Moment**: Not all operations benefit from SIMD! Simple conditional updates may be slower due to the overhead of masked operations. Profile your code!

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

### Expected Results

```
Scalar time: ~11 ms
SIMD time:   ~2 ms
Speedup:     ~5x

Results:
  Scalar count: 8390381
  SIMD count:   8390381
  Match:        YES
```

---

# Advanced Examples

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

### Expected Results

```
Scalar time: ~5.4 ms
SIMD time:   ~1.3 ms
Speedup:     ~4x
```

---

## Example 6: 1D Convolution with FMA

**File**: [`06_conv.cpp`](06_conv.cpp)

Fused Multiply-Add (FMA) computes `a*b + c` in a single CPU instruction rather than two separate operations (multiply then add). This provides two benefits: higher precision because there's only one rounding instead of two, and higher throughput since it's a single instruction. The `stdx::fma()` function makes this available in portable C++. This example applies a 1D convolution kernel to an input signal—each output is the weighted sum of neighboring inputs. For small kernels like this 3-element filter, the overhead of setting up SIMD operations can outweigh the benefits, which is why you might see the SIMD version run slower than expected. In practice, FMA truly shines with larger kernels or in compute-heavy scenarios like neural network inference.

### Key Concepts

1. **FMA**: `stdx::fma(a, b, c)` = a*b + c with single rounding
2. **Benefits**: Higher precision (one rounding) and throughput
3. **Applications**: DSP, ML, image processing

### Code

First, the scalar version:

```cpp
// Simple 1D convolution: y[i] = sum(x[i+j] * k[j]) for j in [0, K)
template<int K>
void conv1d_scalar(const float* x, const float* k, float* y, size_t n) {
    size_t end = n - K + 1;
    for (size_t i = 0; i < end; ++i) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            s += x[i + j] * k[j];
        }
        y[i] = s;
    }
}
```

Now the SIMD version:

```cpp
template<int K>
void conv1d_simd(const float* x, const float* k, float* y, size_t n) {
    using V = native_simd<float>;
    constexpr size_t W = V::size();
    
    for (size_t i = 0; i + W <= n - K + 1; i += W) {
        V acc = 0.f;
        
        for (int j = 0; j < K; ++j) {
            V kv(k[j]);
            V xv;
            xv.copy_from(x + i + j, stdx::element_aligned);
            acc = stdx::fma(xv, kv, acc);  // acc += xv * kv
        }
        
        acc.copy_to(y + i, stdx::element_aligned);
    }
}
```

### Expected Results

```
Scalar time:  ~0.34 ms
SIMD time:    1.93 ms
Speedup:      ~0.17x (slower!)
Max diff:     0
```

---

## Example 7: 3x3 Image Filter

**File**: [`07_filter.cpp`](07_filter.cpp)

This example demonstrates applying a 3x3 box filter (horizontal pass) to an image, a common operation in computer vision for smoothing or blurring. The key challenge is handling image borders: pixels at the edges have fewer neighbors than interior pixels, so they require special treatment—we handle these with scalar code. This example also introduces stride access patterns: images are stored row-by-row, and rows may have padding for alignment (stride differs from width). The kernel weights are pre-loaded into SIMD vectors before the loop to avoid redundant loads. Finally, we use 64-byte aligned memory allocation (via `posix_memalign`) which is optimal for AVX-512 loads and stores.

### Key Concepts

1. **Border handling**: Edge pixels use fewer neighbors
2. **Stride access**: Image rows may have padding
3. **Memory alignment**: 64-byte aligned for AVX-512

### Code

First, the scalar version:

```cpp
// Horizontal pass of 3x3 box filter
void box3x3_horizontal_scalar(const Image& in, Image& out) {
    for (int y = 0; y < in.h; ++y) {
        const float* src = in.data + y * in.stride;
        float* dst = out.data + y * out.stride;
        
        // Left edge: only 2 neighbors available
        if (in.w >= 1) dst[0] = (src[0] + src[1]) * 0.5f;
        if (in.w >= 2) dst[1] = (src[0] + src[1] + src[2]) / 3.f;
        
        // Main body: 3 neighbors
        for (int x = 2; x < in.w - 2; ++x) {
            dst[x] = (src[x-1] + src[x] + src[x+1]) / 3.f;
        }
        
        // Right edge
        if (in.w >= 2) dst[in.w - 2] = (src[in.w - 3] + src[in.w - 2] + src[in.w - 1]) / 3.f;
        if (in.w >= 1) dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}
```

Now the SIMD version:

```cpp
void box3x3_horizontal_simd(const Image& in, Image& out) {
    using V = native_simd<float>;
    constexpr size_t W = V::size();
    
    const float k[3] = {1.f/3, 1.f/3, 1.f/3};
    const V k0(k[0]), k1(k[1]), k2(k[2]);
    
    for (int y = 0; y < in.h; ++y) {
        const float* src = in.data + y * in.stride;
        float* dst = out.data + y * out.stride;
        
        // Left edge (scalar)
        if (in.w >= 1) dst[0] = (src[0] + src[1]) * 0.5f;
        
        // Main body (SIMD)
        for (int x = 2; x + W + 1 <= in.w; x += W) {
            V a, b, c;
            a.copy_from(src + x - 1, stdx::element_aligned);
            b.copy_from(src + x,     stdx::element_aligned);
            c.copy_from(src + x + 1, stdx::element_aligned);
            
            V r = a * k0 + b * k1 + c * k2;
            r.copy_to(dst + x, stdx::element_aligned);
        }
        
        // Right edge (scalar)
        // ...
    }
}
```

### Expected Results

```
Scalar time: 0.67 ms
SIMD time:   0.64 ms
Speedup:     ~1.04x
```

---

## Key Takeaways

### When to Use std::simd

✅ **Use explicit SIMD when**:
- Complex tail handling is needed
- Operations are compute-intensive (reductions, math functions)
- Specific alignment guarantees are required
- Compiler autovectorization fails

❌ **Skip explicit SIMD when**:
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

### Further Reading

- [cppreference: std::simd](https://en.cppreference.com/w/cpp/experimental/simd)
- [GCC libstdc++ SIMD documentation](https://gcc.gnu.org/onlinedocs/libstdc++/manual/simd_support.html)
- [C++26 Proposal P1928](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p1928r3.html)

---

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
