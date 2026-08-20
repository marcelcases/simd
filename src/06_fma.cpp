#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 6: FMA (Fused Multiply-Add)
// ============================================================================
// FMA computes a*b + c in a single instruction with:
//   - Single rounding (better precision than separate multiply + add)
//   - Higher throughput (one instruction instead of two)
//
// Key insight: FMA shows speedup when the operation is COMPUTE-BOUND.
// If memory bandwidth is the bottleneck, SIMD won't help much.
//
// This example demonstrates BOTH cases:
// 1. Memory-bound: y[i] = a[i] * b[i] + c[i]  (3 loads, 1 store per FMA)
// 2. Compute-bound: accumulate dot product (1 load, no store per iteration)

// ============ Memory-bound example: y = a*b + c ============

// Prevent auto-vectorization for fair comparison
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void fma_membound_scalar(const float* a, const float* b, const float* c, 
                          float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = a[i] * b[i] + c[i];
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void fma_membound_scalar(const float* a, const float* b, const float* c, 
                          float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = a[i] * b[i] + c[i];
    }
}
#pragma GCC pop_options
#endif

void fma_membound_simd(const float* a, const float* b, const float* c, 
                        float* y, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    std::size_t i = 0;
    
    for (; i + W <= n; i += W) {
        V va, vb, vc;
        va.copy_from(&a[i], stdx::element_aligned);
        vb.copy_from(&b[i], stdx::element_aligned);
        vc.copy_from(&c[i], stdx::element_aligned);
        V r = va * vb + vc;  // Compiler generates FMA
        r.copy_to(&y[i], stdx::element_aligned);
    }
    for (; i < n; ++i) y[i] = a[i] * b[i] + c[i];
}

// ============ Compute-bound example: dot product ============

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
// Intel/Clang: use optnone to truly disable all optimizations for fair comparison
__attribute__((noinline, optnone))
float dot_scalar(const float* a, const float* b, std::size_t n) {
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
float dot_scalar(const float* a, const float* b, std::size_t n) {
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];  // FMA: sum = sum + a[i]*b[i]
    }
    return sum;
}
#pragma GCC pop_options
#endif

float dot_simd(const float* a, const float* b, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    V acc = 0.f;
    std::size_t i = 0;
    
    for (; i + W <= n; i += W) {
        V va, vb;
        va.copy_from(&a[i], stdx::element_aligned);
        vb.copy_from(&b[i], stdx::element_aligned);
        // Note: Intel compiler generates better code with stdx::fma(),
        // while GCC generates better code with acc + va * vb.
        // Here we use the form that works well with both compilers.
#if defined(__INTEL_LLVM_COMPILER)
        acc = stdx::fma(va, vb, acc);
#else
        acc = acc + va * vb;
#endif
    }
    
    float sum = stdx::reduce(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

int main() {
    std::cout << "=== Example 6: FMA (Fused Multiply-Add) ===\n\n";
    
    const std::size_t N = 1ULL << 24;
    
    std::vector<float> a(N), b(N), c(N), y(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        c[i] = dist(rng);
    }
    
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    
    // Test 1: Memory-bound (y = a*b + c)
    std::cout << "TEST 1: Memory-bound (y = a*b + c)\n";
    std::cout << "  3 loads + 1 store per FMA = memory bottleneck\n";
    
    double t1_scalar = bench_ms([&]() {
        fma_membound_scalar(a.data(), b.data(), c.data(), y.data(), N);
    });
    double t1_simd = bench_ms([&]() {
        fma_membound_simd(a.data(), b.data(), c.data(), y.data(), N);
    });
    
    std::cout << "  Scalar: " << t1_scalar << " ms\n";
    std::cout << "  SIMD:   " << t1_simd << " ms\n";
    std::cout << "  Speedup: " << (t1_scalar / t1_simd) << "x\n\n";
    
    // Test 2: Compute-bound (dot product)
    std::cout << "TEST 2: Compute-bound (dot product)\n";
    std::cout << "  2 loads per FMA, accumulating in registers\n";
    
    double t2_scalar = bench_ms([&]() -> float {
        return dot_scalar(a.data(), b.data(), N);
    });
    double t2_simd = bench_ms([&]() -> float {
        return dot_simd(a.data(), b.data(), N);
    });
    
    std::cout << "  Scalar: " << t2_scalar << " ms\n";
    std::cout << "  SIMD:   " << t2_simd << " ms\n";
    std::cout << "  Speedup: " << (t2_scalar / t2_simd) << "x\n\n";
    
    std::cout << "KEY LESSON:\n";
    std::cout << "-----------\n";
    std::cout << "SIMD shines when the operation is COMPUTE-BOUND.\n";
    std::cout << "If you're waiting on memory, wider registers don't help.\n";
    std::cout << "\nRule of thumb: more computation per byte loaded = better speedup.\n";
    
    return 0;
}
