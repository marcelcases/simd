#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 8: When SIMD Makes Things WORSE - 1D Convolution
// ============================================================================
// This example demonstrates a case where explicit SIMD can actually be SLOWER
// than letting the compiler auto-vectorize scalar code.
//
// Why? Several factors work against us:
// 1. Small kernel size (K=3) means little compute per output
// 2. Overlapping memory accesses (each output needs K consecutive inputs)
// 3. The scalar loop is simple enough that the compiler optimizes it well
// 4. SIMD overhead (setup, tail handling) dominates for simple operations
//
// LESSON: Always benchmark! SIMD is not a magic speedup button.

// ============ 1D Convolution: y[i] = sum(x[i+j] * k[j]) for j in [0, K) ============

// Scalar convolution - let the compiler do its thing
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
#endif
void conv1d_scalar_novec(const float* x, const float* k, float* y, 
                          std::size_t n, int K) {
    std::size_t end = n - K + 1;
    for (std::size_t i = 0; i < end; ++i) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            s += x[i + j] * k[j];
        }
        y[i] = s;
    }
}
#if !defined(__INTEL_LLVM_COMPILER) && !defined(__clang__)
#pragma GCC pop_options
#endif

// Scalar convolution - auto-vectorized by compiler
void conv1d_scalar_autovec(const float* __restrict x, const float* __restrict k, 
                            float* __restrict y, std::size_t n, int K) {
    std::size_t end = n - K + 1;
    for (std::size_t i = 0; i < end; ++i) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            s += x[i + j] * k[j];
        }
        y[i] = s;
    }
}

// Explicit SIMD convolution using FMA
template<int K>
void conv1d_simd(const float* x, const float* k, float* y, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    std::size_t end = n - K + 1;
    std::size_t i = 0;
    
    // Main SIMD loop
    for (; i + W <= end; i += W) {
        V acc = 0.f;
        
        for (int j = 0; j < K; ++j) {
            V kv(k[j]);  // Broadcast kernel value to all lanes
            V xv;
            xv.copy_from(x + i + j, stdx::element_aligned);
            // Use FMA: acc = acc + xv * kv
#if defined(__INTEL_LLVM_COMPILER)
            acc = stdx::fma(xv, kv, acc);
#else
            acc = acc + xv * kv;  // GCC generates better code this way
#endif
        }
        
        acc.copy_to(y + i, stdx::element_aligned);
    }
    
    // Tail: scalar fallback
    for (; i < end; ++i) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            s += x[i + j] * k[j];
        }
        y[i] = s;
    }
}

// Verify results match
float max_diff(const float* a, const float* b, std::size_t n) {
    float maxd = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        maxd = std::max(maxd, std::abs(a[i] - b[i]));
    }
    return maxd;
}

int main() {
    std::cout << "=== Example 8: When SIMD Makes Things WORSE ===\n\n";
    
    const std::size_t N = 1ULL << 20;  // 1M elements
    constexpr int K = 3;  // Small kernel
    
    std::vector<float> x(N), y_novec(N), y_autovec(N), y_simd(N);
    float kernel[K] = {0.25f, 0.5f, 0.25f};  // Simple smoothing kernel
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = dist(rng);
    }
    
    std::cout << "1D Convolution: y[i] = sum(x[i+j] * k[j]) for j in [0, " << K << ")\n";
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Kernel size: " << K << " (small!)\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    
    // Benchmark scalar (no vectorization)
    double t_novec = bench_ms([&]() {
        conv1d_scalar_novec(x.data(), kernel, y_novec.data(), N, K);
    });
    
    // Benchmark scalar (auto-vectorized)
    double t_autovec = bench_ms([&]() {
        conv1d_scalar_autovec(x.data(), kernel, y_autovec.data(), N, K);
    });
    
    // Benchmark explicit SIMD
    double t_simd = bench_ms([&]() {
        conv1d_simd<K>(x.data(), kernel, y_simd.data(), N);
    });
    
    // Verify correctness
    float diff_autovec = max_diff(y_novec.data(), y_autovec.data(), N - K + 1);
    float diff_simd = max_diff(y_novec.data(), y_simd.data(), N - K + 1);
    
    std::cout << "RESULTS:\n";
    std::cout << "--------\n";
    std::cout << "Scalar (no vectorization): " << t_novec << " ms\n";
    std::cout << "Scalar (auto-vectorized):  " << t_autovec << " ms";
    if (t_autovec < t_novec) {
        std::cout << " (" << (t_novec / t_autovec) << "x faster)";
    }
    std::cout << "\n";
    std::cout << "Explicit SIMD:             " << t_simd << " ms";
    if (t_simd < t_novec) {
        std::cout << " (" << (t_novec / t_simd) << "x faster)";
    } else {
        std::cout << " (" << (t_simd / t_novec) << "x SLOWER!)";
    }
    std::cout << "\n\n";
    
    std::cout << "Correctness check:\n";
    std::cout << "  Auto-vec vs no-vec max diff: " << diff_autovec << "\n";
    std::cout << "  SIMD vs no-vec max diff:     " << diff_simd << "\n\n";
    
    std::cout << "WHY IS EXPLICIT SIMD SLOWER?\n";
    std::cout << "----------------------------\n";
    std::cout << "1. SMALL KERNEL: Only " << K << " multiplies per output = little compute\n";
    std::cout << "2. MEMORY BOUND: Each output loads " << K << " overlapping values\n";
    std::cout << "3. COMPILER IS SMART: Auto-vectorization often beats hand-written SIMD\n";
    std::cout << "4. OVERHEAD: SIMD setup/tail handling costs dominate\n\n";
    
    std::cout << "LESSON: Don't assume SIMD = faster. Always benchmark!\n";
    std::cout << "        The compiler's auto-vectorizer is often good enough.\n";
    std::cout << "        Explicit SIMD shines for COMPLEX operations the compiler\n";
    std::cout << "        can't optimize well (reductions, masked ops, etc.)\n";
    
    return 0;
}
