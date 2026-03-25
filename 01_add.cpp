#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 1: Basic Vector Add
// ============================================================================
// This example demonstrates the fundamental pattern of SIMD programming:
// - Load W elements from memory into a SIMD register
// - Perform element-wise operation (add in this case)
// - Store W elements back to memory
// - Handle any remaining elements (tail) with a simple loop

// Scalar (non-vectorized) baseline - one element at a time
// The pragma prevents the compiler from auto-vectorizing this function
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void add_scalar(float* __restrict dst, const float* __restrict src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}
#pragma GCC pop_options

// SIMD version using std::simd
// Works on x86 (SSE/AVX/AVX-512), ARM (NEON), and RISC-V (RVV) without changes
void add_simd(float* __restrict dst, const float* __restrict src, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();  // 16 on AVX-512, 8 on AVX2, 4 on NEON
    
    std::size_t i = 0;
    
    // Main vectorized loop: process W elements per iteration
    for (; i + W <= n; i += W) {
        V a; 
        a.copy_from(dst + i, stdx::element_aligned);
        V b; 
        b.copy_from(src + i, stdx::element_aligned);
        
        a += b;  // Element-wise addition - single instruction!
        
        a.copy_to(dst + i, stdx::element_aligned);
    }
    
    // Tail: handle remaining elements one at a time
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

int main() {
    std::cout << "=== Example 1: Basic Vector Add ===\n\n";
    
    const std::size_t N = 1ULL << 24;  // 16 million elements
    std::vector<float> src(N), dst_scalar(N), dst_simd(N);
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& x : src) x = dist(rng);
    
    // Benchmark scalar version
    dst_scalar = src;
    double t_scalar = bench_ms([&]() -> float {
        add_scalar(dst_scalar.data(), src.data(), N);
        return checksum(dst_scalar.begin(), dst_scalar.end());
    });
    
    // Benchmark SIMD version
    dst_simd = src;
    double t_simd = bench_ms([&]() -> float {
        add_simd(dst_simd.data(), src.data(), N);
        return checksum(dst_simd.begin(), dst_simd.end());
    });
    
    // Verify correctness
    float max_diff = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(dst_scalar[i] - dst_simd[i]));
    }
    
    std::cout << "Array size: " << N << " elements (" << (N * 4 / 1024 / 1024) << " MB)\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n";
    std::cout << "Max diff:    " << max_diff << "\n\n";
    
    // Key insight: For simple operations like add, the compiler often 
    // auto-vectorizes anyway. std::simd becomes essential when:
    // - Operations are complex (e.g., reductions)
    // - Tail handling is needed
    // - Specific alignment guarantees are required
    
    std::cout << "KEY INSIGHT: This operation is memory-bandwidth limited,\n";
    std::cout << "so speedup is modest. See sum reduction for compute-limited example.\n";
    
    return 0;
}