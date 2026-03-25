#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 4: Count with Popcount
// ============================================================================
// This demonstrates counting elements satisfying a condition using SIMD masks
// and popcount. It's useful for histogram computation, threshold counting, etc.
//
// stdx::popcount returns the number of true elements in a mask
// This is implemented efficiently using SIMD instructions on most CPUs

// Scalar count: simple loop
// The pragma prevents auto-vectorization to show true scalar performance
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
std::size_t count_scalar(const float* a, std::size_t n, float thr) {
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] > thr) ++cnt;
    }
    return cnt;
}
#pragma GCC pop_options

// SIMD count using popcount
std::size_t count_simd(const float* a, std::size_t n, float thr) {
    using V = native_simd<float>;
    using M = native_mask<float>;
    constexpr std::size_t W = V::size();
    
    std::size_t total = 0;
    V vthr(thr);
    std::size_t i = 0;
    
    for (; i + W <= n; i += W) {
        V v;
        v.copy_from(a + i, stdx::element_aligned);
        
        M mask = v > vthr;
        total += stdx::popcount(mask);
    }
    
    // Tail
    for (; i < n; ++i) {
        if (a[i] > thr) ++total;
    }
    
    return total;
}

int main() {
    std::cout << "=== Example 4: Count with Popcount ===\n\n";
    
    const std::size_t N = 1ULL << 24;
    const float threshold = 0.0f;  // count elements > 0
    
    std::vector<float> a(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : a) x = dist(rng);
    
    double t_scalar = bench_ms([&]() -> std::size_t {
        return count_scalar(a.data(), N, threshold);
    });
    
    double t_simd = bench_ms([&]() -> std::size_t {
        return count_simd(a.data(), N, threshold);
    });
    
    std::size_t cnt_scalar = count_scalar(a.data(), N, threshold);
    std::size_t cnt_simd = count_simd(a.data(), N, threshold);
    
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Threshold:  " << threshold << " (count elements > threshold)\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n\n";
    
    std::cout << "Results:\n";
    std::cout << "  Scalar count: " << cnt_scalar << "\n";
    std::cout << "  SIMD count:   " << cnt_simd << "\n";
    std::cout << "  Match:        " << (cnt_scalar == cnt_simd ? "YES" : "NO") << "\n\n";
    
    std::cout << "HOW POPCOUNT WORKS:\n";
    std::cout << "-------------------\n";
    std::cout << "M mask = v > vthr;         // Element-wise comparison\n";
    std::cout << "total += stdx::popcount(mask); // Count true bits\n";
    std::cout << "\nOn x86 this uses VPOPCNTD (AVX-512) or equivalent.\n";
    std::cout << "Much faster than loop with branches!\n";
    
    return 0;
}