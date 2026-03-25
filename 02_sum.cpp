#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 2: Sum Reduction
// ============================================================================
// This demonstrates horizontal reduction - summing all elements in a SIMD register.
// This is harder for compilers to auto-vectorize, so explicit SIMD shows clear gains.
//
// The key operation is stdx::reduce() which performs:
//   result = v[0] + v[1] + ... + v[W-1]

// Scalar sum: simple loop
float sum_scalar(const float* a, std::size_t n) {
    float s = 0.f;
    for (std::size_t i = 0; i < n; ++i) s += a[i];
    return s;
}

// SIMD sum: accumulate in SIMD registers, then reduce
float sum_simd(const float* a, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    // Start with zero vector
    V acc = 0.f;
    std::size_t i = 0;
    
    // Main loop: accumulate W elements at a time
    for (; i + W <= n; i += W) {
        V v;
        v.copy_from(a + i, stdx::element_aligned);
        acc += v;  // Element-wise addition
    }
    
    // Horizontal reduction: sum all elements in the SIMD register
    float s = stdx::reduce(acc);
    
    // Tail: handle remaining elements
    for (; i < n; ++i) s += a[i];
    
    return s;
}

int main() {
    std::cout << "=== Example 2: Sum Reduction ===\n\n";
    
    const std::size_t N = 1ULL << 24;  // 16 million
    std::vector<float> a(N);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& x : a) x = dist(rng);
    
    // Benchmark scalar
    double t_scalar = bench_ms([&]() -> float {
        return sum_scalar(a.data(), N);
    });
    
    // Benchmark SIMD
    double t_simd = bench_ms([&]() -> float {
        return sum_simd(a.data(), N);
    });
    
    float chk_scalar = sum_scalar(a.data(), N);
    float chk_simd = sum_simd(a.data(), N);
    
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n\n";
    
    std::cout << "IMPORTANT: Note that sums match = NO\n";
    std::cout << "This is due to floating-point non-associativity:\n";
    std::cout << "  (a + b) + c  !=  a + (b + c) in floating point\n";
    std::cout << "The SIMD version adds elements in a different order,\n";
    std::cout << "causing different rounding errors.\n";
    std::cout << "For exact reproducibility, use Kahan summation or\n";
    std::cout << "enforce a specific order.\n\n";
    
    std::cout << "Scalar sum: " << chk_scalar << "\n";
    std::cout << "SIMD sum:   " << chk_simd << "\n";
    std::cout << "Difference: " << std::abs(chk_scalar - chk_simd) << "\n";
    
    return 0;
}