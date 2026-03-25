#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 3: Clamp with Masks
// ============================================================================
// This demonstrates masked operations - applying operations only to elements
// that satisfy a condition. This is essential for:
// - Conditional updates without branches
// - Efficient tail handling
// - Filtering operations
//
// Key functions:
// - stdx::where(mask, value) - returns a masked view for conditional updates
// - mask operations are element-wise comparisons

// Scalar clamp: simple loop with if
void clamp_scalar(float* a, std::size_t n, float hi) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] > hi) a[i] = hi;
    }
}

// SIMD clamp using masks
void clamp_simd(float* a, std::size_t n, float hi) {
    using V = native_simd<float>;
    using M = native_mask<float>;
    constexpr std::size_t W = V::size();
    
    V vhi(hi);
    std::size_t i = 0;
    
    for (; i + W <= n; i += W) {
        V v;
        v.copy_from(a + i, stdx::element_aligned);
        
        // Create mask: elements where v > hi
        M mask = v > vhi;
        
        // Conditional update: only elements where mask is true get set to hi
        stdx::where(mask, v) = vhi;
        
        v.copy_to(a + i, stdx::element_aligned);
    }
    
    // Tail
    for (; i < n; ++i) {
        if (a[i] > hi) a[i] = hi;
    }
}

int main() {
    std::cout << "=== Example 3: Clamp with Masks ===\n\n";
    
    const std::size_t N = 1ULL << 24;
    std::vector<float> a_scalar(N), a_simd(N);
    const float hi = 0.5f;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : a_scalar) x = dist(rng);
    a_simd = a_scalar;
    
    double t_scalar = bench_ms([&]() {
        clamp_scalar(a_scalar.data(), N, hi);
    });
    
    double t_simd = bench_ms([&]() {
        clamp_simd(a_simd.data(), N, hi);
    });
    
    // Verify
    int diff_count = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (a_scalar[i] != a_simd[i]) ++diff_count;
    }
    
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Clamp upper bound: " << hi << "\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n";
    std::cout << "Different elements: " << diff_count << "\n\n";
    
    // Show how masks work
    std::cout << "HOW MASKS WORK:\n";
    std::cout << "---------------\n";
    std::cout << "M mask = v > vhi;           // Creates element-wise mask\n";
    std::cout << "stdx::where(mask, v) = vhi; // Updates only where mask is true\n";
    std::cout << "\nThis compiles to predicated instructions on CPUs that support them:\n";
    std::cout << "  - x86: blendvps / blendvpd\n";
    std::cout << "  - ARM: bitwise select\n";
    std::cout << "  - No branches = better branch prediction\n";
    
    return 0;
}