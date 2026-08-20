#include "common.h"
#include <limits>

using namespace simd_examples;

// ============================================================================
// Example 5: Numerically Stable Softmax
// ============================================================================
// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// This is numerically stable because subtracting the max prevents overflow.
// The algorithm has 3 passes:
//   1. Find maximum value
//   2. Compute exp(x - max) and sum
//   3. Normalize by sum
//
// This demonstrates multi-pass algorithms and polynomial approximation.

// Polynomial approximation for exp(z) where z <= 0
// Using Horner's method: 1 + z + z^2/2! + z^3/3! + ...
// This is needed because std::exp may not have SIMD overloads in TS v2
template<class V>
V exp_poly(V z) noexcept {
    const V c1(1.f), c2(1.f), c3(.5f), c4(1.f/6.f), c5(1.f/24.f), c6(1.f/120.f);
    return c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * c6))));
}

// Pass 1: Find maximum (for numerical stability)
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
float find_max_scalar(const float* x, std::size_t n) {
    float m = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, x[i]);
    return m;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
float find_max_scalar(const float* x, std::size_t n) {
    float m = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, x[i]);
    return m;
}
#pragma GCC pop_options
#endif

float find_max_simd(const float* x, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    V vmax = std::numeric_limits<float>::lowest();
    std::size_t i = 0;
    
    for (; i + W <= n; i += W) {
        V v; v.copy_from(x + i, stdx::element_aligned);
        vmax = stdx::max(vmax, v);
    }
    
    float m = stdx::hmax(vmax);
    for (; i < n; ++i) m = std::max(m, x[i]);
    return m;
}

// Scalar softmax
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void softmax_scalar(float* x, std::size_t n) {
    float maxv = find_max_scalar(x, n);
    float sum = 0.f;
    
    for (std::size_t i = 0; i < n; ++i) {
        float z = std::exp(x[i] - maxv);
        x[i] = z;
        sum += z;
    }
    
    for (std::size_t i = 0; i < n; ++i) x[i] /= sum;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void softmax_scalar(float* x, std::size_t n) {
    float maxv = find_max_scalar(x, n);
    float sum = 0.f;
    
    for (std::size_t i = 0; i < n; ++i) {
        float z = std::exp(x[i] - maxv);
        x[i] = z;
        sum += z;
    }
    
    for (std::size_t i = 0; i < n; ++i) x[i] /= sum;
}
#pragma GCC pop_options
#endif

// SIMD softmax
void softmax_simd(float* x, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    // Pass 1: find max
    float maxv = find_max_simd(x, n);
    V vmax(maxv);
    
    // Pass 2: exp(x - max) and sum
    V vsum = 0.f;
    std::size_t i = 0;
    
    for (; i + W <= n; i += W) {
        V v; v.copy_from(x + i, stdx::element_aligned);
        v -= vmax;
        v = exp_poly(v);  // SIMD polynomial
        v.copy_to(x + i, stdx::element_aligned);
        vsum += v;
    }
    
    float sum = stdx::reduce(vsum);
    for (; i < n; ++i) {
        float z = std::exp(x[i] - maxv);
        x[i] = z;
        sum += z;
    }
    
    // Pass 3: normalize
    V vsumv(sum);
    i = 0;
    for (; i + W <= n; i += W) {
        V v; v.copy_from(x + i, stdx::element_aligned);
        v /= vsumv;
        v.copy_to(x + i, stdx::element_aligned);
    }
    for (; i < n; ++i) x[i] /= sum;
}

int main() {
    std::cout << "=== Example 5: Numerically Stable Softmax ===\n\n";
    
    const std::size_t N = 1ULL << 20;  // 1 million
    std::vector<float> x_scalar(N), x_simd(N);
    
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);  // typical logits
    for (auto& x : x_scalar) x = nd(rng);
    x_simd = x_scalar;
    
    double t_scalar = bench_ms([&]() -> float {
        softmax_scalar(x_scalar.data(), N);
        return checksum(x_scalar.begin(), x_scalar.end());
    });
    
    double t_simd = bench_ms([&]() -> float {
        softmax_simd(x_simd.data(), N);
        return checksum(x_simd.begin(), x_simd.end());
    });
    
    float max_diff = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(x_scalar[i] - x_simd[i]));
    }
    
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n";
    std::cout << "Max diff:    " << max_diff << "\n\n";
    
    std::cout << "First 5 values (should sum to ~1.0):\n";
    std::cout << "Scalar: ";
    for (std::size_t i = 0; i < 5; ++i) std::cout << x_scalar[i] << " ";
    std::cout << "\nSIMD:   ";
    for (std::size_t i = 0; i < 5; ++i) std::cout << x_simd[i] << " ";
    std::cout << "\n\n";
    
    std::cout << "Sum of all values (should be ~1.0):\n";
    std::cout << "Scalar: " << checksum(x_scalar.begin(), x_scalar.end()) << "\n";
    std::cout << "SIMD:   " << checksum(x_simd.begin(), x_simd.end()) << "\n";
    
    return 0;
}