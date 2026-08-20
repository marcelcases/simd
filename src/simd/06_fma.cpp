#include "../simd_common.h"

using namespace simd_examples;

// Example 6: SIMD FMA workloads.
void fma_membound_simd(const float* a, const float* b, const float* c,
                       float* y, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V va, vb, vc;
        va.copy_from(a + i, stdx::element_aligned);
        vb.copy_from(b + i, stdx::element_aligned);
        vc.copy_from(c + i, stdx::element_aligned);
        (va * vb + vc).copy_to(y + i, stdx::element_aligned);
    }
    for (; i < n; ++i) y[i] = a[i] * b[i] + c[i];
}

float dot_simd(const float* a, const float* b, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    V accumulator = 0.f;
    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V va, vb;
        va.copy_from(a + i, stdx::element_aligned);
        vb.copy_from(b + i, stdx::element_aligned);
#if defined(__INTEL_LLVM_COMPILER)
        accumulator = stdx::fma(va, vb, accumulator);
#else
        accumulator = accumulator + va * vb;
#endif
    }

    float sum = stdx::reduce(accumulator);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

int main() {
    std::cout << "=== Example 6: SIMD FMA ===\n\n";

    const std::size_t N = 1ULL << 24;
    std::vector<float> a(N), b(N), c(N), y(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = distribution(rng);
        b[i] = distribution(rng);
        c[i] = distribution(rng);
    }

    const double memory_time = bench_ms([&]() {
        fma_membound_simd(a.data(), b.data(), c.data(), y.data(), N);
    });
    const double dot_time = bench_ms([&]() -> float {
        return dot_simd(a.data(), b.data(), N);
    });

    std::cout << "Array size: " << N << " elements\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Memory-bound y=a*b+c: " << memory_time << " ms\n";
    std::cout << "Compute-bound dot:    " << dot_time << " ms\n";
    return 0;
}
