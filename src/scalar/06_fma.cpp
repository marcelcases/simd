#include "../common.h"

using namespace simd_examples;

// Example 6: scalar FMA workloads.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void fma_membound_scalar(const float* a, const float* b, const float* c,
                         float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = a[i] * b[i] + c[i];
}

__attribute__((noinline, optnone))
float dot_scalar(const float* a, const float* b, std::size_t n) {
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void fma_membound_scalar(const float* a, const float* b, const float* c,
                         float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = a[i] * b[i] + c[i];
}

float dot_scalar(const float* a, const float* b, std::size_t n) {
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 6: Scalar FMA ===\n\n";

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
        fma_membound_scalar(a.data(), b.data(), c.data(), y.data(), N);
    });
    const double dot_time = bench_ms([&]() -> float {
        return dot_scalar(a.data(), b.data(), N);
    });

    std::cout << "Array size: " << N << " elements\n\n";
    std::cout << "Memory-bound y=a*b+c: " << memory_time << " ms\n";
    std::cout << "Compute-bound dot:    " << dot_time << " ms\n";
    return 0;
}
