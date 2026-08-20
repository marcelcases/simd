#include "../common.h"

using namespace simd_examples;

// Example 2: scalar sum reduction.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
float sum_scalar(const float* a, std::size_t n) {
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i];
    return sum;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
float sum_scalar(const float* a, std::size_t n) {
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i];
    return sum;
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 2: Scalar Sum Reduction ===\n\n";

    const std::size_t N = 1ULL << 24;
    std::vector<float> a(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& x : a) x = dist(rng);

    const double time = bench_ms([&]() -> float {
        return sum_scalar(a.data(), N);
    });
    const float sum = sum_scalar(a.data(), N);

    std::cout << "Array size:  " << N << " elements\n";
    std::cout << "Scalar time: " << time << " ms\n";
    std::cout << "Scalar sum:  " << sum << "\n";
    return 0;
}
