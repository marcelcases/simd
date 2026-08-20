#include "../common.h"

using namespace simd_examples;

// Example 3: scalar clamp with a branch.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void clamp_scalar(float* a, std::size_t n, float hi) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] > hi) a[i] = hi;
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void clamp_scalar(float* a, std::size_t n, float hi) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] > hi) a[i] = hi;
    }
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 3: Scalar Clamp ===\n\n";

    const std::size_t N = 1ULL << 24;
    const float hi = 0.5f;
    std::vector<float> values(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : values) x = dist(rng);

    const double time = bench_ms([&]() {
        clamp_scalar(values.data(), N, hi);
    });

    std::cout << "Array size:   " << N << " elements\n";
    std::cout << "Upper bound:  " << hi << "\n";
    std::cout << "Scalar time:  " << time << " ms\n";
    return 0;
}
