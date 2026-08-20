#include "../common.h"

using namespace simd_examples;

// Example 4: scalar threshold count.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
std::size_t count_scalar(const float* a, std::size_t n, float threshold) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] > threshold) ++count;
    }
    return count;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
std::size_t count_scalar(const float* a, std::size_t n, float threshold) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] > threshold) ++count;
    }
    return count;
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 4: Scalar Threshold Count ===\n\n";

    const std::size_t N = 1ULL << 24;
    const float threshold = 0.f;
    std::vector<float> values(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : values) x = dist(rng);

    const double time = bench_ms([&]() -> std::size_t {
        return count_scalar(values.data(), N, threshold);
    });
    const std::size_t count = count_scalar(values.data(), N, threshold);

    std::cout << "Array size:   " << N << " elements\n";
    std::cout << "Threshold:    " << threshold << "\n";
    std::cout << "Scalar time:  " << time << " ms\n";
    std::cout << "Scalar count: " << count << "\n";
    return 0;
}
