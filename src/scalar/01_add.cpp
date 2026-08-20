#include "../common.h"

using namespace simd_examples;

// Example 1: scalar vector addition.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void add_scalar(float* __restrict dst, const float* __restrict src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] += src[i];
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void add_scalar(float* __restrict dst, const float* __restrict src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] += src[i];
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 1: Scalar Vector Add ===\n\n";

    const std::size_t N = 1ULL << 24;
    std::vector<float> src(N), dst(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& x : src) x = dist(rng);
    dst = src;

    const double time = bench_ms([&]() -> float {
        add_scalar(dst.data(), src.data(), N);
        return checksum(dst.begin(), dst.end());
    });

    std::cout << "Array size:  " << N << " elements\n";
    std::cout << "Scalar time: " << time << " ms\n";
    std::cout << "Checksum:     " << checksum(dst.begin(), dst.end()) << "\n";
    return 0;
}
