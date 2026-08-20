#include "../common.h"

using namespace simd_examples;

// Example 8: scalar 1D convolution.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
#endif
void conv1d_scalar(const float* x, const float* kernel, float* y,
                   std::size_t n, int kernel_size) {
    const std::size_t end = n - kernel_size + 1;
    for (std::size_t i = 0; i < end; ++i) {
        float sum = 0.f;
        for (int j = 0; j < kernel_size; ++j) {
            sum += x[i + j] * kernel[j];
        }
        y[i] = sum;
    }
}
#if !defined(__INTEL_LLVM_COMPILER) && !defined(__clang__)
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 8: Scalar 1D Convolution ===\n\n";

    const std::size_t N = 1ULL << 20;
    constexpr int K = 3;
    const float kernel[K] = {0.25f, 0.5f, 0.25f};
    std::vector<float> x(N), y(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : x) value = distribution(rng);

    const double time = bench_ms([&]() {
        conv1d_scalar(x.data(), kernel, y.data(), N, K);
    });

    std::cout << "Array size:  " << N << " elements\n";
    std::cout << "Kernel size: " << K << "\n";
    std::cout << "Scalar time: " << time << " ms\n";
    return 0;
}
