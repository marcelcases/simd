#include "../common.h"

using namespace simd_examples;

// Example 5: numerically stable scalar softmax.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
float find_max_scalar(const float* x, std::size_t n) {
    float max_value = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < n; ++i) {
        max_value = std::max(max_value, x[i]);
    }
    return max_value;
}

__attribute__((noinline, optnone))
void softmax_scalar(float* x, std::size_t n) {
    const float max_value = find_max_scalar(x, n);
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        const float value = std::exp(x[i] - max_value);
        x[i] = value;
        sum += value;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] /= sum;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
float find_max_scalar(const float* x, std::size_t n) {
    float max_value = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < n; ++i) {
        max_value = std::max(max_value, x[i]);
    }
    return max_value;
}

void softmax_scalar(float* x, std::size_t n) {
    const float max_value = find_max_scalar(x, n);
    float sum = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        const float value = std::exp(x[i] - max_value);
        x[i] = value;
        sum += value;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] /= sum;
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 5: Scalar Softmax ===\n\n";

    const std::size_t N = 1ULL << 20;
    std::vector<float> values(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> distribution(0.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const double time = bench_ms([&]() -> float {
        softmax_scalar(values.data(), N);
        return checksum(values.begin(), values.end());
    });

    std::cout << "Array size:  " << N << " elements\n";
    std::cout << "Scalar time: " << time << " ms\n";
    std::cout << "Probability sum: " << checksum(values.begin(), values.end()) << "\n";
    return 0;
}
