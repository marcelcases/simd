#include "../simd_common.h"

using namespace simd_examples;

// Example 2: explicit SIMD sum reduction.
float sum_simd(const float* a, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    V accumulator = 0.f;
    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V values;
        values.copy_from(a + i, stdx::element_aligned);
        accumulator += values;
    }

    float sum = stdx::reduce(accumulator);
    for (; i < n; ++i) sum += a[i];
    return sum;
}

int main() {
    std::cout << "=== Example 2: SIMD Sum Reduction ===\n\n";

    const std::size_t N = 1ULL << 24;
    std::vector<float> a(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : a) value = distribution(rng);

    const double time = bench_ms([&]() -> float {
        return sum_simd(a.data(), N);
    });
    const float sum = sum_simd(a.data(), N);

    std::cout << "Array size: " << N << " elements\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:  " << time << " ms\n";
    std::cout << "SIMD sum:   " << sum << "\n";
    return 0;
}
