#include "../simd_common.h"

using namespace simd_examples;

// Example 4: explicit SIMD threshold count with mask popcount.
std::size_t count_simd(const float* a, std::size_t n, float threshold) {
    using V = native_simd<float>;
    using M = native_mask<float>;
    constexpr std::size_t width = V::size();

    const V threshold_vector(threshold);
    std::size_t count = 0;
    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V values;
        values.copy_from(a + i, stdx::element_aligned);
        const M mask = values > threshold_vector;
        count += stdx::popcount(mask);
    }
    for (; i < n; ++i) {
        if (a[i] > threshold) ++count;
    }
    return count;
}

int main() {
    std::cout << "=== Example 4: SIMD Threshold Count ===\n\n";

    const std::size_t N = 1ULL << 24;
    const float threshold = 0.f;
    std::vector<float> values(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const double time = bench_ms([&]() -> std::size_t {
        return count_simd(values.data(), N, threshold);
    });
    const std::size_t count = count_simd(values.data(), N, threshold);

    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Threshold:  " << threshold << "\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:  " << time << " ms\n";
    std::cout << "SIMD count: " << count << "\n";
    return 0;
}
