#include "../simd_common.h"

using namespace simd_examples;

// Example 3: explicit SIMD clamp using a mask.
void clamp_simd(float* a, std::size_t n, float hi) {
    using V = native_simd<float>;
    using M = native_mask<float>;
    constexpr std::size_t width = V::size();

    const V upper_bound(hi);
    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V values;
        values.copy_from(a + i, stdx::element_aligned);
        const M mask = values > upper_bound;
        stdx::where(mask, values) = upper_bound;
        values.copy_to(a + i, stdx::element_aligned);
    }
    for (; i < n; ++i) {
        if (a[i] > hi) a[i] = hi;
    }
}

int main() {
    std::cout << "=== Example 3: SIMD Clamp ===\n\n";

    const std::size_t N = 1ULL << 24;
    const float hi = 0.5f;
    std::vector<float> values(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const double time = bench_ms([&]() {
        clamp_simd(values.data(), N, hi);
    });

    std::cout << "Array size:  " << N << " elements\n";
    std::cout << "Upper bound: " << hi << "\n";
    std::cout << "SIMD width:  " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:   " << time << " ms\n";
    return 0;
}
