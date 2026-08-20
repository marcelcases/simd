#include "../simd_common.h"

using namespace simd_examples;

// Example 5: numerically stable SIMD softmax.
template<class V>
V exp_poly(V z) noexcept {
    const V c1(1.f), c2(1.f), c3(.5f), c4(1.f / 6.f);
    const V c5(1.f / 24.f), c6(1.f / 120.f);
    return c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * c6))));
}

float find_max_simd(const float* x, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    V maximum = std::numeric_limits<float>::lowest();
    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V values;
        values.copy_from(x + i, stdx::element_aligned);
        maximum = stdx::max(maximum, values);
    }

    float max_value = stdx::hmax(maximum);
    for (; i < n; ++i) max_value = std::max(max_value, x[i]);
    return max_value;
}

void softmax_simd(float* x, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    const float max_value = find_max_simd(x, n);
    const V max_vector(max_value);
    V vector_sum = 0.f;
    std::size_t i = 0;

    // Pass 1: exp(x - max) and vector sum.
    for (; i + width <= n; i += width) {
        V values;
        values.copy_from(x + i, stdx::element_aligned);
        values = exp_poly(values - max_vector);
        values.copy_to(x + i, stdx::element_aligned);
        vector_sum += values;
    }

    float sum = stdx::reduce(vector_sum);
    for (; i < n; ++i) {
        x[i] = std::exp(x[i] - max_value);
        sum += x[i];
    }

    // Pass 2: normalize.
    const V sum_vector(sum);
    i = 0;
    for (; i + width <= n; i += width) {
        V values;
        values.copy_from(x + i, stdx::element_aligned);
        (values / sum_vector).copy_to(x + i, stdx::element_aligned);
    }
    for (; i < n; ++i) x[i] /= sum;
}

int main() {
    std::cout << "=== Example 5: SIMD Softmax ===\n\n";

    const std::size_t N = 1ULL << 20;
    std::vector<float> values(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> distribution(0.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const double time = bench_ms([&]() -> float {
        softmax_simd(values.data(), N);
        return checksum(values.begin(), values.end());
    });

    std::cout << "Array size:      " << N << " elements\n";
    std::cout << "SIMD width:      " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:       " << time << " ms\n";
    std::cout << "Probability sum: " << checksum(values.begin(), values.end()) << "\n";
    return 0;
}
