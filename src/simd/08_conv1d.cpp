#include "../simd_common.h"

using namespace simd_examples;

// Example 8: explicit SIMD 1D convolution.
template<int kernel_size>
void conv1d_simd(const float* x, const float* kernel, float* y, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    const std::size_t end = n - kernel_size + 1;
    std::size_t i = 0;
    for (; i + width <= end; i += width) {
        V accumulator = 0.f;
        for (int j = 0; j < kernel_size; ++j) {
            V values;
            values.copy_from(x + i + j, stdx::element_aligned);
            const V kernel_value(kernel[j]);
#if defined(__INTEL_LLVM_COMPILER)
            accumulator = stdx::fma(values, kernel_value, accumulator);
#else
            accumulator = accumulator + values * kernel_value;
#endif
        }
        accumulator.copy_to(y + i, stdx::element_aligned);
    }

    for (; i < end; ++i) {
        float sum = 0.f;
        for (int j = 0; j < kernel_size; ++j) sum += x[i + j] * kernel[j];
        y[i] = sum;
    }
}

int main() {
    std::cout << "=== Example 8: SIMD 1D Convolution ===\n\n";

    const std::size_t N = 1ULL << 20;
    constexpr int K = 3;
    const float kernel[K] = {0.25f, 0.5f, 0.25f};
    std::vector<float> x(N), y(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : x) value = distribution(rng);

    const double time = bench_ms([&]() {
        conv1d_simd<K>(x.data(), kernel, y.data(), N);
    });

    std::cout << "Array size:  " << N << " elements\n";
    std::cout << "Kernel size: " << K << "\n";
    std::cout << "SIMD width:  " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:   " << time << " ms\n";
    return 0;
}
