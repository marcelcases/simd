#include "simd_examples/08_conv1d.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, std::size_t kernel_size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    if (kernel_size == 0 || size < kernel_size) {
        return;
    }

    const std::size_t output_size =
        size - kernel_size + 1;
    std::size_t i = 0;
    for (; i + width <= output_size; i += width) {
        vector_type accumulator = 0.f;
        for (std::size_t j = 0; j < kernel_size; ++j) {
            vector_type values;
            values.copy_from(input + i + j, stdx::element_aligned);
            const vector_type kernel_value(kernel[kernel_size - 1 - j]);
#if defined(__INTEL_LLVM_COMPILER)
            accumulator = stdx::fma(values, kernel_value, accumulator);
#else
            accumulator = accumulator + values * kernel_value;
#endif
        }
        accumulator.copy_to(output + i, stdx::element_aligned);
    }

    for (; i < output_size; ++i) {
        float sum = 0.f;
        for (std::size_t j = 0; j < kernel_size; ++j) {
            sum += input[i + j] * kernel[kernel_size - 1 - j];
        }
        output[i] = sum;
    }
}

}
