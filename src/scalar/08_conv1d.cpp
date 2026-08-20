#include "simd_examples/08_conv1d.hpp"

namespace simd_examples::scalar {

void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, std::size_t kernel_size) noexcept {
    if (kernel_size == 0 || size < kernel_size) {
        return;
    }

    const std::size_t output_size =
        size - kernel_size + 1;
    for (std::size_t i = 0; i < output_size; ++i) {
        float result = 0.f;
        for (std::size_t j = 0; j < kernel_size; ++j) {
            result += input[i + j] * kernel[kernel_size - 1 - j];
        }
        output[i] = result;
    }
}

}
