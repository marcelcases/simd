// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "exercises/07_conv1d.hpp"
#include "simd_common.h"

namespace exercises::simd {

void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t input_size, std::size_t kernel_size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();
    const std::size_t output_size = input_size - kernel_size + 1;
    std::size_t i = 0;
    for (; i + width <= output_size; i += width) {
        vector_type sum(0.f);

        for (std::size_t j = 0; j < kernel_size; ++j) {
            vector_type input_vector;
            input_vector.copy_from(input + i + j, stdx::element_aligned);
            const vector_type kernel_value(kernel[kernel_size - 1 - j]);
            sum += input_vector * kernel_value;
        }
        sum.copy_to(output + i, stdx::element_aligned);
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
