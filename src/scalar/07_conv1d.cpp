// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "exercises/07_conv1d.hpp"

namespace exercises::scalar {

void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t input_size, std::size_t kernel_size) noexcept {
    const std::size_t output_size = input_size - kernel_size + 1;
    for (std::size_t i = 0; i < output_size; ++i) {
        float sum = 0.f;

        for (std::size_t j = 0; j < kernel_size; ++j) {
            sum += input[i + j] * kernel[kernel_size - 1 - j];
        }
        output[i] = sum;
    }
}

}
