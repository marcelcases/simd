// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "exercises/06_filter.hpp"

namespace exercises::scalar {

void blur_horizontal(const float* input, float* output,
                     int width, int height) noexcept {
    constexpr float inverse_three = 1.f / 3.f;
    for (int row = 0; row < height; ++row) {
        const float* source = input + row * width;
        float* destination = output + row * width;

        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int column = 1; column < width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) * inverse_three;
        }
        destination[width - 1] = (source[width - 2] + source[width - 1]) * 0.5f;
    }
}

}
