// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/07_filter.hpp"

namespace simd_examples::scalar {

void blur_horizontal(ConstImageView input, ImageView output) noexcept {
    const int width = input.width;
    if (width <= 0 || input.height <= 0) {
        return;
    }

    const float inverse_three = 1.f / 3.f;
    for (int row = 0; row < input.height; ++row) {
        const float* source = input.data + row * width;
        float* destination = output.data + row * width;

        if (width == 1) {
            destination[0] = source[0];
            continue;
        }

        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int column = 1; column < width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) * inverse_three;
        }
        destination[width - 1] =
            (source[width - 2] + source[width - 1]) * 0.5f;
    }
}

}
