// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/07_filter.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

void blur_horizontal(ConstImageView input, ImageView output) noexcept {
    using vector_type = native_simd<float>;
    constexpr int vector_width = vector_type::size();
    const int width = input.width;
    if (width <= 0 || input.height <= 0) {
        return;
    }

    const vector_type one_third(1.f / 3.f);
    for (int row = 0; row < input.height; ++row) {
        const float* source = input.data + row * width;
        float* destination = output.data + row * width;

        if (width == 1) {
            destination[0] = source[0];
            continue;
        }

        destination[0] = (source[0] + source[1]) * 0.5f;
        int column = 1;
        for (; column + vector_width < width; column += vector_width) {
            vector_type left;
            vector_type center;
            vector_type right;
            left.copy_from(source + column - 1, stdx::element_aligned);
            center.copy_from(source + column, stdx::element_aligned);
            right.copy_from(source + column + 1, stdx::element_aligned);
            vector_type result = (left + center + right) * one_third;
            result.copy_to(destination + column, stdx::element_aligned);
        }

        for (; column < width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) / 3.f;
        }
        destination[width - 1] =
            (source[width - 2] + source[width - 1]) * 0.5f;
    }
}

}
