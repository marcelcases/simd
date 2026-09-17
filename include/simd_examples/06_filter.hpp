// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

namespace simd_examples {

struct ConstImageView {
    int width;
    int height;
    const float* data;
};

struct ImageView {
    int width;
    int height;
    float* data;
};

namespace scalar {
void blur_horizontal(ConstImageView input, ImageView output) noexcept;
}

namespace simd {
void blur_horizontal(ConstImageView input, ImageView output) noexcept;
}

} // namespace simd_examples
