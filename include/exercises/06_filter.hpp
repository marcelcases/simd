// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

namespace exercises {

// Preconditions: width >= 2, height >= 1; valid, non-overlapping buffers
// containing width * height floats in row-major order.

namespace scalar {
void blur_horizontal(const float* input, float* output,
                     int width, int height) noexcept;
}

namespace simd {
void blur_horizontal(const float* input, float* output,
                     int width, int height) noexcept;
}

} // namespace exercises
