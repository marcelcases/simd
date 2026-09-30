// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace exercises::scalar {
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, std::size_t kernel_size) noexcept;
}

namespace exercises::simd {
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, std::size_t kernel_size) noexcept;
}
