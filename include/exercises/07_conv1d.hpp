// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace exercises::scalar {
// Preconditions: 1 <= kernel_size <= input_size; valid input and kernel buffers;
// output holds input_size - kernel_size + 1 floats and overlaps neither buffer.
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t input_size, std::size_t kernel_size) noexcept;
}

namespace exercises::simd {
// Same buffer and size preconditions as the scalar kernel.
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t input_size, std::size_t kernel_size) noexcept;
}
