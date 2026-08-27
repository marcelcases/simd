// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace simd_examples::scalar {
void softmax(float* values, std::size_t size) noexcept;
}

namespace simd_examples::simd {
void softmax(float* values, std::size_t size) noexcept;
}
