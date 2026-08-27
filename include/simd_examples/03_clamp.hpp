// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace simd_examples::scalar {
void clamp(float* values, std::size_t size, float upper_bound) noexcept;
}

namespace simd_examples::simd {
void clamp(float* values, std::size_t size, float upper_bound) noexcept;
}
