// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace exercises::scalar {
void clamp(float* values, std::size_t size, float upper_bound) noexcept;
}

namespace exercises::simd {
void clamp(float* values, std::size_t size, float upper_bound) noexcept;
}
