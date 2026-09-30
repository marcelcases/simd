// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace exercises::scalar {
std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept;
}

namespace exercises::simd {
std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept;
}
