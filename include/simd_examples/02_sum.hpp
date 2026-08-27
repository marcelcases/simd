// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace simd_examples::scalar {
float sum(const float* values, std::size_t size) noexcept;
}

namespace simd_examples::simd {
float sum(const float* values, std::size_t size) noexcept;
}
