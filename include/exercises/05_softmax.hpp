// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace exercises::scalar {
void softmax(float* values, std::size_t size) noexcept;
}

namespace exercises::simd {
void softmax(float* values, std::size_t size) noexcept;
}
