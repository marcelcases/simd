// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace simd_examples::scalar {
void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept;
float dot_product(const float* a, const float* b, std::size_t size) noexcept;
}

namespace simd_examples::simd {
void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept;
float dot_product(const float* a, const float* b, std::size_t size) noexcept;
}
