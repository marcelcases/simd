// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <cstddef>

namespace exercises::scalar {

void add(float* destination, const float* source, std::size_t size) noexcept;
void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept;

} // namespace exercises::scalar

namespace exercises::simd {

void add(float* destination, const float* source, std::size_t size) noexcept;
void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept;

} // namespace exercises::simd
