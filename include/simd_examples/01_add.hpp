#pragma once

#include <cstddef>

namespace simd_examples::scalar {

void add(float* destination, const float* source, std::size_t size) noexcept;

} // namespace simd_examples::scalar

namespace simd_examples::simd {

void add(float* destination, const float* source, std::size_t size) noexcept;

} // namespace simd_examples::simd
