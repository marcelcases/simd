#pragma once

#include <cstddef>

namespace simd_examples::scalar {
std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept;
}

namespace simd_examples::simd {
std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept;
}
