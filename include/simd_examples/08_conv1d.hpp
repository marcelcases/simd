#pragma once

#include <cstddef>

namespace simd_examples::scalar {
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, int kernel_size) noexcept;
}

namespace simd_examples::simd {
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, int kernel_size) noexcept;
}
