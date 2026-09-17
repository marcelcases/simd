// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/02_reduction_dot.hpp"

namespace simd_examples::scalar {

float sum(const float* values, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        result += values[i];
    }
    return result;
}

float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

}
