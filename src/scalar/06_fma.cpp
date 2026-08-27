// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/06_fma.hpp"

namespace simd_examples::scalar {

void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = a[i] * b[i] + c[i];
    }
}

float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

}
