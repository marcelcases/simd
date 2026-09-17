// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/01_add_fma.hpp"

namespace simd_examples::scalar {

void add(float* destination, const float* source, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        destination[i] += source[i];
    }
}

void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = a[i] * b[i] + c[i];
    }
}

}
