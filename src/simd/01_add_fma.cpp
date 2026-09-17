// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/01_add_fma.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

void add(float* destination, const float* source, std::size_t size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type destination_vector;
        vector_type source_vector;
        destination_vector.copy_from(destination + i, stdx::element_aligned);
        source_vector.copy_from(source + i, stdx::element_aligned);
        destination_vector += source_vector;
        destination_vector.copy_to(destination + i, stdx::element_aligned);
    }

    for (; i < size; ++i) {
        destination[i] += source[i];
    }
}

void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type a_vector;
        vector_type b_vector;
        vector_type c_vector;
        a_vector.copy_from(a + i, stdx::element_aligned);
        b_vector.copy_from(b + i, stdx::element_aligned);
        c_vector.copy_from(c + i, stdx::element_aligned);
        stdx::fma(a_vector, b_vector, c_vector)
            .copy_to(output + i, stdx::element_aligned);
    }

    for (; i < size; ++i) {
        output[i] = a[i] * b[i] + c[i];
    }
}

}
