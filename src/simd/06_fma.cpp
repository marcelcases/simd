// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/06_fma.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

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

float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    vector_type accumulator(0.f);
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type a_vector;
        vector_type b_vector;
        a_vector.copy_from(a + i, stdx::element_aligned);
        b_vector.copy_from(b + i, stdx::element_aligned);
        accumulator = stdx::fma(a_vector, b_vector, accumulator);
    }

    float result = stdx::reduce(accumulator);
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

}
