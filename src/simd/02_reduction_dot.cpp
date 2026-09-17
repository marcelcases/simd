// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/02_reduction_dot.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

float sum(const float* values, std::size_t size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    vector_type accumulator(0.f);
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type values_vector;
        values_vector.copy_from(values + i, stdx::element_aligned);
        accumulator += values_vector;
    }

    float result = stdx::reduce(accumulator);
    for (; i < size; ++i) {
        result += values[i];
    }
    return result;
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
