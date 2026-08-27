// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/05_softmax.hpp"
#include "simd_common.h"

#include <algorithm>
#include <cmath>

namespace {

float find_max(const float* values, std::size_t size) noexcept {
    using vector_type = simd_examples::native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    vector_type maximum(values[0]);
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type values_vector;
        values_vector.copy_from(values + i, stdx::element_aligned);
        maximum = stdx::max(maximum, values_vector);
    }

    float result = stdx::hmax(maximum);
    for (; i < size; ++i) {
        result = std::max(result, values[i]);
    }
    return result;
}

}

namespace simd_examples::simd {

void softmax(float* values, std::size_t size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();
    if (size == 0) {
        return;
    }

    const float maximum = find_max(values, size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
    }

    vector_type total_vector(0.f);
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type values_vector;
        values_vector.copy_from(values + i, stdx::element_aligned);
        total_vector += values_vector;
    }

    float total = stdx::reduce(total_vector);
    for (; i < size; ++i) {
        total += values[i];
    }

    const vector_type total_value(total);
    i = 0;
    for (; i + width <= size; i += width) {
        vector_type values_vector;
        values_vector.copy_from(values + i, stdx::element_aligned);
        (values_vector / total_value).copy_to(values + i, stdx::element_aligned);
    }
    for (; i < size; ++i) {
        values[i] /= total;
    }
}

}
