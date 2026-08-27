// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "simd_examples/03_clamp.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

void clamp(float* values, std::size_t size, float upper_bound) noexcept {
    using vector_type = native_simd<float>;
    using mask_type = native_mask<float>;
    constexpr std::size_t width = vector_type::size();

    const vector_type upper(upper_bound);
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type values_vector;
        values_vector.copy_from(values + i, stdx::element_aligned);
        const mask_type above_upper = values_vector > upper;
        stdx::where(above_upper, values_vector) = upper;
        values_vector.copy_to(values + i, stdx::element_aligned);
    }

    for (; i < size; ++i) {
        if (values[i] > upper_bound) {
            values[i] = upper_bound;
        }
    }
}

}
