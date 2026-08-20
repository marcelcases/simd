#include "simd_examples/05_softmax.hpp"
#include "simd_common.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

template<class Vector>
Vector exp_poly(Vector values) noexcept {
    const Vector c1(1.f), c2(1.f), c3(.5f), c4(1.f / 6.f);
    const Vector c5(1.f / 24.f), c6(1.f / 120.f);
    return c1 + values * (c2 + values * (c3 + values * (c4 + values * (c5 + values * c6))));
}

float find_max(const float* values, std::size_t size) noexcept {
    using vector_type = simd_examples::native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    vector_type maximum = std::numeric_limits<float>::lowest();
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type vector;
        vector.copy_from(values + i, stdx::element_aligned);
        maximum = stdx::max(maximum, vector);
    }

    float result = stdx::hmax(maximum);
    for (; i < size; ++i) result = std::max(result, values[i]);
    return result;
}

} // namespace

namespace simd_examples::simd {

void softmax(float* values, std::size_t size) noexcept {
    using vector_type = native_simd<float>;
    constexpr std::size_t width = vector_type::size();

    const float maximum = find_max(values, size);
    const vector_type maximum_vector(maximum);
    vector_type vector_sum = 0.f;
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type vector;
        vector.copy_from(values + i, stdx::element_aligned);
        vector = exp_poly(vector - maximum_vector);
        vector.copy_to(values + i, stdx::element_aligned);
        vector_sum += vector;
    }

    float sum = stdx::reduce(vector_sum);
    for (; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }

    const vector_type sum_vector(sum);
    i = 0;
    for (; i + width <= size; i += width) {
        vector_type vector;
        vector.copy_from(values + i, stdx::element_aligned);
        (vector / sum_vector).copy_to(values + i, stdx::element_aligned);
    }
    for (; i < size; ++i) values[i] /= sum;
}

} // namespace simd_examples::simd
