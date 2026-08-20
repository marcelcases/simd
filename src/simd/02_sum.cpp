#include "simd_examples/02_sum.hpp"
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

}
