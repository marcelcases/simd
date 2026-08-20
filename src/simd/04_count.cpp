#include "simd_examples/04_count.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept {
    using vector_type = native_simd<float>;
    using mask_type = native_mask<float>;
    constexpr std::size_t width = vector_type::size();

    const vector_type threshold_vector(threshold);
    std::size_t count = 0;
    std::size_t i = 0;
    for (; i + width <= size; i += width) {
        vector_type vector;
        vector.copy_from(values + i, stdx::element_aligned);
        const mask_type mask = vector > threshold_vector;
        count += stdx::popcount(mask);
    }

    for (; i < size; ++i) {
        if (values[i] > threshold) ++count;
    }
    return count;
}

}
