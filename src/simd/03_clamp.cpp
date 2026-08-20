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
        vector_type vector;
        vector.copy_from(values + i, stdx::element_aligned);
        const mask_type mask = vector > upper;
        stdx::where(mask, vector) = upper;
        vector.copy_to(values + i, stdx::element_aligned);
    }

    for (; i < size; ++i) {
        if (values[i] > upper_bound) values[i] = upper_bound;
    }
}

} // namespace simd_examples::simd
