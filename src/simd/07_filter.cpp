#include "simd_examples/07_filter.hpp"
#include "simd_common.h"

namespace simd_examples::simd {

void blur_horizontal(ConstImageView input, ImageView output) noexcept {
    using vector_type = native_simd<float>;
    constexpr int width = vector_type::size();
    const vector_type inverse_three(1.f / 3.f);

    for (int row = 0; row < input.height; ++row) {
        const float* source = &input.at(row, 0);
        float* destination = &output.at(row, 0);
        destination[0] = (source[0] + source[1]) * 0.5f;

        int column = 1;
        for (; column + width < input.width; column += width) {
            vector_type left, center, right;
            left.copy_from(source + column - 1, stdx::element_aligned);
            center.copy_from(source + column, stdx::element_aligned);
            right.copy_from(source + column + 1, stdx::element_aligned);
            ((left + center + right) * inverse_three)
                .copy_to(destination + column, stdx::element_aligned);
        }
        for (; column < input.width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) / 3.f;
        }
        destination[input.width - 1] =
            (source[input.width - 2] + source[input.width - 1]) * 0.5f;
    }
}

} // namespace simd_examples::simd
