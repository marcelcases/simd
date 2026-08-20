#include "simd_examples/07_filter.hpp"

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void blur_horizontal(ConstImageView input, ImageView output) noexcept {
    const float inverse_three = 1.f / 3.f;
    for (int row = 0; row < input.height; ++row) {
        const float* source = &input.at(row, 0);
        float* destination = &output.at(row, 0);
        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int column = 1; column < input.width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) * inverse_three;
        }
        destination[input.width - 1] =
            (source[input.width - 2] + source[input.width - 1]) * 0.5f;
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void blur_horizontal(ConstImageView input, ImageView output) noexcept {
    const float inverse_three = 1.f / 3.f;
    for (int row = 0; row < input.height; ++row) {
        const float* source = &input.at(row, 0);
        float* destination = &output.at(row, 0);
        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int column = 1; column < input.width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) * inverse_three;
        }
        destination[input.width - 1] =
            (source[input.width - 2] + source[input.width - 1]) * 0.5f;
    }
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
