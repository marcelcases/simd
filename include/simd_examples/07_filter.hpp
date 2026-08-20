#pragma once

namespace simd_examples {

struct ConstImageView {
    int width;
    int height;
    const float* data;

    const float& at(int row, int column) const {
        return data[row * width + column];
    }
};

struct ImageView {
    int width;
    int height;
    float* data;

    float& at(int row, int column) {
        return data[row * width + column];
    }
};

namespace scalar {
void blur_horizontal(ConstImageView input, ImageView output) noexcept;
}

namespace simd {
void blur_horizontal(ConstImageView input, ImageView output) noexcept;
}

} // namespace simd_examples
