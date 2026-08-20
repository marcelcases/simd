#include "../simd_common.h"
#include <cstdlib>

using namespace simd_examples;

// Example 7: SIMD horizontal image blur.
struct Image {
    int width, height;
    float* data;

    float& at(int y, int x) { return data[y * width + x]; }
    const float& at(int y, int x) const { return data[y * width + x]; }
};

void blur_horizontal_simd(const Image& input, Image& output) {
    using V = native_simd<float>;
    constexpr int width = V::size();
    const V inverse_three(1.f / 3.f);

    for (int y = 0; y < input.height; ++y) {
        const float* source = &input.at(y, 0);
        float* destination = &output.at(y, 0);

        destination[0] = (source[0] + source[1]) * 0.5f;
        int x = 1;
        for (; x + width < input.width; x += width) {
            V left, center, right;
            left.copy_from(source + x - 1, stdx::element_aligned);
            center.copy_from(source + x, stdx::element_aligned);
            right.copy_from(source + x + 1, stdx::element_aligned);
            ((left + center + right) * inverse_three)
                .copy_to(destination + x, stdx::element_aligned);
        }
        for (; x < input.width - 1; ++x) {
            destination[x] =
                (source[x - 1] + source[x] + source[x + 1]) / 3.f;
        }
        destination[input.width - 1] =
            (source[input.width - 2] + source[input.width - 1]) * 0.5f;
    }
}

int main() {
    std::cout << "=== Example 7: SIMD Horizontal Blur ===\n\n";

    const int width = 1920;
    const int height = 1080;
    Image input{width, height, nullptr}, output{width, height, nullptr};
    input.data = static_cast<float*>(std::malloc(width * height * sizeof(float)));
    output.data = static_cast<float*>(std::malloc(width * height * sizeof(float)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (int i = 0; i < width * height; ++i) input.data[i] = distribution(rng);

    const double time = bench_ms([&]() {
        blur_horizontal_simd(input, output);
    });

    std::cout << "Image size: " << width << " x " << height << "\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:  " << time << " ms\n";

    std::free(input.data);
    std::free(output.data);
    return 0;
}
