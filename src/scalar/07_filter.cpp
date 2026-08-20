#include "../common.h"
#include <cstdlib>

using namespace simd_examples;

// Example 7: scalar horizontal image blur.
struct Image {
    int width, height;
    float* data;

    float& at(int y, int x) { return data[y * width + x]; }
    const float& at(int y, int x) const { return data[y * width + x]; }
};

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void blur_horizontal_scalar(const Image& input, Image& output) {
    const float inverse_three = 1.f / 3.f;
    for (int y = 0; y < input.height; ++y) {
        const float* source = &input.at(y, 0);
        float* destination = &output.at(y, 0);
        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int x = 1; x < input.width - 1; ++x) {
            destination[x] =
                (source[x - 1] + source[x] + source[x + 1]) * inverse_three;
        }
        destination[input.width - 1] =
            (source[input.width - 2] + source[input.width - 1]) * 0.5f;
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void blur_horizontal_scalar(const Image& input, Image& output) {
    const float inverse_three = 1.f / 3.f;
    for (int y = 0; y < input.height; ++y) {
        const float* source = &input.at(y, 0);
        float* destination = &output.at(y, 0);
        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int x = 1; x < input.width - 1; ++x) {
            destination[x] =
                (source[x - 1] + source[x] + source[x + 1]) * inverse_three;
        }
        destination[input.width - 1] =
            (source[input.width - 2] + source[input.width - 1]) * 0.5f;
    }
}
#pragma GCC pop_options
#endif

int main() {
    std::cout << "=== Example 7: Scalar Horizontal Blur ===\n\n";

    const int width = 1920;
    const int height = 1080;
    Image input{width, height, nullptr}, output{width, height, nullptr};
    input.data = static_cast<float*>(std::malloc(width * height * sizeof(float)));
    output.data = static_cast<float*>(std::malloc(width * height * sizeof(float)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (int i = 0; i < width * height; ++i) input.data[i] = distribution(rng);

    const double time = bench_ms([&]() {
        blur_horizontal_scalar(input, output);
    });

    std::cout << "Image size:  " << width << " x " << height << "\n";
    std::cout << "Scalar time: " << time << " ms\n";

    std::free(input.data);
    std::free(output.data);
    return 0;
}
