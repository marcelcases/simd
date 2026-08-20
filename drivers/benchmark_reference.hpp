#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
#endif

namespace simd_examples::benchmark::reference {

inline void add(float* destination, const float* source, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) destination[i] += source[i];
}

inline float sum(const float* values, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) result += values[i];
    return result;
}

inline void clamp(float* values, std::size_t size, float upper_bound) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > upper_bound) values[i] = upper_bound;
    }
}

inline std::size_t count_above(const float* values, std::size_t size,
                        float threshold) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > threshold) ++count;
    }
    return count;
}

inline void softmax(float* values, std::size_t size) noexcept {
    if (size == 0) {
        return;
    }

    float maximum = values[0];
    for (std::size_t i = 1; i < size; ++i) maximum = std::max(maximum, values[i]);

    float total = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        total += values[i];
    }
    for (std::size_t i = 0; i < size; ++i) values[i] /= total;
}

inline void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) output[i] = a[i] * b[i] + c[i];
}

inline float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) result += a[i] * b[i];
    return result;
}

inline void blur_horizontal(const float* input, float* output,
                             int width, int height) noexcept {
    if (width <= 0 || height <= 0) {
        return;
    }

    for (int row = 0; row < height; ++row) {
        const float* source = input + row * width;
        float* destination = output + row * width;
        if (width == 1) {
            destination[0] = source[0];
            continue;
        }

        destination[0] = (source[0] + source[1]) * 0.5f;
        for (int column = 1; column < width - 1; ++column) {
            destination[column] =
                (source[column - 1] + source[column] + source[column + 1]) / 3.f;
        }
        destination[width - 1] =
            (source[width - 2] + source[width - 1]) * 0.5f;
    }
}

inline void convolve_1d(const float* input, const float* kernel, float* output,
                          std::size_t size, std::size_t kernel_size) noexcept {
    if (kernel_size == 0 || size < kernel_size) {
        return;
    }

    const std::size_t output_size =
        size - kernel_size + 1;
    for (std::size_t i = 0; i < output_size; ++i) {
        float result = 0.f;
        for (std::size_t j = 0; j < kernel_size; ++j) {
            result += input[i + j] * kernel[kernel_size - 1 - j];
        }
        output[i] = result;
    }
}

} // namespace simd_examples::benchmark::reference

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
