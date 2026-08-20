#include "simd_examples/08_conv1d.hpp"

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, int kernel_size) noexcept {
    const std::size_t end = size - kernel_size + 1;
    for (std::size_t i = 0; i < end; ++i) {
        float sum = 0.f;
        for (int j = 0; j < kernel_size; ++j) sum += input[i + j] * kernel[j];
        output[i] = sum;
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void convolve_1d(const float* input, const float* kernel, float* output,
                 std::size_t size, int kernel_size) noexcept {
    const std::size_t end = size - kernel_size + 1;
    for (std::size_t i = 0; i < end; ++i) {
        float sum = 0.f;
        for (int j = 0; j < kernel_size; ++j) sum += input[i + j] * kernel[j];
        output[i] = sum;
    }
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
