#include "simd_examples/05_softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void softmax(float* values, std::size_t size) noexcept {
    float maximum = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < size; ++i) maximum = std::max(maximum, values[i]);

    float sum = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }
    for (std::size_t i = 0; i < size; ++i) values[i] /= sum;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void softmax(float* values, std::size_t size) noexcept {
    float maximum = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < size; ++i) maximum = std::max(maximum, values[i]);

    float sum = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }
    for (std::size_t i = 0; i < size; ++i) values[i] /= sum;
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
