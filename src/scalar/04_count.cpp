#include "simd_examples/04_count.hpp"

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > threshold) ++count;
    }
    return count;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
std::size_t count_above(const float* values, std::size_t size, float threshold) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > threshold) ++count;
    }
    return count;
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
