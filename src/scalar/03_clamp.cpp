#include "simd_examples/03_clamp.hpp"

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void clamp(float* values, std::size_t size, float upper_bound) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > upper_bound) values[i] = upper_bound;
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void clamp(float* values, std::size_t size, float upper_bound) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > upper_bound) values[i] = upper_bound;
    }
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
