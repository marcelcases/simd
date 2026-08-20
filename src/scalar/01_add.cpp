#include "simd_examples/01_add.hpp"

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void add(float* destination, const float* source, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        destination[i] += source[i];
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void add(float* destination, const float* source, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        destination[i] += source[i];
    }
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
