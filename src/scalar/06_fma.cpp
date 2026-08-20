#include "simd_examples/06_fma.hpp"

namespace simd_examples::scalar {

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) output[i] = a[i] * b[i] + c[i];
}

__attribute__((noinline, optnone))
float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) result += a[i] * b[i];
    return result;
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void fma_memory_bound(const float* a, const float* b, const float* c,
                      float* output, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) output[i] = a[i] * b[i] + c[i];
}

float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) result += a[i] * b[i];
    return result;
}
#pragma GCC pop_options
#endif

} // namespace simd_examples::scalar
