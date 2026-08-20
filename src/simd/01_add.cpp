#include "../simd_common.h"

using namespace simd_examples;

// Example 1: explicit SIMD vector addition.
void add_simd(float* __restrict dst, const float* __restrict src, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t width = V::size();

    std::size_t i = 0;
    for (; i + width <= n; i += width) {
        V destination;
        V source;
        destination.copy_from(dst + i, stdx::element_aligned);
        source.copy_from(src + i, stdx::element_aligned);
        destination += source;
        destination.copy_to(dst + i, stdx::element_aligned);
    }
    for (; i < n; ++i) dst[i] += src[i];
}

int main() {
    std::cout << "=== Example 1: SIMD Vector Add ===\n\n";

    const std::size_t N = 1ULL << 24;
    std::vector<float> src(N), dst(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : src) value = distribution(rng);
    dst = src;

    const double time = bench_ms([&]() -> float {
        add_simd(dst.data(), src.data(), N);
        return checksum(dst.begin(), dst.end());
    });

    std::cout << "Array size: " << N << " elements\n";
    std::cout << "SIMD width: " << native_simd<float>::size() << " floats\n";
    std::cout << "SIMD time:  " << time << " ms\n";
    std::cout << "Checksum:   " << checksum(dst.begin(), dst.end()) << "\n";
    return 0;
}
