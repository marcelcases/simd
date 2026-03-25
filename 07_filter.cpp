#include "common.h"
#include <cstring>
#include <cstdlib>

using namespace simd_examples;

// ============================================================================
// Example 7: 3x3 Image Box Filter
// ============================================================================
// This demonstrates 2D image processing with a sliding window.
// The horizontal pass loads 3 consecutive values per position.
//
// Key concepts:
// - Handling image borders (first/last pixels)
// - Stride access patterns (stride != width for alignment)

struct Image {
    int w, h, stride;
    float* data;
};

// Horizontal pass of 3x3 box filter
// Each output pixel is average of 3 horizontal neighbors
void box3x3_horizontal_scalar(const Image& in, Image& out) {
    for (int y = 0; y < in.h; ++y) {
        const float* src = in.data + y * in.stride;
        float* dst = out.data + y * out.stride;
        
        // Left edge: only 2 neighbors available
        if (in.w >= 1) dst[0] = (src[0] + src[1]) * 0.5f;
        if (in.w >= 2) dst[1] = (src[0] + src[1] + src[2]) / 3.f;
        
        // Main body: 3 neighbors
        for (int x = 2; x < in.w - 2; ++x) {
            dst[x] = (src[x-1] + src[x] + src[x+1]) / 3.f;
        }
        
        // Right edge
        if (in.w >= 2) dst[in.w - 2] = (src[in.w - 3] + src[in.w - 2] + src[in.w - 1]) / 3.f;
        if (in.w >= 1) dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}

void box3x3_horizontal_simd(const Image& in, Image& out) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    const float k[3] = {1.f/3, 1.f/3, 1.f/3};
    const V k0(k[0]), k1(k[1]), k2(k[2]);
    
    for (int y = 0; y < in.h; ++y) {
        const float* src = in.data + y * in.stride;
        float* dst = out.data + y * out.stride;
        
        // Left edge handling (scalar)
        if (in.w >= 1) dst[0] = (src[0] + src[1]) * 0.5f;
        if (in.w >= 2) dst[1] = (src[0] + src[1] + src[2]) / 3.f;
        
        int x = 2;
        
        // Main vectorized section: process W pixels at a time
        for (; x + (int)W + 1 <= in.w; x += (int)W) {
            V a, b, c;
            a.copy_from(src + x - 1, stdx::element_aligned);
            b.copy_from(src + x,     stdx::element_aligned);
            c.copy_from(src + x + 1, stdx::element_aligned);
            
            // Weighted sum: a*k0 + b*k1 + c*k2
            V r = a * k0 + b * k1 + c * k2;
            r.copy_to(dst + x, stdx::element_aligned);
        }
        
        // Tail: simple scalar loop
        for (; x < in.w - 2; ++x) {
            dst[x] = (src[x-1] + src[x] + src[x+1]) / 3.f;
        }
        
        // Right edge handling (scalar)
        if (in.w >= 2) dst[in.w - 2] = (src[in.w - 3] + src[in.w - 2] + src[in.w - 1]) / 3.f;
        if (in.w >= 1) dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}

int main() {
    std::cout << "=== Example 7: 3x3 Box Filter (Horizontal Pass) ===\n\n";
    
    const int W = 1920, H = 1080;  // Full HD
    Image in{W, H, W, nullptr}, tmp{W, H, W, nullptr};
    
    // Allocate aligned memory for SIMD efficiency
    if (posix_memalign((void**)&in.data, 64, W * H * sizeof(float)) != 0) return 1;
    if (posix_memalign((void**)&tmp.data, 64, W * H * sizeof(float)) != 0) return 1;
    
    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < W * H; ++i) in.data[i] = dist(rng);
    
    double t_scalar = bench_ms([&]() {
        box3x3_horizontal_scalar(in, tmp);
    });
    
    double t_simd = bench_ms([&]() {
        box3x3_horizontal_simd(in, tmp);
    });
    
    std::cout << "Image size:  " << W << " x " << H << " = " << (W*H) << " pixels\n";
    std::cout << "SIMD width:  " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n\n";
    
    std::cout << "KEY CONCEPTS:\n";
    std::cout << "-------------\n";
    std::cout << "1. Border handling: First/last pixels use fewer neighbors\n";
    std::cout << "2. Stride access: Image rows may have padding (stride != width)\n";
    std::cout << "3. Kernel weights: Pre-loaded into SIMD vectors for efficiency\n";
    std::cout << "4. Memory alignment: 64-byte aligned for AVX-512\n";
    
    std::free(in.data);
    std::free(tmp.data);
    
    return 0;
}