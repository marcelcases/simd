#include "common.h"
#include <cstdlib>

using namespace simd_examples;

// ============================================================================
// Example 7: 2D Image Processing - Horizontal Box Filter
// ============================================================================
// This demonstrates SIMD applied to image processing with a sliding window.
//
// Key concepts:
// - Processing images row-by-row
// - Handling borders (edges need special treatment)
// - Why overlapping loads limit speedup (memory-bound operations)

// Simple grayscale image structure
struct Image {
    int w, h;
    float* data;
    
    float& at(int y, int x) { return data[y * w + x]; }
    const float& at(int y, int x) const { return data[y * w + x]; }
};

// ---- Scalar horizontal blur: out[x] = (in[x-1] + in[x] + in[x+1]) / 3 ----
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
__attribute__((noinline, optnone))
void blur_horizontal_scalar(const Image& in, Image& out) {
    const float inv3 = 1.f / 3.f;
    
    for (int y = 0; y < in.h; ++y) {
        const float* src = &in.at(y, 0);
        float* dst = &out.at(y, 0);
        
        // Left edge
        dst[0] = (src[0] + src[1]) * 0.5f;
        
        // Main body
        for (int x = 1; x < in.w - 1; ++x) {
            dst[x] = (src[x-1] + src[x] + src[x+1]) * inv3;
        }
        
        // Right edge
        dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}
#else
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")
void blur_horizontal_scalar(const Image& in, Image& out) {
    const float inv3 = 1.f / 3.f;
    
    for (int y = 0; y < in.h; ++y) {
        const float* src = &in.at(y, 0);
        float* dst = &out.at(y, 0);
        
        // Left edge
        dst[0] = (src[0] + src[1]) * 0.5f;
        
        // Main body
        for (int x = 1; x < in.w - 1; ++x) {
            dst[x] = (src[x-1] + src[x] + src[x+1]) * inv3;
        }
        
        // Right edge
        dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}
#pragma GCC pop_options
#endif

// ---- SIMD horizontal blur ----
void blur_horizontal_simd(const Image& in, Image& out) {
    using V = native_simd<float>;
    constexpr int W = V::size();
    const V inv3(1.f / 3.f);
    
    for (int y = 0; y < in.h; ++y) {
        const float* src = &in.at(y, 0);
        float* dst = &out.at(y, 0);
        
        // Left edge (scalar)
        dst[0] = (src[0] + src[1]) * 0.5f;
        
        // Main SIMD body
        int x = 1;
        for (; x + W < in.w; x += W) {
            V left, center, right;
            left.copy_from(src + x - 1, stdx::element_aligned);
            center.copy_from(src + x, stdx::element_aligned);
            right.copy_from(src + x + 1, stdx::element_aligned);
            
            V result = (left + center + right) * inv3;
            result.copy_to(dst + x, stdx::element_aligned);
        }
        
        // Tail (scalar)
        for (; x < in.w - 1; ++x) {
            dst[x] = (src[x-1] + src[x] + src[x+1]) / 3.f;
        }
        
        // Right edge (scalar)
        dst[in.w - 1] = (src[in.w - 2] + src[in.w - 1]) * 0.5f;
    }
}

int main() {
    std::cout << "=== Example 7: Image Processing (Horizontal Blur) ===\n\n";
    
    // Full HD image
    const int W = 1920, H = 1080;
    Image in{W, H, nullptr}, out{W, H, nullptr};
    
    // Aligned allocation
    posix_memalign((void**)&in.data, 64, W * H * sizeof(float));
    posix_memalign((void**)&out.data, 64, W * H * sizeof(float));
    
    // Random pixel values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < W * H; ++i) in.data[i] = dist(rng);
    
    double t_scalar = bench_ms([&]() {
        blur_horizontal_scalar(in, out);
    });
    
    double t_simd = bench_ms([&]() {
        blur_horizontal_simd(in, out);
    });
    
    std::cout << "Image size:  " << W << " x " << H << " (" << (W*H/1e6) << "M pixels)\n";
    std::cout << "SIMD width:  " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time: " << t_scalar << " ms\n";
    std::cout << "SIMD time:   " << t_simd << " ms\n";
    std::cout << "Speedup:     " << (t_scalar / t_simd) << "x\n\n";
    
    std::cout << "WHY SPEEDUP IS MODEST:\n";
    std::cout << "----------------------\n";
    std::cout << "This operation loads 3 overlapping vectors per output:\n";
    std::cout << "  left:   [x-1, x,   x+1, x+2, ...]\n";
    std::cout << "  center: [x,   x+1, x+2, x+3, ...]\n";
    std::cout << "  right:  [x+1, x+2, x+3, x+4, ...]\n\n";
    std::cout << "Memory bandwidth becomes the bottleneck, not compute.\n";
    std::cout << "For better performance, use separable filters or tiling.\n";
    
    std::free(in.data);
    std::free(out.data);
    
    return 0;
}
