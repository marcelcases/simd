#include "common.h"

using namespace simd_examples;

// ============================================================================
// Example 6: 1D Convolution with FMA
// ============================================================================
// This demonstrates FMA (Fused Multiply-Add) which computes a*b + c in a single
// instruction. This is crucial for:
// - Convolution operations
// - Polynomial evaluation
// - Matrix operations
//
// stdx::fma(a, b, c) = a * b + c with single rounding

// Simple 1D convolution: y[i] = sum(x[i+j] * k[j]) for j in [0, K)
// This is the "valid" region where the kernel fits fully
template<int K>
void conv1d_scalar(const float* x, const float* k, float* y, std::size_t n) {
    std::size_t end = n - K + 1;
    for (std::size_t i = 0; i < end; ++i) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            s += x[i + j] * k[j];
        }
        y[i] = s;
    }
}

// SIMD convolution using FMA
// Note: This is a simplified version for demonstration
template<int K>
void conv1d_simd(const float* x, const float* k, float* y, std::size_t n) {
    using V = native_simd<float>;
    constexpr std::size_t W = V::size();
    
    std::size_t end = n - K + 1;
    std::size_t i = 0;
    
    // Pre-load kernel into SIMD vectors
    // For simplicity, we reload each time (in production, cache this)
    for (; i + W <= end; i += W) {
        V acc = 0.f;
        
        for (int j = 0; j < K; ++j) {
            V kv(k[j]);  // Broadcast kernel value
            V xv;
            xv.copy_from(x + i + j, stdx::element_aligned);
            acc = stdx::fma(xv, kv, acc);  // acc += xv * kv
        }
        
        acc.copy_to(y + i, stdx::element_aligned);
    }
    
    // Tail: scalar
    for (; i < end; ++i) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            s += x[i + j] * k[j];
        }
        y[i] = s;
    }
}

int main() {
    std::cout << "=== Example 6: 1D Convolution with FMA ===\n\n";
    
    constexpr int K = 3;  // Kernel size
    const std::size_t N = 1ULL << 20;
    const std::size_t out_n = N - K + 1;
    
    std::vector<float> x(N), k(K), y_scalar(out_n), y_simd(out_n);
    
    // Input signal
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : x) v = dist(rng);
    
    // Simple averaging kernel [1/3, 1/3, 1/3]
    k[0] = k[1] = k[2] = 1.f / K;
    
    double t_scalar = bench_ms([&]() {
        conv1d_scalar<K>(x.data(), k.data(), y_scalar.data(), N);
    });
    
    double t_simd = bench_ms([&]() {
        conv1d_simd<K>(x.data(), k.data(), y_simd.data(), N);
    });
    
    // Verify
    float max_diff = 0.f;
    for (std::size_t i = 0; i < out_n; ++i) {
        max_diff = std::max(max_diff, std::abs(y_scalar[i] - y_simd[i]));
    }
    
    std::cout << "Input size:   " << N << " elements\n";
    std::cout << "Output size:  " << out_n << " elements\n";
    std::cout << "Kernel size:  " << K << "\n";
    std::cout << "SIMD width:   " << native_simd<float>::size() << " floats\n\n";
    std::cout << "Scalar time:  " << t_scalar << " ms\n";
    std::cout << "SIMD time:    " << t_simd << " ms\n";
    std::cout << "Speedup:      " << (t_scalar / t_simd) << "x\n";
    std::cout << "Max diff:     " << max_diff << "\n\n";
    
    std::cout << "HOW FMA WORKS:\n";
    std::cout << "--------------\n";
    std::cout << "stdx::fma(a, b, c) computes a * b + c in ONE instruction\n";
    std::cout << "vs. separate multiply and add which would have two roundings.\n";
    std::cout << "\nBenefits:\n";
    std::cout << "  - Single rounding (better precision)\n";
    std::cout << "  - Higher throughput (1 instruction vs 2)\n";
    std::cout << "  - Common in DSP, ML, image processing\n";
    
    return 0;
}