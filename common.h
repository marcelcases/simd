#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <experimental/simd>

namespace stdx = std::experimental;

namespace simd_examples {
    // Simple benchmark helper: returns best time in milliseconds
    // Runs the function 'iters' times and returns the minimum time
    //
    // To prevent the compiler from optimizing away the function call,
    // we use a global volatile variable. This forces the compiler to 
    // actually execute the function because the global variable's value
    // might affect observable program behavior.
    
    // Global counter - prevents optimization of function calls
    static volatile std::size_t g_benchmark_sink = 0;
    
    // Single benchmark function that works with both void and non-void lambdas
    template<class F, class R = decltype(std::declval<F>()())>
    double bench_ms_impl(F&& f, int iters, std::true_type /* is void */) {
        using clk = std::chrono::steady_clock;
        double best = 1e300;
        
        for (int i = 0; i < iters; ++i) {
            auto t0 = clk::now();
            f();  // Void function
            auto t1 = clk::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best = std::min(best, ms);
        }
        
        return best;
    }
    
    template<class F, class R = decltype(std::declval<F>()())>
    double bench_ms_impl(F&& f, int iters, std::false_type /* not void */) {
        using clk = std::chrono::steady_clock;
        double best = 1e300;
        
        for (int i = 0; i < iters; ++i) {
            auto t0 = clk::now();
            g_benchmark_sink = f();  // Store result to prevent optimization
            auto t1 = clk::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best = std::min(best, ms);
        }
        
        return best;
    }
    
    // Main benchmark function - dispatches to appropriate implementation
    template<class F>
    double bench_ms(F&& f, int iters = 10) {
        return bench_ms_impl(f, iters, std::is_same<typename std::result_of<F()>::type, void>{});
    }

    // Simple checksum for verification (sum of all elements)
    template<class Iter>
    float checksum(Iter first, Iter last) {
        float s = 0.f;
        for (; first != last; ++first) s += *first;
        return s;
    }

    // Native SIMD type for current architecture
    // On AVX-512: 16 floats, On AVX2: 8 floats, On NEON: 4 floats
    template<class T>
    using native_simd = stdx::native_simd<T>;
    
    // Native SIMD mask type
    template<class T>
    using native_mask = stdx::simd_mask<T, stdx::simd_abi::native<T>>;
}

#endif