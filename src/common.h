#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace simd_examples {

// Prevent the compiler from removing a measured non-void computation.
static volatile std::size_t g_benchmark_sink = 0;

// Return the best execution time in milliseconds.
template<class F>
double bench_ms(F&& f, int iters = 10) {
    using result_type = std::invoke_result_t<F&>;
    using clock = std::chrono::steady_clock;
    double best = std::numeric_limits<double>::max();

    for (int i = 0; i < iters; ++i) {
        const auto start = clock::now();
        if constexpr (std::is_void_v<result_type>) {
            f();
        } else {
            g_benchmark_sink = f();
        }
        const auto stop = clock::now();
        const double elapsed =
            std::chrono::duration<double, std::milli>(stop - start).count();
        best = std::min(best, elapsed);
    }
    return best;
}

// Simple checksum for result verification.
template<class Iter>
float checksum(Iter first, Iter last) {
    float sum = 0.f;
    for (; first != last; ++first) sum += *first;
    return sum;
}

} // namespace simd_examples

#endif
