#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace simd_examples::benchmark {

static volatile std::size_t benchmark_sink = 0;

template<class F>
double best_time_ms(F&& function, int repetitions) {
    using result_type = std::invoke_result_t<F&>;
    using clock = std::chrono::steady_clock;
    double best = std::numeric_limits<double>::max();

    for (int i = 0; i < repetitions; ++i) {
        const auto start = clock::now();
        if constexpr (std::is_void_v<result_type>) {
            function();
        } else {
            benchmark_sink = function();
        }
        const auto stop = clock::now();
        const double elapsed =
            std::chrono::duration<double, std::milli>(stop - start).count();
        best = std::min(best, elapsed);
    }
    return best;
}


template<class Setup, class F>
double best_time_ms(Setup&& setup, F&& function, int repetitions) {
    using result_type = std::invoke_result_t<F&>;
    using clock = std::chrono::steady_clock;
    double best = std::numeric_limits<double>::max();

    for (int i = 0; i < repetitions; ++i) {
        setup();
        const auto start = clock::now();
        if constexpr (std::is_void_v<result_type>) {
            function();
        } else {
            benchmark_sink = function();
        }
        const auto stop = clock::now();
        const double elapsed =
            std::chrono::duration<double, std::milli>(stop - start).count();
        best = std::min(best, elapsed);
    }
    return best;
}

inline bool within_tolerance(float actual, float expected,
                             float relative = 1e-3f,
                             float absolute = 1e-5f) {
    return std::abs(actual - expected) <=
        absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

template<class Writer>
bool write_output(std::string_view path, Writer&& writer) {
    if (path.empty()) {
        writer(std::cout);
        return true;
    }

    std::ofstream output{std::string(path)};
    if (!output) {
        std::cerr << "Cannot write " << path << "\n";
        return false;
    }
    writer(output);
    return true;
}

template<class Iter>
float checksum(Iter first, Iter last) {
    float sum = 0.f;
    for (; first != last; ++first) sum += *first;
    return sum;
}

inline float max_abs_difference(const float* a, const float* b, std::size_t size) {
    float maximum = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        maximum = std::max(maximum, std::abs(a[i] - b[i]));
    }
    return maximum;
}

struct OneDimOptions {
    std::size_t size = 1ULL << 24;
    int repetitions = 10;
    std::string output;
};

enum class ParseResult { success, help, error };

inline ParseResult parse_one_dim_options(int argc, char** argv,
                                         OneDimOptions& options,
                                         [[maybe_unused]] std::string_view program) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--size" && i + 1 < argc) {
            options.size = std::stoull(argv[++i]);
        } else if (argument == "--repetitions" && i + 1 < argc) {
            options.repetitions = std::stoi(argv[++i]);
        } else if (argument == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (argument == "--help") {
            return ParseResult::help;
        } else {
            return ParseResult::error;
        }
    }

    if (options.size == 0 || options.repetitions <= 0) {
        return ParseResult::error;
    }
    return ParseResult::success;
}

inline void print_one_dim_usage([[maybe_unused]] std::string_view program) {
    std::cerr << "Usage: " << program
              << " [--size N] [--repetitions N] [--output FILE]\n";
}


struct ImageOptions {
    int width = 1920;
    int height = 1080;
    int repetitions = 10;
    std::string output;
};

inline ParseResult parse_image_options(int argc, char** argv,
                                       ImageOptions& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--width" && i + 1 < argc) {
            options.width = std::stoi(argv[++i]);
        } else if (argument == "--height" && i + 1 < argc) {
            options.height = std::stoi(argv[++i]);
        } else if (argument == "--repetitions" && i + 1 < argc) {
            options.repetitions = std::stoi(argv[++i]);
        } else if (argument == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (argument == "--help") {
            return ParseResult::help;
        } else {
            return ParseResult::error;
        }
    }

    if (options.width < 3 || options.height < 1 || options.repetitions <= 0) {
        return ParseResult::error;
    }
    return ParseResult::success;
}

inline void print_image_usage([[maybe_unused]] std::string_view program) {
    std::cerr << "Usage: " << program
              << " [--width N] [--height N] [--repetitions N] [--output FILE]\n";
}

} // namespace simd_examples::benchmark
