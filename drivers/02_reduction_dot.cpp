// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "simd_examples/02_reduction_dot.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_row(std::ostream& output, const OneDimOptions& options,
               const char* kernel, double time, double result, float difference) {
    output << "02_reduction_dot," << kernel << ","
           << simd_examples::benchmark::implementation_name << ","
           << options.size << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "02_reduction_dot");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("02_reduction_dot");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    std::vector<float> values(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const float expected = simd_examples::benchmark::reference::sum(
        values.data(), options.size);
    const double time = simd_examples::benchmark::best_time_ms(
        [&]() -> float {
            return simd_examples::benchmark::implementation::sum(
                values.data(), options.size);
        }, options.repetitions);
    const float result = simd_examples::benchmark::implementation::sum(
        values.data(), options.size);
    const float difference = std::abs(result - expected);

    std::vector<float> a(options.size), b(options.size);
    std::mt19937 kernel_rng(42);
    std::uniform_real_distribution<float> kernel_distribution(-1.f, 1.f);
    for (std::size_t i = 0; i < options.size; ++i) {
        a[i] = kernel_distribution(kernel_rng);
        b[i] = kernel_distribution(kernel_rng);
        (void)kernel_distribution(kernel_rng); // Preserve the original input sequence.
    }

    const float expected_dot = simd_examples::benchmark::reference::dot_product(
        a.data(), b.data(), options.size);
    const double dot_time = simd_examples::benchmark::best_time_ms(
        [&]() -> float {
            return simd_examples::benchmark::implementation::dot_product(
                a.data(), b.data(), options.size);
        }, options.repetitions);
    const float dot_result = simd_examples::benchmark::implementation::dot_product(
        a.data(), b.data(), options.size);
    const float dot_difference = std::abs(dot_result - expected_dot);

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output) {
            output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
            write_row(output, options, "sum", time, result, difference);
            write_row(output, options, "dot_product", dot_time,
                      dot_result, dot_difference);
        });
    return written && simd_examples::benchmark::within_tolerance(dot_result, expected_dot) &&
        simd_examples::benchmark::within_tolerance(result, expected)
        ? 0 : 1;
}
