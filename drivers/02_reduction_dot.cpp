// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "exercises/02_reduction_dot.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using exercises::benchmark::OneDimOptions;
using exercises::benchmark::ParseResult;
using exercises::benchmark::TimingResult;

void write_row(std::ostream& output, const OneDimOptions& options,
               const char* kernel, const TimingResult& timing,
               double result, float difference) {
    output << "02_reduction_dot," << kernel << ","
           << exercises::benchmark::implementation_name << ","
           << options.size << "," << options.warmups << ","
           << options.iterations << "," << options.samples << ","
           << timing.median_ms << "," << timing.minimum_ms << ","
           << timing.maximum_ms << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = exercises::benchmark::parse_one_dim_options(
        argc, argv, options, "02_reduction_dot");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            exercises::benchmark::print_one_dim_usage("02_reduction_dot");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    std::vector<float> values(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const float expected = exercises::benchmark::reference::sum(
        values.data(), options.size);
    const auto sum_timing = exercises::benchmark::measure_kernel_ms(
        [] {},
        [&]() -> float {
            return exercises::benchmark::implementation::sum(
                values.data(), options.size);
        }, options.warmups, options.iterations, options.samples);
    const float result = exercises::benchmark::implementation::sum(
        values.data(), options.size);
    const float difference = std::abs(result - expected);

    std::vector<float> a(options.size), b(options.size);
    std::mt19937 kernel_rng(42);
    std::uniform_real_distribution<float> kernel_distribution(-1.f, 1.f);
    for (std::size_t i = 0; i < options.size; ++i) {
        a[i] = kernel_distribution(kernel_rng);
        b[i] = kernel_distribution(kernel_rng);
        (void)kernel_distribution(kernel_rng);
    }

    const float expected_dot = exercises::benchmark::reference::dot_product(
        a.data(), b.data(), options.size);
    const auto dot_timing = exercises::benchmark::measure_kernel_ms(
        [] {},
        [&]() -> float {
            return exercises::benchmark::implementation::dot_product(
                a.data(), b.data(), options.size);
        }, options.warmups, options.iterations, options.samples);
    const float dot_result = exercises::benchmark::implementation::dot_product(
        a.data(), b.data(), options.size);
    const float dot_difference = std::abs(dot_result - expected_dot);

    const bool written = exercises::benchmark::write_output(
        options.output, [&](std::ostream& output_stream) {
            output_stream << "exercise,kernel,implementation,size,warmups,iterations,samples,median_time_ms,min_time_ms,max_time_ms,result,max_abs_difference\n";
            write_row(output_stream, options, "sum", sum_timing,
                      result, difference);
            write_row(output_stream, options, "dot_product", dot_timing,
                      dot_result, dot_difference);
        });
    return written &&
        exercises::benchmark::within_tolerance(result, expected) &&
        exercises::benchmark::within_tolerance(dot_result, expected_dot)
        ? 0 : 1;
}
