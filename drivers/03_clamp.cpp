// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "simd_examples/03_clamp.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;
using simd_examples::benchmark::TimingResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               const TimingResult& timing, double result, float difference) {
    output << "exercise,kernel,implementation,size,warmups,iterations,samples,median_time_ms,min_time_ms,max_time_ms,result,max_abs_difference\n";
    output << "03_clamp,clamp," << simd_examples::benchmark::implementation_name
           << "," << options.size << "," << options.warmups << ","
           << options.iterations << "," << options.samples << ","
           << timing.median_ms << "," << timing.minimum_ms << ","
           << timing.maximum_ms << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "03_clamp");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("03_clamp");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr float upper_bound = 0.5f;
    std::vector<float> input(options.size), values, expected;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : input) value = distribution(rng);
    values = input;
    expected = input;
    simd_examples::benchmark::reference::clamp(
        expected.data(), options.size, upper_bound);

    std::vector<std::vector<float>> sample_values(
        static_cast<std::size_t>(options.iterations), input);
    std::size_t current_iteration = 0;
    const auto timing = simd_examples::benchmark::measure_kernel_ms(
        [&] {
            for (auto& sample : sample_values) {
                sample = input;
            }
            current_iteration = 0;
        },
        [&] {
            simd_examples::benchmark::implementation::clamp(
                sample_values[current_iteration].data(), options.size,
                upper_bound);
            ++current_iteration;
        }, options.warmups, options.iterations, options.samples);

    values = input;
    simd_examples::benchmark::implementation::clamp(
        values.data(), options.size, upper_bound);
    const double result = simd_examples::benchmark::checksum(
        values.begin(), values.end());
    const float difference = simd_examples::benchmark::max_abs_difference(
        values.data(), expected.data(), options.size);

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output) {
            write_csv(output, options, timing, result, difference);
        });
    return written && difference <= 1e-6f ? 0 : 1;
}
