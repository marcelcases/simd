// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "exercises/07_conv1d.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using exercises::benchmark::OneDimOptions;
using exercises::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double time, double result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "07_conv1d,convolution,"
           << exercises::benchmark::implementation_name << ","
           << options.size << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    options.size = 1ULL << 20;
    const auto parsed = exercises::benchmark::parse_one_dim_options(
        argc, argv, options, "07_conv1d");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            exercises::benchmark::print_one_dim_usage("07_conv1d");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr std::size_t kernel_size = 3;
    const float kernel[kernel_size] = {0.25f, 0.5f, 0.125f};
    if (options.size < kernel_size) return 1;

    const std::size_t output_size = options.size - kernel_size + 1;
    std::vector<float> input(options.size), output(options.size), expected(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : input) value = distribution(rng);

    exercises::benchmark::reference::convolve_1d(
        input.data(), kernel, expected.data(), options.size, kernel_size);
    const double time = exercises::benchmark::best_time_ms(
        [&] {
            exercises::benchmark::implementation::convolve_1d(
                input.data(), kernel, output.data(), options.size, kernel_size);
        }, options.repetitions);
    const double result = exercises::benchmark::checksum(
        output.begin(), output.begin() + output_size);
    const float difference = exercises::benchmark::max_abs_difference(
        output.data(), expected.data(), output_size);

    const bool written = exercises::benchmark::write_output(
        options.output, [&](std::ostream& output_stream) {
            write_csv(output_stream, options, time, result, difference);
        });
    return written && difference <= 1e-5f ? 0 : 1;
}
