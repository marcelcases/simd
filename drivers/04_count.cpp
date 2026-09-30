// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "exercises/04_count.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using exercises::benchmark::OneDimOptions;
using exercises::benchmark::ParseResult;
using exercises::benchmark::TimingResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               const TimingResult& timing, std::size_t result,
               std::size_t difference) {
    output << "exercise,kernel,implementation,size,warmups,iterations,samples,median_time_ms,min_time_ms,max_time_ms,result,max_abs_difference\n";
    output << "04_count,count," << exercises::benchmark::implementation_name
           << "," << options.size << "," << options.warmups << ","
           << options.iterations << "," << options.samples << ","
           << timing.median_ms << "," << timing.minimum_ms << ","
           << timing.maximum_ms << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = exercises::benchmark::parse_one_dim_options(
        argc, argv, options, "04_count");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            exercises::benchmark::print_one_dim_usage("04_count");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr float threshold = 0.f;
    std::vector<float> values(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const std::size_t expected = exercises::benchmark::reference::count_above(
        values.data(), options.size, threshold);
    const auto timing = exercises::benchmark::measure_kernel_ms(
        [] {},
        [&]() -> std::size_t {
            return exercises::benchmark::implementation::count_above(
                values.data(), options.size, threshold);
        }, options.warmups, options.iterations, options.samples);
    const std::size_t result = exercises::benchmark::implementation::count_above(
        values.data(), options.size, threshold);
    const std::size_t difference = result > expected
        ? result - expected : expected - result;

    const bool written = exercises::benchmark::write_output(
        options.output, [&](std::ostream& output) {
            write_csv(output, options, timing, result, difference);
        });
    return written && difference == 0 ? 0 : 1;
}
