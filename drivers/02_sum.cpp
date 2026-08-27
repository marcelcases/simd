// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "simd_examples/02_sum.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double time, float result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "02_sum,sum," << simd_examples::benchmark::implementation_name << ","
           << options.size << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "02_sum");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("02_sum");
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

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output) {
            write_csv(output, options, time, result, difference);
        });
    return written && simd_examples::benchmark::within_tolerance(result, expected)
        ? 0 : 1;
}
