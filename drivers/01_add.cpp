// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "simd_examples/01_add.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double time, double result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "01_add,add," << simd_examples::benchmark::implementation_name << ","
           << options.size << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "01_add");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("01_add");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    std::vector<float> source(options.size), destination, expected;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : source) value = distribution(rng);
    destination = source;
    expected = source;
    simd_examples::benchmark::reference::add(
        expected.data(), source.data(), options.size);

    const double time = simd_examples::benchmark::best_time_ms(
        [&] { destination = source; },
        [&]() -> double {
            simd_examples::benchmark::implementation::add(
                destination.data(), source.data(), options.size);
            return simd_examples::benchmark::checksum(
                destination.begin(), destination.end());
        }, options.repetitions);
    const double result = simd_examples::benchmark::checksum(
        destination.begin(), destination.end());
    const float difference = simd_examples::benchmark::max_abs_difference(
        destination.data(), expected.data(), options.size);

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output) {
            write_csv(output, options, time, result, difference);
        });
    return written && difference <= 1e-6f ? 0 : 1;
}
