// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "simd_examples/01_add_fma.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;
using simd_examples::benchmark::TimingResult;

void write_row(std::ostream& output, const OneDimOptions& options,
               const char* kernel, const TimingResult& timing,
               double result, float difference) {
    output << "01_add_fma," << kernel << ","
           << simd_examples::benchmark::implementation_name << ","
           << options.size << "," << options.warmups << ","
           << options.iterations << "," << options.samples << ","
           << timing.median_ms << "," << timing.minimum_ms << ","
           << timing.maximum_ms << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "01_add_fma");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("01_add_fma");
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

    const auto add_timing = simd_examples::benchmark::measure_kernel_ms(
        [&] { destination = source; },
        [&] {
            simd_examples::benchmark::implementation::add(
                destination.data(), source.data(), options.size);
        }, options.warmups, options.iterations, options.samples);

    destination = source;
    simd_examples::benchmark::implementation::add(
        destination.data(), source.data(), options.size);
    const double add_result = simd_examples::benchmark::checksum(
        destination.begin(), destination.end());
    const float add_difference = simd_examples::benchmark::max_abs_difference(
        destination.data(), expected.data(), options.size);

    std::vector<float> a(options.size), b(options.size), c(options.size);
    std::vector<float> output(options.size), expected_output(options.size);
    std::mt19937 kernel_rng(42);
    std::uniform_real_distribution<float> kernel_distribution(-1.f, 1.f);
    for (std::size_t i = 0; i < options.size; ++i) {
        a[i] = kernel_distribution(kernel_rng);
        b[i] = kernel_distribution(kernel_rng);
        c[i] = kernel_distribution(kernel_rng);
    }

    simd_examples::benchmark::reference::fma_memory_bound(
        a.data(), b.data(), c.data(), expected_output.data(), options.size);
    const auto fma_timing = simd_examples::benchmark::measure_kernel_ms(
        [] {},
        [&] {
            simd_examples::benchmark::implementation::fma_memory_bound(
                a.data(), b.data(), c.data(), output.data(), options.size);
        }, options.warmups, options.iterations, options.samples);

    simd_examples::benchmark::implementation::fma_memory_bound(
        a.data(), b.data(), c.data(), output.data(), options.size);
    const double fma_result = simd_examples::benchmark::checksum(
        output.begin(), output.end());
    const float fma_difference = simd_examples::benchmark::max_abs_difference(
        output.data(), expected_output.data(), options.size);

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output_stream) {
            output_stream << "exercise,kernel,implementation,size,warmups,iterations,samples,median_time_ms,min_time_ms,max_time_ms,result,max_abs_difference\n";
            write_row(output_stream, options, "add", add_timing,
                      add_result, add_difference);
            write_row(output_stream, options, "memory_fma", fma_timing,
                      fma_result, fma_difference);
        });
    return written && add_difference <= 1e-6f && fma_difference <= 1e-5f
        ? 0 : 1;
}
