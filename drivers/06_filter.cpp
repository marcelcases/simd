// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "benchmark_common.hpp"
#include "exercises/06_filter.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using exercises::benchmark::ImageOptions;
using exercises::benchmark::ParseResult;

void write_csv(std::ostream& output, const ImageOptions& options,
               double time, double result, float difference) {
    const std::size_t pixels =
        static_cast<std::size_t>(options.width) * options.height;
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "06_filter,horizontal_blur,"
           << exercises::benchmark::implementation_name << ","
           << pixels << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    ImageOptions options;
    const auto parsed = exercises::benchmark::parse_image_options(argc, argv, options);
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            exercises::benchmark::print_image_usage("06_filter");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    const std::size_t pixels =
        static_cast<std::size_t>(options.width) * options.height;
    std::vector<float> input(pixels), output(pixels), expected(pixels);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : input) value = distribution(rng);

    exercises::benchmark::reference::blur_horizontal(
        input.data(), expected.data(), options.width, options.height);

    const double time = exercises::benchmark::best_time_ms(
        [&] {
            exercises::benchmark::implementation::blur_horizontal(
                input.data(), output.data(), options.width, options.height);
        }, options.repetitions);
    const double result = exercises::benchmark::checksum(
        output.begin(), output.end());
    const float difference = exercises::benchmark::max_abs_difference(
        output.data(), expected.data(), pixels);

    const bool written = exercises::benchmark::write_output(
        options.output, [&](std::ostream& output_stream) {
            write_csv(output_stream, options, time, result, difference);
        });
    return written && difference <= 1e-5f ? 0 : 1;
}
