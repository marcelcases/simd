#include "benchmark_common.hpp"
#include "simd_examples/08_conv1d.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double scalar_time, double simd_time,
               float scalar_result, float simd_result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "08_conv1d,convolution,scalar," << options.size << "," << options.repetitions << ","
           << scalar_time << "," << scalar_result << "," << difference << "\n";
    output << "08_conv1d,convolution,simd," << options.size << "," << options.repetitions << ","
           << simd_time << "," << simd_result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    options.size = 1ULL << 20;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "08_conv1d_bench");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("08_conv1d_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr int kernel_size = 3;
    const float kernel[kernel_size] = {0.25f, 0.5f, 0.25f};
    if (options.size < static_cast<std::size_t>(kernel_size)) return 1;

    std::vector<float> input(options.size), scalar_output(options.size), simd_output(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : input) value = distribution(rng);

    const double scalar_time = simd_examples::benchmark::best_time_ms([&] {
        simd_examples::scalar::convolve_1d(input.data(), kernel, scalar_output.data(),
                                            options.size, kernel_size);
    }, options.repetitions);
    const double simd_time = simd_examples::benchmark::best_time_ms([&] {
        simd_examples::simd::convolve_1d(input.data(), kernel, simd_output.data(),
                                          options.size, kernel_size);
    }, options.repetitions);

    const std::size_t output_size = options.size - kernel_size + 1;
    const float difference = simd_examples::benchmark::max_abs_difference(
        scalar_output.data(), simd_output.data(), output_size);
    const float scalar_result = simd_examples::benchmark::checksum(
        scalar_output.begin(), scalar_output.begin() + output_size);
    const float simd_result = simd_examples::benchmark::checksum(
        simd_output.begin(), simd_output.begin() + output_size);

    if (options.output.empty()) {
        write_csv(std::cout, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    } else {
        std::ofstream output(options.output);
        if (!output) return 1;
        write_csv(output, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    }

    return difference < 1e-5f ? 0 : 1;
}
