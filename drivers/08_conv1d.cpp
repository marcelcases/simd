#include "benchmark_common.hpp"
#include "simd_examples/08_conv1d.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double time, float result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "08_conv1d,convolution,"
           << simd_examples::benchmark::implementation_name << ","
           << options.size << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    options.size = 1ULL << 20;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "08_conv1d");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("08_conv1d");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr int kernel_size = 3;
    const float kernel[kernel_size] = {0.25f, 0.5f, 0.25f};
    if (options.size < static_cast<std::size_t>(kernel_size)) return 1;

    const std::size_t output_size = options.size - kernel_size + 1;
    std::vector<float> input(options.size), output(options.size), expected(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : input) value = distribution(rng);

    simd_examples::benchmark::reference::convolve_1d(
        input.data(), kernel, expected.data(), options.size, kernel_size);
    const double time = simd_examples::benchmark::best_time_ms(
        [&] {
            simd_examples::benchmark::implementation::convolve_1d(
                input.data(), kernel, output.data(), options.size, kernel_size);
        }, options.repetitions);
    const float result = simd_examples::benchmark::checksum(
        output.begin(), output.begin() + output_size);
    const float difference = simd_examples::benchmark::max_abs_difference(
        output.data(), expected.data(), output_size);

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output_stream) {
            write_csv(output_stream, options, time, result, difference);
        });
    return written && difference <= 1e-5f ? 0 : 1;
}
