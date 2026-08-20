#include "benchmark_common.hpp"
#include "simd_examples/07_filter.hpp"
#include "benchmark_implementation.hpp"
#include "benchmark_reference.hpp"

namespace {

using simd_examples::benchmark::ImageOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const ImageOptions& options,
               double time, double result, float difference) {
    const std::size_t pixels =
        static_cast<std::size_t>(options.width) * options.height;
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "07_filter,horizontal_blur,"
           << simd_examples::benchmark::implementation_name << ","
           << pixels << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    ImageOptions options;
    const auto parsed = simd_examples::benchmark::parse_image_options(argc, argv, options);
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_image_usage("07_filter");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    const std::size_t pixels =
        static_cast<std::size_t>(options.width) * options.height;
    std::vector<float> input(pixels), output(pixels), expected(pixels);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : input) value = distribution(rng);

    simd_examples::benchmark::reference::blur_horizontal(
        input.data(), expected.data(), options.width, options.height);
    const simd_examples::ConstImageView input_view{
        options.width, options.height, input.data()};
    const simd_examples::ImageView output_view{
        options.width, options.height, output.data()};

    const double time = simd_examples::benchmark::best_time_ms(
        [&] {
            simd_examples::benchmark::implementation::blur_horizontal(
                input_view, output_view);
        }, options.repetitions);
    const double result = simd_examples::benchmark::checksum(
        output.begin(), output.end());
    const float difference = simd_examples::benchmark::max_abs_difference(
        output.data(), expected.data(), pixels);

    const bool written = simd_examples::benchmark::write_output(
        options.output, [&](std::ostream& output_stream) {
            write_csv(output_stream, options, time, result, difference);
        });
    return written && difference <= 1e-5f ? 0 : 1;
}
