#include "benchmark_common.hpp"
#include "simd_examples/07_filter.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::ImageOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const ImageOptions& options,
               double scalar_time, double simd_time,
               float scalar_result, float simd_result, float difference) {
    const std::size_t pixels = static_cast<std::size_t>(options.width) * options.height;
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "07_filter,horizontal_blur,scalar," << pixels << "," << options.repetitions << ","
           << scalar_time << "," << scalar_result << "," << difference << "\n";
    output << "07_filter,horizontal_blur,simd," << pixels << "," << options.repetitions << ","
           << simd_time << "," << simd_result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    ImageOptions options;
    const auto parsed = simd_examples::benchmark::parse_image_options(argc, argv, options);
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_image_usage("07_filter_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    const std::size_t pixels = static_cast<std::size_t>(options.width) * options.height;
    std::vector<float> input(pixels), scalar_output(pixels), simd_output(pixels);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : input) value = distribution(rng);

    const simd_examples::ConstImageView input_view{
        options.width, options.height, input.data()};
    const simd_examples::ImageView scalar_view{
        options.width, options.height, scalar_output.data()};
    const simd_examples::ImageView simd_view{
        options.width, options.height, simd_output.data()};

    const double scalar_time = simd_examples::benchmark::best_time_ms([&] {
        simd_examples::scalar::blur_horizontal(input_view, scalar_view);
    }, options.repetitions);
    const double simd_time = simd_examples::benchmark::best_time_ms([&] {
        simd_examples::simd::blur_horizontal(input_view, simd_view);
    }, options.repetitions);

    const float difference = simd_examples::benchmark::max_abs_difference(
        scalar_output.data(), simd_output.data(), pixels);
    const float scalar_result = simd_examples::benchmark::checksum(
        scalar_output.begin(), scalar_output.end());
    const float simd_result = simd_examples::benchmark::checksum(
        simd_output.begin(), simd_output.end());

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
