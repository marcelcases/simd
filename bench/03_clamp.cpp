#include "benchmark_common.hpp"
#include "simd_examples/03_clamp.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double scalar_time, double simd_time,
               float scalar_result, float simd_result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "03_clamp,clamp,scalar," << options.size << "," << options.repetitions << ","
           << scalar_time << "," << scalar_result << "," << difference << "\n";
    output << "03_clamp,clamp,simd," << options.size << "," << options.repetitions << ","
           << simd_time << "," << simd_result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "03_clamp_bench");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("03_clamp_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr float upper_bound = 0.5f;
    std::vector<float> input(options.size), scalar_values, simd_values;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : input) value = distribution(rng);
    scalar_values = input;
    simd_values = input;

    const double scalar_time = simd_examples::benchmark::best_time_ms([&] { scalar_values = input; }, [&] {
        simd_examples::scalar::clamp(scalar_values.data(), options.size, upper_bound);
    }, options.repetitions);
    const double simd_time = simd_examples::benchmark::best_time_ms([&] { simd_values = input; }, [&] {
        simd_examples::simd::clamp(simd_values.data(), options.size, upper_bound);
    }, options.repetitions);

    const float difference = simd_examples::benchmark::max_abs_difference(
        scalar_values.data(), simd_values.data(), options.size);
    const float scalar_result = simd_examples::benchmark::checksum(
        scalar_values.begin(), scalar_values.end());
    const float simd_result = simd_examples::benchmark::checksum(
        simd_values.begin(), simd_values.end());

    if (options.output.empty()) {
        write_csv(std::cout, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    } else {
        std::ofstream output(options.output);
        if (!output) return 1;
        write_csv(output, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    }

    return difference == 0.f ? 0 : 1;
}
