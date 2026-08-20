#include "benchmark_common.hpp"
#include "simd_examples/05_softmax.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double scalar_time, double simd_time,
               float scalar_sum, float simd_sum, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "05_softmax,softmax,scalar," << options.size << "," << options.repetitions << ","
           << scalar_time << "," << scalar_sum << "," << difference << "\n";
    output << "05_softmax,softmax,simd," << options.size << "," << options.repetitions << ","
           << simd_time << "," << simd_sum << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    options.size = 1ULL << 20;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "05_softmax_bench");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("05_softmax_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    std::vector<float> input(options.size), scalar_values, simd_values;
    std::mt19937 rng(42);
    std::normal_distribution<float> distribution(0.f, 1.f);
    for (auto& value : input) value = distribution(rng);
    scalar_values = input;
    simd_values = input;

    const double scalar_time = simd_examples::benchmark::best_time_ms([&] { scalar_values = input; }, [&] {
        simd_examples::scalar::softmax(scalar_values.data(), options.size);
    }, options.repetitions);
    const double simd_time = simd_examples::benchmark::best_time_ms([&] { simd_values = input; }, [&] {
        simd_examples::simd::softmax(simd_values.data(), options.size);
    }, options.repetitions);

    const float scalar_sum = simd_examples::benchmark::checksum(
        scalar_values.begin(), scalar_values.end());
    const float simd_sum = simd_examples::benchmark::checksum(
        simd_values.begin(), simd_values.end());
    const float difference = simd_examples::benchmark::max_abs_difference(
        scalar_values.data(), simd_values.data(), options.size);

    if (options.output.empty()) {
        write_csv(std::cout, options, scalar_time, simd_time,
                  scalar_sum, simd_sum, difference);
    } else {
        std::ofstream output(options.output);
        if (!output) return 1;
        write_csv(output, options, scalar_time, simd_time,
                  scalar_sum, simd_sum, difference);
    }

    return std::abs(scalar_sum - 1.f) < 1e-3f && std::abs(simd_sum - 1.f) < 1e-3f ? 0 : 1;
}
