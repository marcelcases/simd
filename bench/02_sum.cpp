#include "benchmark_common.hpp"
#include "simd_examples/02_sum.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double scalar_time, double simd_time,
               float scalar_result, float simd_result, float difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "02_sum,sum,scalar," << options.size << "," << options.repetitions << ","
           << scalar_time << "," << scalar_result << "," << difference << "\n";
    output << "02_sum,sum,simd," << options.size << "," << options.repetitions << ","
           << simd_time << "," << simd_result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "02_sum_bench");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("02_sum_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    std::vector<float> values(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const double scalar_time = simd_examples::benchmark::best_time_ms([&] {
        return simd_examples::scalar::sum(values.data(), options.size);
    }, options.repetitions);
    const double simd_time = simd_examples::benchmark::best_time_ms([&] {
        return simd_examples::simd::sum(values.data(), options.size);
    }, options.repetitions);

    const float scalar_result = simd_examples::scalar::sum(values.data(), options.size);
    const float simd_result = simd_examples::simd::sum(values.data(), options.size);
    const float difference = std::abs(scalar_result - simd_result);

    if (options.output.empty()) {
        write_csv(std::cout, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    } else {
        std::ofstream output(options.output);
        if (!output) return 1;
        write_csv(output, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    }

    return std::isfinite(scalar_result) && std::isfinite(simd_result) ? 0 : 1;
}
