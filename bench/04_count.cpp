#include "benchmark_common.hpp"
#include "simd_examples/04_count.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_csv(std::ostream& output, const OneDimOptions& options,
               double scalar_time, double simd_time,
               std::size_t scalar_result, std::size_t simd_result,
               std::size_t difference) {
    output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    output << "04_count,count,scalar," << options.size << "," << options.repetitions << ","
           << scalar_time << "," << scalar_result << "," << difference << "\n";
    output << "04_count,count,simd," << options.size << "," << options.repetitions << ","
           << simd_time << "," << simd_result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "04_count_bench");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("04_count_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    constexpr float threshold = 0.f;
    std::vector<float> values(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (auto& value : values) value = distribution(rng);

    const double scalar_time = simd_examples::benchmark::best_time_ms([&] {
        return simd_examples::scalar::count_above(values.data(), options.size, threshold);
    }, options.repetitions);
    const double simd_time = simd_examples::benchmark::best_time_ms([&] {
        return simd_examples::simd::count_above(values.data(), options.size, threshold);
    }, options.repetitions);

    const std::size_t scalar_result =
        simd_examples::scalar::count_above(values.data(), options.size, threshold);
    const std::size_t simd_result =
        simd_examples::simd::count_above(values.data(), options.size, threshold);
    const std::size_t difference = scalar_result > simd_result
        ? scalar_result - simd_result
        : simd_result - scalar_result;

    if (options.output.empty()) {
        write_csv(std::cout, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    } else {
        std::ofstream output(options.output);
        if (!output) return 1;
        write_csv(output, options, scalar_time, simd_time,
                  scalar_result, simd_result, difference);
    }

    return difference == 0 ? 0 : 1;
}
