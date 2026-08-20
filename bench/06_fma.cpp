#include "benchmark_common.hpp"
#include "simd_examples/06_fma.hpp"

#include <fstream>

namespace {

using simd_examples::benchmark::OneDimOptions;
using simd_examples::benchmark::ParseResult;

void write_row(std::ostream& output, const OneDimOptions& options,
               const char* kernel, const char* implementation,
               double time, float result, float difference) {
    output << "06_fma," << kernel << "," << implementation << ","
           << options.size << "," << options.repetitions << ","
           << time << "," << result << "," << difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    OneDimOptions options;
    const auto parsed = simd_examples::benchmark::parse_one_dim_options(
        argc, argv, options, "06_fma_bench");
    if (parsed != ParseResult::success) {
        if (parsed == ParseResult::error) {
            simd_examples::benchmark::print_one_dim_usage("06_fma_bench");
        }
        return parsed == ParseResult::help ? 0 : 1;
    }

    std::vector<float> a(options.size), b(options.size), c(options.size);
    std::vector<float> scalar_output(options.size), simd_output(options.size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (std::size_t i = 0; i < options.size; ++i) {
        a[i] = distribution(rng);
        b[i] = distribution(rng);
        c[i] = distribution(rng);
    }

    const double scalar_memory_time = simd_examples::benchmark::best_time_ms([&] {
        simd_examples::scalar::fma_memory_bound(
            a.data(), b.data(), c.data(), scalar_output.data(), options.size);
    }, options.repetitions);
    const double simd_memory_time = simd_examples::benchmark::best_time_ms([&] {
        simd_examples::simd::fma_memory_bound(
            a.data(), b.data(), c.data(), simd_output.data(), options.size);
    }, options.repetitions);
    const float memory_difference = simd_examples::benchmark::max_abs_difference(
        scalar_output.data(), simd_output.data(), options.size);
    const float scalar_memory_result = simd_examples::benchmark::checksum(
        scalar_output.begin(), scalar_output.end());
    const float simd_memory_result = simd_examples::benchmark::checksum(
        simd_output.begin(), simd_output.end());

    const double scalar_dot_time = simd_examples::benchmark::best_time_ms([&] {
        return simd_examples::scalar::dot_product(a.data(), b.data(), options.size);
    }, options.repetitions);
    const double simd_dot_time = simd_examples::benchmark::best_time_ms([&] {
        return simd_examples::simd::dot_product(a.data(), b.data(), options.size);
    }, options.repetitions);
    const float scalar_dot_result =
        simd_examples::scalar::dot_product(a.data(), b.data(), options.size);
    const float simd_dot_result =
        simd_examples::simd::dot_product(a.data(), b.data(), options.size);
    const float dot_difference = std::abs(scalar_dot_result - simd_dot_result);

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!options.output.empty()) {
        file.open(options.output);
        if (!file) return 1;
        output = &file;
    }

    *output << "exercise,kernel,implementation,size,repetitions,time_ms,result,max_abs_difference\n";
    write_row(*output, options, "memory_fma", "scalar", scalar_memory_time,
              scalar_memory_result, memory_difference);
    write_row(*output, options, "memory_fma", "simd", simd_memory_time,
              simd_memory_result, memory_difference);
    write_row(*output, options, "dot_product", "scalar", scalar_dot_time,
              scalar_dot_result, dot_difference);
    write_row(*output, options, "dot_product", "simd", simd_dot_time,
              simd_dot_result, dot_difference);

    return memory_difference < 1e-5f && std::isfinite(dot_difference) ? 0 : 1;
}
