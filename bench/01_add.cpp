#include "common.h"
#include "simd_examples/01_add.hpp"
#include "simd_common.h"

#include <fstream>
#include <string_view>

namespace {

struct Options {
    std::size_t size = 1ULL << 24;
    int repetitions = 10;
    std::string output;
};

void print_usage() {
    std::cerr << "Usage: 01_add_bench [--size N] [--repetitions N] [--output FILE]\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--size" && i + 1 < argc) {
            options.size = std::stoull(argv[++i]);
        } else if (argument == "--repetitions" && i + 1 < argc) {
            options.repetitions = std::stoi(argv[++i]);
        } else if (argument == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (argument == "--help") {
            print_usage();
            return false;
        } else {
            std::cerr << "Unknown or incomplete option: " << argument << "\n";
            print_usage();
            return false;
        }
    }

    if (options.size == 0 || options.repetitions <= 0) {
        std::cerr << "--size and --repetitions must be positive\n";
        return false;
    }
    return true;
}

void write_csv(std::ostream& output, const Options& options,
               double scalar_time, double simd_time,
               float scalar_checksum, float simd_checksum, float max_difference) {
    output << "exercise,implementation,size,repetitions,time_ms,checksum,max_abs_difference\n";
    output << "01_add,scalar," << options.size << "," << options.repetitions << ","
           << scalar_time << "," << scalar_checksum << "," << max_difference << "\n";
    output << "01_add,simd," << options.size << "," << options.repetitions << ","
           << simd_time << "," << simd_checksum << "," << max_difference << "\n";
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) return 1;

    std::vector<float> source(options.size);
    std::vector<float> scalar_destination(options.size);
    std::vector<float> simd_destination(options.size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    for (auto& value : source) value = distribution(rng);
    scalar_destination = source;
    simd_destination = source;

    const double scalar_time = simd_examples::bench_ms([&]() -> float {
        simd_examples::scalar::add(
            scalar_destination.data(), source.data(), options.size);
        return simd_examples::checksum(
            scalar_destination.begin(), scalar_destination.end());
    }, options.repetitions);

    const double simd_time = simd_examples::bench_ms([&]() -> float {
        simd_examples::simd::add(
            simd_destination.data(), source.data(), options.size);
        return simd_examples::checksum(
            simd_destination.begin(), simd_destination.end());
    }, options.repetitions);

    float max_difference = 0.f;
    for (std::size_t i = 0; i < options.size; ++i) {
        max_difference = std::max(
            max_difference,
            std::abs(scalar_destination[i] - simd_destination[i]));
    }

    const float scalar_checksum = simd_examples::checksum(
        scalar_destination.begin(), scalar_destination.end());
    const float simd_checksum = simd_examples::checksum(
        simd_destination.begin(), simd_destination.end());

    if (options.output.empty()) {
        write_csv(std::cout, options, scalar_time, simd_time,
                  scalar_checksum, simd_checksum, max_difference);
    } else {
        std::ofstream output(options.output);
        if (!output) {
            std::cerr << "Cannot write " << options.output << "\n";
            return 1;
        }
        write_csv(output, options, scalar_time, simd_time,
                  scalar_checksum, simd_checksum, max_difference);
    }

    return max_difference == 0.f ? 0 : 1;
}
