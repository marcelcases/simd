#include "simd_examples/04_count.hpp"

namespace simd_examples::scalar {

std::size_t count_above(const float* values, std::size_t size,
                        float threshold) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > threshold) {
            ++count;
        }
    }
    return count;
}

}
