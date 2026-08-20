#include "simd_examples/02_sum.hpp"

namespace simd_examples::scalar {

float sum(const float* values, std::size_t size) noexcept {
    float result = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        result += values[i];
    }
    return result;
}

}
