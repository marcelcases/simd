#include "simd_examples/03_clamp.hpp"

namespace simd_examples::scalar {

void clamp(float* values, std::size_t size, float upper_bound) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > upper_bound) {
            values[i] = upper_bound;
        }
    }
}

}
