#include "simd_examples/05_softmax.hpp"

#include <algorithm>
#include <cmath>

namespace simd_examples::scalar {

void softmax(float* values, std::size_t size) noexcept {
    if (size == 0) {
        return;
    }

    float maximum = values[0];
    for (std::size_t i = 1; i < size; ++i) {
        maximum = std::max(maximum, values[i]);
    }

    float total = 0.f;
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        total += values[i];
    }

    for (std::size_t i = 0; i < size; ++i) {
        values[i] /= total;
    }
}

}
