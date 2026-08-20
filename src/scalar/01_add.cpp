#include "simd_examples/01_add.hpp"

namespace simd_examples::scalar {

void add(float* destination, const float* source, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        destination[i] += source[i];
    }
}

}
