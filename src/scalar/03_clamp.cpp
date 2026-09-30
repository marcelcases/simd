// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#include "exercises/03_clamp.hpp"

namespace exercises::scalar {

void clamp(float* values, std::size_t size, float upper_bound) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > upper_bound) {
            values[i] = upper_bound;
        }
    }
}

}
