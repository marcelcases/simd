#pragma once

#include <cstddef>
#include <experimental/simd>

namespace stdx = std::experimental;

namespace simd_examples {

template<class T>
using native_simd = stdx::native_simd<T>;

template<class T>
using native_mask = stdx::simd_mask<T, stdx::simd_abi::native<T>>;

}
