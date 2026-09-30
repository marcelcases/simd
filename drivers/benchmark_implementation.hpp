// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Marcel Cases Freixenet

#pragma once

#include <string_view>

#if defined(SIMD_EXAMPLES_SCALAR) && defined(SIMD_EXAMPLES_SIMD)
#error "Select exactly one implementation"
#elif !defined(SIMD_EXAMPLES_SCALAR) && !defined(SIMD_EXAMPLES_SIMD)
#error "Select an implementation"
#endif

namespace exercises::benchmark {

#if defined(SIMD_EXAMPLES_SCALAR)
namespace implementation = ::exercises::scalar;
inline constexpr std::string_view implementation_name = "scalar";
#else
namespace implementation = ::exercises::simd;
inline constexpr std::string_view implementation_name = "simd";
#endif

} // namespace exercises::benchmark
