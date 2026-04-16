# RISC-V Support Status

## Summary

The RISC-V emulation environment is **fully functional** for running the `std::simd` examples, but there's an important limitation: **GCC 13.3's `std::experimental::simd` implementation does not yet support RISC-V Vector extensions (RVV)**. The code falls back to scalar operations.

## Current Setup

✅ **Working:**
- QEMU 8.2.2 with RISC-V user-mode emulation
- GCC 13.3 RISC-V cross-compiler with RVV support (`-march=rv64gcv`)
- All 7 examples compile and run successfully
- Code is portable and ready for future RVV support

❌ **Limitation:**
- `std::experimental::simd` falls back to `simd_abi::scalar` on RISC-V
- SIMD width = 1 (no vectorization through std::simd API)
- Generated assembly uses scalar FP instructions (`flw`, `fadd.s`, `fsw`) instead of vector instructions

## Installation (Ubuntu 24.04)

```bash
# Install QEMU and RISC-V toolchain
sudo apt install -y qemu-user-static gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Verify installation
qemu-riscv64-static --version  # Should show QEMU 8.2.2+
riscv64-linux-gnu-g++ --version  # Should show GCC 13.3+
```

## Building and Running

```bash
# Build all RISC-V binaries
make riscv

# Run examples (VLEN doesn't matter currently since using scalar ABI)
make run-riscv-128

# Generate assembly to verify instructions
make verify-riscv
```

## Results (QEMU Emulation on x86_64 Host)

| Example | Scalar time | SIMD time | Speedup | Notes |
|---------|-------------|-----------|---------|-------|
| 01_add | 266.8 ms | 268.3 ms | **1.00x** | No vectorization |
| 02_sum | 129.1 ms | 129.0 ms | **1.00x** | No vectorization |
| 03_clamp | 86.8 ms | 86.9 ms | **1.00x** | No vectorization |
| 04_count | 221.8 ms | 221.5 ms | **1.00x** | No vectorization |
| 05_softmax | 153.0 ms | 82.3 ms | **1.86x** | Unexpectedly faster! |
| 06_fma | 34.6 ms | 34.5 ms | **1.00x** | No vectorization |
| 07_filter | 39.4 ms | 52.7 ms | **0.75x** | Overhead without vectorization |

**Important:** Times are from QEMU emulation (slow) and not representative of real RISC-V hardware performance.

## Technical Details

### Why No Vectorization?

When compiled for RISC-V, `std::experimental::native_simd<float>` resolves to:
- ABI type: `std::experimental::simd_abi::_Scalar`
- SIMD width: 1
- This is because GCC's libstdc++ doesn't yet map the RISC-V Vector extension to the `std::simd` ABI

### Verification

```bash
# Check what ABI is used
$ cat test.cpp
#include <experimental/simd>
#include <iostream>
int main() {
    using V = std::experimental::native_simd<float>;
    std::cout << "Width: " << V::size() << std::endl;
    std::cout << "Is scalar: " << std::is_same_v<typename V::abi_type, 
        std::experimental::simd_abi::scalar> << std::endl;
}

$ riscv64-linux-gnu-g++ -std=c++2b -O3 -march=rv64gcv -static test.cpp -o test.riscv
$ qemu-riscv64-static -cpu rv64,v=true,vlen=128 ./test.riscv
Width: 1
Is scalar: 1
```

### Expected RVV Instructions (Not Generated)

If `std::simd` supported RVV, we would see:
- `vsetvli` - Set vector length
- `vle32.v` - Vector load (32-bit elements)
- `vse32.v` - Vector store
- `vfadd.vv` - Vector float add
- `vfmul.vv` - Vector float multiply
- `vfmacc.vv` - Vector FMA

Instead, we see scalar instructions:
- `flw` - Float load word
- `fadd.s` - Scalar float add
- `fsw` - Float store word

## Future Outlook

### When Will RVV Be Supported?

The C++26 `std::simd` specification is still evolving, and compiler support is incremental:

1. **x86 AVX/AVX-512**: Well supported in GCC (experimental)
2. **ARM NEON**: Supported in GCC (experimental)
3. **RISC-V RVV**: Not yet supported in `std::experimental::simd`

Potential timeline:
- **GCC 14/15**: May add RVV support to `std::experimental::simd`
- **C++26 finalization**: Official `std::simd` (not experimental)
- **2026-2027**: Broader compiler support

### Workarounds for Now

If you need actual RVV vectorization today:

1. **Use intrinsics directly** (not portable):
   ```cpp
   #include <riscv_vector.h>
   vfloat32m1_t va = vle32_v_f32m1(ptr, vl);
   ```

2. **Wait for GCC updates**: Monitor GCC development for RVV `std::simd` support

3. **Use auto-vectorization**: GCC can auto-vectorize simple loops with `-march=rv64gcv -ftree-vectorize`

### Testing Auto-Vectorization

```bash
# Try enabling auto-vectorization for scalar functions
riscv64-linux-gnu-g++ -std=c++2b -O3 -march=rv64gcv -ftree-vectorize \
  -fopt-info-vec-optimized -S 01_add.cpp -o 01_add.s
```

This may vectorize simple loops automatically, though it won't use the `std::simd` API.

## Value of Current Setup

Even though vectorization isn't working yet, this setup is valuable because:

1. ✅ **Code portability verified** - Same source compiles for x86, ARM, and RISC-V
2. ✅ **Ready for future support** - When GCC adds RVV support, recompile and it works
3. ✅ **Educational** - Shows how `std::simd` gracefully degrades to scalar
4. ✅ **Emulation infrastructure** - QEMU environment ready for testing
5. ✅ **Baseline established** - Can compare future vectorized performance

## References

- [GCC RISC-V Vector Extension Support](https://gcc.gnu.org/projects/riscv-vector.html)
- [RISC-V Vector Extension Specification](https://github.com/riscv/riscv-v-spec)
- [GCC libstdc++ SIMD Support](https://gcc.gnu.org/onlinedocs/libstdc++/manual/simd_support.html)
- [C++26 std::simd Proposal P1928](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p1928r3.html)
