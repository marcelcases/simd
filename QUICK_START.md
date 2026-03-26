# Quick Start Guide

This guide gets you running SIMD examples on all supported platforms in 5 minutes.

## Platform-Specific Setup

### Linux (x86-64 with AVX-512)

```bash
# No additional setup needed - use system GCC
make          # Build all examples
make run      # Run all examples
```

### macOS (Apple Silicon or Intel)

```bash
# Install Homebrew GCC (Apple Clang doesn't support std::simd)
brew install gcc

# Build and run
make          # Automatically uses g++-15
make run
```

### RISC-V Emulation (Ubuntu/Debian)

```bash
# Install QEMU and RISC-V cross-compiler (one-time setup)
sudo apt install -y qemu-user-static gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Build RISC-V binaries
make riscv

# Run with VLEN=128 (4 floats per register, like ARM NEON)
make run-riscv-128

# Run with VLEN=512 (16 floats per register, like AVX-512)
make run-riscv-512

# Run both configurations
make run-riscv-both
```

**Note**: Currently falls back to scalar code due to lack of GCC support for RVV in `std::simd`. See [RISCV_STATUS.md](RISCV_STATUS.md).

## Understanding the Examples

### Basic Examples (Start Here!)

1. **01_add** - Vector addition (`c[i] = a[i] + b[i]`)
   - Demonstrates load/compute/store pattern
   - Shows why memory bandwidth matters
   - Speedup: ~1.2x (bandwidth-limited)

2. **02_sum** - Sum reduction
   - Demonstrates horizontal reduction
   - Shows floating-point non-associativity
   - Speedup: ~5x (compute-bound)

3. **03_clamp** - Conditional clamping with masks
   - Demonstrates masked operations
   - Shows when SIMD overhead hurts
   - Speedup: ~0.8-2.7x (varies by platform)

4. **04_count** - Count elements above threshold
   - Demonstrates popcount for efficient counting
   - Shows predicate/mask usage
   - Speedup: ~5-24x

### Advanced Examples

5. **05_softmax** - Numerically stable softmax
   - Multi-pass algorithm (max, exp, normalize)
   - Polynomial approximation for exp()
   - Speedup: ~2.7-4.2x

6. **06_conv** - 1D convolution with FMA
   - Fused multiply-add operations
   - Shows when overhead dominates
   - Speedup: ~0.2-1.1x (small kernel overhead)

7. **07_filter** - 3x3 image box filter
   - 2D stride access patterns
   - Border handling
   - Speedup: ~0.8-1.0x (overhead issue)

## Verifying SIMD is Working

```bash
# Generate assembly for inspection
g++ -std=c++2b -O3 -march=native -S 01_add.cpp -o 01_add.s

# Look for vector instructions
grep -E 'vaddps|fadd.*\.4s|vfadd' 01_add.s

# ARM NEON: Look for instructions like "fadd v30.4s"
# x86 AVX:   Look for instructions like "vaddps zmm0"
# RISC-V:    Look for instructions like "vfadd.vv" (not yet working)
```

See [ASSEMBLY_EXAMPLES.md](ASSEMBLY_EXAMPLES.md) for annotated assembly with detailed explanations.

## Common Issues

### "No such file or directory: experimental/simd"

You need GCC 9+ with libstdc++ experimental support:
```bash
# Check GCC version
g++ --version  # Should be 9.0 or higher

# On macOS, make sure you're using Homebrew GCC, not Apple Clang
which g++      # Should NOT be /usr/bin/g++
```

### "SIMD width: 1" on RISC-V

This is expected! GCC 13's `std::experimental::simd` doesn't support RISC-V Vector yet. The code falls back to scalar operations. See [RISCV_STATUS.md](RISCV_STATUS.md).

### Speedup is less than expected

Common reasons:
1. **Memory bandwidth**: Simple operations like add are bottlenecked by memory, not compute
2. **Small arrays**: SIMD overhead dominates for tiny arrays
3. **Auto-vectorization**: Compiler may vectorize both versions
4. **Thermal throttling**: CPU may reduce frequency under load

## Learning Path

**Beginner:**
1. Read the [README.md](README.md) introduction
2. Run `make run` and observe speedups
3. Look at the source code for examples 01-04
4. Generate assembly and find vector instructions

**Intermediate:**
5. Read [Understanding Vector Instructions](README.md#understanding-vector-instructions-low-level-details)
6. Study [ASSEMBLY_EXAMPLES.md](ASSEMBLY_EXAMPLES.md)
7. Modify examples and observe assembly changes
8. Experiment with alignment (`element_aligned` vs `vector_aligned`)

**Advanced:**
9. Profile with `perf` or similar tools
10. Try writing your own SIMD algorithm
11. Compare auto-vectorization vs explicit `std::simd`
12. Explore the advanced examples (05-07)

## Makefile Targets Reference

```bash
make                  # Build all examples for native platform
make run              # Build and run all examples
make clean            # Remove all build artifacts

make riscv            # Build all RISC-V binaries
make run-riscv-128    # Run RISC-V with VLEN=128
make run-riscv-512    # Run RISC-V with VLEN=512
make run-riscv-both   # Run both VLEN configurations
make verify-riscv     # Generate assembly and check for RVV instructions

make 01_add           # Build specific example
make 01_add.riscv     # Build specific RISC-V example
```

## Documentation Index

- **[README.md](README.md)** - Main guide with all examples and concepts
- **[ASSEMBLY_EXAMPLES.md](ASSEMBLY_EXAMPLES.md)** - Annotated assembly for all architectures
- **[RISCV_STATUS.md](RISCV_STATUS.md)** - RISC-V support status and technical details
- **[RISCV_SETUP.md](RISCV_SETUP.md)** - RISC-V emulation setup alternatives
- **[QUICK_START.md](QUICK_START.md)** - This document

## Getting Help

1. Check the [Key Takeaways](README.md#key-takeaways) section
2. Review the [Quick Reference](README.md#appendix-quick-reference)
3. Search for your specific instruction in the assembly examples
4. Read the inline comments in the source code

## Next Steps

- Try modifying an example and observe the performance change
- Experiment with different array sizes
- Compare results with `-march=native` vs specific flags
- Profile your code with `perf stat -e instructions,cycles ./01_add`
- Read about your CPU's SIMD capabilities in the vendor documentation

Happy vectorizing! 🚀
