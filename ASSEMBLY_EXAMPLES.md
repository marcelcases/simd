# Annotated Assembly Examples

This document shows real assembly code generated from the `std::simd` examples, with detailed annotations explaining what each instruction does. This helps you understand what's happening at the hardware level.

## Example: Vector Addition (`a[i] += b[i]`)

The same C++ code:
```cpp
void add_simd(float* dst, const float* src, size_t n) {
    using V = stdx::native_simd<float>;
    constexpr size_t W = V::size();
    
    for (size_t i = 0; i + W <= n; i += W) {
        V a, b;
        a.copy_from(dst + i, stdx::element_aligned);
        b.copy_from(src + i, stdx::element_aligned);
        a += b;
        a.copy_to(dst + i, stdx::element_aligned);
    }
}
```

Compiles to three completely different implementations:

---

## ARM NEON Assembly (128-bit, 4 floats per iteration)

Generated on Apple M1 Pro with `g++-15 -O3 -mcpu=apple-m1`:

```assembly
add_simd:
    # Function prologue
    cbz     x2, .L1              # if (n == 0) return
    
    # Setup loop counter
    lsr     x3, x2, #2           # x3 = n / 4 (number of iterations)
    cbz     x3, .L3              # if (iterations == 0) goto tail
    
.L4:  # Main SIMD loop - processes 4 floats per iteration
    # LOAD: Read 4 floats from dst array into vector register v30
    ld1     {v30.4s}, [x0]       # v30 = [dst[i], dst[i+1], dst[i+2], dst[i+3]]
                                 # x0 = pointer to dst
                                 # .4s = treat as 4 single-precision floats
    
    # LOAD: Read 4 floats from src array into vector register v31
    ld1     {v31.4s}, [x1], #16  # v31 = [src[i], src[i+1], src[i+2], src[i+3]]
                                 # x1 = pointer to src
                                 # #16 = post-increment x1 by 16 bytes (4 floats)
    
    # COMPUTE: Vector addition - all 4 lanes in parallel!
    fadd    v30.4s, v31.4s, v30.4s  # v30[0] = v31[0] + v30[0]
                                     # v30[1] = v31[1] + v30[1]
                                     # v30[2] = v31[2] + v30[2]
                                     # v30[3] = v31[3] + v30[3]
                                     # All done in ONE instruction!
    
    # STORE: Write 4 results back to dst array
    st1     {v30.4s}, [x0], #16  # Store v30 to memory
                                 # Post-increment x0 by 16 bytes
    
    # Loop control
    subs    x3, x3, #1           # Decrement counter
    bne     .L4                  # if (counter != 0) continue loop
    
.L3:  # Tail loop for remaining elements
    # ... scalar code for n % 4 elements ...

.L1:  # Return
    ret
```

**Key insights:**
- Processes **4 floats per iteration** (128 bits / 32 bits per float)
- **1 load + 1 load + 1 add + 1 store** = 4 instructions for 4 operations
- Post-increment addressing (`[x0], #16`) updates pointers automatically
- Loop overhead: Just `subs` and `bne` - very efficient!

**Performance:**
- **4x parallelism** vs scalar
- Memory bandwidth becomes the bottleneck, not computation

---

## x86 AVX-512 Assembly (512-bit, 16 floats per iteration)

Generated on Intel Xeon 8480+ with `g++ -O3 -march=native -mavx512f`:

```assembly
add_simd:
    # Check if n == 0
    test    rdx, rdx
    je      .L1
    
    # Setup loop
    mov     rax, rdx
    shr     rax, 4               # rax = n / 16 (iterations)
    je      .L3                  # if (iterations == 0) goto tail
    sal     rax, 6               # rax *= 64 (bytes per iteration)
    xor     ecx, ecx             # ecx = 0 (loop counter)
    
.L4:  # Main SIMD loop - processes 16 floats per iteration
    # LOAD: Read 16 floats from dst array into zmm register
    vmovups zmm0, zmmword ptr [rdi + rcx]
                                 # zmm0 = 16 floats from dst[i..i+15]
                                 # rdi = base pointer to dst
                                 # rcx = byte offset
                                 # vmovups = unaligned load
                                 # zmmword = 64 bytes (512 bits)
    
    # LOAD: Read 16 floats from src array
    vmovups zmm1, zmmword ptr [rsi + rcx]
                                 # zmm1 = 16 floats from src[i..i+15]
                                 # rsi = base pointer to src
    
    # COMPUTE: Vector addition - all 16 lanes in parallel!
    vaddps  zmm0, zmm1, zmm0     # zmm0[i] = zmm1[i] + zmm0[i] for i=0..15
                                 # vaddps = vector add packed singles
                                 # Executes 16 additions simultaneously!
    
    # STORE: Write 16 results back to dst
    vmovups zmmword ptr [rdi + rcx], zmm0
                                 # Store all 16 floats back
    
    # Loop control
    add     rcx, 64              # Advance by 64 bytes (16 floats)
    cmp     rcx, rax             # Check if done
    jne     .L4                  # Continue if not done
    
.L3:  # Tail loop
    # ... scalar code for n % 16 elements ...

.L1:  # Return
    vzeroupper                   # Clear upper bits of vector registers
    ret
```

**Key insights:**
- Processes **16 floats per iteration** (512 bits / 32 bits per float)
- **4x more data per iteration** than ARM NEON!
- Uses `zmm` registers (512-bit) instead of `ymm` (256-bit) or `xmm` (128-bit)
- `vzeroupper` prevents performance penalties when mixing AVX and non-AVX code

**Performance:**
- **16x parallelism** vs scalar
- **4x more throughput** than NEON (same architecture)
- Still memory-bandwidth limited for simple operations

**Instruction variants:**
- `vmovups` = unaligned move (works with any address)
- `vmovaps` = aligned move (requires 64-byte alignment, faster)
- Choice depends on `element_aligned` vs `vector_aligned` in `std::simd`

---

## RISC-V Scalar Assembly (Currently No Vectorization)

Generated with `riscv64-linux-gnu-g++ -O3 -march=rv64gcv`:

```assembly
add_simd:
    # Check if n == 0
    beq     a2, zero, .L1        # if (n == 0) return
                                 # a2 = n parameter
    
    # Setup: calculate end pointer
    slli    a2, a2, 2            # a2 = n * 4 (convert to bytes)
    add     a5, a0, a2           # a5 = dst + (n * 4) = end pointer
    
.L3:  # Loop - processes 1 float per iteration (SCALAR!)
    # LOAD: Read single float from dst
    flw     fa5, 0(a0)           # fa5 = *dst (floating-point register)
                                 # a0 = current pointer to dst
                                 # flw = float load word (32-bit)
    
    # LOAD: Read single float from src
    flw     fa4, 0(a1)           # fa4 = *src
                                 # a1 = current pointer to src
    
    # Increment pointers BEFORE computation (to hide latency)
    addi    a0, a0, 4            # dst++
    addi    a1, a1, 4            # src++
    
    # COMPUTE: Scalar addition (only 1 float!)
    fadd.s  fa5, fa5, fa4        # fa5 = fa5 + fa4
                                 # .s suffix = single-precision scalar
                                 # This is NOT a vector instruction!
    
    # STORE: Write single float back
    fsw     fa5, -4(a0)          # Store to dst[-1] (we already incremented)
                                 # fsw = float store word
    
    # Loop control
    bne     a0, a5, .L3          # if (dst != end) continue
    
.L1:  # Return
    ret
```

**Why no vectorization?**
```assembly
# What we SHOULD see with RVV support:
vsetvli t0, a2, e32, m1      # Configure: 32-bit elements, 1 register group
vle32.v v1, (a0)             # Load vector of floats from dst
vle32.v v2, (a1)             # Load vector of floats from src
vfadd.vv v3, v1, v2          # Vector add: v3 = v1 + v2
vse32.v v3, (a0)             # Store vector to dst

# But GCC's std::simd doesn't support RVV yet, so we get scalar code instead!
```

**Key insights:**
- Only **1 float per iteration** - no parallelism!
- Uses scalar FP instructions: `flw`, `fadd.s`, `fsw`
- Should use vector instructions: `vle32.v`, `vfadd.vv`, `vse32.v`
- `std::experimental::simd` falls back to `simd_abi::scalar` on RISC-V
- Performance: **Same as scalar code** because it IS scalar code!

**When RVV support is added** (future GCC versions):
- With VLEN=128: Would process 4 floats/iteration (like NEON)
- With VLEN=512: Would process 16 floats/iteration (like AVX-512)
- RISC-V advantage: **Scalable** - same code adapts to different VLENs

---

## Comparison: Instructions Per 64 Floats

To process 64 floating-point additions (`c[i] = a[i] + b[i]` for i=0..63):

| Architecture | Width | Iterations | Loads | Adds | Stores | Total |
|--------------|-------|------------|-------|------|--------|-------|
| **Scalar** | 32-bit | 64 | 128 | 64 | 64 | 256 |
| **ARM NEON** | 128-bit | 16 | 32 | 16 | 16 | 64 |
| **x86 AVX-512** | 512-bit | 4 | 8 | 4 | 4 | 16 |
| **RISC-V (current)** | 32-bit | 64 | 128 | 64 | 64 | 256 |
| **RISC-V (VLEN=128)** | 128-bit | 16 | 32 | 16 | 16 | 64 |
| **RISC-V (VLEN=512)** | 512-bit | 4 | 8 | 4 | 4 | 16 |

**Instruction reduction:**
- NEON vs Scalar: **4x fewer instructions**
- AVX-512 vs Scalar: **16x fewer instructions**
- AVX-512 vs NEON: **4x fewer instructions**

This is the fundamental reason SIMD is faster!

---

## Advanced Example: Fused Multiply-Add (FMA)

C++ code from `06_conv.cpp`:
```cpp
for (int j = 0; j < K; ++j) {
    V kv(k[j]);
    V xv;
    xv.copy_from(x + i + j, stdx::element_aligned);
    acc = stdx::fma(xv, kv, acc);  // acc += xv * kv
}
```

### ARM NEON Assembly
```assembly
# FMA: acc = acc + (xv * kv)
fmla    v31.4s, v27.4s, v30.4s   # v31 += v27 * v30
                                 # fmla = floating multiply-accumulate
                                 # Computes: v31[i] = v31[i] + (v27[i] * v30[i])
                                 # For i = 0, 1, 2, 3
```

This single instruction replaces:
```cpp
// What it does (4 times in parallel):
acc[0] = acc[0] + (xv[0] * kv[0])
acc[1] = acc[1] + (xv[1] * kv[1])
acc[2] = acc[2] + (xv[2] * kv[2])
acc[3] = acc[3] + (xv[3] * kv[3])
```

### x86 AVX-512 Assembly
```assembly
# FMA: acc = acc + (xv * kv)
vfmadd231ps zmm0, zmm1, zmm2     # zmm0 += zmm1 * zmm2
                                 # vfmadd231ps = vector FMA (231 order)
                                 # 231 means: dest += src2 * src1
                                 # Processes 16 floats at once!
```

### Benefits of FMA
1. **Higher precision**: Only one rounding instead of two (multiply, then add)
2. **Higher throughput**: One instruction instead of two
3. **Critical for ML/DSP**: Dot products are very common

---

## How to Generate These Examples Yourself

```bash
# ARM NEON (on Apple Silicon Mac)
g++-15 -std=c++2b -O3 -mcpu=apple-m1 -S 01_add.cpp -o 01_add.s
less 01_add.s  # Search for add_simd function

# x86 AVX-512 (on Linux)
g++ -std=c++2b -O3 -march=native -mavx512f -S 01_add.cpp -o 01_add.s
less 01_add.s  # Search for add_simd function

# RISC-V (cross-compile)
riscv64-linux-gnu-g++ -std=c++2b -O3 -march=rv64gcv -S 01_add.cpp -o 01_add.s
less 01_add.s  # Search for add_simd function

# Clean up C++ name mangling
c++filt _Z8add_simdPfPKfm
# Output: add_simd(float*, float const*, unsigned long)
```

---

## Learning Resources

**ARM NEON:**
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- Search for instructions like `fadd`, `fmla`, `ld1`

**x86 AVX-512:**
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- Search for instructions like `vaddps`, `vmovups`

**RISC-V Vector:**
- [RISC-V Vector Extension Specification](https://github.com/riscv/riscv-v-spec)
- [RISC-V Vector Intrinsics](https://github.com/riscv-non-isa/rvv-intrinsic-doc)

**Understanding Assembly:**
- Register names tell you the width: `zmm` (512), `ymm` (256), `xmm` (128), `v` (varies)
- Suffixes tell you the data type: `.4s` (4 singles), `.2d` (2 doubles)
- Prefixes tell you vector vs scalar: `v` prefix in x86 means vector
- Post-increment addressing saves instructions: `[x0], #16` = load + increment
