# 1. Addition and Fused Multiply-Add (FMA)

[Scalar source](../../src/scalar/01_add_fma.cpp) | [SIMD source](../../src/simd/01_add_fma.cpp)

Two element-wise kernels update an output array:

- `add`: `destination[i] += source[i]`.
- `fma_memory_bound`: `output[i] = a[i] * b[i] + c[i]`.

Fused multiply-add (FMA) combines multiplication and addition with one rounding
of the result. Each output requires reading `a[i]`, `b[i]`, and `c[i]`, then
writing `output[i]`. With little arithmetic per byte transferred, memory traffic
can limit SIMD gains.

For 32-bit floats, counting multiplication and addition as two FLOPs:

```text
Fused multiply-add: 2 FLOPs / 16 bytes = 0.125 FLOPs per byte
```

This estimates array traffic, ignoring cache effects and write allocation.
[Exercise 2](../02_reduction_dot/README.md) contrasts this with a dot product
whose accumulator stays in registers.

## Used in

- Addition: combining arrays, vectors, and tensors.
- Fused multiply-add: element-wise tensor operations, scaling and biasing arrays,
  and signal-processing kernels.

## Kernel workflow

| Kernel | Scalar | SIMD |
|---|---|---|
| Addition | Read a pair, add, and store | Load two vectors, add corresponding lanes, and store |
| Fused multiply-add | Compute `a[i] * b[i] + c[i]` and store | Load three vectors, apply `stdx::fma`, and store |

## SIMD notes

- Each lane computes an independent output; no horizontal reduction is needed.
- `copy_from` and `copy_to` load and store consecutive array elements.
- Scalar tails handle elements left after complete vector groups.
- `stdx::fma` explicitly requests fusion; scalar `a * b + c` may be fused by
  the compiler, so rounding can differ.
