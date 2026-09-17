# 2. Reduction and Dot Product

[Scalar source](../../src/scalar/02_reduction_dot.cpp) | [SIMD source](../../src/simd/02_reduction_dot.cpp)

Two reduction kernels return one scalar result:

- `sum`: adds all array elements.
- `dot_product`: computes `sum(a[i] * b[i])`.

The dot product reads only `a[i]` and `b[i]`; its SIMD accumulator stays in
**registers** during the loop. Unlike the element-wise fused multiply-add in
[exercise 1](../01_add_fma/README.md), there is no per-element output store.

For 32-bit floats, counting multiplication and addition as two FLOPs:

```text
Element-wise fused multiply-add: 2 FLOPs / 16 bytes = 0.125 FLOPs per byte
Dot product:                     2 FLOPs /  8 bytes = 0.25  FLOPs per byte
```

Ignoring cache effects, write allocation, and the final reduction, the dot
product has twice the arithmetic intensity. It can still be memory-bound;
higher arithmetic intensity does not guarantee a compute-bound kernel.

## Used in

- Sum: totals, averages, and other numerical reductions.
- Dot product: BLAS operations, matrix multiplication, neural-network layers,
  vector similarity, and correlation.

## Kernel workflow

| Step | Scalar | SIMD |
|---|---|---|
| Sum | Add each value to one scalar total | Accumulate vector groups lane-wise |
| Dot product | Multiply each pair and add to one scalar total | Accumulate products with `stdx::fma` (fused multiply-add) |
| Finish | Return the total | Combine lane totals with `stdx::reduce`, then add the scalar tail |

## SIMD notes

- Lane accumulators hold partial sums rather than independent final outputs.
- Scalar tails handle elements left after complete vector groups.
- Reduction order and fused arithmetic can change floating-point rounding.
