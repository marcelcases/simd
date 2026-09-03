# 6. Fused Multiply-Add (FMA) and Dot Product

[Scalar source](../../src/scalar/06_fma.cpp) | [SIMD source](../../src/simd/06_fma.cpp)

This exercise contains two kernels that share multiplication and addition but
have different dataflows. One of them is a fused multiply-add (FMA) kernel:

- `fma_memory_bound` computes `output[i] = a[i] * b[i] + c[i]` for every index
  and stores an output array.
- `dot_product` computes `sum(a[i] * b[i])` and returns one scalar result.

The kernels perform the same fused multiply-add arithmetic but move different
amounts of data. For each index, `fma_memory_bound` must read `a[i]`, `b[i]`,
and `c[i]`, then write `output[i]`. The dot product only reads `a[i]` and
`b[i]`; its accumulator stays in **registers** during the loop, and it produces
only one final scalar result.

This makes the FMA kernel more strongly memory-bound. Ignoring cache effects and
amortizing the single final scalar over all elements, a simple estimate using
32-bit floats is:

```text
Fused multiply-add:  2 FLOPs / 16 bytes = 0.125 FLOPs per byte
Dot product:         2 FLOPs /  8 bytes = 0.25   FLOPs per byte
```

The dot product therefore has twice the arithmetic intensity. It is less
strongly dominated by memory traffic and makes SIMD arithmetic and reduction
more visible, although it can still be memory-bound depending on the hardware
and cache behavior.

## Used in

- Fused multiply-add (FMA): element-wise tensor operations, scaling and biasing arrays,
  and signal-processing kernels.
- Dot product: BLAS operations, matrix multiplication, neural-network layers,
  vector similarity, and correlation.

## Kernel workflow

The scalar version handles one index at a time. The SIMD version handles complete
vector-width groups and uses a scalar tail.

| Kernel | Scalar | SIMD |
|---|---|---|
| Fused multiply-add (FMA) | Compute `a[i] * b[i] + c[i]` and store the result | Load three vector groups, apply `stdx::fma`, and store the result |
| Dot product | Multiply each pair and add it to a scalar accumulator | Load two vector groups, accumulate with `stdx::fma`, then reduce the lanes |

## SIMD notes

- `stdx::fma` expresses lane-wise fused multiply-add: `a * b + c`.
- `stdx::reduce` combines the dot-product accumulator lanes into one scalar.
- The scalar tail handles elements left after the complete vector groups.
