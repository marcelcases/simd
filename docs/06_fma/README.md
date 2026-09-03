# 6. FMA and Dot Product

[Scalar source](../../src/scalar/06_fma.cpp) | [SIMD source](../../src/simd/06_fma.cpp)

This exercise contains two kernels that share multiplication and addition but
have different dataflows:

- `fma_memory_bound` computes `output[i] = a[i] * b[i] + c[i]` for every index
  and stores an output array.
- `dot_product` computes `sum(a[i] * b[i])` and returns one scalar result.

They are intentionally together: the first is dominated by memory traffic, while
the second emphasizes arithmetic and a final reduction.

## Used in

- Memory-bound FMA: element-wise tensor operations, scaling and biasing arrays,
  and signal-processing kernels.
- Dot product: BLAS operations, matrix multiplication, neural-network layers,
  vector similarity, and correlation.

## Kernel workflow

The scalar version handles one index at a time. The SIMD version handles complete
vector-width groups and uses a scalar tail.

| Kernel | Scalar | SIMD |
|---|---|---|
| Memory-bound FMA | Compute `a[i] * b[i] + c[i]` and store the result | Load three vector groups, apply `stdx::fma`, and store the result |
| Dot product | Multiply each pair and add it to a scalar accumulator | Load two vector groups, accumulate with `stdx::fma`, then reduce the lanes |

## SIMD notes

- `stdx::fma` expresses lane-wise fused multiply-add: `a * b + c`.
- `stdx::reduce` combines the dot-product accumulator lanes into one scalar.
- The scalar tail handles elements left after the complete vector groups.
