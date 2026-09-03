# 2. Sum Reduction

[Scalar source](../../src/scalar/02_sum.cpp) | [SIMD source](../../src/simd/02_sum.cpp)

Adds all elements of an array and returns one scalar total.

## Used in

- Computing totals and averages.
- Building norms and other reductions.

## Kernel workflow

Both kernels perform the same operation. The scalar version processes one value
at a time; the SIMD version processes full vector-width groups and uses a scalar
tail.

| Step | Scalar | SIMD |
|---|---|---|
| Accumulate | Add each value to one scalar total | Add vector groups to a vector accumulator |
| Finish | Return the scalar total | Use `reduce` on the lanes, then add the tail |

## SIMD notes

- `stdx::reduce` combines the partial lane sums into one scalar result.
- The scalar tail handles elements left after the complete vector groups.
