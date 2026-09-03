# 1. Element-wise Array Addition

[Scalar source](../../src/scalar/01_add.cpp) | [SIMD source](../../src/simd/01_add.cpp)

Adds two arrays element by element. The result is stored in `destination`:
`destination[i] += source[i]`.

## Used in

- Combining vectors, arrays, or buffers element by element.
- Vector and tensor addition in numerical programs.

## Kernel workflow

Both kernels perform the same operation. The scalar version processes one
corresponding pair at a time; the SIMD version processes full vector-width
groups and uses a scalar tail.

| Step | Scalar | SIMD |
|---|---|---|
| Load | Read one element from each array | Load groups with `copy_from` |
| Add | Add the two scalar values | Add corresponding lanes |
| Store | Write one result | Store the vector with `copy_to` |

## SIMD notes

- Each vector lane computes one independent array addition.
- The scalar tail handles elements left after the complete vector groups.
