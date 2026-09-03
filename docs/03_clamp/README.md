# 3. Upper-bound Clamp

[Scalar source](../../src/scalar/03_clamp.cpp) | [SIMD source](../../src/simd/03_clamp.cpp)

Replaces every value above `upper_bound` with `upper_bound`.

## Used in

- Enforcing upper limits on numerical values.
- Limiting values before further computation.

## Kernel workflow

Both kernels perform the same operation. The scalar version processes one value
at a time; the SIMD version processes full vector-width groups and uses a scalar
tail.

| Step | Scalar | SIMD |
|---|---|---|
| Compare | Test one value against the bound | Compare all lanes with the bound |
| Update | Replace the value if it is above the bound | Use a mask and `where` to update selected lanes |

## SIMD notes

- A comparison creates one Boolean mask for each vector lane.
- `stdx::where` updates only the lanes selected by the mask.
- The scalar tail handles elements left after the complete vector groups.
