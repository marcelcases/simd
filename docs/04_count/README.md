# 4. Count Above Threshold

[Scalar source](../../src/scalar/04_count.cpp) | [SIMD source](../../src/simd/04_count.cpp)

Returns how many values are greater than `threshold`.

## Used in

- Counting values that pass a threshold.
- Detecting selected values in data-processing loops.

## Kernel workflow

Both kernels perform the same operation. The scalar version processes one value
at a time; the SIMD version processes full vector-width groups and uses a scalar
tail.

| Step | Scalar | SIMD |
|---|---|---|
| Compare | Test one value against the threshold | Compare all lanes with the threshold |
| Count | Increment the counter when the test is true | Count true mask lanes with `popcount` |

## SIMD notes

- A comparison mask records which lanes are above the threshold.
- `stdx::popcount` counts the true lanes in the mask.
- The scalar tail handles elements left after the complete vector groups.
