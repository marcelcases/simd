# 5. Numerically Stable Softmax

[Scalar source](../../src/scalar/05_softmax.cpp) | [SIMD source](../../src/simd/05_softmax.cpp)

A logit is an unnormalized, real-valued score produced by a model before it
is converted into a probability. Softmax converts a vector of logits into
probabilities: values between zero and one that add up to one.

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**Numerical stability.** Softmax is shift-invariant:
`softmax(x - c) = softmax(x)`. Subtracting the maximum makes the shifted
maximum zero, so every finite exponential is at most one. This avoids exponential overflow without changing
the resulting probabilities.

## Used in

- Classification: maps class scores to class probabilities.
- Language models / LLMs: turns one logit per vocabulary token into the
  probabilities used to select or sample the next token.
- Attention in Transformers: normalizes query-key similarity scores into
  attention weights.
- Mixture-of-experts and routing: turns expert scores into weights that
  determine each expert's contribution.

## Kernel workflow

Both kernels compute the same four steps. The scalar version handles one value
at a time; the SIMD version handles full vector-width groups and finishes with a
scalar tail.

| Step | Scalar | SIMD |
|---|---|---|
| Find maximum | Compare elements one at a time | Compare each vector's lanes, then `hmax` to obtain one scalar maximum |
| Compute exponentials | Compute `exp(value - maximum)` per element | Same computation; the source loop uses scalar `std::exp` |
| Sum exponentials | Add each value to one scalar total | Add vector values lane-wise, then `reduce` to obtain one scalar total |
| Normalize | Divide each value by the total | Broadcast the total, divide vector values, and store |

## Numerical example

These input logits differ only by a constant offset:

| Step | Input A | Input B |
|---|---|---|
| Input | `[0, 1, 3]` | `[5, 6, 8]` |
| Maximum | `3` | `8` |
| Subtract maximum | `[-3, -2, 0]` | `[-3, -2, 0]` |
| Exponentiate | `[0.0498, 0.1353, 1]` | `[0.0498, 0.1353, 1]` |
| Sum | `1.1851` | `1.1851` |
| Normalize | `[0.0420, 0.1142, 0.8438]` | `[0.0420, 0.1142, 0.8438]` |

Softmax depends on relative differences, so both produce the same probabilities.
Values are rounded. Neither small input would overflow directly; subtracting
the maximum also protects against overflow for large logits.

## SIMD notes

- The SIMD API used here provides portable vector arithmetic, comparisons, and
  reductions, but **no portable vector overload for `std::exp`**. GCC 16.2
  provides an experimental `stdx::exp`, but our x86-64 and RVV builds
  scalarize it rather than using vector math. The exponential loop is therefore
  scalar in the source; a compiler or math library may still auto-vectorize it.
- `copy_from` loads consecutive array elements into a SIMD vector, typically
  backed by vector registers rather than a newly allocated memory buffer.
- `stdx::hmax(maximum)` horizontally reduces all vector lanes to one scalar
  maximum. `stdx::reduce` performs the analogous sum reduction.
- Scalar tails handle input sizes that are not multiples of the vector width,
  avoiding masked-load/store overhead while keeping bounds handling simple.
