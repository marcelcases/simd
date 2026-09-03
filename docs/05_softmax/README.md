# 5. Numerically Stable Softmax

[Scalar source](../../src/scalar/05_softmax.cpp) | [SIMD source](../../src/simd/05_softmax.cpp)

A logit is an unnormalized, real-valued score produced by a model before it
is converted into a probability. Softmax converts a vector of logits into
probabilities: values between zero and one that add up to one.

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**Numerical stability.** Softmax is shift-invariant:
`softmax(x - c) = softmax(x)`. For example, subtracting the maximum from
`[0, 1, 3]` produces `[-3, -2, 0]`; both inputs produce
`[0.0420, 0.1142, 0.8438]`. The shifted maximum is zero, so every finite
exponential is at most one. This avoids exponential overflow without changing
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
| Find maximum | Scalar comparisons | Lane-wise maximum, then `hmax` |
| Compute exponentials | Scalar `std::exp` | Scalar `std::exp` |
| Sum exponentials | Scalar accumulator | SIMD accumulator, then `reduce` |
| Normalize | Scalar division | SIMD division and store |

## SIMD notes

- `copy_from` loads consecutive array elements into a SIMD vector, typically
  backed by vector registers rather than a newly allocated memory buffer.
- `stdx::hmax(maximum)` horizontally reduces all vector lanes to one scalar
  maximum. `stdx::reduce` performs the analogous sum reduction.
- Scalar tails handle input sizes that are not multiples of the vector width,
  avoiding masked-load/store overhead while keeping bounds handling simple.
- The SIMD API provides portable vector arithmetic, comparisons, and reductions,
  but no portable vector overload for `std::exp`. The exponential loop is
  therefore scalar in the source; a compiler or math library may still
  auto-vectorize it.
