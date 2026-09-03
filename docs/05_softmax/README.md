# 5. Numerically Stable Softmax

## Description

Softmax converts logits into probabilities:

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Subtracting the maximum logit before taking exponentials does not change the
probabilities because `softmax(x - c) = softmax(x)`. Choosing `c = max(x)`
prevents exponential overflow and gives a numerically stable softmax.

The SIMD implementation uses four phases: find the maximum, compute
exponentials, reduce their sum, and normalize the values.

## Why it matters

Softmax is widely used to turn model scores into probabilities in classification,
attention and transformer models, and other probabilistic selection methods.

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

## Example

```text
softmax([0, 1, 3]) = softmax([-3, -2, 0])
                   = [0.0420, 0.1142, 0.8438]
```

## Run

```bash
./build/05_softmax_scalar --size 4194304 --repetitions 5
./build/05_softmax_simd   --size 4194304 --repetitions 5
```
