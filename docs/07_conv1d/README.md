# 7. 1D Mathematical Convolution

[Scalar source](../../src/scalar/07_conv1d.cpp) | [SIMD source](../../src/simd/07_conv1d.cpp)

Slides a kernel (an array of weights) across a 1D signal. For each complete
window, multiply the input values by the **reversed kernel** and add the
products to produce one output value.

```text
output[i] = sum(input[i + j] * kernel[kernel_size - 1 - j])
            for j = 0, ..., kernel_size - 1
```

This is **valid convolution**: no padding, only complete windows. The output
contains `input_size - kernel_size + 1` values. Kernel size and weights are
arguments, not hardcoded into the computation.

## Used in

- Audio: FIR filters for smoothing noise or selecting frequency bands.
- Sensors: smoothing temperature, vibration, or accelerometer measurements.
- Communications: pulse shaping and filtering received signals.
- Physical systems: computing a response from an input and an impulse response.
- Time series: weighted moving averages.

Many neural-network libraries call their operation convolution but compute
**correlation**, without reversing the kernel. Our implementation performs
mathematical convolution. An asymmetric kernel makes the difference visible.

## Numerical example

Smooth a signal containing one isolated peak with a symmetric kernel:

```text
Input:  [0, 0, 8, 0, 0]
Kernel: [0.25, 0.5, 0.25]
```

| Input window | Weighted sum | Output |
|---|---|---:|
| `[0, 0, 8]` | `0 × 0.25 + 0 × 0.5 + 8 × 0.25` | 2 |
| `[0, 8, 0]` | `0 × 0.25 + 8 × 0.5 + 0 × 0.25` | 4 |
| `[8, 0, 0]` | `8 × 0.25 + 0 × 0.5 + 0 × 0.25` | 2 |

```text
Output: [2, 4, 2]
```

The sharp peak becomes smaller and spreads across neighbouring positions.
There are three outputs because `5 - 3 + 1 = 3`; no partial edge windows are
computed. This kernel is symmetric, so reversing it does not change its weights.
For an asymmetric kernel such as `[10, 20]`, the applied weights are `[20, 10]`.

## Kernel workflow

| Step | Scalar | SIMD |
|---|---|---|
| Start | Set one sum to zero for an output window | Set every accumulator lane to zero |
| Accumulate | Read input forwards and kernel backwards; multiply and add | Load shifted input values, broadcast one reversed weight, multiply and add lane-wise |
| Store | Write one output value | Store one vector of output values |
| Finish | Continue through all complete windows | Process remaining outputs with a scalar tail |

The outer index `i` selects an output window; the inner index `j` selects a
position within that window. Each output starts with a fresh sum.

Both kernels require `1 <= kernel_size <= input_size`, valid input and kernel
buffers, and enough output storage. Output must overlap neither input nor kernel.
The caller owns the arrays; the kernels only perform computation.

## SIMD notes

- Each lane computes a different output window, not a partial sum for one
  shared output. No horizontal reduction is needed.
- At inner step `j`, the load starts at `input + i + j`. Consecutive lanes
  receive samples from consecutive output windows.
- The reversed kernel weight is broadcast to every lane.
- Neighbouring windows overlap and reuse input samples.
- The vector loop processes complete output groups; the scalar tail handles
  leftovers. All loads remain inside the input under the stated preconditions.
