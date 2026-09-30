# 6. Horizontal Image Blur

[Scalar source](../../src/scalar/06_filter.cpp) | [SIMD source](../../src/simd/06_filter.cpp)

Smooths each row of a grayscale image independently. The horizontal window is
**hardcoded to three pixels**: the left neighbour, the current pixel, and the
right neighbour. Its size is not configurable.

```text
output[row, column] = (left + center + right) / 3
```

The first and last pixels average the two available pixels instead. There is
no vertical averaging.

## Used in

- Simple image smoothing.
- Learning stencil operations with overlapping input windows.

## Numerical example

Consider one row with five pixel intensities:

```text
Input: [2, 4, 9, 5, 1]
```

| Position | Calculation | Output |
|---|---|---:|
| First | `(2 + 4) / 2` | 3 |
| Interior | `(2 + 4 + 9) / 3` | 5 |
| Interior | `(4 + 9 + 5) / 3` | 6 |
| Interior | `(9 + 5 + 1) / 3` | 5 |
| Last | `(5 + 1) / 2` | 3 |

```text
Input:  [2, 4, 9, 5, 1]
Output: [3, 5, 6, 5, 3]
```

The peak drops from 9 to 6, while lower neighbouring values rise. Sharp
differences are softened. The row still contains five pixels: edges average
two pixels, while the interior uses the fixed three-pixel window.
Every image row is processed independently.

## Kernel workflow

| Step | Scalar | SIMD |
|---|---|---|
| Select a row | Point to its input and output pixels | Same |
| Left edge | Average the first two pixels | Same scalar operation |
| Interior | Average three pixels for one output | Load left, center, and right vectors; average lane-wise |
| Finish | Average the last two pixels | Process the scalar tail, then the right edge |

The image is logically a 2D matrix stored in a flat array of floats, row by row.
The driver owns the arrays; the kernel receives pointers and dimensions:

```cpp
void blur_horizontal(const float* input, float* output,
                     int width, int height) noexcept;
```

`source = input + row * width` selects the start of a row. Then
`source[column]` is equivalent to `input[row * width + column]`.
Input and output must be valid, non-overlapping buffers, with `width >= 2` and
`height >= 1`. The driver validates the dimensions.

## SIMD notes

- Each lane produces one output pixel; SIMD width does not change the
  three-pixel window.
- Three shifted loads align each lane's left, center, and right neighbours.
- `scale` contains `1.f / 3.f` in every lane.
- `element_aligned` permits the shifted addresses without requiring full
  vector alignment; the experimental SIMD API requires this argument.
- The vector loop stops before a right-neighbour load would cross the row.
  A scalar tail handles remaining interior pixels.
- Neighbouring outputs reuse input pixels through caches; three loads in the
  source do not imply three independent main-memory reads per output.
