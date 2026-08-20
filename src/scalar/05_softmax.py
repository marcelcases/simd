"""Numerically stable scalar softmax reference."""

import math
from typing import Sequence


def softmax(values: Sequence[float]) -> list[float]:
    """Return softmax(values), subtracting the maximum for stability."""
    if not values:
        return []

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
