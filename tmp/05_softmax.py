import math
from typing import Sequence


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
