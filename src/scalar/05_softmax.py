# Softmax Formula
# softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

import math
from typing import List

def softmax(x: List[float]) -> List[float]:
    """
    Numerically stable softmax that returns a new list.
    """
    if not x:
        return []
    
    maxv = max(x)
    exps = [math.exp(xi - maxv) for xi in x]
    total = sum(exps)
    return [e / total for e in exps]


# Usage:
scores = [100, 9, 8, 2.0, 1.0, 0.1, -1.0, -10]
probs = softmax(scores)
print(probs)
print(sum(probs))  # ≈ 1.0