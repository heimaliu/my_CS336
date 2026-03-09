from __future__ import annotations

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine LR schedule with linear warmup."""
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_term * (max_learning_rate - min_learning_rate)

    return min_learning_rate
