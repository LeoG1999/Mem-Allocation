"""Statistical utilities for confidence intervals and paired tests."""

from __future__ import annotations

import math
import random
from statistics import mean, pstdev
from typing import Sequence

import numpy as np


def summarize_scalar(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def bootstrap_ci(
    values: Sequence[float],
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    if not values:
        return {"low": 0.0, "high": 0.0, "alpha": alpha, "num_bootstrap": num_bootstrap}
    if len(values) == 1:
        v = float(values[0])
        return {"low": v, "high": v, "alpha": alpha, "num_bootstrap": num_bootstrap}

    rng = random.Random(seed)
    arr = list(values)
    samples = []
    for _ in range(num_bootstrap):
        sample = [arr[rng.randrange(len(arr))] for _ in range(len(arr))]
        samples.append(float(mean(sample)))
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return {"low": low, "high": high, "alpha": alpha, "num_bootstrap": num_bootstrap}


def paired_permutation_test(
    xs: Sequence[float],
    ys: Sequence[float],
    num_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Approximate two-sided paired permutation test.
    """
    if len(xs) != len(ys) or not xs:
        return 1.0
    diffs = [x - y for x, y in zip(xs, ys)]
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)

    count = 0
    for _ in range(num_samples):
        signed = [d if rng.random() < 0.5 else -d for d in diffs]
        stat = abs(sum(signed) / len(signed))
        if stat >= observed:
            count += 1
    return (count + 1) / (num_samples + 1)


def mcnemar_exact_test(xs: Sequence[bool], ys: Sequence[bool]) -> float:
    """
    Two-sided exact McNemar test over discordant pairs.
    """
    if len(xs) != len(ys) or not xs:
        return 1.0

    b = 0  # x correct, y wrong
    c = 0  # x wrong, y correct
    for x, y in zip(xs, ys):
        if x and not y:
            b += 1
        elif y and not x:
            c += 1

    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2 * prob)
    return p
