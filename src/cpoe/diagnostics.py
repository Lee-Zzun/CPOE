"""CPOE 적합 진단 — 표본 pmf와 L1 거리.

객체:
  empirical_pmf  — 절단 지지대 위 표본 pmf
  l1_distance    — Σ_x|p−q|
"""

from __future__ import annotations

import numpy as np


def empirical_pmf(sample: np.ndarray, M: int) -> np.ndarray:
    """{0..M-1} 위 표본 pmf. M 이상 값은 M-1로 clip."""
    sample = np.clip(np.asarray(sample, dtype=int), 0, M - 1)
    counts = np.bincount(sample, minlength=M)
    total = counts.sum()
    if total == 0:
        raise ValueError("empirical_pmf: empty sample.")
    return counts.astype(float) / float(total)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """‖p−q‖₁ = Σ_x|p(x)−q(x)|."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"l1_distance: shape mismatch {p.shape} vs {q.shape}")
    return float(np.sum(np.abs(p - q)))
