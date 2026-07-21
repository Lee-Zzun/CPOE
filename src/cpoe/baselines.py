"""성능비교용 베이스라인 카운트 분포 적합 (Poisson / Negative Binomial / COM-Poisson).

선행연구 `Proofread__Skewness_and_Kurtosis_Counting_Data.pdf` 및 GitHub
`silontee/skewness-and-kurtosis@branch_jihun`의 적률법(MoM) 방식을 포팅하여, CPOE와의
교차모델 L1 비교에서 동일한 추정 원리(MoM)를 쓴다.

  Poisson:  μ = 표본평균  (Poisson은 MoM ≡ MLE).
  NB(β,c):  c = 1 − μ/σ²,  β = μ²/(σ²−μ)  (과대산포 σ²>μ 필요; 아니면 Poisson 폴백).
            scipy 모수화 nbinom(n=β, p=1−c).
  CMP(λ,ν): 적률매칭 (λ,ν)는 호출자가 우리 MoM(stage-1)에서 받아 넘긴다 →
            CPOE-MoM과 동일 베이스라인 공유(= CPOE의 K=0). pmf만 평가.

모든 적합 pmf는 주어진 grid 위에서 합=1로 정규화한다(공정한 L1 비교; branch_jihun 관례 일치).
공개 API: poisson_fitted_pmf, nb_fitted_pmf, cmp_fitted_pmf.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import nbinom, poisson

from cpoe import compoisson as _cmp

_EPS = 1e-300


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > _EPS else p


def poisson_fitted_pmf(data: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Poisson MoM(=MLE) 적합 pmf. μ=표본평균."""
    data = np.asarray(data, dtype=float)
    mu = float(data.mean())
    p = np.asarray(poisson.pmf(np.asarray(grid), mu), dtype=float)
    return _normalize(p), {"model": "Poisson", "mu": mu}


def nb_fitted_pmf(data: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Negative Binomial MoM 적합 pmf (β,c 모수화).

    σ²≤μ(과대산포 아님)이면 Poisson으로 폴백.
    """
    data = np.asarray(data, dtype=float)
    mu = float(data.mean())
    var = float(data.var(ddof=0))
    if var <= mu + 1e-12:
        p, info = poisson_fitted_pmf(data, grid)
        info = {"model": "NB(fallback=Poisson)", "mu": mu, "var": var, "fallback": True}
        return p, info
    c = 1.0 - mu / var
    beta = mu * mu / (var - mu)
    p = np.asarray(nbinom.pmf(np.asarray(grid), beta, 1.0 - c), dtype=float)
    return _normalize(p), {"model": "NB", "beta": beta, "c": c, "fallback": False}


def cmp_fitted_pmf(lam: float, nu: float, grid: np.ndarray) -> Tuple[np.ndarray, dict]:
    """COM-Poisson 적합 pmf. (λ,ν)는 호출자가 MoM(stage-1)에서 받아 넘긴다.

    CPOE-MoM과 *동일* (λ,ν)를 공유하므로 CMP = CPOE의 K=0(틸트 없음)에 해당한다.
    """
    grid = np.asarray(grid)
    p = np.asarray(_cmp.pmf(grid, float(lam), float(nu)), dtype=float)
    return _normalize(p), {"model": "CMP", "lam": float(lam), "nu": float(nu)}
