"""§4.4 Adaptive Selection of the Truncation Order — 차수선택 규칙 구현.

논문 대응:
  Proposition (정확 위험분해): R(K)=Σ_{n=3}^K σ_n²/N + Σ_{n>K} θ_n²  (eq. risk-decomposition)
  불편-위험 규칙 (eq:risk-profile):
      R̂(K) = Σ_{n=3}^K (2σ̂_n²/N − θ̂_n²),   K̂_RISK = argmin_K R̂(K)
      항별: θ̂_n² > 2σ̂_n²/N (계수>√2·SE)면 채택

규약: θ̂_n(=raw MoM 투영)·σ̂_n²(=Var_p[ψ_n])는 고정기저(§4.1 MoM). basis.psi_matrix의
열 j(0-based)는 ψ_{j+1}. 활성 전개는 n≥3 (θ_1=θ_2=0, Lemma theta12).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def momproj_stats(sample: np.ndarray, lam: float, nu: float, k_max: int,
                  *, basis, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """(λ̂,ν̂) 위에서 raw MoM 투영 θ̂_n 과 고정기저 분산 σ̂_n² (n=1..k_max) 반환.

    θ̂_n = (1/N)Σ_i ψ_n(X_i),   σ̂_n² = Var_p[ψ_n(X)] (divisor N−1, 논문 §4.4 정의).
    shape (k_max,); index n−1 ↔ 차수 n.
    """
    s = np.clip(np.asarray(sample, dtype=int), 0, M - 1)
    psi = np.asarray(basis.psi_matrix(lam, nu, k_max, M), dtype=float)  # (M, k_max)
    psi_x = psi[s]                                                       # (N, k_max)
    theta_hat = psi_x.mean(axis=0)
    sigma2 = psi_x.var(axis=0, ddof=1)
    return theta_hat, sigma2


def risk_profile(theta_hat: np.ndarray, sigma2: np.ndarray, N: int,
                 k_max: int) -> Dict[int, float]:
    """불편-위험 곡선 R̂(K), K=2..k_max. R̂(2)=0 (활성 전개 시작 전)."""
    theta_hat = np.asarray(theta_hat, float)
    sigma2 = np.asarray(sigma2, float)
    incr = 2.0 * sigma2 / N - theta_hat ** 2      # index n−1
    R: Dict[int, float] = {2: 0.0}
    cum = 0.0
    for K in range(3, k_max + 1):
        cum += float(incr[K - 1])                 # 항 n=K
        R[K] = cum
    return R


def select_k_risk(theta_hat: np.ndarray, sigma2: np.ndarray, N: int,
                  k_max: int) -> int:
    """K̂_RISK = argmin_K R̂(K)  (동률 시 최소 K)."""
    R = risk_profile(theta_hat, sigma2, N, k_max)
    kmin = min(R, key=lambda k: (R[k], k))
    return int(kmin)
