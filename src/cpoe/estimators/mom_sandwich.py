"""§4.1 two-step sandwich variance for MoM (eq:mom-stacked, eq:mom-asymptotic).

mom.py의 eq.(18) naive 분산 Var_p[ψ]은 (λ̂,ν̂)를 기지로 취급한다. 여기서는 1단계
플러그인 오차를 ψ_n을 통해 전파하는 정확한 2단계 sandwich를 계산한다.

Stacked just-identified Z-추정 (η = (λ,ν,θ_3,…,θ_K), 차원 K):
    g(x;η) = [ x − μ(β);  (x−μ(β))² − σ²(β);  ψ_n(x;β) − θ_n  (n=3..K) ] ∈ R^K
    √N(η̂−η*) →d N(0, G⁻¹ Ω G⁻ᵀ),   G = E[∂g/∂η],  Ω = Var[g].
θ-블록 대각의 제곱근이 SE(θ̂_n) (2단계 보정 포함). fixed-basis se(=eq.18)와의 비교로
검증(§5 tab:mc-se-check).

∂μ/∂β·∂σ²/∂β·∂ψ_n/∂β 는 중심차분(Joint score와 동일한 수치미분 방식)으로 계산한다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _mean_var(lam: float, nu: float, *, cmp_mod, M: int):
    """절단 지지대 위 COM-Poisson 평균·분산 (eq:cmp-mean/variance)."""
    x = np.arange(M, dtype=float)
    w = np.asarray(cmp_mod.pmf(x, lam, nu, M=M), dtype=float)
    mu = float((x * w).sum())
    var = float(((x - mu) ** 2 * w).sum())
    return mu, var


def sandwich_cov(sample: np.ndarray, lam: float, nu: float, K: int,
                 *, cmp_mod, basis, M: int,
                 h_lam: Optional[float] = None,
                 h_nu: Optional[float] = None):
    """(λ̂,ν̂,θ̂)에서 2단계 sandwich 점근공분산 V=(1/N)G⁻¹ΩG⁻ᵀ 반환.

    반환: (V, se) — V는 (K,K) η̂ 공분산(η 순서 λ,ν,θ_3..θ_K),
    se는 (K,) 배열로 se[n−1]=SE(θ̂_n) (n=1,2 자리는 nan; θ_1=θ_2=0 고정).
    """
    if K < 3:
        return np.zeros((0, 0)), np.full(K, np.nan)
    s = np.clip(np.asarray(sample, dtype=int), 0, M - 1)
    N = int(s.shape[0])
    n_free = K - 2
    dim = 2 + n_free  # = K

    if h_lam is None:
        h_lam = max(1e-5, 1e-4 * abs(lam))
    if h_nu is None:
        h_nu = max(1e-5, 1e-4 * abs(nu))

    # ψ_n at data, columns ψ_3..ψ_K (0-based col index 2..K-1)
    def psi_free_at(l, n):
        return np.asarray(basis.psi_matrix(l, n, K, M), dtype=float)[s][:, 2:K]

    psi_free_x = psi_free_at(lam, nu)            # (N, n_free)
    theta_free = psi_free_x.mean(axis=0)         # θ̂_3..θ̂_K

    # ---- Ω = Var[g] (표본공분산, η̂에서 평가) ----
    mu, var = _mean_var(lam, nu, cmp_mod=cmp_mod, M=M)
    g_mean = s.astype(float) - mu
    g_var = (s.astype(float) - mu) ** 2 - var
    g_proj = psi_free_x - theta_free
    g = np.column_stack([g_mean, g_var, g_proj])  # (N, dim)
    Omega = np.cov(g, rowvar=False, bias=False)
    Omega = np.atleast_2d(Omega)

    # ---- G = E[∂g/∂η] (플러그인) ----
    mu_lp, var_lp = _mean_var(lam + h_lam, nu, cmp_mod=cmp_mod, M=M)
    mu_lm, var_lm = _mean_var(lam - h_lam, nu, cmp_mod=cmp_mod, M=M)
    mu_np, var_np = _mean_var(lam, nu + h_nu, cmp_mod=cmp_mod, M=M)
    mu_nm, var_nm = _mean_var(lam, nu - h_nu, cmp_mod=cmp_mod, M=M)
    dmu_dl = (mu_lp - mu_lm) / (2 * h_lam)
    dmu_dn = (mu_np - mu_nm) / (2 * h_nu)
    dvar_dl = (var_lp - var_lm) / (2 * h_lam)
    dvar_dn = (var_np - var_nm) / (2 * h_nu)

    dpsi_dl = (psi_free_at(lam + h_lam, nu) - psi_free_at(lam - h_lam, nu)) / (2 * h_lam)
    dpsi_dn = (psi_free_at(lam, nu + h_nu) - psi_free_at(lam, nu - h_nu)) / (2 * h_nu)
    Epsi_dl = dpsi_dl.mean(axis=0)   # (n_free,)  ≈ E[∂ψ_n/∂λ]
    Epsi_dn = dpsi_dn.mean(axis=0)

    G = np.zeros((dim, dim))
    G[0, 0] = -dmu_dl
    G[0, 1] = -dmu_dn
    G[1, 0] = -dvar_dl          # E[∂g_var/∂λ] = -∂σ²/∂λ  (E[x-μ]=0 at MoM fit)
    G[1, 1] = -dvar_dn
    for j in range(n_free):
        G[2 + j, 0] = Epsi_dl[j]
        G[2 + j, 1] = Epsi_dn[j]
        G[2 + j, 2 + j] = -1.0

    Ginv = np.linalg.inv(G)
    V = (Ginv @ Omega @ Ginv.T) / N

    se = np.full(K, np.nan)
    dV = np.clip(np.diag(V), 0.0, None)
    for j in range(n_free):
        se[2 + j] = float(np.sqrt(dV[2 + j]))
    return V, se
