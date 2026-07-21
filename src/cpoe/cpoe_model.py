"""§3.1 eq.(12)-(13)·§3.2 eq.(14) — CPOE pmf와 투영항등식.

CPOE pmf (eq.12)
    p_θ(x;λ,ν) = w_CMP(x;λ,ν)·(1 + θ⊤ψ(x;λ,ν))
⟨ψ_n,ψ_0⟩=0 (n≥1, Prop.1)이라 tilt는 ℕ 위에서 w_CMP에 대해 1로 적분;
절단 지지대 S_M에선 잔차만큼 어긋나며 truncation_mass로 노출.

투영항등식 (eq.14)
    θ_n = ⟨p/w_CMP, ψ_n⟩ = E_p[ψ_n(X)]
MoM(표본평균)·MLE(최적점 score)의 *공통 앵커*.

객체: cpoe_pmf/cpoe_log_pmf (eq.12/log), truncation_mass (Σp_θ 진단),
empirical_theta (eq.14 plug-in), expected_psi_under_model (E_{p_θ}[ψ]=θ 검증).

백엔드 프로토콜(duck-typed):
    cmp_mod.{pmf,log_pmf}(x, lam, nu, M)
    basis.psi_matrix(lam, nu, K, M) -> (M, K), 열 n=1..K
"""

from __future__ import annotations

import numpy as np


_TILT_EPS = 1e-300


def cpoe_pmf(theta: np.ndarray, lam: float, nu: float, *, cmp_mod, basis, M: int) -> np.ndarray:
    """x=0..M-1에서 p_θ(x) 반환. shape (M,).

    재정규화 안 함: θ∈C_K면 (Prop.2) tilt가 w_CMP에 대해 1로 적분. θ가 C_K 밖이면
    일부 음수 가능 — 조용히 clip 말고 feasibility.feasibility_diagnostic으로 진단할 것.
    """
    theta = np.asarray(theta, dtype=float)
    x = np.arange(M)
    w = np.asarray(cmp_mod.pmf(x, lam, nu, M=M), dtype=float)
    psi = np.asarray(basis.psi_matrix(lam, nu, theta.shape[0], M), dtype=float)
    tilt = 1.0 + psi @ theta
    return w * tilt


def cpoe_log_pmf(theta: np.ndarray, lam: float, nu: float, *, cmp_mod, basis, M: int) -> np.ndarray:
    """절단 지지대 위 log p_θ(x). 1+θ⊤ψ(x)≤0이면 -inf.

    smooth surrogate가 필요한 호출자는 log 전에 tilt를 clip할 것
    (MLE는 자체 _neg_log_lik_smoothed를 씀).
    """
    theta = np.asarray(theta, dtype=float)
    x = np.arange(M)
    log_w = np.asarray(cmp_mod.log_pmf(x, lam, nu, M=M), dtype=float)
    psi = np.asarray(basis.psi_matrix(lam, nu, theta.shape[0], M), dtype=float)
    tilt = 1.0 + psi @ theta
    out = np.full(M, -np.inf, dtype=float)
    pos = tilt > _TILT_EPS
    out[pos] = log_w[pos] + np.log(tilt[pos])
    return out


def truncation_mass(theta: np.ndarray, lam: float, nu: float, *, cmp_mod, basis, M: int) -> float:
    """Σ_{x<M} p_θ(x). 1과의 차이가 절단오차 진단. 보통 1-mass<1e-6이어야 L1/J-test 신뢰."""
    return float(cpoe_pmf(theta, lam, nu, cmp_mod=cmp_mod, basis=basis, M=M).sum())


def empirical_theta(sample: np.ndarray, lam: float, nu: float, K: int,
                    *, basis, M: int) -> np.ndarray:
    """eq.(14) plug-in: θ̂_n = (1/N)Σ_i ψ_n(X_i;λ,ν). shape (K,).

    MoM 2단계, MLE warm-start에서 사용.
    """
    sample = np.clip(np.asarray(sample, dtype=int), 0, M - 1)
    psi_full = np.asarray(basis.psi_matrix(lam, nu, K, M), dtype=float)
    return psi_full[sample].mean(axis=0)


def expected_psi_under_model(theta: np.ndarray, lam: float, nu: float,
                             *, cmp_mod, basis, M: int) -> np.ndarray:
    """절단 지지대 위 E_{p_θ}[ψ_n(X)]. eq.(14)로 θ_n과 같아야 함.

    차이 e_n = E_{p_θ}[ψ_n] − θ_n 는 절단오차 → M 충분성 점검용.
    """
    theta = np.asarray(theta, dtype=float)
    p = cpoe_pmf(theta, lam, nu, cmp_mod=cmp_mod, basis=basis, M=M)
    psi = np.asarray(basis.psi_matrix(lam, nu, theta.shape[0], M), dtype=float)
    return psi.T @ p
