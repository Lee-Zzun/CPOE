"""§4.1 eq.(15)-(18) — 적률법 (MoM, 논문 충실 v1).

알고리즘:
  Step 1 (CMP):  (λ̂,ν̂) ← solve X̄=μ(λ,ν), S²=σ²(λ,ν)        (eq.15-16)
  Step 2 (CPOE): θ̂_n = (1/N)Σ_i ψ_n(X_i;λ̂,ν̂)              (eq.17)
  분산:          Σ_MoM = Var_p[ψ(X;λ̂,ν̂)] — (λ̂,ν̂)를 기지로 취급 (eq.18)

공개 API: mom_fit (step 2 + eq.18 분산 + 실행가능성), MoMResult (출력 데이터클래스).

v2 백로그 (개정/확장): (λ̂,ν̂) 플러그인 오차를 ψ_n을 통해 전파하는 샌드위치 분산
V=(1/N)G⁻¹ΩG⁻ᵀ는 논문 범위를 넘는 확장이다 → v1은 eq.(18)만 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import fsolve

from cpoe.compoisson import CMPDivergenceError
from cpoe.cpoe_model import empirical_theta
from cpoe.feasibility import (
    FeasibilityReport,
    feasibility_diagnostic,
    project_onto_C_K,
)


def _pad_theta(free: np.ndarray, K: int) -> np.ndarray:
    """자유 계수 (θ_3..θ_K)를 길이-K 벡터 [0, 0, θ_3,…,θ_K]에 끼워 넣는다."""
    K = int(K)
    full = np.zeros(K)
    free = np.asarray(free, dtype=float).ravel()
    if K > 2 and free.size:
        full[2:2 + free.size] = free
    return full


@dataclass
class MoMResult:
    """MoM 출력 + eq.(18) 분산 컨테이너."""

    lam_hat: float
    nu_hat: float
    theta_hat: np.ndarray            # 원시 eq.(17) 추정치
    theta_hat_projected: np.ndarray  # C_K로 사영 (실행가능하면 원시와 동일)
    feasibility_raw: FeasibilityReport
    feasibility_projected: FeasibilityReport
    var_eq18: np.ndarray             # K×K, eq.(18), (λ̂,ν̂)를 기지로 취급
    se_theta_eq18: np.ndarray        # K, sqrt(diag(var_eq18)/N)
    diagnostics: dict = field(default_factory=dict)

    @property
    def pre_projection_feasible(self) -> bool:
        return bool(self.feasibility_raw.feasible)

    @property
    def projected(self) -> bool:
        return bool(self.diagnostics.get("projection", {}).get("projected", False))


def _cmp_mean_var(lam: float, nu: float, *, cmp_mod, M: int):
    """절단 지지대 S_M 위에서 CMP의 (μ, σ²)."""
    x = np.arange(M)
    w = np.asarray(cmp_mod.pmf(x, lam, nu, M=M), dtype=float)
    mu = float((x * w).sum())
    var = float(((x - mu) ** 2 * w).sum())
    return mu, var


def _solve_step1(xbar: float, s2: float, *, cmp_mod, M: int,
                 lam0: float, nu0: float):
    """로그 매개변수화한 fsolve로 μ(lam,nu)=xbar, σ²(lam,nu)=s²를 푼다."""

    def residuals(p):
        lam = float(np.exp(p[0]))
        nu = float(np.exp(p[1]))
        try:
            mu, var = _cmp_mean_var(lam, nu, cmp_mod=cmp_mod, M=M)
        except CMPDivergenceError:
            return np.array([1e6, 1e6])
        return np.array([mu - xbar, var - s2])

    sol, info, ier, msg = fsolve(
        residuals,
        x0=np.array([np.log(max(lam0, 1e-6)), np.log(max(nu0, 1e-6))]),
        full_output=True,
        xtol=1e-10,
    )
    lam_hat = float(np.exp(sol[0]))
    nu_hat = float(np.exp(sol[1]))
    return lam_hat, nu_hat, {
        "ier": int(ier),
        "fsolve_msg": str(msg),
        "nfev": int(info.get("nfev", 0)),
        "final_residual": [float(v) for v in residuals(sol)],
    }


def mom_fit(
    sample: np.ndarray,
    K: int,
    *,
    cmp_mod,
    basis,
    M: int,
    lam0: Optional[float] = None,
    nu0: Optional[float] = None,
    project_to_feasible: bool = True,
) -> MoMResult:
    """2단계 MoM 실행 → eq.(18) 분산을 포함한 MoMResult 반환.

    sample: (N,) 정수.  K: 절단 차수.  cmp_mod, basis: 백엔드 (cpoe_model 도크스트링).
    M: 절단 지지대 크기 (sample은 [0,M)로 클리핑됨).  lam0, nu0: step-1 fsolve 웜스타트.
    project_to_feasible: True (기본값)이면 원시 θ̂를 C_K로 사영하여 원시와 함께 저장 (논문 충실).
    """
    sample = np.asarray(sample, dtype=int)
    N = sample.shape[0]
    if N < 4:
        raise ValueError("MoM needs N >= 4 to estimate (lam, nu) and theta.")
    xbar = float(sample.mean())
    s2 = float(sample.var(ddof=1))

    # 웜스타트 (§4.1): 호출자가 lam0/nu0를 생략하면 (xbar,s2)의 Shmueli 점근 역변환을 사용한다.
    if lam0 is None or nu0 is None:
        if hasattr(cmp_mod, "asymptotic_start"):
            a_lam, a_nu = cmp_mod.asymptotic_start(xbar, s2)
        else:  # pragma: no cover - 선택적 메서드가 없는 백엔드
            a_lam, a_nu = max(xbar, 1e-3), 1.0
        lam0 = a_lam if lam0 is None else lam0
        nu0 = a_nu if nu0 is None else nu0

    lam_hat, nu_hat, info_step1 = _solve_step1(
        xbar, s2, cmp_mod=cmp_mod, M=M, lam0=lam0, nu0=nu0,
    )

    # K=0 특수 경우: CMP 전용 기준선. 기저, 사영, eq.(18) 모두 없음.
    if K == 0:
        empty = np.zeros(0)
        empty_var = np.zeros((0, 0))
        from cpoe.feasibility import FeasibilityReport
        empty_feas = FeasibilityReport(
            feasible=True, min_tilt=1.0, argmin_x=0, n_violations=0,
            margin_to_boundary=1.0,
        )
        return MoMResult(
            lam_hat=lam_hat, nu_hat=nu_hat,
            theta_hat=empty, theta_hat_projected=empty,
            feasibility_raw=empty_feas, feasibility_projected=empty_feas,
            var_eq18=empty_var, se_theta_eq18=empty,
            diagnostics={
                "N": int(N), "xbar": xbar, "s_squared": s2,
                "step1": info_step1,
                "projection": {"projected": False, "message": "K=0 baseline"},
            },
        )

    # 자유 계수 θ_3..θ_K에 대한 eq.(17) 사영 (θ_1=θ_2=0 고정, B0).
    n_free = max(0, K - 2)
    theta_free = empirical_theta(sample, lam_hat, nu_hat, K, basis=basis, M=M)[2:].copy()
    theta_hat = _pad_theta(theta_free, K)            # [0, 0, θ̂_3,…,θ̂_K]

    psi_support = np.asarray(basis.psi_matrix(lam_hat, nu_hat, K, M), dtype=float)  # (M, K)
    psi_free = psi_support[:, 2:K] if K > 2 else psi_support[:, :0]                  # (M, n_free)
    feas_raw = feasibility_diagnostic(theta_hat, psi_support)
    if project_to_feasible and n_free and not feas_raw.feasible:
        theta_proj_free, proj_info = project_onto_C_K(theta_free, psi_free)
        theta_proj = _pad_theta(theta_proj_free, K)
    else:
        theta_proj = theta_hat.copy()
        proj_info = {"projected": False, "message": "raw estimate already feasible"}
    feas_proj = feasibility_diagnostic(theta_proj, psi_support)

    # eq.(18): Σ_MoM = Var_p[ψ_n(X;λ̂,ν̂)] (자유 계수 n=3..K). (λ̂,ν̂)를 기지로 취급;
    # 2단계 플러그인 샌드위치는 v2로 미룬다. θ_1,θ_2 자리에 0 행/열을 넣어 K×K로 패딩.
    var_eq18 = np.zeros((K, K))
    se_theta_eq18 = np.zeros(K)
    if n_free:
        psi_x = psi_free[np.clip(sample, 0, M - 1)]
        psi_demeaned = psi_x - psi_x.mean(axis=0, keepdims=True)
        var_free = (psi_demeaned.T @ psi_demeaned) / N
        var_eq18[2:, 2:] = var_free
        se_theta_eq18 = _pad_theta(np.sqrt(np.maximum(np.diag(var_free) / N, 0.0)), K)

    diagnostics = {
        "N": int(N),
        "xbar": xbar,
        "s_squared": s2,
        "step1": info_step1,
        "projection": proj_info,
    }
    return MoMResult(
        lam_hat=lam_hat,
        nu_hat=nu_hat,
        theta_hat=theta_hat,
        theta_hat_projected=theta_proj,
        feasibility_raw=feas_raw,
        feasibility_projected=feas_proj,
        var_eq18=var_eq18,
        se_theta_eq18=se_theta_eq18,
        diagnostics=diagnostics,
    )


def fit_mom(
    data,
    K: int,
    *,
    M: Optional[int] = None,
    cmp_mod=None,
    basis=None,
    project: bool = True,
    **kwargs,
) -> MoMResult:
    """상위 수준 진입점: 원시 데이터 배열로부터 2단계 MoM 적합.

    기본 백엔드와 절단 M을 채워 주는 mom_fit의 얇은 래퍼.
    """
    from cpoe.backends import CMP_BACKEND, BASIS_BACKEND, default_M

    sample = np.asarray(data)
    if cmp_mod is None:
        cmp_mod = CMP_BACKEND
    if basis is None:
        basis = BASIS_BACKEND
    if M is None:
        M = default_M(sample)
    return mom_fit(
        sample.astype(int), K,
        cmp_mod=cmp_mod, basis=basis, M=M,
        project_to_feasible=project, **kwargs,
    )
