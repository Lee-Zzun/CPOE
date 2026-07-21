"""§3.1 Proposition 2 — feasibility 집합 C_K(λ,ν)와 L2 투영.

CPOE pmf p_θ(x)=w_CMP(x)(1+θ⊤ψ(x)) (eq.12)이 유효 pmf이려면 모든 x에서
1+θ⊤ψ(x)≥0. 절단 지지대 S_M={0..M-1}에선 반공간들의 유한 교집합:

    A θ ≥ -1,   A[i,n] = ψ_n(x_i),   i∈S_M, n=1..K.

C_K는 닫힌 볼록집합이며 원점 포함(Prop.2, ac-04). 이 모듈은 절단 지지대 진단과
MoM(ac-07) 후처리에 쓰는 L2 투영을 제공. FeasibilityReport는 위반 지지대
인덱스·위반 수까지 노출.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class FeasibilityReport:
    """제약 1+θ⊤ψ(x)≥0 (절단 지지대) 진단."""

    feasible: bool
    min_tilt: float
    argmin_x: int
    n_violations: int
    margin_to_boundary: float

    def to_dict(self) -> dict:
        return {
            "feasible": bool(self.feasible),
            "min_tilt": float(self.min_tilt),
            "argmin_x": int(self.argmin_x),
            "n_violations": int(self.n_violations),
            "margin_to_boundary": float(self.margin_to_boundary),
        }


def _coerce_psi_support(psi_support) -> np.ndarray:
    """(M,K) psi 행렬 또는 OrthonormalBasis를 받아 (M,K) 행렬로 강제.

    추정기는 행렬을 직접 넘기고, 고수준 호출자(케이스 스터디·ac-10)는 basis 객체만
    가진 경우가 많아 여기서 ψ_1..ψ_K를 평가해 강제. lazy import로 순환 import 회피.
    """
    if hasattr(psi_support, "coeffs") and hasattr(psi_support, "a"):
        from cpoe import basis as _basis_mod
        from cpoe import compoisson as _cmp
        b = psi_support
        M = _cmp.support_upper(b.lam, b.nu)
        x = np.arange(M + 1)
        psi_all = _basis_mod.eval_psi_all(x, b)      # (K+1, M+1)
        return np.asarray(psi_all[1:].T, dtype=float)  # (M+1, K), drop psi_0
    return np.asarray(psi_support, dtype=float)


def build_constraint_matrix(psi_support: np.ndarray) -> np.ndarray:
    """feasibility 제약이 Aθ≥-1이 되도록 A 검증·반환.

    psi_support: (M,K). 행=ψ(x_i) (x_i=0..M-1), 열=기저 n=1..K (상수 n=0 제외, Prop.1).
    """
    A = np.asarray(psi_support, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"psi_support must be 2-D (M, K); got shape {A.shape}")
    return A


def feasibility_diagnostic(theta: np.ndarray, psi_support) -> FeasibilityReport:
    """min tilt 값·argmin 지지대 인덱스·강한 위반 수 보고. psi_support는 (M,K) 또는 OrthonormalBasis."""
    theta = np.asarray(theta, dtype=float)
    A = _coerce_psi_support(psi_support)
    tilt = 1.0 + A @ theta
    idx = int(np.argmin(tilt))
    return FeasibilityReport(
        feasible=bool(np.all(tilt >= 0.0)),
        min_tilt=float(tilt[idx]),
        argmin_x=idx,
        n_violations=int(np.sum(tilt < 0.0)),
        margin_to_boundary=float(tilt[idx]),
    )


def project_onto_C_K(
    theta_hat: np.ndarray,
    psi_support: np.ndarray,
    *,
    shrink_eps: float = 1e-8,
    method: str = "SLSQP",
    max_iter: int = 500,
    ftol: float = 1e-12,
) -> tuple:
    """θ̂를 절단 feasibility 다면체 C_K에 L2 투영.

    QP: min_θ ½‖θ-θ̂‖²  s.t.  Aθ ≥ -1+shrink_eps.
    shrink_eps 여유로 θ를 (엄격) 내부에 둬 downstream log p_θ가 수치적으로 정의되게 함
    (MoM 후처리용).

    반환: (theta_proj, info). info 키: projected(이미 내부면 False)·success·iterations·
    pre_min_tilt·post_min_tilt·message.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    K = theta_hat.shape[0]
    A = build_constraint_matrix(psi_support)
    rhs = -1.0 + shrink_eps

    pre = feasibility_diagnostic(theta_hat, psi_support)
    if pre.min_tilt >= shrink_eps:
        return theta_hat.copy(), {
            "projected": False,
            "success": True,
            "iterations": 0,
            "pre_min_tilt": pre.min_tilt,
            "post_min_tilt": pre.min_tilt,
            "message": "already feasible with margin",
        }

    # 각 제약 a_i⊤θ - rhs ≥ 0.
    constraints = [
        {
            "type": "ineq",
            "fun": (lambda th, a=A[i], r=rhs: float(a @ th - r)),
            "jac": (lambda th, a=A[i]: a),
        }
        for i in range(A.shape[0])
    ]

    res = minimize(
        fun=lambda th: 0.5 * float(np.dot(th - theta_hat, th - theta_hat)),
        x0=theta_hat.copy(),  # SLSQP은 infeasible 시작점을 restoration으로 처리
        jac=lambda th: th - theta_hat,
        constraints=constraints,
        method=method,
        options={"ftol": ftol, "maxiter": max_iter, "disp": False},
    )

    theta_proj = np.asarray(res.x, dtype=float)
    post = feasibility_diagnostic(theta_proj, psi_support)
    info = {
        "projected": True,
        "success": bool(res.success),
        "iterations": int(getattr(res, "nit", -1)),
        "pre_min_tilt": pre.min_tilt,
        "post_min_tilt": post.min_tilt,
        "message": str(res.message),
    }
    return theta_proj, info
