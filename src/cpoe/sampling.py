"""CPOE 시뮬레이션 실험용 표본 추출 (논문 §6.1 DGP).

절단 CMP 샘플러는 S1/S3 성분(CMP + 오염/긴꼬리 혼합)을, sample_cpoe_tilt는 임의 tilt θ로
직접 CPOE 표본을 생성(투영항등식 eq.14로 참 계수=θ). S2는 cubic θ_3 tilt로 평균·분산을
보존(등산포)하면서 참값을 아는 비-Poisson DGP를 만든다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cpoe import basis as _basis_mod
from cpoe import compoisson as _cmp


def sample_cmp(
    lam: float, nu: float, N: int, rng: np.random.Generator,
    M: Optional[int] = None,
) -> np.ndarray:
    """(절단) COM-Poisson pmf에서 inverse-CDF 추출.

    M 기본값 = support_upper(lam,nu)+1 (1e-12 잔여질량 절단 → 절단 편향 무시 가능).
    """
    if M is None:
        M = int(_cmp.support_upper(float(lam), float(nu))) + 1
    x = np.arange(M)
    p = np.asarray(_cmp.pmf(x, float(lam), float(nu)), dtype=float)
    p = p / p.sum()
    cdf = np.cumsum(p)
    u = rng.random(N)
    return np.searchsorted(cdf, u).astype(int)


def sample_cpoe_tilt(
    lam: float, nu: float, theta, N: int, rng: np.random.Generator,
    M: Optional[int] = None,
) -> np.ndarray:
    """임의 tilt 벡터 θ를 가진 CPOE 분포에서 추출.

    p_θ(x) = w_CMP(x;λ,ν)·(1 + Σ_{n≥1} θ_n ψ_n(x)). 투영항등식(eq.14)으로
    표본 DGP의 참 CPOE 계수는 정확히 θ.

    모멘트: ψ_n(n≥3)은 {1,x,x²}에 직교 → n≥3 tilt만 쓰면 평균·분산 불변.
    시나리오 S2가 이 방식으로 *정확한* 등산포(V/M=베이스라인)를 유지하면서
    참값을 아는 비-Poisson·CPOE 표현가능 분포를 만든다.

    절단 지지대에서 tilt pmf에 음수가 있으면(θ가 C_K 밖) ValueError.
    """
    theta = np.asarray(theta, dtype=float)
    K = int(theta.shape[0])
    if M is None:
        M = int(_cmp.support_upper(float(lam), float(nu))) + 1
    x = np.arange(M)
    w = np.asarray(_cmp.pmf(x, float(lam), float(nu)), dtype=float)
    b = _basis_mod.build_basis(float(lam), float(nu), K=K)
    psi_all = _basis_mod.eval_psi_all(x, b)        # (K+1, M); 행 ψ_0..ψ_K
    tilt = 1.0 + theta @ np.asarray(psi_all[1:], dtype=float)  # Σ_{n=1}^K θ_n ψ_n
    if tilt.min() < -1e-12:
        raise ValueError(
            f"sample_cpoe_tilt: theta={theta} infeasible at (lam={lam}, nu={nu}); "
            f"min tilt = {tilt.min():.3e} < 0."
        )
    p = w * np.maximum(tilt, 0.0)
    p = p / p.sum()
    cdf = np.cumsum(p)
    u = rng.random(N)
    return np.searchsorted(cdf, u).astype(int)
