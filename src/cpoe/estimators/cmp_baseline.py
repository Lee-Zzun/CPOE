"""순수 COM-Poisson 베이스라인 (λ,ν)의 최대우도 추정 (틸트 θ 없음).

Sequential MLE의 1단계(베이스라인 적합)로 재사용되는 core 모듈. 로그우도
    ℓ(λ,ν) = T1·log λ − ν·T2 − N·log Z(λ,ν),  T1=Σ X_i, T2=Σ log(X_i!)
를 (log λ, log ν)에서 L-BFGS-B로 최대화한다 (Shmueli warm-start). 관측 Fisher는
자연 파라미터에서의 중앙차분 Hessian으로 산출한다.

공개 API: cmp_mle_fit(sample) -> dict.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from cpoe import compoisson as cmp_


def _neg_loglik_log(p: np.ndarray, T1: float, T2: float, N: int) -> float:
    """−ℓ(λ,ν), p=(log λ, log ν). 발산 영역에서는 큰 값으로 되돌린다."""
    lam = math.exp(p[0]); nu = math.exp(p[1])
    try:
        lz = cmp_.log_Z(lam, nu)
    except (cmp_.CMPDivergenceError, ValueError):
        return 1e30
    return -(T1 * math.log(lam) - nu * T2 - N * lz)


def _neg_loglik_nat(lam: float, nu: float, T1: float, T2: float, N: int) -> float:
    return -(T1 * math.log(lam) - nu * T2 - N * cmp_.log_Z(lam, nu))


def _hessian2(lam: float, nu: float, T1: float, T2: float, N: int) -> np.ndarray:
    """자연 파라미터 (λ,ν)에서 −ℓ의 2×2 중앙차분 Hessian = 관측 Fisher.

    경계 안전: λ,ν>0 도메인을 벗어나지 않도록 스텝을 절반-거리로 클램프한다(특히 강한
    과대산포에서 ν̂→0일 때 중앙차분이 ν<0를 평가해 log_Z가 ValueError를 던지는 것 방지).
    """
    hl = 1e-4 * max(abs(lam), 1.0)
    hn = 1e-4 * max(abs(nu), 1.0)
    if lam > 0:
        hl = min(hl, 0.49 * lam)
    if nu > 0:
        hn = min(hn, 0.49 * nu)
    f = lambda a, b: _neg_loglik_nat(a, b, T1, T2, N)
    f0 = f(lam, nu)
    fll = (f(lam + hl, nu) - 2 * f0 + f(lam - hl, nu)) / (hl * hl)
    fnn = (f(lam, nu + hn) - 2 * f0 + f(lam, nu - hn)) / (hn * hn)
    fln = (f(lam + hl, nu + hn) - f(lam + hl, nu - hn)
           - f(lam - hl, nu + hn) + f(lam - hl, nu - hn)) / (4 * hl * hn)
    return np.array([[fll, fln], [fln, fnn]], dtype=float)


def cmp_mle_fit(sample: np.ndarray) -> dict:
    """순수 COM-Poisson (λ,ν) MLE + 관측 Fisher SE.

    반환: {lam_hat, nu_hat, se_lam, se_nu, log_lik, converged}.
    """
    sample = np.asarray(sample, dtype=int)
    N = int(sample.shape[0])
    T1 = float(sample.sum())
    T2 = float(np.sum(gammaln(sample.astype(float) + 1.0)))
    xbar = float(sample.mean()); s2 = float(sample.var(ddof=1))
    lam0, nu0 = cmp_.asymptotic_start(xbar, s2)
    res = minimize(
        _neg_loglik_log, x0=np.array([math.log(lam0), math.log(nu0)]),
        args=(T1, T2, N), method="L-BFGS-B",
    )
    lam_hat = float(math.exp(res.x[0])); nu_hat = float(math.exp(res.x[1]))
    loglik = float(-_neg_loglik_nat(lam_hat, nu_hat, T1, T2, N))
    # 관측 Fisher → cov.
    try:
        I_N = _hessian2(lam_hat, nu_hat, T1, T2, N)
        cov = np.linalg.inv(I_N)
        se_lam = float(math.sqrt(max(cov[0, 0], 0.0)))
        se_nu = float(math.sqrt(max(cov[1, 1], 0.0)))
    except (np.linalg.LinAlgError, ValueError, cmp_.CMPDivergenceError):
        # SE 계산 실패(경계·발산·특이행렬)는 점추정을 막지 않는다 → SE만 NaN.
        se_lam = se_nu = float("nan")
    return {
        "lam_hat": lam_hat, "nu_hat": nu_hat,
        "se_lam": se_lam, "se_nu": se_nu,
        "log_lik": loglik, "converged": bool(res.success),
    }
