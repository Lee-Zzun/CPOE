"""Sequential MLE (= two-step / limited-information MLE) for the CPOE model.

Joint MLE(= full-information MLE; mle.mle_fit)와 달리, (λ,ν)와 θ를 *순차*로 추정한다:
  1단계: 순수 CMP 베이스라인 (λ,ν)를 MLE로 적합 (θ=0 가정) — cmp_baseline.cmp_mle_fit.
  2단계: (λ̂,ν̂)를 *고정*하고, 그 베이스라인 위 직교전개 틸트 θ를 제약 MLE로 적합 —
         mle.solve_theta_at_baseline (Joint nested의 내부 θ-해와 *동일* 경성 KKT 제약).

CPOE 로그우도는 직교성으로 분리된다:
    ℓ(λ,ν,θ) = ℓ_CMP(λ,ν) + Σ_i log(1 + θ⊤ψ(X_i;λ,ν)).
Joint은 둘째 항의 (λ,ν)-되먹임까지 포함해 동시 최대화하는 반면, Sequential은 1단계에서
그 항을 무시(θ=0)하므로 일반적으로 (λ̂,ν̂)_seq ≠ (λ̂,ν̂)_joint이고 효율이 낮다
(일치성은 유지). 표준 명칭: two-step/two-stage MLE, limited-information ML
(Newey–McFadden 1994 §6; Murphy–Topel 1985).

feasibility(θ∈C_K) 강제는 2단계의 경성 선형제약(solve_theta_at_baseline)으로 Joint(nested)와
*완전히 동일*하다 — 두 추정법의 유일한 차이는 (λ,ν) 동시최적화 여부뿐이다.

본 모듈은 성능비교(L1) 용도이므로 SE/Fisher는 산출하지 않는다(MLEResult의 해당 필드는
NaN 플레이스홀더). SE가 필요하면 비모수 부트스트랩이 권고 기본값이다(2단계 전파 →
Murphy–Topel 보정 필요).

공개 API: sequential_mle_fit (core), fit_sequential_mle (고수준 래퍼).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cpoe.estimators.cmp_baseline import cmp_mle_fit
from cpoe.estimators.mle import (
    MLEResult,
    _LOG_CLIP,
    _neg_log_lik_smoothed,
    _pad_theta,
    solve_theta_at_baseline,
)
from cpoe.feasibility import feasibility_diagnostic


def sequential_mle_fit(
    sample: np.ndarray,
    K: int,
    *,
    cmp_mod,
    basis,
    M: int,
) -> MLEResult:
    """2단계(Sequential) MLE: CMP-MLE로 (λ,ν) → 고정 후 θ 제약 MLE.

    sample, K, cmp_mod, basis, M: 표준 CPOE 인자(mle_fit과 동일).
    반환: MLEResult (lam/nu/theta/log_likelihood/feasibility/converged 유효;
    SE·Fisher 필드는 NaN 플레이스홀더 — 본 비교는 L1만 사용).
    """
    sample = np.asarray(sample, dtype=int)
    N = int(sample.shape[0])
    if N < K + 3:
        raise ValueError(f"Sequential MLE requires N ≥ K+3 (got N={N}, K={K}).")

    # ---- 1단계: 순수 CMP 베이스라인 (λ,ν) MLE (θ=0) ----
    cmp_res = cmp_mle_fit(sample)
    lam_hat = float(cmp_res["lam_hat"])
    nu_hat = float(cmp_res["nu_hat"])

    # ---- 2단계: (λ̂,ν̂) 고정, 틸트 θ 제약 MLE (Joint nested와 동일 경성 KKT 제약) ----
    theta_free, inner_ok = solve_theta_at_baseline(
        sample, lam_hat, nu_hat, K, basis=basis, M=M,
    )
    theta_hat = _pad_theta(theta_free, K)            # [0, 0, θ̂_3,…,θ̂_K]
    eta = np.concatenate([[lam_hat, nu_hat], theta_hat])

    # ---- η̂에서의 로그우도 ----
    log_lik = -_neg_log_lik_smoothed(
        sample, eta, K, cmp_mod=cmp_mod, basis=basis, M=M, log_clip=_LOG_CLIP,
    )

    # ---- feasibility 진단 ----
    psi_support = np.asarray(basis.psi_matrix(lam_hat, nu_hat, K, M), dtype=float)
    feas = feasibility_diagnostic(theta_hat, psi_support)

    # ---- SE/Fisher: 본 비교에서 미사용 → NaN 플레이스홀더 (dataclass 충족) ----
    nan_full = np.full((2 + K, 2 + K), np.nan)
    se_theta = np.full(K, np.nan)

    converged = bool(cmp_res["converged"]) and bool(inner_ok) and bool(feas.feasible)
    diagnostics = {
        "N": int(N),
        "method": "sequential (two-step: CMP-MLE then constrained tilt-MLE)",
        "stage1_cmp": {
            "lam_hat": lam_hat, "nu_hat": nu_hat,
            "log_lik": float(cmp_res["log_lik"]),
            "converged": bool(cmp_res["converged"]),
        },
        "stage2_inner_success": bool(inner_ok),
        "min_tilt": float(feas.min_tilt),
    }
    return MLEResult(
        lam_hat=lam_hat,
        nu_hat=nu_hat,
        theta_hat=theta_hat,
        log_likelihood=float(log_lik),
        fisher_information=nan_full,
        cov_eta=nan_full.copy(),
        se_lam=float("nan"),
        se_nu=float("nan"),
        se_theta=se_theta,
        feasibility=feas,
        outer_iterations=1,
        converged=converged,
        diagnostics=diagnostics,
    )


def fit_sequential_mle(
    data,
    K: int,
    *,
    M: Optional[int] = None,
    cmp_mod=None,
    basis=None,
    L=None,  # 호출부 대칭성(fit_mle/fit_mom)을 위해 받되 무시
    **kwargs,
) -> MLEResult:
    """상위 수준 진입점: 원시 데이터 배열로부터 Sequential MLE.

    기본 백엔드와 절단 M을 채워 주는 sequential_mle_fit의 얇은 래퍼
    (fit_mle/fit_mom과 호출 대칭).
    """
    from cpoe.backends import CMP_BACKEND, BASIS_BACKEND, default_M

    sample = np.asarray(data)
    if cmp_mod is None:
        cmp_mod = CMP_BACKEND
    if basis is None:
        basis = BASIS_BACKEND
    if M is None:
        M = default_M(sample)
    return sequential_mle_fit(sample.astype(int), K, cmp_mod=cmp_mod, basis=basis, M=M)
