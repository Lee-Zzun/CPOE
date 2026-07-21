"""§4.2 eq.(19)-(25) — 제약 MLE (중첩 프로파일 + 정확한 선형 제약).

제약 MLE (eq.19):
    max_η ℓ_N(η) = Σ_i log p_θ(X_i;λ,ν)   s.t. 1+θ⊤ψ(x;λ,ν)≥0 ∀x∈S_M.

해법 (§5 computing; COMPUTING_EXPLANATION §3-(b)): 논문 §5.2의 장벽 증강
벌점 (ρ↑)을 **중첩 프로파일 + 정확한 선형 제약**으로 대체한다.
  - 외부: (λ,ν) 2D L-BFGS-B (프로파일 아웃).
  - 내부: 고정된 (λ,ν)에서 θ-문제는 로그-오목 + 선형 제약 = 볼록.
    trust-constr + LinearConstraint(ψ,−1,∞) + 해석적 grad/Hess로 정확히 푼다.
  프로파일 = 결합 MLE (포락선 정리). 장벽이 할 수 없었던 것 (꼬리-ψ 폭발로 인한
  θ=0 함정)은 제약을 KKT로 정확히 처리하여 해결된다.

관측 Fisher (eq.25): η̂에서 I_N ≈ −∂²ℓ_N/∂η². 벌점 없는 −ℓ_N의 중심 차분
헤시안 + PSD 사영 (중심 차분이 최적점에서 만드는 작은 음의 고유값을 피함).
θ_1=θ_2=0은 정규화로 고정되며 파라미터가 아니므로 곡률/SE를 갖지 않는다 —
자유 (λ,ν,θ_3..θ_K)만 갖는다.

공개 API: MLEResult (출력 데이터클래스), mle_fit (주 진입점).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, LinearConstraint

from cpoe.compoisson import (
    CMPDivergenceError,
    LAM_MAX,
    LAM_MIN,
    NU_MAX,
    NU_MIN,
)
from cpoe.cpoe_model import empirical_theta
from cpoe.feasibility import (
    FeasibilityReport,
    feasibility_diagnostic,
)


_BIG = 1e30
_LOG_CLIP = 1e-12


@dataclass
class MLEResult:
    """MLE 출력 컨테이너."""

    lam_hat: float
    nu_hat: float
    theta_hat: np.ndarray
    log_likelihood: float
    fisher_information: np.ndarray   # (2+K)×(2+K), PSD 사영된 관측 정보
    cov_eta: np.ndarray              # (2+K)×(2+K), fisher_information의 역행렬
    se_lam: float
    se_nu: float
    se_theta: np.ndarray
    feasibility: FeasibilityReport
    outer_iterations: int
    converged: bool
    diagnostics: dict = field(default_factory=dict)

    @property
    def log_lik(self) -> float:
        """상위 수준 호출자 (fit_mle 소비자)를 위한 별칭."""
        return float(self.log_likelihood)


def _neg_log_lik_smoothed(sample: np.ndarray, eta: np.ndarray, K: int,
                          *, cmp_mod, basis, M: int,
                          log_clip: float = _LOG_CLIP) -> float:
    """− ℓ_N(η), log(·)를 아래로부터 클리핑.

    프로파일 목적함수가 NaN 없이 ∂C_K 근방을 통과하도록 한다. 최종
    실행가능한 η̂ (모든 tilt > log_clip)에서는 편향이 없다.
    """
    lam = float(eta[0])
    nu = float(eta[1])
    theta = np.asarray(eta[2:], dtype=float)
    if lam <= 0.0 or nu <= 0.0:
        return _BIG
    try:
        log_w = np.asarray(cmp_mod.log_pmf(np.arange(M), lam, nu, M=M), dtype=float)
        psi = np.asarray(basis.psi_matrix(lam, nu, K, M), dtype=float)
    except (CMPDivergenceError, np.linalg.LinAlgError, ValueError):
        # 실행불가능 탐침 (다루기 힘든 Z, 퇴화 측도): 큰 목적값을 반환하여
        # 최적화기를 계산 가능 영역으로 되밀어 넣는다.
        return _BIG
    tilt = 1.0 + psi @ theta
    s = np.clip(sample, 0, M - 1)
    log_tilt_s = np.log(np.maximum(tilt[s], log_clip))
    log_p_s = log_w[s] + log_tilt_s
    if not np.all(np.isfinite(log_p_s)):
        return _BIG
    return float(-log_p_s.sum())


def _penalty(eta: np.ndarray, K: int, *, basis, M: int) -> float:
    """P(η) = Σ_x [max(−(1+θ⊤ψ(x)),0)]² (절단 지지대). 최종 제약 위반에 대한
    진단 (중첩은 제약을 정확히 만족 → ≈0이어야 함)."""
    lam = float(eta[0])
    nu = float(eta[1])
    theta = np.asarray(eta[2:], dtype=float)
    if lam <= 0.0 or nu <= 0.0:
        return 0.0
    try:
        psi = np.asarray(basis.psi_matrix(lam, nu, K, M), dtype=float)
    except (CMPDivergenceError, np.linalg.LinAlgError, ValueError):
        return _BIG  # 실행불가능 탐침 → 큰 값
    tilt = 1.0 + psi @ theta
    viol = np.maximum(-tilt, 0.0)
    return float((viol * viol).sum())


def _pad_theta(free: np.ndarray, K: int) -> np.ndarray:
    """자유 계수 (θ_3..θ_K)를 길이-K 벡터 [0, 0, θ_3,…,θ_K]에 끼워 넣는다."""
    K = int(K)
    full = np.zeros(K)
    free = np.asarray(free, dtype=float).ravel()
    if K > 2 and free.size:
        full[2:2 + free.size] = free
    return full


def _scatter_free_to_full(M_free: np.ndarray, K: int) -> np.ndarray:
    """(2+nf)×(2+nf) 자유-파라미터 행렬을 (2+K)×(2+K) 전체-파라미터 공간에 끼워 넣는다.

    자유 순서 (λ,ν,θ_3..θ_K), 전체 순서 (λ,ν,θ_1,θ_2,θ_3..θ_K) → 고정된 θ_1,θ_2
    행/열 (전체 인덱스 2,3)은 0으로 유지된다. fisher/cov를 (2+K)×(2+K)로 산출하며, θ_1,θ_2 SE=0.
    """
    M_free = np.asarray(M_free, dtype=float)
    nf = M_free.shape[0] - 2
    full = np.zeros((2 + K, 2 + K))
    idx = np.array([0, 1] + list(range(4, 4 + max(0, nf))), dtype=int)  # λ,ν,θ_3..θ_K
    full[np.ix_(idx, idx)] = M_free
    return full


def _numerical_hessian(f, x: np.ndarray, *, rel_h: float = 1e-4,
                       abs_h: float = 1e-6) -> np.ndarray:
    """x에서 스칼라 f의 중심 차분 헤시안.

    ν 성분 (인덱스 1)은 x[1]-h < NU_MIN이면 전진 차분을 사용한다 — 지지대 바닥
    아래를 탐침하면 _neg_log_lik_smoothed가 _BIG을 반환하여 (ν̂~0 근처에서 헤시안을
    오염시킴), 이를 피한다.
    """
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    H = np.zeros((d, d))
    f0 = float(f(x))
    h = np.array([rel_h * max(1.0, abs(x[i])) + abs_h for i in range(d)])

    # +1 = 중심 차분, +2 = 전진 (단측) 차분. NU_MIN 근처의 ν에 사용.
    direction = np.ones(d, dtype=int)
    if d >= 2 and (x[1] - h[1]) < NU_MIN:
        direction[1] = 2  # ν 방향의 전진 차분

    def _fwd(i: int):
        e = np.zeros(d); e[i] = h[i]
        return float(f(x + e)), float(f(x + 2.0 * e))

    def _back(i: int):
        e = np.zeros(d); e[i] = h[i]
        return float(f(x - e))

    # 대각
    for i in range(d):
        if direction[i] == 1:
            ei = np.zeros(d); ei[i] = h[i]
            fpp = float(f(x + ei))
            fmm = float(f(x - ei))
            H[i, i] = (fpp - 2.0 * f0 + fmm) / (h[i] * h[i])
        else:  # 전진 차분: (f(x+2h) - 2f(x+h) + f(x)) / h²
            fp, fpp2 = _fwd(i)
            H[i, i] = (fpp2 - 2.0 * fp + f0) / (h[i] * h[i])

    # 비대각
    for i in range(d):
        for j in range(i + 1, d):
            ei = np.zeros(d); ei[i] = h[i]
            ej = np.zeros(d); ej[j] = h[j]
            if direction[i] == 1 and direction[j] == 1:
                fpp = float(f(x + ei + ej))
                fpm = float(f(x + ei - ej))
                fmp = float(f(x - ei + ej))
                fmm = float(f(x - ei - ej))
                H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
            else:
                # 경계에 닿는 방향은 전진, 나머지는 중심을 사용. 한쪽이 전진이면
                # 스텐실 일관성을 위해 양쪽 모두 전진으로:
                # ∂²f/∂xi∂xj ~ (f(x+ei+ej)-f(x+ei)-f(x+ej)+f(x))/(hi·hj)
                f_pp = float(f(x + ei + ej))
                f_p0 = float(f(x + ei))
                f_0p = float(f(x + ej))
                H[i, j] = (f_pp - f_p0 - f_0p + f0) / (h[i] * h[j])
            H[j, i] = H[i, j]

    return H


def _psd_project(H: np.ndarray, rel_floor: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, dict]:
    """작은 고유값을 바닥값으로 올려 대칭 H를 PSD로 사영한다. (H_psd, H_psd_inv, info)를 반환."""
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    eps_floor = rel_floor * (float(np.abs(w).max()) + 1.0)
    w_floored = np.where(w > eps_floor, w, eps_floor)
    H_psd = (V * w_floored) @ V.T
    H_inv = (V * (1.0 / w_floored)) @ V.T
    H_psd = 0.5 * (H_psd + H_psd.T)
    H_inv = 0.5 * (H_inv + H_inv.T)
    info = {
        "min_eigenvalue_raw": float(w.min()),
        "max_eigenvalue_raw": float(w.max()),
        "eps_floor_used": float(eps_floor),
        "n_floored": int(np.sum(w <= eps_floor)),
    }
    return H_psd, H_inv, info


def solve_theta_at_baseline(
    sample: np.ndarray, lam: float, nu: float, K: int, *, basis, M: int,
) -> Tuple[np.ndarray, bool]:
    """고정 (λ,ν)에서 자유계수 θ_3..θ_K의 제약 MLE (내부 θ-부분문제).

        max_θ Σ_i log(1+θ⊤ψ(X_i;λ,ν))  s.t.  1+θ⊤ψ(x;λ,ν) ≥ 0  ∀x∈S_M,

    θ_1=θ_2=0 (B0)으로 자유 열 ψ_3..ψ_K만 사용. 고정 (λ,ν)에서 목적은 로그-오목,
    제약은 θ에 선형(볼록 계획) → trust-constr + LinearConstraint(ψ_free,−1,∞) +
    해석적 grad/Hess로 정확히 푼다. (자유계수[길이 n_free], 성공 플래그) 반환.

    nested Joint MLE(mle_fit 내부)와 Sequential MLE(mle_sequential)의 *공용* θ-해 —
    feasibility 강제가 두 추정법에서 동일하도록 단일 출처로 둔다.
    """
    n_free = max(0, int(K) - 2)
    if n_free == 0:
        return np.zeros(0), True
    sample = np.asarray(sample, dtype=int)
    try:
        psi = np.asarray(basis.psi_matrix(lam, nu, K, M), dtype=float)
    except (CMPDivergenceError, np.linalg.LinAlgError, ValueError):
        return np.zeros(n_free), False
    psi_free = psi[:, 2:K]                          # (M, n_free) = ψ_3..ψ_K
    psid = psi_free[np.clip(sample, 0, M - 1)]
    lc = LinearConstraint(psi_free, -1.0, np.inf)

    def n_ll(th):
        t = 1.0 + psid @ th
        if t.min() <= 1e-12:
            return 1e30
        return float(-np.log(t).sum())

    def n_g(th):
        t = 1.0 + psid @ th
        return -(psid / t[:, None]).sum(0)

    def n_h(th):
        t = 1.0 + psid @ th
        return (psid / (t * t)[:, None]).T @ psid

    rr = minimize(n_ll, np.zeros(n_free), jac=n_g, hess=n_h, method="trust-constr",
                  constraints=[lc],
                  options={"maxiter": 400, "gtol": 1e-8, "xtol": 1e-12, "verbose": 0})
    return np.asarray(rr.x, dtype=float), bool(rr.success)


def mle_fit(
    sample: np.ndarray,
    K: int,
    *,
    cmp_mod,
    basis,
    M: int,
    eta_init: Optional[np.ndarray] = None,
    max_outer: int = 12,
    inner_max_iter: int = 500,
    inner_gtol: float = 1e-6,
    inner_ftol: float = 1e-12,
    feas_tol: float = 1e-8,
    lam_bounds: Tuple[float, float] = (LAM_MIN, LAM_MAX),
    nu_bounds: Tuple[float, float] = (NU_MIN, NU_MAX),
    theta_bound: float = 50.0,
    theta_init: str = "origin",
    grad_h: float = 1e-6,
) -> MLEResult:
    """중첩 제약 MLE 실행 (외부 (λ,ν) 프로파일 / 내부 θ 볼록 선형 제약).

    sample, K, cmp_mod, basis, M: 표준 CPOE 인자.
    eta_init: (2+K,) 시작점. None이면 (λ_0,ν_0)←asymptotic_start이며 θ_0은 theta_init으로 설정된다.
    theta_init: {"origin","empirical"}. "origin" (기본값) θ_0=0 (항상 실행가능);
        "empirical"은 empirical_theta로 시드 (실행불가능하면 원점 방향으로 선 탐색; 레거시).
    max_outer, inner_max_iter, inner_gtol, inner_ftol: 외부/내부 최적화기 옵션.
    feas_tol: 수렴 판정을 위한 min-tilt 허용오차.
    lam_bounds, nu_bounds, theta_bound: 박스 경계 (θ 실행가능성은 선형 제약으로 처리됨).
    """
    sample = np.asarray(sample, dtype=int)
    N = sample.shape[0]
    if N < K + 3:
        raise ValueError(f"MLE requires N ≥ K+3 (got N={N}, K={K}).")
    n_free = max(0, K - 2)            # 자유 계수 θ_3..θ_K (θ_1=θ_2=0 고정, B0)

    # ---- 초기화 (자유 계수 θ_3..θ_K만) ----
    if eta_init is None:
        xbar = float(sample.mean())
        s2 = float(sample.var(ddof=1))
        # Shmueli 점근 웜스타트 (§4.1), 박스로 클리핑. (λ,ν)를 계산 가능 영역의 해
        # 근처에 안착시킨다.
        if hasattr(cmp_mod, "asymptotic_start"):
            a_lam, a_nu = cmp_mod.asymptotic_start(xbar, s2)
        else:  # pragma: no cover
            a_lam, a_nu = max(xbar, 1e-2), 1.0
        lam0 = float(np.clip(a_lam, lam_bounds[0], lam_bounds[1]))
        nu0 = float(np.clip(a_nu, nu_bounds[0], nu_bounds[1]))
        if theta_init == "origin":
            # v1 기본값: θ_0=0은 항상 C_K 내부에 있다 (1 + 0ᵀψ = 1 > 0).
            theta0 = np.zeros(n_free)
        elif theta_init == "empirical":
            theta0 = empirical_theta(sample, lam0, nu0, K, basis=basis, M=M)[2:].copy()
            psi_at_init = np.asarray(basis.psi_matrix(lam0, nu0, K, M), dtype=float)
            psi_free0 = psi_at_init[:, 2:K] if K > 2 else psi_at_init[:, :0]
            if n_free and (1.0 + psi_free0 @ theta0).min() <= 1e-3:
                alpha = 1.0
                for _ in range(30):
                    alpha *= 0.5
                    if (1.0 + psi_free0 @ (alpha * theta0)).min() > 1e-3:
                        theta0 = alpha * theta0
                        break
                else:
                    theta0 = np.zeros(n_free)
        else:
            raise ValueError(f"theta_init must be 'origin' or 'empirical'; got {theta_init!r}")
        eta_init = np.concatenate([[lam0, nu0], theta0])
    eta = np.asarray(eta_init, dtype=float).copy()

    # ---- 제약 MLE: 중첩 최적화 (외부 λ,ν / 내부 θ) ----
    # 고정된 (λ,ν)에서 C_K (eq.13)는 θ에 선형이고 ℓ_N은 θ에 오목 → θ-부분문제는
    # 정확한 선형 제약 + 해석적 grad/Hess (trust-constr)로 푸는 잘 조건화된 볼록
    # 계획법이다. 외부 벌점 / 로그-장벽이 할 수 없었던 것: 이들은 전체 지지대
    # 제약을 하나의 매끄러운 항으로 뭉뚱그려, 기울기가 꼬리-ψ 폭발에 지배되어
    # 실행가능한 가능도 이득이 클 때조차 θ=0에 갇힌다.
    # 외부 수준은 2D (λ,ν)만 프로파일한다. C_K는 변하지 않는다
    # (엄격, 전체 지지대); θ_0=0은 중립적 시작이다. (§3 이론이 아닌 §5 computing.)
    def _solve_theta(lam_i, nu_i):
        # 모듈 레벨 solve_theta_at_baseline로 위임 (Joint·Sequential 공용; 거동 불변).
        return solve_theta_at_baseline(sample, lam_i, nu_i, K, basis=basis, M=M)

    def _negll_fixed_theta(lam_i, nu_i, th_free):
        """고정 θ*에서의 −ℓ. 내부 재최적화가 없어 (λ,ν)에 매끄러운 해석적 함수."""
        return _neg_log_lik_smoothed(
            sample, np.concatenate([[lam_i, nu_i], _pad_theta(th_free, K)]), K,
            cmp_mod=cmp_mod, basis=basis, M=M,
        )

    def _profile_val_grad(p):
        """프로파일 −ℓ 값과 포락선(Danskin) 기울기를 함께 반환 (jac=True).

        ∇g(λ,ν) = ∂(−ℓ)/∂(λ,ν)|_{θ* 고정} (envelope theorem; Danskin 1967;
        프로파일 우도 Murphy–van der Vaart 2000). θ*를 \emph{고정}해 중심차분하므로
        외부가 매 평가마다 내부를 재최적화하며 생기던 \emph{잡음}(원래의 niter=1 조기종료
        버그 원인)이 사라지고, 매끄러운 −ℓ을 미분한다. 관측점 tilt는 우도가 강제로
        >0으로 유지하므로 작은 (λ,ν) 섭동에도 유효(내부해 interior).
        """
        lam_i, nu_i = float(p[0]), float(p[1])
        if not (lam_bounds[0] < lam_i < lam_bounds[1] and nu_bounds[0] < nu_i < nu_bounds[1]):
            return _BIG, np.zeros(2)
        th_free, _ = _solve_theta(lam_i, nu_i)
        f0 = _negll_fixed_theta(lam_i, nu_i, th_free)
        # 유효성 가드: 정규화된 pmf는 p(x)≤1 ⟹ ℓ_N≤0 ⟺ f0=−ℓ_N≥0. f0<0이면 고차 기저
        # ill-conditioning으로 정규화가 붕괴한 퇴화해이므로 큰 값으로 막아 최적화기를 유효역에 가둔다.
        if not np.isfinite(f0) or f0 >= _BIG or f0 < -1e-6:
            return _BIG, np.zeros(2)
        grad = np.zeros(2)
        center = (lam_i, nu_i)
        lo = (lam_bounds[0], nu_bounds[0])
        hi = (lam_bounds[1], nu_bounds[1])
        for j in range(2):
            h = grad_h * max(1.0, abs(center[j]))
            xp = min(center[j] + h, hi[j] - 1e-12)
            xm = max(center[j] - h, lo[j] + 1e-12)
            pp = list(center); pm = list(center)
            pp[j] = xp; pm[j] = xm
            fp = _negll_fixed_theta(pp[0], pp[1], th_free)
            fm = _negll_fixed_theta(pm[0], pm[1], th_free)
            denom = xp - xm
            grad[j] = ((fp - fm) / denom
                       if (denom > 0 and np.isfinite(fp) and np.isfinite(fm)) else 0.0)
        return f0, grad

    # 외부 (λ,ν) 최적화: 절단 뉴턴 TNC(Nash 1984) + 해석적 포락선 기울기(jac=True). L-BFGS-B는
    # 기울기 크기(~2e2) 대비 (λ,ν) 스케일(~1e-1)이 작아 오버슈팅으로 niter=1 조기종료했으나 TNC는 견딘다.
    # 다중 시작점(multistart; Nocedal–Wright 2006): 주 시작점 asymptotic_start(요청대로 유지)에 더해
    # 안전 시작점 CMP-MLE를 추가해 외부가 국소최적에 갇히는 것을 막는다 — CMP-MLE 시작은 최소한
    # Sequential 해를 가용점으로 주므로 ℓ_Joint ≥ ℓ_Seq를 보장한다(예: S3/S4 K=8 국소최적 회피).
    _tnc_opts = {"maxfun": max(200, max_outer * 40), "ftol": inner_ftol,
                 "xtol": inner_ftol, "gtol": inner_gtol}
    _starts = [np.array([float(eta[0]), float(eta[1])])]
    try:
        from cpoe.estimators.cmp_baseline import cmp_mle_fit as _cmp_mle_fit
        _c = _cmp_mle_fit(sample)
        _cm0 = float(np.clip(_c["lam_hat"], lam_bounds[0], lam_bounds[1]))
        _cm1 = float(np.clip(_c["nu_hat"], nu_bounds[0], nu_bounds[1]))
        if np.isfinite(_cm0) and np.isfinite(_cm1):
            _starts.append(np.array([_cm0, _cm1]))
    except Exception:  # noqa: BLE001 - CMP-MLE 실패해도 주 시작점으로 진행
        pass
    outer = None
    for _x0 in _starts:
        _r = minimize(_profile_val_grad, _x0, method="TNC", jac=True,
                      bounds=[lam_bounds, nu_bounds], options=_tnc_opts)
        if outer is None or (np.isfinite(_r.fun) and _r.fun < outer.fun):
            outer = _r
    lam_o, nu_o = float(outer.x[0]), float(outer.x[1])
    theta_o_free, inner_ok = _solve_theta(lam_o, nu_o)
    theta_o = _pad_theta(theta_o_free, K)               # [0, 0, θ_3,…,θ_K]
    eta = np.concatenate([[lam_o, nu_o], theta_o])
    psi_fin = np.asarray(basis.psi_matrix(lam_o, nu_o, K, M), dtype=float)
    min_tilt_fin = float((1.0 + psi_fin @ theta_o).min())
    outer_iter = int(getattr(outer, "nit", -1))
    inner_success = bool(inner_ok)
    inner_message = str(getattr(outer, "message", ""))
    # 수렴 판정: TNC의 success 플래그(고차에서 'Linear search failed'를 자주 띄움)에 의존하지 않고
    # 결과의 *타당성*으로 판정한다. 정규화된 pmf는 p(x)≤1 ⟹ ℓ_N=Σ_i log p(X_i) ≤ 0이어야 하므로,
    # ℓ_N>0이면 고차 기저 ill-conditioning으로 정규화가 붕괴한 퇴화해 → 미수렴 처리(신뢰성 게이트에서 제외).
    _ll_chk = -_neg_log_lik_smoothed(sample, eta, K, cmp_mod=cmp_mod, basis=basis, M=M)
    valid_ll = bool(np.isfinite(_ll_chk)) and (_ll_chk <= 1e-6)
    converged = bool(inner_ok) and (min_tilt_fin >= -feas_tol) and valid_ll
    history = [{
        "solver": "nested(TNC outer / trust-constr linear inner; envelope grad)",
        "outer_success": bool(outer.success),
        "outer_niter": outer_iter,
        "inner_success": bool(inner_ok),
        "neg_log_lik": float(_neg_log_lik_smoothed(sample, eta, K, cmp_mod=cmp_mod, basis=basis, M=M)),
        "min_tilt": min_tilt_fin,
        "message": inner_message,
    }]

    lam_hat = float(eta[0])
    nu_hat = float(eta[1])
    theta_hat = eta[2:].copy()

    # ---- η̂에서의 정확한 로그-가능도 ----
    log_lik = -_neg_log_lik_smoothed(
        sample, eta, K, cmp_mod=cmp_mod, basis=basis, M=M, log_clip=_LOG_CLIP,
    )

    # ---- 관측 Fisher 정보: η̂에서 I_obs = ∂²(−ℓ_N)/∂η² ----
    # 자유 파라미터 (λ,ν,θ_3..θ_K)에 대해서만 — θ_1=θ_2=0은 정규화로 고정되며
    # 곡률/SE를 갖지 않는다 (파라미터 아님).
    def _nll_free(zf):
        eta_full = np.concatenate([[zf[0], zf[1]], _pad_theta(zf[2:], K)])
        return _neg_log_lik_smoothed(
            sample, eta_full, K, cmp_mod=cmp_mod, basis=basis, M=M, log_clip=_LOG_CLIP,
        )

    z_free = np.concatenate([[lam_hat, nu_hat], theta_o_free])
    H_free = _numerical_hessian(_nll_free, z_free)
    I_free, cov_free, psd_info = _psd_project(H_free)
    I_obs = _scatter_free_to_full(I_free, K)     # (2+K)×(2+K), θ_1,θ_2 자리에 0
    cov_eta = _scatter_free_to_full(cov_free, K)

    se_free = np.sqrt(np.maximum(np.diag(cov_free), 0.0))
    se_lam = float(se_free[0])
    se_nu = float(se_free[1])
    se_theta = _pad_theta(se_free[2:], K)        # 길이 K, θ_1,θ_2 자리에 0

    psi_support = np.asarray(basis.psi_matrix(lam_hat, nu_hat, K, M), dtype=float)
    feas = feasibility_diagnostic(theta_hat, psi_support)

    diagnostics = {
        "N": int(N),
        "outer_history": history,
        "hessian_psd": psd_info,
        "final_penalty": float(_penalty(eta, K, basis=basis, M=M)),
        "final_inner_message": inner_message,
        "final_inner_success": bool(inner_success),
    }
    return MLEResult(
        lam_hat=lam_hat,
        nu_hat=nu_hat,
        theta_hat=theta_hat,
        log_likelihood=float(log_lik),
        fisher_information=I_obs,
        cov_eta=cov_eta,
        se_lam=se_lam,
        se_nu=se_nu,
        se_theta=se_theta,
        feasibility=feas,
        outer_iterations=int(outer_iter),
        converged=bool(converged),
        diagnostics=diagnostics,
    )


def fit_mle(
    data,
    K: int,
    *,
    M: int | None = None,
    cmp_mod=None,
    basis=None,
    method: str = "nested",
    L=None,  # 호출부 대칭성을 위해 받아들임; 무시됨
    **kwargs,
) -> MLEResult:
    """상위 수준 진입점: 원시 데이터 배열로부터 중첩(nested) KKT 제약 MLE(완전정보).

    외부 (λ,ν) TNC 프로파일(해석적 포락선 기울기 + 다중 시작점) / 내부 θ 경성 KKT 제약해.
    기본 백엔드와 절단 M을 채워 mle_fit에 위임하는 얇은 래퍼다.
    """
    from cpoe.backends import CMP_BACKEND, BASIS_BACKEND, default_M

    sample = np.asarray(data)
    if cmp_mod is None:
        cmp_mod = CMP_BACKEND
    if basis is None:
        basis = BASIS_BACKEND
    if M is None:
        M = default_M(sample)
    if method != "nested":
        raise ValueError(f"unknown MLE method {method!r}; only 'nested' is supported.")
    return mle_fit(sample.astype(int), K, cmp_mod=cmp_mod, basis=basis, M=M, **kwargs)
