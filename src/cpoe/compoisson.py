"""COM-Poisson 베이스라인: Z(λ,ν)·모멘트(mu, var)·pmf·점근 warm-start

논문 §5(Algorithm 1)의 CMP 수치 경로를 구현한다.
  eq(1) pmf  w_CMP(x;λ,ν) = λ^x / ((x!)^ν · Z(λ,ν))
  eq(2) Z    = Σ_j λ^j/(j!)^ν       — 적응적 절단 + log-sum-exp
  eq(3) μ=E[X],  eq(4) Var[X]

극한(§2.1): ν=1→Poisson, ν=0&λ<1→기하, ν→∞→Bernoulli.

하이퍼파라미터(config/hyperparams.yaml; 하드코딩 금지 — NU_MIN 아티팩트 정책):
  TOL, DECAY_STREAK_REQUIRED, M_HARD_MAX, 허용 모수 박스(NU/LAM_MIN·MAX).
  tol은 호출별 키워드로 override 가능.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import gammaln

try:
    import yaml
except ImportError:  # pragma: no cover -- pyyaml은 pyproject.toml 의존성
    yaml = None


def _load_hyperparams() -> dict:
    """config/hyperparams.yaml 로드. 없으면 경고 후 기본값(유닛테스트 단독 import 허용)."""
    if yaml is None:
        return {}
    cfg = Path(__file__).resolve().parents[2] / "config" / "hyperparams.yaml"
    if not cfg.exists():
        warnings.warn(
            f"config/hyperparams.yaml를 {cfg}에서 못 찾음; 기본값으로 폴백. "
            "유닛테스트엔 무방하나 paper-regression 실행은 반드시 config를 가리켜야 함 "
            "(민감도 스윕은 config 없이는 동작 불가).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    with open(cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_HP = _load_hyperparams()
TOL_DEFAULT: float = float(_HP.get("TOL", 1e-12))
_DECAY_STREAK_REQUIRED: int = int(_HP.get("DECAY_STREAK_REQUIRED", 5))
M_HARD_MAX: int = int(_HP.get("M_HARD_MAX", 100_000))

# 허용 모수 박스(config). asymptotic_start와 추정기 탐색범위(mom/mle)에 주입돼
# 옵티마이저가 수치적으로 다루기 힘든 (큰 λ, 작은 ν) 구석으로 새는 것을 막는다.
NU_MIN: float = float(_HP.get("NU_MIN", 0.02))
NU_MAX: float = float(_HP.get("NU_MAX", 20.0))
LAM_MIN: float = float(_HP.get("LAM_MIN", 1.0e-3))
LAM_MAX: float = float(_HP.get("LAM_MAX", 1.0e3))


class CMPDivergenceError(RuntimeError):
    """Z(λ,ν)를 M_HARD_MAX 항 안에 절단하지 못할 때 발생.

    ν>0이면 급수는 항상 수렴하지만 모드 λ^(1/ν)가 절단 예산을 넘으면 부분합이
    꼬리에 닿지 못한다. 추정기 목적함수가 이를 잡아 큰 목적값(_BIG)을 반환함으로써
    옵티마이저를 계산가능 영역으로 되돌린다(steer-back). 발산 영역은 답이 될 수 없어
    별도 점근 치환을 두지 않으며, 애초에 허용 박스 + asymptotic_start가 탐색을
    계산가능 영역에 묶어둔다.
    """

    def __init__(self, lam: float, nu: float, budget: int):
        self.lam = float(lam)
        self.nu = float(nu)
        self.budget = int(budget)
        super().__init__(
            f"Z(lam={lam}, nu={nu})의 모드 ~ lam**(1/nu)가 절단 예산 "
            f"M_HARD_MAX={budget}를 초과; (lam, nu)를 허용 박스 안에 유지하라."
        )


@dataclass(frozen=True)
class CMPCache:
    """한 (λ,ν,tol)의 절단 지지대 j=0..M 위 로그항 캐시."""

    lam: float
    nu: float
    M: int
    log_terms: np.ndarray  # shape (M+1,), log(λ^j/(j!)^ν), 정규화 전
    log_Z: float


def _choose_truncation(
    lam: float, nu: float, tol: float
) -> tuple[int, np.ndarray, float]:
    """적응적 절단. (M, log_terms[0..M], log_Z) 반환.

    정책(§5.1 빈 본문 채움): j=0,1,...로 진행하며 log-sum-exp로 누적.
    모드를 안전하게 지난 뒤(항이 _DECAY_STREAK_REQUIRED회 연속 감소)
    log(항_j) - log(누적합) < log(tol)이면 종료.
    """
    if lam <= 0:
        raise ValueError(f"lambda는 > 0이어야 함, got {lam}")
    if nu < 0:
        raise ValueError(f"nu는 >= 0이어야 함, got {nu}")
    if nu == 0.0 and lam >= 1.0:
        raise ValueError(
            f"nu=0이면 eq(2) 급수는 lambda < 1에서만 수렴 (got lambda={lam})"
        )

    # 싼 발산 조기탐지: 모드 λ^(1/ν)가 예산을 넘으면 M_HARD_MAX 항을 다 돌려도
    # 꼬리에 닿지 못하므로, 예산을 갈지 말고 즉시 실패시킨다.
    if nu > 0.0 and (math.log(lam) / nu) > math.log(M_HARD_MAX):
        raise CMPDivergenceError(lam, nu, M_HARD_MAX)

    log_lam = math.log(lam)
    log_tol = math.log(tol)

    log_terms: list[float] = []
    log_running_sum = -math.inf
    log_running_max = -math.inf
    decay_streak = 0

    for j in range(M_HARD_MAX + 1):
        log_t = j * log_lam - nu * gammaln(j + 1)
        log_terms.append(float(log_t))
        log_running_sum = float(np.logaddexp(log_running_sum, log_t))

        if log_t < log_running_max:
            decay_streak += 1
        else:
            log_running_max = float(log_t)
            decay_streak = 0

        if decay_streak >= _DECAY_STREAK_REQUIRED and (log_t - log_running_sum) < log_tol:
            return j, np.asarray(log_terms, dtype=np.float64), log_running_sum

    raise CMPDivergenceError(lam, nu, M_HARD_MAX)


@lru_cache(maxsize=4096)
def _cache(lam: float, nu: float, tol: float) -> CMPCache:
    M, log_terms, log_Z_val = _choose_truncation(float(lam), float(nu), float(tol))
    return CMPCache(
        lam=float(lam),
        nu=float(nu),
        M=int(M),
        log_terms=log_terms,
        log_Z=float(log_Z_val),
    )


# ---------- log Z, Z, support_upper ----------


def log_Z(lam: float, nu: float, *, tol: float = TOL_DEFAULT) -> float:
    """log Z(λ,ν). 절단 지지대 위 log-sum-exp 값."""
    return _cache(float(lam), float(nu), float(tol)).log_Z


def Z(lam: float, nu: float, *, tol: float = TOL_DEFAULT) -> float:
    """eq(2): Z(λ,ν) = Σ_{j≥0} λ^j/(j!)^ν, 적응적 절단."""
    return float(math.exp(_cache(float(lam), float(nu), float(tol)).log_Z))


def support_upper(lam: float, nu: float, *, tol: float = TOL_DEFAULT) -> int:
    """내부 절단 상한 M. 다운스트림 기저 구축이 재사용."""
    return _cache(float(lam), float(nu), float(tol)).M


# ---------- 모멘트: mu, var ----------


def _weights_normalized(
    lam: float, nu: float, tol: float
) -> tuple[np.ndarray, int]:
    """정규화 가중치 w_j = exp(log_terms - log_Z)와 절단 상한 M."""
    c = _cache(float(lam), float(nu), float(tol))
    return np.exp(c.log_terms - c.log_Z), c.M


def mu(lam: float, nu: float, *, tol: float = TOL_DEFAULT) -> float:
    """eq(3): E_{w_CMP}[X]."""
    w, M = _weights_normalized(lam, nu, tol)
    js = np.arange(M + 1, dtype=np.float64)
    return float(np.sum(js * w))


def var(lam: float, nu: float, *, tol: float = TOL_DEFAULT) -> float:
    """eq(4): Var_{w_CMP}[X] = E[X²] - (E[X])²."""
    w, M = _weights_normalized(lam, nu, tol)
    js = np.arange(M + 1, dtype=np.float64)
    m1 = float(np.sum(js * w))
    m2 = float(np.sum(js * js * w))
    return m2 - m1 * m1


# ---------- pmf / log pmf ----------


def pmf(
    x: int | np.ndarray,
    lam: float,
    nu: float,
    *,
    M: int | None = None,
    tol: float = TOL_DEFAULT,
):
    """eq(1): w_CMP(x;λ,ν). eq(12) 선형 tilt 밀도(cpoe_model.py)의 첫 인자.

    M: 추정기/모델층이 넘기는 절단 상한 힌트. pmf는 적응적 Z(λ,ν)에 대해 x에서
    직접 평가하므로 반환값은 M에 영향받지 않음 — 실험층 어댑터와 (lam, nu, M=)
    프로토콜을 맞추려고 받기만 한다.
    """
    c = _cache(float(lam), float(nu), float(tol))
    x_arr = np.asarray(x, dtype=np.int64)
    if np.any(x_arr < 0):
        raise ValueError(f"x는 음이 아닌 정수여야 함; got {x}")
    log_p = x_arr * math.log(lam) - nu * gammaln(x_arr + 1) - c.log_Z
    out = np.exp(log_p)
    if np.isscalar(x):
        return float(out)
    return out


def log_pmf(
    x: int | np.ndarray,
    lam: float,
    nu: float,
    *,
    M: int | None = None,
    tol: float = TOL_DEFAULT,
):
    """log w_CMP(x;λ,ν) = x·log λ - ν·log x! - log Z.

    MLE 목적함수(estimators/mle.py)와 cpoe_model.cpoe_log_pmf가 직접 호출.
    로그공간 1차 계산이라 꼬리에서 pmf가 0으로 언더플로돼도 유한값을 유지한다
    (log(pmf)로 대체하면 -inf로 깨져 우도 합이 오염됨). M은 프로토콜 호환용 더미.
    """
    c = _cache(float(lam), float(nu), float(tol))
    x_arr = np.asarray(x, dtype=np.int64)
    if np.any(x_arr < 0):
        raise ValueError(f"x는 음이 아닌 정수여야 함; got {x}")
    out = x_arr * math.log(lam) - nu * gammaln(x_arr + 1) - c.log_Z
    if np.isscalar(x):
        return float(out)
    return out


# ---------- 점근 warm-start 헬퍼 ----------
# v1: 호출처 0인 헬퍼는 제거 — 닫힌형 극한(poisson/geometric/bernoulli_limit_log_Z)과
# 대-λ 점근 logZ(asymptotic_log_Z). 후자가 폴백한다던 발산 영역은 실제론 옵티마이저가
# 후퇴(steer-back)하므로 점근값 치환이 불필요했다(박스 전략과 중복).
# ν=1 Poisson 극한(logZ≈λ)은 tests/test_compoisson_log_Z.py에서 일반 log_Z 경로 대비 검증.


def asymptotic_start(
    xbar: float,
    s2: float,
    *,
    nu_min: float | None = None,
    nu_max: float | None = None,
    lam_min: float | None = None,
    lam_max: float | None = None,
) -> tuple[float, float]:
    """Shmueli(2005) 선두차수 역산: (표본평균, 분산) → (λ0, ν0). 세 추정기 공통 warm-start.

    §4.1이 MoM/Newton 시작값을 "λ large에 대한 Shmueli 점근 근사"로 지정.
    대-λ 점근 모멘트 E[X]~λ^(1/ν), Var[X]~λ^(1/ν)/ν 에서
        ν0 ~ mean/var = 1/(V/M),   λ0 ~ mean^ν0.
    이 역산은 함의 모드 λ0^(1/ν0)~mean이 모든 산포영역(과대 ν0<1·등 ν0~1·과소 ν0>1)에서
    계산가능 영역(CMPDivergenceError 참조)에 머물도록 의도적으로 택한 것이다.

    표본 모멘트가 퇴화하면 Poisson-like(λ0=mean, ν0=1)로 폴백. 결과는 허용 박스로 clip;
    박스 경계는 호출 시점에 모듈 상수를 읽으므로 NU_MIN을 monkeypatch하는 스윕이 반영된다.
    """
    nu_min = NU_MIN if nu_min is None else nu_min
    nu_max = NU_MAX if nu_max is None else nu_max
    lam_min = LAM_MIN if lam_min is None else lam_min
    lam_max = LAM_MAX if lam_max is None else lam_max
    if not (xbar > 0.0 and np.isfinite(xbar) and s2 > 0.0 and np.isfinite(s2)):
        return (float(np.clip(max(xbar, lam_min), lam_min, lam_max)), 1.0)
    nu0 = float(np.clip(xbar / s2, nu_min, nu_max))
    lam0 = float(np.clip(xbar ** nu0, lam_min, lam_max))
    return lam0, nu0
