"""COM-Poisson 가중치에 대해 직교정규인 다항식 기저 {ψ_n}의 수치적 구성 (논문 §2.3·§5.1, ledger I9).

  eq(6)       ⟨f,g⟩_CMP = Σ_x f(x)g(x) w_CMP(x)                     (내적)
  eq(10)-(11) 삼항 점화식  x·ψ_n = b_n·ψ_{n+1} + a_n·ψ_n + b_{n-1}·ψ_{n-1}

COM-Poisson는 Askey scheme 밖에 있으므로 (ac-03) 닫힌 형태의 점화식이나
생성함수가 없으며, 따라서 기저를 **수치적으로** 구성해야 한다.

구성 (단일 경로): 점화식 계수 (a_n,b_n)은 절단된 COM-Poisson 측도 위에서의
이산화-Stieltjes/Lanczos 절차로 생성되며 (build_basis→stieltjes_recurrence),
매 단계마다 완전 재직교화를 수행한다.
논문 §2.3은 중간~큰 K에 대해 이 경로를 권장하는데, 교과서적인
원시 모멘트 Hankel/Cholesky 구성 (eq.8-9)은 지수적으로
악조건이기 때문이다 (극단 과대산포 사례의 K=9에서 cond~1e23; 모멘트 문제 자체는
유일하게 결정되지만 수치적 분해는 그렇지 않다 — Gautschi 2004). Stieltjes는
모멘트 행렬을 아예 형성하지 않아 이를 피한다. 기저가 하류 Gram과 *동일한*
이산 측도 위에서 만들어지므로, 극단적 과대산포 하에서도 ⟨ψ_i,ψ_j⟩_CMP=δ_ij가
~1e-13 수준으로 성립한다. 평가는 항상 점화식을 사용하며
(eval_psi/eval_psi_all), eval_psi_monomial은 동일한 점화식에서 유도된
계수에 대한 Horner 교차검증이다.

하이퍼파라미터 (config/hyperparams.yaml): TOL. 하드코딩 금지 — NU_MIN 류의 인공산물을 방지한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as _npp
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from cpoe import compoisson as cmp_

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_hp() -> dict:
    if yaml is None:
        return {}
    cfg = Path(__file__).resolve().parents[2] / "config" / "hyperparams.yaml"
    if not cfg.exists():
        return {}
    with open(cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_HP = _load_hp()
TOL_DEFAULT: float = float(_HP.get("TOL", 1e-12))


@dataclass(frozen=True)
class OrthonormalBasis:
    """(lam, nu)에서 차수 K까지의 직교정규 다항식 기저.

    lam, nu : CMP 기저 측도의 가중치 파라미터.
    K       : 최대 차수 (기저는 ψ_0..ψ_K).
    coeffs  : (K+1, K+1) 하삼각, 양의 대각. coeffs[n,k] = ψ_n(x)의
              x^k 계수. 양의 대각 규약이 ψ_n의 부호를 고정 → (lam, nu) 전반에서
              일관됨.
    a, b    : (K,) 삼항 점화식 계수 a_n, b_n (n=0..K-1). b_n>0
              (직교정규 규약).
    """

    lam: float
    nu: float
    K: int
    coeffs: np.ndarray
    a: np.ndarray
    b: np.ndarray


# ---------- 이산화-Stieltjes 구성 (단일 경로) ----------


def _discrete_measure(
    lam: float, nu: float, tol: float, *, min_points: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """절단된 COM-Poisson 측도의 (노드 x, 정규화 가중치 w), x=0..M (M=support_upper).

    이것은 하류 Gram이 평가되는 바로 그 이산 측도이므로, 기저를 *이* 격자
    위에서 만들면 ⟨ψ_i,ψ_j⟩=δ_ij가 기계 정밀도 수준으로 성립한다 (악조건의
    Hankel 경로에 의존하지 않음).

    min_points: 적응적 절단이 고차 기저가 필요로 하는 자유도보다 짧을 때 노드를
    확장한다. 확장 노드들은 정확한 (아주 작은) CMP
    가중치를 가지므로, 측도를 교란하지 않으면서 자유도를 추가한다. 그래도
    불충분하면 (집중된 측도, 예: Bernoulli 극한 근처) 다음 Stieltjes 단계가
    LinAlgError를 발생 → 추정기가 이를 실행불가능 탐침으로 취급한다.
    """
    c = cmp_._cache(float(lam), float(nu), float(tol))
    n_pts = max(c.M + 1, int(min_points))
    x = np.arange(n_pts, dtype=np.float64)
    if n_pts <= c.M + 1:
        w = np.exp(c.log_terms[:n_pts] - c.log_Z)
    else:
        log_terms_ext = x * math.log(float(lam)) - float(nu) * gammaln(x + 1.0)
        w = np.exp(log_terms_ext - c.log_Z)
    return x, w


def stieltjes_recurrence(
    x: np.ndarray, w: np.ndarray, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """완전 재직교화를 적용한 이산화-Stieltjes (Lanczos) 절차.

    이산 측도 (x,w)에 대해 직교정규인 다항식의 삼항 점화식 계수 (a_n,b_n),
    n=0..K-1을 구성한다:
        x·ψ_n = b_n·ψ_{n+1} + a_n·ψ_n + b_{n-1}·ψ_{n-1}
        a_n = ⟨x·ψ_n, ψ_n⟩_w,   b_n = ‖(x-a_n)ψ_n - b_{n-1}ψ_{n-1}‖_w

    (노드,가중치) 격자 위에서 직접 실행되며 Hankel 행렬을 결코 형성하지 않는다
    (원시 모멘트 Hankel/Cholesky는 극단 과대산포 사례의 K=9에서 cond~1e23). 매 단계의 수정
    Gram-Schmidt 재직교화가 극단적 과대산포 하에서도 ⟨ψ_i,ψ_j⟩_w=δ_ij를
    ~1e-13 수준으로 유지한다.
    """
    a = np.zeros(K, dtype=np.float64)
    b = np.zeros(K, dtype=np.float64)

    def ip(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(w, u * v))

    ones = np.ones_like(x)
    psi = [ones / np.sqrt(ip(ones, ones))]  # ψ_0 정규화: ‖1‖_w = sqrt(Σw)
    a[0] = ip(x * psi[0], psi[0])
    for k in range(K):
        r = (x - a[k]) * psi[k]
        if k > 0:
            r = r - b[k - 1] * psi[k - 1]
        # ψ_0..ψ_k에 대한 완전 재직교화 (반올림 표류를 제거)
        for j in range(k + 1):
            r = r - ip(r, psi[j]) * psi[j]
        nrm = np.sqrt(ip(r, r))
        if not np.isfinite(nrm) or nrm <= 0.0:
            raise np.linalg.LinAlgError(
                f"차수 {k + 1}에서 Stieltjes 붕괴 (norm={nrm}); 이산 측도가 직교다항식을 "
                f"{K + 1}개 미만만 지지함. K를 줄이거나 tol을 낮춰 지지대를 넓혀라."
            )
        b[k] = nrm
        psi.append(r / nrm)
        if k + 1 < K:
            a[k + 1] = ip(x * psi[k + 1], psi[k + 1])
    return a, b


def _coeffs_from_recurrence(a: np.ndarray, b: np.ndarray, K: int) -> np.ndarray:
    """단항식 계수 C[n,k]=[x^k]ψ_n. 평가에 사용되는 *동일한* 점화식에서
    유도된다 (따라서 eval_psi와 eval_psi_monomial이 구성상 일치한다).
    하삼각, 양의 대각.

        ψ_0=1,  ψ_1=(x-a_0)/b_0,  ψ_{k+1}=((x-a_k)ψ_k - b_{k-1}ψ_{k-1})/b_k
    """
    C = np.zeros((K + 1, K + 1), dtype=np.float64)
    C[0, 0] = 1.0
    C[1, 0] = -a[0] / b[0]
    C[1, 1] = 1.0 / b[0]
    for k in range(1, K):
        shifted = np.zeros(K + 1, dtype=np.float64)
        shifted[1:] = C[k, :-1]  # ψ_k에 x를 곱함
        C[k + 1] = (shifted - a[k] * C[k] - b[k - 1] * C[k - 1]) / b[k]
    return C


# ---------- 상위 수준 빌더 ----------


def build_basis(
    lam: float, nu: float, K: int, *, tol: float = TOL_DEFAULT
) -> OrthonormalBasis:
    """(lam, nu)에서 차수 K까지의 직교정규 기저를 구성한다.

    점화식 계수 (a_n,b_n)은 절단된 측도 위의 이산화-Stieltjes로 생성된다
    (§2.3) — 원시 모멘트 Hankel의 Cholesky 분해가 아니다 (극단 과대산포 사례의
    K=9에서 cond~1e23). 단항식 계수는 동일한 점화식에서 유도되므로 두 평가 경로
    (eval_psi/eval_psi_monomial)가 일치한다. 모멘트 행렬은 결코 형성되지 않는다.
    """
    if K < 1:
        raise ValueError(f"K는 >= 1이어야 함, got {K}")
    x, w = _discrete_measure(lam, nu, tol, min_points=K + 1)
    if x.shape[0] < K + 1:
        raise ValueError(
            f"절단 지지대가 {x.shape[0]}점뿐이라 직교다항식 {K + 1}개를 못 만듦; "
            "tol을 낮춰 지지대를 넓혀라."
        )
    a, b = stieltjes_recurrence(x, w, K)
    coeffs = _coeffs_from_recurrence(a, b, K)
    return OrthonormalBasis(
        lam=float(lam),
        nu=float(nu),
        K=int(K),
        coeffs=coeffs,
        a=a,
        b=b,
    )


# ---------- 다항식 평가 (점화식 + 단항식 교차검증) ----------


def eval_psi(
    x: int | float | np.ndarray, n: int, basis: OrthonormalBasis
):
    """삼항 점화식을 통해 ψ_n(x)를 평가한다 (수치적으로 안정적).

      ψ_0=1,  ψ_1=(x-a_0)/b_0,  ψ_{k+1}=((x-a_k)ψ_k - b_{k-1}ψ_{k-1})/b_k
    """
    if not 0 <= n <= basis.K:
        raise ValueError(f"n은 0 <= n <= K={basis.K}를 만족해야 함, got n={n}")
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    psi_prev = np.zeros_like(x_arr)
    psi_curr = np.ones_like(x_arr)  # ψ_0 = 1
    if n == 0:
        out = psi_curr
    else:
        for k in range(n):
            b_minus = basis.b[k - 1] if k > 0 else 0.0
            psi_next = (
                (x_arr - basis.a[k]) * psi_curr - b_minus * psi_prev
            ) / basis.b[k]
            psi_prev = psi_curr
            psi_curr = psi_next
        out = psi_curr
    return float(out[0]) if np.isscalar(x) else out


def eval_psi_all(
    x: int | float | np.ndarray, basis: OrthonormalBasis
) -> np.ndarray:
    """ψ_0(x)..ψ_K(x)를 한 번에 일괄 평가한다. (K+1,) 또는 (K+1, len(x))를 반환.

    x에 대해 벡터화하고 차수 k에 대해 반복한다. 절단 지지대 {0..M} 위에서의
    하류 일괄 평가의 핵심 일꾼.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    K = basis.K
    out = np.empty((K + 1, x_arr.shape[0]), dtype=np.float64)
    out[0] = 1.0
    if K >= 1:
        out[1] = (x_arr - basis.a[0]) / basis.b[0]
    for k in range(1, K):
        out[k + 1] = (
            (x_arr - basis.a[k]) * out[k] - basis.b[k - 1] * out[k - 1]
        ) / basis.b[k]
    if np.isscalar(x):
        return out[:, 0]
    return out


def eval_psi_monomial(
    x: int | float | np.ndarray, n: int, basis: OrthonormalBasis
):
    """저장된 단항식 계수로부터 ψ_n(x)의 Horner 평가.

    높은 n과 큰 |x|에서 계수의 동적 범위가 넓을 때, numpy polyval (Horner)은
    순진한 Σ_k coeffs[n,k]·x^k보다 훨씬 안정적이다. 점화식 (eval_psi)에 대한
    교차검증 경로
    (tests/test_basis_orthogonality.py::test_recurrence_eval_matches_monomial_coeffs).
    """
    if not 0 <= n <= basis.K:
        raise ValueError(f"n은 0 <= n <= K={basis.K}를 만족해야 함, got n={n}")
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = _npp.polyval(x_arr, basis.coeffs[n])
    return float(out[0]) if np.isscalar(x) else out


# ---------- 교과서적 Hankel/Cholesky 구성 (논문 eq.8-9; 비교/참조 경로) ----------
#
# 이 경로는 모멘트 행렬을 *명시적으로* 형성하고 Cholesky 분해한다.
# build_basis (Stieltjes)는 행렬을 결코 형성하지 않아 악조건을 피하므로,
# 둘을 비교하면 "수치적으로 구성한다"는 것이 무엇을 의미하는지와 고차에서
# 방법 선택이 왜 중요한지를 드러낸다 (task 1; Gautschi 2004).
#
# 용어: 아래의 H는 우리 논문의 "Gram 행렬" (eq.8)과 동일한 대상이다 —
# 단항식 기저 {1,x,…,x^K}의 ⟨·,·⟩_CMP Gram 행렬로, 그 성분 m_{i+j}는
# i+j에만 의존하므로 Hankel 구조를 갖는다. 수치 직교다항식 문헌의 표준
# 명명을 따라 이를 "Hankel 행렬"이라 부른다 (Gautschi 2004).


@dataclass(frozen=True)
class CholeskyBasis:
    """Hankel/Cholesky 경로 (eq.8-9)로부터의 직교정규 기저 + 수치 진단.

    lam, nu, K : CMP 파라미터와 최대 차수.
    coeffs     : (K+1, K+1) 하삼각. coeffs[n,k] = ψ_n(x) = (L^{-1})_{nk}의
                 x^k 계수. 분해가 실패하면 (양정치 아님) None.
    hankel     : (K+1, K+1) Hankel/Gram 행렬 H = [m_{i+j}].
    orth_err   : max_{i,j} |⟨ψ_i, ψ_j⟩_CMP − δ_ij|. Cholesky 경로의 직교성
                 손실을 측정.
    ok         : Cholesky 분해가 성공했는지 여부.
    """

    lam: float
    nu: float
    K: int
    coeffs: np.ndarray | None
    hankel: np.ndarray
    orth_err: float
    ok: bool


def hankel_moment_matrix(
    lam: float, nu: float, K: int, *, tol: float = TOL_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """절단된 COM-Poisson 측도의 원시 모멘트로부터 Hankel/Gram 행렬 H=[m_{i+j}] (eq.8)을 만든다.

    build_basis (Stieltjes)와 *동일한* 절단 측도 (노드 x, 가중치 w) 위에서
    m_k=Σ_x x^k w(x), k=0..2K를 계산하므로, 두 경로를 비교하면 측도가 아닌
    *수치적 구성* 만의 차이를 분리해낸다.

    반환: (H (K+1,K+1), m (2K+1,)).
    참고: 고차 모멘트 m_{2K}는 꼬리에 민감하다 — 바로 이 민감성이 모멘트→다항식
    사상이 악조건인 이유이며 (Gautschi 2004), task 1이 보이고자 하는 바이다.
    """
    if K < 1:
        raise ValueError(f"K는 >= 1이어야 함, got {K}")
    x, w = _discrete_measure(lam, nu, tol, min_points=2 * K + 1)
    powers = np.vander(x, N=2 * K + 1, increasing=True)  # (n_pts, 2K+1): x^0..x^{2K}
    m = w @ powers                                        # 원시 모멘트 m_0..m_{2K}
    H = np.empty((K + 1, K + 1), dtype=np.float64)
    for i in range(K + 1):
        H[i] = m[i:i + K + 1]
    return H, m


def _orthonormality_error(
    coeffs: np.ndarray, lam: float, nu: float, *, tol: float = TOL_DEFAULT
) -> float:
    """절단 지지대 x=0..M 위에서의 max_{i,j}|⟨ψ_i,ψ_j⟩_CMP − δ_ij| (직교성 손실)."""
    c = cmp_._cache(float(lam), float(nu), float(tol))
    x = np.arange(c.M + 1, dtype=np.float64)
    w = np.exp(c.log_terms - c.log_Z)
    K1 = coeffs.shape[0]
    P = np.empty((K1, x.shape[0]), dtype=np.float64)
    for n in range(K1):
        P[n] = _npp.polyval(x, coeffs[n])
    G = (P * w) @ P.T                                    # G[i,j] = ⟨ψ_i, ψ_j⟩_CMP
    return float(np.max(np.abs(G - np.eye(K1))))


def gram_schmidt_cholesky(
    lam: float, nu: float, K: int, *, tol: float = TOL_DEFAULT
) -> CholeskyBasis:
    """논문 eq.(8)-(9): Hankel/Gram 행렬 H의 Cholesky 분해로부터의 직교정규 다항식 계수.

        H = L Lᵀ (하삼각, 양의 대각),   ψ_n(x) = Σ_k (L⁻¹)_{nk} x^k,   coeffs[n,k]=(L⁻¹)_{nk}.

    A=L⁻¹로 두면 A H Aᵀ = I이므로 {ψ_n}은 직교정규이다. A는 하삼각이므로
    deg(ψ_n)=n, ψ_0=1/√m_0=1. 고차에서 cond(H)의 폭발은
    (a) 분해가 양정치 아님으로 실패하거나, (b) 성공하되 큰 직교성 오차를
    동반하게 만든다 — build_basis (Stieltjes)는 H를 결코 형성하지 않아 이를
    피한다.
    """
    H, _m = hankel_moment_matrix(lam, nu, K, tol=tol)
    try:
        L = np.linalg.cholesky(H)                        # H = L Lᵀ, 하삼각
        coeffs = solve_triangular(L, np.eye(K + 1), lower=True)  # L⁻¹, 하삼각
        orth_err = _orthonormality_error(coeffs, lam, nu, tol=tol)
        ok = True
    except np.linalg.LinAlgError:
        coeffs = None
        orth_err = float("inf")
        ok = False
    return CholeskyBasis(
        lam=float(lam), nu=float(nu), K=int(K),
        coeffs=coeffs, hankel=H, orth_err=orth_err, ok=ok,
    )


def orthonormality_error(basis: OrthonormalBasis, *, tol: float = TOL_DEFAULT) -> float:
    """build_basis (TTR) 결과의 직교성 오차 — Cholesky 경로와 동일한 잣대로 비교."""
    return _orthonormality_error(basis.coeffs, basis.lam, basis.nu, tol=tol)

