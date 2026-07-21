"""cpoe.compoisson / cpoe.basis를 추정기가 쓰는 duck-typed (lam, nu, M=) 프로토콜로
잇는 공유 백엔드 어댑터 (프로토콜은 cpoe.cpoe_model docstring 참조).

이전엔 실험 스크립트마다 _CMPBackend/_BasisBackend를 사적으로 재정의(중복)했는데 여기로
단일화. 추정기 편의 래퍼(estimators.{mom,mle_sequential,mle}.fit_*)가 기본값으로 이 어댑터를 쓰므로,
데이터 배열만 있는 호출자는 백엔드를 직접 만들 필요가 없다.

선택적 프로토콜 메서드(소비자가 hasattr로 탐지):
    CMPBackend.asymptotic_start(xbar, s2) -> (lam0, nu0)   [MoM/MLE warm-start]
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

from cpoe import basis as _basis_mod
from cpoe import compoisson as _cmp


class CMPBackend:
    """COM-Poisson 베이스 측도 어댑터 (eq.1 pmf/log_pmf, §4.1 warm-start)."""

    @staticmethod
    def pmf(x, lam, nu, *, M=None):
        x_arr = np.asarray(x, dtype=int)
        return _cmp.pmf(x_arr, float(lam), float(nu))

    @staticmethod
    def log_pmf(x, lam, nu, *, M=None):
        x_arr = np.asarray(x, dtype=int)
        log_Z_val = _cmp.log_Z(float(lam), float(nu))
        return x_arr * math.log(float(lam)) - float(nu) * gammaln(x_arr + 1) - log_Z_val

    @staticmethod
    def asymptotic_start(xbar, s2):
        return _cmp.asymptotic_start(float(xbar), float(s2))


class BasisBackend:
    """직교기저 어댑터: psi_matrix(lam, nu, K, M) -> (M, K).

    열은 절단 지지대 x=0..M-1 위 ψ_1..ψ_K (상수 ψ_0=1은 Prop.1에 따라 제외).
    """

    @staticmethod
    def psi_matrix(lam, nu, K, M):
        b = _basis_mod.build_basis(float(lam), float(nu), int(K))
        x = np.arange(int(M))
        psi_all = _basis_mod.eval_psi_all(x, b)        # (K+1, M)
        return np.asarray(psi_all[1:].T, dtype=float)  # (M, K), drop psi_0


# 편의 import용 모듈 싱글턴.
CMP_BACKEND = CMPBackend()
BASIS_BACKEND = BasisBackend()


def default_M(sample, *, tol: float = 1e-12) -> int:
    """fit_* 편의 래퍼가 쓰는 절단 지지대 크기.

    M = max(support_upper(lam_0, nu_0, tol), max(data)+1), fit 시작 시 asymptotic_start로
    1회 산정. 옵티마이저 루프 내내 고정(루프 안에서 재산정하면 격자/clip/기저 모양이
    스텝마다 바뀌어 목적함수 불연속 → fsolve/L-BFGS-B 붕괴).

    max(data)+1 하한은 관측 카운트가 clip되지 않게 함(꼬리 휴리스틱 아님);
    support_upper 항은 tol=1e-12 잔여질량 절단 기준(compoisson.support_upper).
    """
    sample = np.asarray(sample)
    xbar = float(sample.mean())
    s2 = float(sample.var(ddof=1))
    lam0, nu0 = _cmp.asymptotic_start(xbar, s2)
    try:
        M_theory = int(_cmp.support_upper(lam0, nu0, tol=tol))
    except Exception:  # noqa: BLE001 - any divergence in the asymptotic start corner
        M_theory = 0
    M_data = int(sample.max()) + 1
    return max(M_theory, M_data)
