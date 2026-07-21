"""고차 K 후속 평가 — 차수별 in-sample L1 적합의 코어 (최종보고서 파이프라인).

데이터·추정법·차수별 단일 적합(`_fit_one`)과 그 pmf(`_pmf`)·in-sample L1 곡선
(`insample_l1_curve`)과 지표 `_l1`·CPOE pmf `_cpoe_fitted_pmf`를 제공한다.

K=0,1,2는 틸트(θ는 n≥3부터) 부재 → CMP 베이스라인과 동일(평탄). K≥3부터 CPOE.
모든 적합은 try/except로 감싸 실패(LinAlgError/infeasible) 시 reliable=False.
"""
from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List

import numpy as np


@contextmanager
def _quiet():
    """고차 K에서 trust-constr/CMP의 overflow·수렴 경고 억제(불안정 적합은 별도 게이트로 처리)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cpoe import compoisson as cmp_  # noqa: E402
from cpoe.diagnostics import empirical_pmf  # noqa: E402
from cpoe.cpoe_model import cpoe_pmf  # noqa: E402
from cpoe.backends import CMP_BACKEND, BASIS_BACKEND  # noqa: E402
from cpoe.estimators import mom as mom_mod  # noqa: E402
from cpoe.estimators import mle as mle_mod  # noqa: E402
from cpoe.estimators.mle_sequential import fit_sequential_mle  # noqa: E402
from cpoe.estimators.cmp_baseline import cmp_mle_fit  # noqa: E402

ESTIMATORS = ("MoM", "Sequential MLE", "Joint MLE")


def _cpoe_fitted_pmf(lam: float, nu: float, theta: np.ndarray, max_x: int) -> np.ndarray:
    """CPOE 적합 pmf(eq.12), 음수 clip. K=0이면 CMP-only."""
    theta = np.asarray(theta, dtype=float)
    K = int(theta.shape[0])
    grid = np.arange(max_x + 1, dtype=np.float64)
    if K == 0:
        return np.asarray(cmp_.pmf(grid, lam, nu), dtype=float)
    p = cpoe_pmf(theta, lam, nu, cmp_mod=CMP_BACKEND, basis=BASIS_BACKEND, M=max_x + 1)
    return np.clip(p, 0.0, None)


def _l1(emp: np.ndarray, p: np.ndarray) -> float:
    """동일 grid·동일 정규화(합=1)로 L1 = Σ|emp − p|."""
    p = np.asarray(p, dtype=float)
    s = p.sum()
    pn = p / s if s > 0 else p
    return float(np.sum(np.abs(np.asarray(emp, dtype=float) - pn)))


# ----------------------------------------------------------------------------
# 단일 적합 (추정법×K) → (lam, nu, theta, in_CK)
# ----------------------------------------------------------------------------
def _fit_one(sample: np.ndarray, estimator: str, K: int):
    """추정법 1회 적합. K<3은 틸트 부재라 CMP 베이스라인(theta=None).

    MoM in_CK = raw θ̂의 feasibility(선행 보고서 관행), Seq/Joint는 경성제약 θ.
    """
    s = np.asarray(sample, dtype=np.float64)
    with _quiet():
        if K >= 3:
            if estimator == "MoM":
                r = mom_mod.fit_mom(s, K=K, project=True)
                return float(r.lam_hat), float(r.nu_hat), np.asarray(r.theta_hat_projected, float), bool(r.feasibility_raw.feasible)
            if estimator == "Sequential MLE":
                r = fit_sequential_mle(s, K=K)
                if not bool(getattr(r, "converged", True)):
                    raise RuntimeError(f"Sequential MLE not converged at K={K}")
                return float(r.lam_hat), float(r.nu_hat), np.asarray(r.theta_hat, float), bool(r.feasibility.feasible)
            if estimator == "Joint MLE":
                r = mle_mod.fit_mle(s, K=K, method="nested")
                if not bool(getattr(r, "converged", True)):
                    raise RuntimeError(f"Joint MLE not converged at K={K}")
                return float(r.lam_hat), float(r.nu_hat), np.asarray(r.theta_hat, float), bool(r.feasibility.feasible)
            raise ValueError(f"unknown estimator {estimator!r}")
        # K<3: 베이스라인 (λ,ν)만. MoM=적률매칭, MLE계열=CMP-MLE.
        if estimator == "MoM":
            r = mom_mod.fit_mom(s, K=3, project=True)
            return float(r.lam_hat), float(r.nu_hat), None, True
        c = cmp_mle_fit(s)
        return float(c["lam_hat"]), float(c["nu_hat"]), None, True


def _pmf(lam: float, nu: float, theta, max_x: int) -> np.ndarray:
    """적합 pmf(grid 0..max_x). theta=None/빈 → CMP 베이스라인."""
    if theta is None or np.asarray(theta).shape[0] == 0:
        return _cpoe_fitted_pmf(lam, nu, np.zeros(0), max_x)
    return _cpoe_fitted_pmf(lam, nu, np.asarray(theta, float), max_x)


def _k_list(k_max: int) -> List[int]:
    return [0, 1, 2] + list(range(3, k_max + 1))


# ----------------------------------------------------------------------------
# in-sample L1 곡선 (진단·그래프·표용)
# ----------------------------------------------------------------------------
def insample_l1_curve(sample: np.ndarray, estimator: str, k_max: int = 14) -> List[dict]:
    """[{K, L1, theta(len K or None), in_CK, reliable}] for K=0..k_max."""
    s = np.asarray(sample, dtype=int)
    max_x = int(s.max())
    emp = empirical_pmf(s, max_x + 1)
    rows: List[dict] = []
    base = None
    try:
        base = _fit_one(s, estimator, 0)  # 베이스라인 (λ,ν) 1회
    except Exception as e:  # noqa: BLE001
        base = ("ERR", e)
    for K in _k_list(k_max):
        try:
            if K < 3:
                if base[0] == "ERR":
                    raise base[1]
                lam, nu, theta, inck = base
            else:
                lam, nu, theta, inck = _fit_one(s, estimator, K)
            p = _pmf(lam, nu, theta, max_x)
            l1 = _l1(emp, p)
            tvec = np.zeros(K) if theta is None else np.asarray(theta, float)
            rows.append(dict(K=K, L1=l1, theta=tvec, in_CK=bool(inck), reliable=True))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(K=K, L1=float("nan"), theta=None, in_CK=False,
                             reliable=False, err=f"{type(e).__name__}: {e}"))
    return rows
