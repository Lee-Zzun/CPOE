"""CPOE 평가용 시뮬레이션 시나리오 — S1~S4 (모두 혼합비 5:5의 유한혼합).

혼합 생성(Cross et al. 2024): 성분 지시자 z_i~Bernoulli(0.5) 추출 → 성분별 표본 →
정수 shift(S4의 +9) → 결합·셔플. CMP 파라미터화는 Shmueli et al.(2005)(λ,ν).

  S1 (과소산포, V/M≈0.75): 0.5·Poisson(λ=5) + 0.5·Binomial(n=10, p=0.5)
  S2 (과대산포, V/M≈2.1):  0.5·Poisson(λ=4) + 0.5·NB(r=4, p=0.4)
  S3 (거의 단봉+올라간 꼬리): 0.5·Poisson(λ=2) + 0.5·Poisson(λ=8)
  S4 (강한 과대산포):       0.5·NB(r=2, p=0.4) + 0.5·(CMP(λ=3, ν=0.5)+9 shift)

NB는 numpy/scipy 모수화 negative_binomial(r, p): 실패 횟수, 평균 r(1-p)/p, 분산 r(1-p)/p².

`true_pmf(name, grid)`는 혼합의 **정확한** 모집단 pmf를 반환한다(oracle-K 검증용).
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cpoe import compoisson as _cmp  # noqa: E402
from cpoe.sampling import sample_cmp  # noqa: E402
@dataclass
class Scenario:
    """시뮬레이션 시나리오: 이름·목표 V/M·설명·샘플러·(있으면) 참 θ."""
    name: str
    paper_target_vm: Optional[float]
    description: str
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    true_theta: Optional[np.ndarray] = None


# ---- 혼합 성분 파라미터 (단일 출처: 샘플러·true_pmf 공유) ----
_S1 = dict(pois_lam=5.0, binom_n=10, binom_p=0.5)
_S2 = dict(pois_lam=4.0, nb_r=4, nb_p=0.4)
_S3 = dict(pois_lam_a=2.0, pois_lam_b=8.0)
_S4 = dict(nb_r=2, nb_p=0.4, cmp_lam=3.0, cmp_nu=0.5, shift=9)


def _mix_sampler(sample_a, sample_b):
    """5:5 혼합 샘플러 팩토리. sample_a/b(n, rng)→정수배열."""
    def sampler(N: int, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(N) < 0.5
        n_a = int(mask.sum())
        n_b = N - n_a
        a = sample_a(n_a, rng) if n_a > 0 else np.zeros(0, dtype=int)
        b = sample_b(n_b, rng) if n_b > 0 else np.zeros(0, dtype=int)
        out = np.concatenate([a, b]).astype(int)
        rng.shuffle(out)
        return out
    return sampler


def _s1():
    s = _S1
    sampler = _mix_sampler(
        lambda n, rng: rng.poisson(s["pois_lam"], size=n),
        lambda n, rng: rng.binomial(s["binom_n"], s["binom_p"], size=n),
    )
    return Scenario(name="S1", paper_target_vm=0.75,
                    description="0.5*Poisson(5) + 0.5*Binomial(10,0.5); underdispersion V/M~0.75.",
                    sampler=sampler, true_theta=None)


def _s2():
    s = _S2
    sampler = _mix_sampler(
        lambda n, rng: rng.poisson(s["pois_lam"], size=n),
        lambda n, rng: rng.negative_binomial(s["nb_r"], s["nb_p"], size=n),
    )
    return Scenario(name="S2", paper_target_vm=2.1,
                    description="0.5*Poisson(4) + 0.5*NB(r=4,p=0.4); overdispersion V/M~2.1.",
                    sampler=sampler, true_theta=None)


def _s3():
    s = _S3
    sampler = _mix_sampler(
        lambda n, rng: rng.poisson(s["pois_lam_a"], size=n),
        lambda n, rng: rng.poisson(s["pois_lam_b"], size=n),
    )
    return Scenario(name="S3", paper_target_vm=2.8,
                    description="0.5*Poisson(2) + 0.5*Poisson(8); near-unimodal (very weak 2nd mode) + raised tail (V/M~2.8).",
                    sampler=sampler, true_theta=None)


def _s4():
    s = _S4
    sampler = _mix_sampler(
        lambda n, rng: rng.negative_binomial(s["nb_r"], s["nb_p"], size=n),
        lambda n, rng: sample_cmp(s["cmp_lam"], s["cmp_nu"], n, rng) + s["shift"],
    )
    return Scenario(name="S4", paper_target_vm=None,
                    description="0.5*NB(r=2,p=0.4) + 0.5*(CMP(3,0.5)+9); strong overdispersion.",
                    sampler=sampler, true_theta=None)


HIGHK_SCENARIOS: Dict[str, Scenario] = {
    "S1": _s1(), "S2": _s2(), "S3": _s3(), "S4": _s4(),
}


def true_pmf(name: str, grid: np.ndarray) -> np.ndarray:
    """시나리오 혼합의 정확한 모집단 pmf를 grid(0..max) 위에서 반환(oracle-K 검증용)."""
    grid = np.asarray(grid, dtype=int)
    if name == "S1":
        s = _S1
        p = 0.5 * stats.poisson.pmf(grid, s["pois_lam"]) \
            + 0.5 * stats.binom.pmf(grid, s["binom_n"], s["binom_p"])
    elif name == "S2":
        s = _S2
        p = 0.5 * stats.poisson.pmf(grid, s["pois_lam"]) \
            + 0.5 * stats.nbinom.pmf(grid, s["nb_r"], s["nb_p"])
    elif name == "S3":
        s = _S3
        p = 0.5 * stats.poisson.pmf(grid, s["pois_lam_a"]) \
            + 0.5 * stats.poisson.pmf(grid, s["pois_lam_b"])
    elif name == "S4":
        s = _S4
        nb = stats.nbinom.pmf(grid, s["nb_r"], s["nb_p"])
        xs = grid - s["shift"]
        cmp_part = np.zeros_like(grid, dtype=float)
        ok = xs >= 0
        cmp_part[ok] = np.asarray(_cmp.pmf(xs[ok], s["cmp_lam"], s["cmp_nu"]), dtype=float)
        p = 0.5 * nb + 0.5 * cmp_part
    else:
        raise ValueError(f"unknown scenario {name!r}")
    p = np.asarray(p, dtype=float)
    tot = p.sum()
    return p / tot if tot > 0 else p
