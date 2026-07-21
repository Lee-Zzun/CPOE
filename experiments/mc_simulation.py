"""§6.2 Monte Carlo 시뮬레이션 드라이버.

설계: S1–S4 × N∈{200,500,1500} × K grid × {MoM, Sequential}, R=500.
재현: 시드맵 rng=default_rng([scen_id, N, r]).
각 복제: 참 pmf 대비 L1/L2 위험(tab:mc-risk·fig:mc-risk-profiles의 데이터).
MoM 위험 곡선은 Supplementary 후보로 함께 기록한다.

사용:
  python mc_simulation.py --R 500 --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cpoe.backends import CMP_BACKEND, BASIS_BACKEND, default_M          # noqa: E402
from cpoe.cpoe_model import cpoe_pmf                                     # noqa: E402
from cpoe.estimators import mom as mom_mod                              # noqa: E402
from cpoe.estimators.cmp_baseline import cmp_mle_fit                    # noqa: E402
from cpoe.estimators.mle import solve_theta_at_baseline, _pad_theta     # noqa: E402
from cpoe import compoisson as _cmp                                     # noqa: E402
from highk_scenarios import HIGHK_SCENARIOS, true_pmf                   # noqa: E402

SCEN_ID = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
K_GRID = [0, 4, 6, 8, 10, 12, 14]
TILT_GRID = [4, 6, 8, 10, 12, 14]
N_LIST = [200, 500, 1500]
OUT = ROOT / "experiments" / "mc_out"


def _true_pmf_grid(scen_name):
    """참 pmf를 누적질량 ≥ 1-1e-10 까지 절단한 grid(0..M*)에서 반환."""
    g = np.arange(400)
    p = true_pmf(scen_name, g)
    c = np.cumsum(p)
    Mstar = int(np.searchsorted(c, 1 - 1e-10)) + 1
    Mstar = max(Mstar, 30)
    grid = np.arange(Mstar)
    return grid, true_pmf(scen_name, grid)


def _fitted_pmf(lam, nu, theta, Gmax):
    """모델 pmf(eq.12) 0..Gmax-1, 음수 clip·재정규화(L1/L2 지표용)."""
    if theta is None or np.asarray(theta).shape[0] == 0:
        p = np.asarray(_cmp.pmf(np.arange(Gmax, dtype=float), lam, nu), float)
    else:
        p = cpoe_pmf(np.asarray(theta, float), lam, nu, cmp_mod=CMP_BACKEND,
                     basis=BASIS_BACKEND, M=Gmax)
        p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s > 0 else p


def _dist(p_true, p_hat):
    n = max(len(p_true), len(p_hat))
    a = np.zeros(n); a[:len(p_true)] = p_true
    b = np.zeros(n); b[:len(p_hat)] = p_hat
    d = a - b
    return float(np.abs(d).sum()), float(np.sqrt((d * d).sum()))


def one_rep(task):
    """단일 복제 (scen, N, r) 처리 → 기록 dict."""
    scen_name, N, r = task
    warnings.simplefilter("ignore")
    with np.errstate(all="ignore"):
        scen = HIGHK_SCENARIOS[scen_name]
        rng = np.random.default_rng([SCEN_ID[scen_name], N, r])
        sample = scen.sampler(N, rng).astype(int)
        M = default_M(sample)
        grid_t, p_true = _true_pmf_grid(scen_name)
        Gmax = len(grid_t)
        rec = {"scen": scen_name, "N": N, "r": r}

        # MoM risk at grid
        for K in K_GRID:
            try:
                rm = mom_mod.fit_mom(sample, K=K, project=True)
                pm = _fitted_pmf(rm.lam_hat, rm.nu_hat,
                                 rm.theta_hat_projected if K >= 3 else None, Gmax)
                l1, l2 = _dist(p_true, pm)
                rec[f"l1_mom_{K}"] = l1; rec[f"l2_mom_{K}"] = l2
            except Exception:
                rec[f"l1_mom_{K}"] = np.nan; rec[f"l2_mom_{K}"] = np.nan

        # Sequential risk at grid
        try:
            c = cmp_mle_fit(sample); lam_s = c["lam_hat"]; nu_s = c["nu_hat"]
        except Exception:
            return rec
        try:
            p0 = _fitted_pmf(lam_s, nu_s, None, Gmax)
            rec["l1_seq_0"], rec["l2_seq_0"] = _dist(p_true, p0)
        except Exception:
            rec["l1_seq_0"] = np.nan; rec["l2_seq_0"] = np.nan
        for K in TILT_GRID:
            try:
                th, ok = solve_theta_at_baseline(sample, lam_s, nu_s, K,
                                                 basis=BASIS_BACKEND, M=M)
                ps = _fitted_pmf(lam_s, nu_s, _pad_theta(th, K), Gmax)
                rec[f"l1_seq_{K}"], rec[f"l2_seq_{K}"] = _dist(p_true, ps)
            except Exception:
                rec[f"l1_seq_{K}"] = np.nan; rec[f"l2_seq_{K}"] = np.nan
        return rec


def run(R, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    OUT.mkdir(parents=True, exist_ok=True)
    scen_names = list(SCEN_ID)
    tasks = [(s, N, r) for s in scen_names for N in N_LIST for r in range(R)]
    print(f"[main] {len(tasks)} tasks, R={R}, workers={workers}", flush=True)
    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_rep, t): t for t in tasks}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el:.0f}s  ({el/done*1000:.0f} ms/task, "
                      f"eta {el/done*(len(tasks)-done):.0f}s)", flush=True)
                # 증분 저장
                np.save(OUT / "main_partial.npy", np.array(results, dtype=object), allow_pickle=True)
    np.save(OUT / f"main_R{R}.npy", np.array(results, dtype=object), allow_pickle=True)
    print(f"[main] done in {time.time()-t0:.0f}s -> {OUT / f'main_R{R}.npy'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.R, a.workers)
