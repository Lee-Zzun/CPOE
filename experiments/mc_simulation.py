"""§6.2 Monte Carlo 시뮬레이션 드라이버.

설계: S1–S4 × N∈{200,500,1500} × K grid × {MoM, Sequential}, R=500 (main),
      Joint 축소설계 (joint 모드: R, N=1500, K∈{4,10,14}).
재현: 시드맵 rng=default_rng([scen_id, N, r]).
각 복제(main): 참 pmf 대비 L1/L2 위험(tab:mc-risk·fig:mc-risk-profiles의 데이터),
  raw-MoM feasibility, K̂_RISK/K̂_BIC/oracle K*·regret, (S2,K=4) sandwich vs
  fixed-basis SE 검증. MoM 위험 곡선은 Supplementary 후보로 함께 기록한다.
[2026-07 복원] 선택규칙(K̂_BIC/oracle/regret)·SE-check·momfeas·joint 모드는 논문
  축소 시 삭제되었다가 리뷰 대응용 검증 도구로 복원됨(논문 본문 표와는 무관).

사용:
  python mc_simulation.py --R 500 --workers 6            # main (MoM+Sequential)
  python mc_simulation.py joint --R 100 --workers 6      # Joint 축소설계
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
from cpoe.estimators import mle as mle_mod                              # noqa: E402
from cpoe.estimators.cmp_baseline import cmp_mle_fit                    # noqa: E402
from cpoe.estimators.mle import (                                       # noqa: E402
    solve_theta_at_baseline, _neg_log_lik_smoothed, _LOG_CLIP, _pad_theta,
)
from cpoe.estimators import mom_sandwich as sw_mod                      # noqa: E402
from cpoe import order_selection as osel                                # noqa: E402
from cpoe import compoisson as _cmp                                     # noqa: E402
from highk_scenarios import HIGHK_SCENARIOS, true_pmf                   # noqa: E402

SCEN_ID = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
K_GRID = [0, 4, 6, 8, 10, 12, 14]
TILT_GRID = [4, 6, 8, 10, 12, 14]
K_MAX = 14
K_ALL = list(range(3, K_MAX + 1))         # selection/oracle 스캔 (전 정수차수)
N_LIST = [200, 500, 1500]
JOINT_ORDERS = [4, 10, 14]
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
    """단일 복제 (scen, N, r) 처리 → 기록 dict. mode='main' 또는 'joint'."""
    scen_name, N, r, mode = task
    warnings.simplefilter("ignore")
    with np.errstate(all="ignore"):
        scen = HIGHK_SCENARIOS[scen_name]
        rng = np.random.default_rng([SCEN_ID[scen_name], N, r])
        sample = scen.sampler(N, rng).astype(int)
        M = default_M(sample)
        grid_t, p_true = _true_pmf_grid(scen_name)
        Gmax = len(grid_t)
        rec = {"scen": scen_name, "N": N, "r": r}

        if mode == "joint":
            for K in JOINT_ORDERS:
                try:
                    rj = mle_mod.fit_mle(sample, K=K, method="nested")
                    if not bool(getattr(rj, "converged", True)):
                        raise RuntimeError("joint not converged")
                    pj = _fitted_pmf(rj.lam_hat, rj.nu_hat, rj.theta_hat, Gmax)
                    l1j, _ = _dist(p_true, pj)
                except Exception:
                    l1j = np.nan
                # Sequential at same K for the gap
                try:
                    c = cmp_mle_fit(sample); lam = c["lam_hat"]; nu = c["nu_hat"]
                    th, ok = solve_theta_at_baseline(sample, lam, nu, K, basis=BASIS_BACKEND, M=M)
                    ps = _fitted_pmf(lam, nu, _pad_theta(th, K), Gmax)
                    l1s, _ = _dist(p_true, ps)
                except Exception:
                    l1s = np.nan
                rec[f"l1_joint_{K}"] = l1j
                rec[f"l1_seq_{K}"] = l1s
            return rec

        # ---- mode == 'main': MoM + Sequential ----
        # MoM θ̂,σ̂ (order selection RISK)
        try:
            r0 = mom_mod.fit_mom(sample, K=K_MAX, project=True)
            lam_m, nu_m = float(r0.lam_hat), float(r0.nu_hat)
            theta_hat, sigma2 = osel.momproj_stats(sample, lam_m, nu_m, K_MAX,
                                                    basis=BASIS_BACKEND, M=M)
            rec["khat_risk"] = osel.select_k_risk(theta_hat, sigma2, N, K_MAX)
        except Exception:
            rec["khat_risk"] = -1

        # MoM risk at grid + feasibility of raw projection
        for K in K_GRID:
            try:
                rm = mom_mod.fit_mom(sample, K=K, project=True)
                pm = _fitted_pmf(rm.lam_hat, rm.nu_hat,
                                 rm.theta_hat_projected if K >= 3 else None, Gmax)
                l1, l2 = _dist(p_true, pm)
                rec[f"l1_mom_{K}"] = l1; rec[f"l2_mom_{K}"] = l2
                if K in JOINT_ORDERS:
                    rec[f"momfeas_{K}"] = bool(rm.feasibility_raw.feasible) if K >= 3 else True
            except Exception:
                rec[f"l1_mom_{K}"] = np.nan; rec[f"l2_mom_{K}"] = np.nan
                if K in JOINT_ORDERS:
                    rec[f"momfeas_{K}"] = False

        # Sequential at all integer orders (risk grid + oracle/BIC scan)
        loglik = {}
        l1_seq_all = {}
        try:
            c = cmp_mle_fit(sample); lam_s = c["lam_hat"]; nu_s = c["nu_hat"]
        except Exception:
            rec["khat_bic"] = -1; rec["oracle_k"] = -1
            rec["regret_risk"] = np.nan; rec["regret_bic"] = np.nan
            return rec
        try:
            # baseline K=0
            p0 = _fitted_pmf(lam_s, nu_s, None, Gmax)
            l1_0, l2_0 = _dist(p_true, p0)
            rec["l1_seq_0"] = l1_0; rec["l2_seq_0"] = l2_0
            l1_seq_all[0] = l1_0
            ll0 = -_neg_log_lik_smoothed(sample, np.array([lam_s, nu_s]), 0,
                                         cmp_mod=CMP_BACKEND, basis=BASIS_BACKEND, M=M, log_clip=_LOG_CLIP)
            loglik[0] = float(ll0)
            for K in K_ALL:
                th, ok = solve_theta_at_baseline(sample, lam_s, nu_s, K, basis=BASIS_BACKEND, M=M)
                eta = np.concatenate([[lam_s, nu_s], _pad_theta(th, K)])
                ll = -_neg_log_lik_smoothed(sample, eta, K, cmp_mod=CMP_BACKEND,
                                            basis=BASIS_BACKEND, M=M, log_clip=_LOG_CLIP)
                ps = _fitted_pmf(lam_s, nu_s, _pad_theta(th, K), Gmax)
                l1, l2 = _dist(p_true, ps)
                loglik[K] = float(ll)
                l1_seq_all[K] = l1
                if K in K_GRID:
                    rec[f"l1_seq_{K}"] = l1; rec[f"l2_seq_{K}"] = l2
            # selection: BIC, oracle, regret (Sequential)
            rec["khat_bic"] = osel.select_k_bic(loglik, N)
            rec["oracle_k"] = osel.oracle_k(l1_seq_all)
            kr = rec.get("khat_risk", -1)
            def _l1at(k):
                return l1_seq_all.get(int(k), np.nan)
            rec["regret_risk"] = _l1at(kr) - _l1at(rec["oracle_k"]) if kr >= 0 else np.nan
            rec["regret_bic"] = _l1at(rec["khat_bic"]) - _l1at(rec["oracle_k"])
        except Exception:
            rec["khat_bic"] = -1; rec["oracle_k"] = -1
            rec["regret_risk"] = np.nan; rec["regret_bic"] = np.nan

        # SE-check (S2, K=4 only): sandwich vs fixed-basis(eq.18) vs MC 표준편차
        if scen_name == "S2":
            try:
                rm = mom_mod.fit_mom(sample, K=4, project=False)
                th = np.asarray(rm.theta_hat, float)
                rec["se_th3"] = th[2]; rec["se_th4"] = th[3]
                rec["fb_se3"] = float(rm.se_theta_eq18[2]); rec["fb_se4"] = float(rm.se_theta_eq18[3])
                _, se = sw_mod.sandwich_cov(sample, float(rm.lam_hat), float(rm.nu_hat), 4,
                                            cmp_mod=CMP_BACKEND, basis=BASIS_BACKEND, M=M)
                rec["sw_se3"] = float(se[2]); rec["sw_se4"] = float(se[3])
            except Exception:
                pass
        return rec


def run(mode, R, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    OUT.mkdir(parents=True, exist_ok=True)
    scen_names = list(SCEN_ID)
    Ns = [1500] if mode == "joint" else N_LIST
    tasks = [(s, N, r, mode) for s in scen_names for N in Ns for r in range(R)]
    print(f"[{mode}] {len(tasks)} tasks, R={R}, workers={workers}", flush=True)
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
                np.save(OUT / f"{mode}_partial.npy", np.array(results, dtype=object), allow_pickle=True)
    np.save(OUT / f"{mode}_R{R}.npy", np.array(results, dtype=object), allow_pickle=True)
    print(f"[{mode}] done in {time.time()-t0:.0f}s -> {OUT / f'{mode}_R{R}.npy'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["main", "joint"], default="main")
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.mode, a.R, a.workers)
