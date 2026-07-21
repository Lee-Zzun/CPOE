"""고차 K 후속 보고서 — 오케스트레이션 (고정 격자 in-sample $L^1$ 비교, opt-K 미산정).

데이터당 고정 격자 G=[0,4,6,8,10,12,14]에서 차수별 **in-sample** $L^1$을 비교·해석한다.
참조 논문(Skewness–Kurtosis OE)의 수렴표(Table 5/8/10)·수렴곡선(Fig 2/4/6)과 동일하게
in-sample $L^1$을 차수에 걸쳐 제시한다. (in-sample $L^1$은 $K$에 단조 하락하므로 ``고차=최고''
오독에 유의 — 과적합 인식은 해석 프로즈에서 다룬다.)

사용: python experiments/highk_report.py [--datasets FIFA]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from cpoe import baselines as base_  # noqa: E402
from cpoe.diagnostics import empirical_pmf  # noqa: E402
from _datasets import load_dataset  # noqa: E402
from highk_optimal_k import _l1  # noqa: E402
from highk_scenarios import HIGHK_SCENARIOS  # noqa: E402
import highk_optimal_k as ok_  # noqa: E402
import highk_plots as plt_  # noqa: E402
import highk_tables as tbl_  # noqa: E402

ALL_DATASETS = ["FIFA", "Insurance-smoker", "S1", "S2", "S3", "S4"]
EST_ABBR = {"MoM": "mom", "Sequential MLE": "seq", "Joint MLE": "joint"}
EST_SHORT = {"MoM": "MoM", "Sequential MLE": "Seq", "Joint MLE": "Joint"}
GRID = [0, 4, 6, 8, 10, 12, 14]   # 비교 고정 격자 (K=14까지; 사용자 결정 2026-07-01, 15 제외)
PMF_KS = [10, 14]                 # 동일-K 3추정법 pmf 비교 차수 (K=10·K=14 각각)


def _slug(name: str) -> str:
    return name.lower().replace("-", "_")


def _load(name: str, N: int, seed: int) -> np.ndarray:
    if name in HIGHK_SCENARIOS:
        i = list(HIGHK_SCENARIOS).index(name)
        return HIGHK_SCENARIOS[name].sampler(N, np.random.default_rng(seed + 1000 * (i + 1))).astype(int)
    return load_dataset(name).astype(int)


def _xy(m: Dict[int, float]):
    ks = sorted(k for k, v in m.items() if v is not None and np.isfinite(v))
    return (ks, [m[k] for k in ks], None)


def process_dataset(name: str, args, fig_dir: Path, tex_dir: Path, out_dir: Path) -> None:
    t0 = time.perf_counter()
    s = _load(name, args.N, args.seed)
    max_x = int(s.max())
    grid = np.arange(max_x + 1)
    emp = empirical_pmf(s, max_x + 1)
    slug = _slug(name)
    print(f"\n[{name}] N={s.size} max={max_x}", flush=True)

    # ---- MoM in-sample 곡선 (A/B용; Seq/Joint 전차수 곡선은 §6에서 미사용) ----
    te = time.perf_counter()
    ins_rows = ok_.insample_l1_curve(s, "MoM", k_max=args.kmax)
    mom = {"ins": {r["K"]: r["L1"] for r in ins_rows if r.get("reliable")}}
    print(f"  {'MoM':>15s}: in-sample done ({time.perf_counter()-te:.1f}s)", flush=True)

    # ---- A: MoM in-sample L1-vs-K 곡선 ----
    plt_.plot_l1_curves({"in-sample": _xy(mom["ins"])}, f"{name}: in-sample $L^1$ vs $K$ (MoM)",
                        fig_dir / f"highk_l1curve_{slug}.pdf", grid_marks=GRID)

    # ---- B: MoM in-sample L1 격자표 ----
    tser = {"in-sample $L^1$": [mom["ins"].get(k) for k in GRID]}
    tbl_.write_l1byk_multi(GRID, tser, tex_dir / f"highk_l1byk_{slug}.tex",
                           out_dir / f"highk_l1byk_{slug}.csv",
                           caption=f"{name}: in-sample $L^1$ of the CPOE (MoM) fit by truncation order $K$.",
                           label=f"tab:highk-l1byk-{slug}")

    # ---- 격자 K별 CPOE(MoM) pmf 1회 계산 (C 그래프용) ----
    cpoe_pmf_by_k: Dict[int, np.ndarray] = {}
    for k in GRID:
        try:
            lam, nu, th, _ = ok_._fit_one(s, "MoM", k)
            cpoe_pmf_by_k[k] = ok_._pmf(lam, nu, th, max_x)
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] CPOE(MoM) K={k} 실패: {e}", flush=True)

    # ---- C: 경험 pmf vs 각 격자 K별 CPOE(MoM) pmf ----
    curvesC = {f"$K$={k}": (cpoe_pmf_by_k[k], _l1(emp, cpoe_pmf_by_k[k]))
               for k in GRID if k in cpoe_pmf_by_k}
    plt_.plot_pmf_compare(emp, grid, curvesC,
                          f"{name}: empirical vs CPOE(MoM) pmf by $K$",
                          fig_dir / f"highk_pmf_{slug}.pdf")

    # ---- D: 경험 pmf vs 베이스라인(Poisson/NB/CMP) vs CPOE(K=4,10) (MoM) ----
    p_pois, _ = base_.poisson_fitted_pmf(s, grid)
    p_nb, _ = base_.nb_fitted_pmf(s, grid)
    curvesD = {"Poisson": (p_pois, _l1(emp, p_pois)), "NB": (p_nb, _l1(emp, p_nb))}
    if 0 in cpoe_pmf_by_k:
        curvesD["CMP"] = (cpoe_pmf_by_k[0], _l1(emp, cpoe_pmf_by_k[0]))
    for kk in (4, 10, 14):
        if kk in cpoe_pmf_by_k:
            curvesD[f"CPOE($K$={kk})"] = (cpoe_pmf_by_k[kk], _l1(emp, cpoe_pmf_by_k[kk]))
    plt_.plot_pmf_compare(emp, grid, curvesD,
                          f"{name}: empirical vs baselines vs CPOE($K$=4,10,14; MoM)",
                          fig_dir / f"highk_pmfbase_{slug}.pdf")

    # ---- G: 고정 K(10)에서 세 추정법 pmf ----
    fig_inputs: List[str] = []
    for k in PMF_KS:
        c8 = {}
        for est in ok_.ESTIMATORS:
            try:
                lam, nu, th, _ = ok_._fit_one(s, est, k)
                p = ok_._pmf(lam, nu, th, max_x)
                c8[EST_SHORT[est]] = (p, _l1(emp, p))
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] (G) {est} K={k} 실패: {e}", flush=True)
        plt_.plot_pmf_compare(emp, grid, c8, f"{name}: 3 estimators at $K$={k}",
                              fig_dir / f"highk_estpmf_{slug}_K{k}.pdf")
        fig_inputs.append(
            "\\begin{figure}[H]\\centering"
            f"\\includegraphics[width=\\textwidth]{{highk_estpmf_{slug}_K{k}.pdf}}"
            f"\\caption{{{name}: empirical pmf versus the CPOE pmfs of the three estimators at the same order $K{{=}}{k}$. Left: linear scale; right: tail on a logarithmic $y$-axis.}}"
            f"\\label{{fig:hk-{slug}-estpmf-k{k}}}\\end{{figure}}")
    (tex_dir / f"highk_estpmf_{slug}.tex").write_text("\n".join(fig_inputs) + "\n", encoding="utf-8")

    print(f"[{name}] done ({time.perf_counter()-t0:.1f}s)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS))
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--kmax", type=int, default=14)
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "papers" / "supp" / "figs")
    ap.add_argument("--tex-dir", type=Path, default=ROOT / "papers" / "supp" / "tables")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)

    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for d in (args.fig_dir, args.tex_dir, args.out_dir):
        d.mkdir(parents=True, exist_ok=True)
    for nm in names:
        process_dataset(nm, args, args.fig_dir, args.tex_dir, args.out_dir)
    print("\n[highk] 완료.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
