"""§6.2 Monte Carlo 결과 → LaTeX 표 값 + 위험곡선 그림 생성.

usage: python mc_tables.py [main_npy] [figdir]
tab:mc-risk(Sequential) 표 값과 MoM 위험 표(Supplementary 후보)를 stdout에 출력;
그림은 figdir/mc_risk_profiles.pdf 로 저장.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = [0, 4, 6, 8, 10, 12, 14]
NS = [200, 500, 1500]
SCENS = ["S1", "S2", "S3", "S4"]


def load(p):
    return list(np.load(p, allow_pickle=True))


def cell(recs, scen, N):
    return [r for r in recs if r["scen"] == scen and r["N"] == N]


def mse(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def risk_table(recs, est):
    print(f"\n=== tab:mc-risk ({est}) : mean L1 (se), bold=row-min ===")
    for scen in SCENS:
        for N in NS:
            rr = cell(recs, scen, N)
            means, ses = [], []
            for K in GRID:
                m, s = mse([r.get(f"l1_{est}_{K}") for r in rr])
                means.append(m); ses.append(s)
            kmin = int(np.nanargmin(means))
            cells = []
            for i, K in enumerate(GRID):
                txt = f"{means[i]:.3f} ({ses[i]:.3f})"
                if i == kmin:
                    txt = f"\\textbf{{{means[i]:.3f}}} ({ses[i]:.3f})"
                cells.append(txt)
            lead = f"\\multirow{{3}}{{*}}{{{scen}}}" if N == 200 else ""
            print(f"{lead} & {N} & " + " & ".join(cells) + " \\\\")
        print("\\midrule")


def make_figure(recs, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, scen in zip(axes.ravel(), SCENS):
        for N in NS:
            rr = cell(recs, scen, N)
            m = [mse([r.get(f"l1_seq_{K}") for r in rr])[0] for K in GRID]
            s = [mse([r.get(f"l1_seq_{K}") for r in rr])[1] for K in GRID]
            m = np.array(m); s = np.array(s)
            ax.plot(GRID, m, marker="o", ms=3, label=f"N={N}")
            ax.fill_between(GRID, m - 2 * s, m + 2 * s, alpha=0.15)
        ax.set_title(scen); ax.set_xlabel("order $K$"); ax.set_ylabel(r"mean $L^1(p_{\rm true},\hat p_K)$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(figdir) / "mc_risk_profiles.pdf"
    fig.savefig(out); print(f"\n[figure] saved {out}")


if __name__ == "__main__":
    main_npy = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "experiments/mc_out/main_R500.npy")
    figdir = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "experiments/mc_out")
    recs = load(main_npy)
    print(f"loaded {len(recs)} main records")
    risk_table(recs, "seq")
    risk_table(recs, "mom")
    make_figure(recs, figdir)
