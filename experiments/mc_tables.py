"""§6.2 Monte Carlo 결과 → LaTeX 표 값 + 위험곡선 그림 생성.

usage: python mc_tables.py [main_npy] [joint_npy] [figdir]
tab:mc-risk(Sequential) 표 값과 MoM 위험 표(Supplementary 후보)를 stdout에 출력;
그림은 figdir/mc_risk_profiles.pdf 로 저장.
[2026-07 복원] 선택규칙(selection)·feasibility/동등성·SE-check 표는 논문 축소 시
삭제되었다가 리뷰 대응용 검증 도구로 복원됨(main npy에 해당 필드가 있을 때만 유효;
mc_simulation.py의 복원판으로 재생성 필요).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = [0, 4, 6, 8, 10, 12, 14]
TILT = [4, 6, 8, 10, 12, 14]
NS = [200, 500, 1500]
SCENS = ["S1", "S2", "S3", "S4"]
JORD = [4, 10, 14]


def load(p):
    return list(np.load(p, allow_pickle=True))


def cell(recs, scen, N):
    return [r for r in recs if r["scen"] == scen and r["N"] == N]


def mse(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def mode_freq(vals):
    v = [int(x) for x in vals if x is not None and int(x) >= 0]
    if not v:
        return -1, 0.0
    u, c = np.unique(v, return_counts=True)
    i = int(c.argmax())
    return int(u[i]), float(c[i] / len(v))


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


def selection_table(recs):
    print("\n=== tab:mc-selection ===")
    for scen in SCENS:
        for N in NS:
            rr = cell(recs, scen, N)
            mr, fr = mode_freq([r.get("khat_risk") for r in rr])
            mb, fb = mode_freq([r.get("khat_bic") for r in rr])
            ok, _ = mse([r.get("oracle_k") for r in rr])
            rgr, rgr_se = mse([r.get("regret_risk") for r in rr])
            rgb, rgb_se = mse([r.get("regret_bic") for r in rr])
            lead = f"\\multirow{{3}}{{*}}{{{scen}}}" if N == 200 else ""
            print(f"{lead} & {N} & {mr} ({fr:.0%}) & {mb} ({fb:.0%}) & {ok:.1f} & "
                  f"{rgr:.4f} ({rgr_se:.4f}) & {rgb:.4f} ({rgb_se:.4f}) \\\\".replace("%", "\\%"))
        print("\\midrule")


def feas_equiv_table(recs, jrecs):
    print("\n=== tab:mc-feasibility-equivalence ===")
    for scen in SCENS:
        rr = [r for r in recs if r["scen"] == scen]
        feas = []
        for K in JORD:
            fr = np.mean([1.0 if r.get(f"momfeas_{K}") else 0.0 for r in rr])
            feas.append(fr)
        # Δ_Seq-MoM: mean over reps & tilt grid orders (L1_seq - L1_mom)
        gaps = []
        for r in rr:
            for K in TILT:
                a, b = r.get(f"l1_seq_{K}"), r.get(f"l1_mom_{K}")
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    gaps.append(a - b)
        dsm, dsm_se = mse(gaps)
        # Δ_Joint-Seq from joint run (mean over reps & JORD)
        djs, djs_se = np.nan, np.nan
        if jrecs is not None:
            jj = [r for r in jrecs if r["scen"] == scen]
            jg = []
            for r in jj:
                for K in JORD:
                    a, b = r.get(f"l1_joint_{K}"), r.get(f"l1_seq_{K}")
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                        jg.append(a - b)
            djs, djs_se = mse(jg)
        print(f"{scen} & {feas[0]:.2f} & {feas[1]:.2f} & {feas[2]:.2f} & "
              f"{dsm:+.4f} ({dsm_se:.4f}) & {djs:+.4f} ({djs_se:.4f}) \\\\")


def se_check_table(recs):
    print("\n=== tab:mc-se-check (S2, K=4) ===")
    for N in NS:
        rr = [r for r in cell(recs, "S2", N) if "sw_se3" in r]
        if not rr:
            print(f"{N} & (no SE-check records) \\\\")
            continue
        th3 = [r["se_th3"] for r in rr]; th4 = [r["se_th4"] for r in rr]
        e3 = np.std(th3, ddof=1); e4 = np.std(th4, ddof=1)
        f3 = np.mean([r["fb_se3"] for r in rr]); f4 = np.mean([r["fb_se4"] for r in rr])
        s3 = np.mean([r["sw_se3"] for r in rr]); s4 = np.mean([r["sw_se4"] for r in rr])
        print(f"{N} & 3 & {e3:.4f} & {f3:.4f} & {s3:.4f} \\\\")
        print(f"{'':4s} & 4 & {e4:.4f} & {f4:.4f} & {s4:.4f} \\\\")


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
    joint_npy = sys.argv[2] if len(sys.argv) > 2 else None
    figdir = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "experiments/mc_out")
    recs = load(main_npy)
    jrecs = load(joint_npy) if joint_npy and Path(joint_npy).exists() else None
    print(f"loaded {len(recs)} main records" + (f", {len(jrecs)} joint" if jrecs else ""))
    risk_table(recs, "seq")
    risk_table(recs, "mom")
    selection_table(recs)
    feas_equiv_table(recs, jrecs)
    se_check_table(recs)
    make_figure(recs, figdir)
