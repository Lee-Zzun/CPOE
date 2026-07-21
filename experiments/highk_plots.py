"""고차 K 후속 보고서 — 그래프.

- plot_l1_curves: L1-vs-K 곡선(고정 격자 비교).
- plot_pmf_compare: 경험 pmf 막대 + 적합 pmf 곡선들(범례에 L1).
출력: papers/supp/figs/highk_*.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STYLES = ["-o", "-s", "-^", "-D", "-v", "-x", "-P", "-*", "-h"]

_CURVE_STYLE = {
    "in-sample": ("-o", "0.6"),
    "true": ("-^", "#2e8b57"),
}


def plot_l1_curves(series: Dict[str, tuple], title: str, out_path: Path,
                   grid_marks: Optional[List[int]] = None) -> None:
    """L1-vs-K 곡선(여러 계열). opt-K 수직선 없음(고정 격자 비교 패러다임).

    series: {label: (xs, ys, ses_or_None)} — label ∈ {in-sample, CV, true}.
    grid_marks: 비교 격자 K들을 옅은 수직선으로 표시(선택).
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if grid_marks:
        for k in grid_marks:
            ax.axvline(k, color="0.9", lw=0.7, zorder=0)
    for label, (xs, ys, ses) in series.items():
        st, col = _CURVE_STYLE.get(label, ("-o", "black"))
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        ax.plot(xs, ys, st, ms=3.4, lw=1.4, color=col, label=f"{label} $L^1$", zorder=3)
        if ses is not None:
            se = np.where(np.isfinite(np.asarray(ses, float)), np.asarray(ses, float), 0.0)
            ax.fill_between(xs, ys - se, ys + se, color=col, alpha=0.15, zorder=1)
    ax.set_xlabel("truncation order $K$")
    ax.set_ylabel("$L^1$ discrepancy")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pmf_compare(emp: np.ndarray, grid: np.ndarray,
                     curves: Dict[str, Tuple[np.ndarray, float]],
                     title: str, out_path: Path, x_disp: Optional[int] = None) -> None:
    """경험 pmf 막대 + 적합 pmf 곡선들(범례 L1). 좌=선형 bulk, 우=꼬리 로그-y."""
    out_path = Path(out_path)
    emp = np.asarray(emp, float)
    grid = np.asarray(grid)
    if x_disp is None:
        nz = np.where(emp > 1e-4)[0]
        x_disp = int(grid[nz[-1]]) if nz.size else int(grid[-1])
    mode = int(grid[int(np.argmax(emp))])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.8))
    # 좌: 선형
    axL.bar(grid, emp, width=0.9, color="0.82", label="Empirical", zorder=1)
    for i, (name, (p, l1)) in enumerate(curves.items()):
        ls = _STYLES[i % len(_STYLES)]
        axL.plot(grid, p, ls, ms=3.0, lw=1.2, label=f"{name} ($L^1$={l1:.3f})", zorder=3)
    axL.set_xlim(-0.5, x_disp + 0.5)
    axL.set_ylim(bottom=0.0)
    axL.set_xlabel("x (count)"); axL.set_ylabel("pmf")
    axL.set_title(title, fontsize=10)
    axL.legend(fontsize=7.5, loc="best")
    # 우: 꼬리 로그-y
    axR.bar(grid, emp, width=0.9, color="0.82", zorder=1)
    for i, (name, (p, l1)) in enumerate(curves.items()):
        ls = _STYLES[i % len(_STYLES)]
        p = np.asarray(p, float); m = p > 0
        axR.plot(grid[m], p[m], ls, ms=2.6, lw=1.1, zorder=3)
    axR.set_yscale("log")
    axR.set_xlim(mode - 0.5, x_disp + 0.5)
    axR.set_xlabel("x (count)"); axR.set_title("tail zoom (log-y)", fontsize=10)
    axR.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
