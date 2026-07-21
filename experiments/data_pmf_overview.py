"""섹션 4(데이터)용 — 6종(실데이터 2 + 시뮬레이션 4)의 경험 pmf 개요 그림.

보고서 파이프라인(highk_report.py)과 *동일한* 데이터 재현:
  - 실데이터: _datasets.load_dataset (FIFA, Insurance-smoker)
  - 시뮬레이션: HIGHK_SCENARIOS[name].sampler(N, default_rng(seed+1000*(i+1)))
    seed=20260606, N=1500, i=시나리오 인덱스(S1..S4 → 0..3)
단일 2x3 패널 PDF를 papers/supp/figs/data_pmf_overview.pdf 로 저장.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cpoe.diagnostics import empirical_pmf  # noqa: E402
from _datasets import load_dataset  # noqa: E402
from highk_scenarios import HIGHK_SCENARIOS  # noqa: E402

SEED = 20260606
N = 1500
ORDER = ["FIFA", "Insurance-smoker", "S1", "S2", "S3", "S4"]
TITLES = {
    "FIFA": "FIFA (World Cup goals)",
    "Insurance-smoker": "Insurance-smoker",
    "S1": "S1: 0.5 Pois(5)+0.5 Bin(10,0.5)",
    "S2": "S2: 0.5 Pois(4)+0.5 NB(4,0.4)",
    "S3": "S3: 0.5 Pois(2)+0.5 Pois(8)",
    "S4": "S4: 0.5 NB(2,0.4)+0.5 (CMP(3,0.5)+9)",
}


def load(name: str) -> np.ndarray:
    if name in HIGHK_SCENARIOS:
        i = list(HIGHK_SCENARIOS).index(name)
        return HIGHK_SCENARIOS[name].sampler(
            N, np.random.default_rng(SEED + 1000 * (i + 1))).astype(int)
    return load_dataset(name).astype(int)


def _draw(ax, name: str) -> None:
    s = load(name)
    n = int(s.size)
    mean = float(s.mean()); varr = float(s.var(ddof=1))
    vm = varr / mean if mean > 0 else float("nan")
    max_x = int(s.max())
    emp = empirical_pmf(s, max_x + 1)
    grid = np.arange(max_x + 1)
    nz = np.where(emp > 1e-4)[0]
    x_disp = int(grid[nz[-1]]) if nz.size else max_x
    ax.bar(grid, emp, width=0.9, color="0.72", edgecolor="0.4", linewidth=0.3)
    ax.set_xlim(-0.5, x_disp + 0.5)
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"{TITLES[name]}\n$N$={n}, $V/M$={vm:.2f}", fontsize=9.5)
    ax.set_xlabel("x (count)", fontsize=9)
    ax.set_ylabel("empirical pmf", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)


def _save(fig, stem: str) -> Path:
    out = ROOT / "papers" / "supp" / "figs" / f"{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def main() -> int:
    # 통합(6패널) — 하위호환.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, name in zip(axes.ravel(), ORDER):
        _draw(ax, name)
    fig.tight_layout()
    _save(fig, "data_pmf_overview")

    # 실데이터 2패널 (Data Analysis 서론용).
    figr, axr = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, name in zip(axr.ravel(), ["FIFA", "Insurance-smoker"]):
        _draw(ax, name)
    figr.tight_layout()
    _save(figr, "data_pmf_overview_real")

    # 시뮬레이션 4패널 (Simulation 서론용).
    figs, axs = plt.subplots(2, 2, figsize=(9, 6.2))
    for ax, name in zip(axs.ravel(), ["S1", "S2", "S3", "S4"]):
        _draw(ax, name)
    figs.tight_layout()
    _save(figs, "data_pmf_overview_sim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
