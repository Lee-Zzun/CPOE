"""공용 실데이터 로더 — FIFA/Insurance를 논문과 동일한 전처리로 로드한다.

검증된 로더를 재사용하여 논문과 동일한 전처리(월드컵 필터, $1500 이산화)를 보장한다.
또한 각 데이터셋의 논문 최적 절단 차수 K*와 기대 (λ,ν)를 노출한다(검증 및 표 캡션용).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cpoe.data_prep import (                 # noqa: E402
    load_fifa_counts,
    insurance_bimodal_to_count,
)

# 논문 §6.1: K*와 MoM 보정된 베이스라인 (λ̂,ν̂), V/M.
DATASET_META = [
    {"name": "FIFA", "K_star": 4, "exp_lam": 2.114, "exp_nu": 0.871, "exp_vm": 1.32},
    {"name": "Insurance", "K_star": 7, "exp_lam": 2.81, "exp_nu": 0.38, "exp_vm": 7.80},
]


def load_dataset(name: str) -> np.ndarray:
    """이름으로 정수 카운트 표본을 로드한다 (cpoe.data_prep = GitHub data_prep.py의 미러)."""
    data_dir = ROOT / "data"
    if name == "FIFA":
        return np.asarray(load_fifa_counts(str(data_dir / "FIFA.csv")), dtype=int)
    if name == "Insurance":
        return np.asarray(insurance_bimodal_to_count(str(data_dir / "insurance.csv")), dtype=int)
    if name == "Insurance-smoker":
        return np.asarray(insurance_bimodal_to_count(
            str(data_dir / "insurance.csv"), row_filter=("smoker", "yes")), dtype=int)
    if name == "Insurance-nonsmoker":
        return np.asarray(insurance_bimodal_to_count(
            str(data_dir / "insurance.csv"), row_filter=("smoker", "no")), dtype=int)
    raise ValueError(f"unknown dataset {name!r}")


def load_all() -> List[Tuple[dict, np.ndarray]]:
    """FIFA, Insurance에 대한 [(meta, sample), ...]."""
    return [(m, load_dataset(m["name"])) for m in DATASET_META]


def descriptive(data: np.ndarray) -> dict:
    N = int(data.shape[0])
    mean = float(np.mean(data))
    var = float(np.var(data, ddof=1))
    return {
        "N": N, "mean": mean, "variance": var,
        "V_over_M": var / mean if mean > 0 else float("nan"),
        "max": int(np.max(data)),
    }
