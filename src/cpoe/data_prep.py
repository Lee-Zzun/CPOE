"""실데이터 전처리 — GitHub silontee/skewness-and-kurtosis `src/data_prep.py`의 미러.

두 함수의 시그니처, 기본값, 로직을 원본에 맞춘다 (FIFA World Cup 골 합,
Insurance $1500 구간화). 원본은 polars 기반이며, polars가 없는 환경을 위해
동일한 결과를 내는 pandas 폴백을 제공한다 (두 경로의 출력 배열은 순서를
제외하면 동일함이 확인됨). 이 모듈은 데이터 전처리의 단일 소스이며
experiments/_datasets.py에서 사용된다.
"""

from __future__ import annotations

import numpy as np

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    _HAS_POLARS = False


def load_fifa_counts(results_csv_path: str) -> np.ndarray:
    """FIFA World Cup 경기당 총 골 (home+away). null 제거, 음수 제외."""
    if _HAS_POLARS:
        df = pl.read_csv(results_csv_path)
        df_wc = df.filter(pl.col("tournament") == "FIFA World Cup")
        x = (df_wc["home_score"] + df_wc["away_score"]).drop_nulls().cast(pl.Int64).to_numpy()
    else:
        import pandas as pd
        df = pd.read_csv(results_csv_path)
        wc = df[df["tournament"] == "FIFA World Cup"]
        x = (wc["home_score"] + wc["away_score"]).dropna().astype("int64").to_numpy()
    return x[x >= 0]


def insurance_bimodal_to_count(
    insurance_csv_path: str, col: str = "charges", bin_width: int = 1500,
    row_filter: tuple[str, str] | None = None,
) -> np.ndarray:
    """연속형 `charges`를 폭 bin_width (=1500)의 정수 구간으로 변환 (상한 없음).

    `row_filter=(컬럼, 값)`이 주어지면 binning 전에 해당 행만 남긴다
    (예: `("smoker", "yes")` → 흡연자 부분군). 기본 None이면 전체 행 사용
    (기존 동작 불변).
    """
    if _HAS_POLARS:
        df = pl.read_csv(insurance_csv_path)
        if row_filter is not None:
            df = df.filter(pl.col(row_filter[0]) == row_filter[1])
        x = df[col].cast(pl.Float64, strict=False).drop_nulls().to_numpy()
    else:
        import pandas as pd
        df = pd.read_csv(insurance_csv_path)
        if row_filter is not None:
            df = df[df[row_filter[0]] == row_filter[1]]
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    cats = (x // bin_width).astype(int)
    return cats
