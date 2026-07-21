# Datasets

Two empirical count datasets back the paper's real-data analysis (paper `tab:metadata_specs-new`,
`tab:descriptive_stats-new`). All preprocessing is deterministic and implemented in
`src/cpoe/data_prep.py` (loaded via `experiments/_datasets.py`).

## File layout

```
data/
  FIFA.csv           - international football results (long table); filtered in code
  insurance.csv      - Kaggle medical-insurance records; discretized in code
  eda_datasets.ipynb - exploratory analysis (descriptive stats, smoker/non-smoker split)
  README.md          - this file
```

## 1. FIFA - World Cup total goals (N = 964)

- **Source**: Kaggle `martj42/international-football-results-from-1872-to-2017` (CC BY 4.0;
  accessed 10 February 2026, per paper metadata table).
- **Preprocessing** (`load_fifa_counts`): keep rows with `tournament == "FIFA World Cup"`,
  count = `home_score + away_score` per match, drop nulls/negatives.
- **Shape**: mildly overdispersed unimodal; mean 2.82, variance 3.71, V/M = 1.32.

## 2. Insurance-smoker - discretized charges (N = 274)

- **Source**: Kaggle `mirichoi0218/insurance` (ODbL; accessed 10 February 2026).
- **Preprocessing** (`insurance_bimodal_to_count`): `floor(charges / 1500)`, restricted to
  `smoker == "yes"`. The smoker subsample isolates the genuinely bimodal shape
  (paper `tab:insurance-split`); the full-sample multimodality is a smoker/non-smoker
  mixture artifact.
- **Shape**: bimodal; mean 20.88, variance 59.46, V/M = 2.85.

## Removed dataset

Sepsis (PhysioNet MIMIC-IV lab-test counts) was removed from the project in the 2026-07 cleanup:
the paper excludes it, and the PhysioNet Credentialed Health Data License forbids committing the
CSV. The loader (`load_sepsis_lab_counts`) and `data/Sepsis.csv` were deleted; the file remains
in git history only. If it is ever reintroduced, `data/Sepsis.csv` MUST be gitignored (DUA).

## License roll-up

| Dataset | Source | License | Redistribution in repo? |
|---|---|---|---|
| FIFA | Kaggle martj42 | CC BY 4.0 | yes with attribution |
| insurance | Kaggle mirichoi0218 | ODbL | yes with attribution |
