# CPOE — COM-Poisson 직교전개 (COM-Poisson Orthogonal Expansion)

*다른 언어로 읽기: [English](README.md)*

이 저장소는 아래 논문의 전체 컴퓨팅 파이프라인을 담고 있다.

> Ji-Hun Lee, Chae-Yon Lee, Hyung-Tae Ha (2026),
> **"Semi-Parametric Extensions of the Conway--Maxwell--Poisson Distribution via
> Orthogonal Polynomial Bases"** (심사 중).

논문의 수치 섹션에 등장하는 모든 수·표·그림은 이 저장소의 코드가 결정적으로
생성한다: 파이프라인을 재실행하면 보고된 값이 바이트 단위로 재현된다(고정
시드, 어떤 보고 수치도 실행 시각·환경에 의존하지 않음).

## 1. 모형 한 단락 요약

COM-Poisson(CMP) 분포 `w_CMP(x; λ, ν) ∝ λ^x / (x!)^ν`는 산포 모수 ν로 포아송을
일반화하여 과대산포(ν < 1)와 과소산포(ν > 1)를 모두 수용하지만, 여전히
2모수 가족이다: 평균과 분산은 맞출 수 있어도 왜도·첨도·다봉성·두꺼운 꼬리는
일반적으로 맞출 수 없다. CPOE는 CMP 베이스라인에 다항식 틸트를 곱해 이 한계를
제거한다:

```
p_θ(x) = w_CMP(x; λ, ν) · (1 + Σₙ θₙ ψₙ(x; λ, ν))
```

여기서 `{ψₙ}`은 CMP 가중치에 대한 직교정규 다항식계(닫힌형이 없어 수치적으로
구성)이고, `θ = (θ₁, …, θ_K)`는 전개계수로 정규화 `θ₁ = θ₂ = 0`을 따른다
(평균·분산 방향은 λ, ν가 이미 흡수; θ₃은 왜도, θ₄는 첨도에 작용). 직교성
덕분에 틸트의 총합은 자동으로 1이 되고, 실패할 수 있는 것은 비음수성뿐이라
θ는 볼록 실행가능 폴리토프 `C_K = {θ : 1 + θᵀψ(x) ≥ 0, ∀x}`에 제한된다.
논문은 (λ, ν, θ)에 대한 세 가지 추정법과 절단 차수 K의 위험 기반 선택 규칙을
개발한다.

## 2. 데이터

실데이터 2종과 시뮬레이션 시나리오 4종을 사용한다(논문 §6). 원시 CSV는
저장소에 포함되어 있으며 **모든 전처리는 결정적 코드**(`src/cpoe/data_prep.py`,
`experiments/_datasets.py` 경유)로 수행되므로 수동 데이터 작업이 전혀 필요
없다.

| 데이터셋 | N | 형상 | 출처 / 라이선스 |
|---|---|---|---|
| FIFA | 964 | 경미한 과대산포, 단봉 (V/M = 1.32) | Kaggle `martj42/international-football-results-from-1872-to-2017`, CC BY 4.0 |
| Insurance-smoker | 274 | 진성 쌍봉 (V/M = 2.85) | Kaggle `mirichoi0218/insurance`, ODbL |
| S1–S4 | 각 1500 | 시뮬레이션, 아래 참조 | `experiments/highk_scenarios.py` 생성 |

- **FIFA** — FIFA 월드컵 경기당 총 득점: `tournament == "FIFA World Cup"` 행만
  남기고 카운트는 `home_score + away_score`. CMP 베이스라인이 거의 단독으로
  적합되는 경미한 과대산포 단봉 벤치마크.
- **Insurance-smoker** — 의료비를 $1,500 구간으로 이산화(`floor(charges/1500)`)
  하고 흡연자로 제한. 흡연자 부표본이 단봉 베이스라인으로는 재현 불가능한
  진성 쌍봉 형상을 분리해 낸다 — 논문에서 가장 어려운 실데이터.
- **S1–S4** — 동일가중(50:50) 2성분 혼합으로 네 가지 형상 영역을 커버:
  S1 과소산포, S2 과대산포, S3 약한 제2봉을 가진 두꺼운 꼬리
  (0.5·Poisson(2) + 0.5·Poisson(8)), S4 과대산포+쌍봉
  (0.5·NB(2, 0.4) + 0.5·(CMP(3, 0.5)+9)). 생성 혼합을 정확히 알므로 시뮬레이션
  적합은 **참 pmf** 대비로 채점되어 진짜 위험 추정치를 준다. 모든 in-sample
  분석은 시나리오당 하나의 **고정시드 대표 표본**(N = 1500, 시드
  `20260606 + 1000·(시나리오 순번+1)`)을 쓰고, 몬테카를로는 시드맵
  `default_rng([시나리오 id, N, r])`에서 반복 표본을 뽑는다.

상세와 라이선스 정리는 [`data/README.md`](data/README.md) 참조.

## 3. 파이프라인 개요

파이프라인은 논문 Algorithm 1("CPOE numerical pipeline")을 그대로 반영한다.
정규화상수·모멘트·기저·틸트 pmf·추정량 어느 것도 닫힌형이 없어, 모든 양은
적응적으로 절단된 지지대 `S_M = {0, …, M}` 위에서 수치적으로 계산된다:

```
                 ┌────────────────────────────────────────────────────────┐
                 │  src/cpoe  (라이브러리 — 수치 코어)                       │
                 │                                                        │
  data/*.csv ─►  │  ① compoisson   log Z·모멘트 (log-sum-exp, 허용오차 1e-12)│
  시나리오    ─►  │  ② basis        ψₙ 삼항점화 (discretized Stieltjes)      │
                 │  ③ cpoe_model   틸트 pmf  p_θ = w·(1+θᵀψ)               │
                 │  ④ feasibility  C_K 폴리토프 진단 + L² 사영               │
                 │  ⑤ estimators   MoM │ Sequential MLE │ Joint MLE        │
                 │  ⑥ order_selection   R̂(K) 프로파일 → K̂_RISK            │
                 └───────────────┬────────────────────────────────────────┘
                                 │  공유 적합 코어: experiments/highk_optimal_k
                 ┌───────────────▼────────────────────────────────────────┐
                 │  experiments  (드라이버 — 논문 산출물당 스크립트 하나)      │
                 │  crossmodel_eval · l2_eval · highk_report ·             │
                 │  data_pmf_overview · compute_orderselection ·           │
                 │  mc_simulation → mc_tables · task1_basis_comparison     │
                 └───────────────┬────────────────────────────────────────┘
                                 ▼
        papers/figs (본문 그림) · papers/supp (Supplement) · results/ (CSV)
```

내적 정합성은 두 가지 구조적 사실이 보장한다:

1. **단일 공유 적합 경로.** 모든 표·그림의 in-sample 적합은
   `experiments/highk_optimal_k._fit_one` / `_pmf`를 거치므로 산출물 간 수치가
   구성상 일치한다(몬테카를로는 동일 추정기를 직접 호출).
2. **단일 공유 내부 솔버.** Sequential·Joint MLE는 *동일한* 볼록 θ-솔버
   (`solve_theta_at_baseline`, 해석적 grad/Hess의 trust-region interior
   Newton, KKT 조건 정확 충족)를 쓰므로 feasibility 강제가 동일하고, 두
   추정량의 차이는 "(λ, ν)가 틸트에 반응하는가" 하나로 좁혀진다.

## 4. 라이브러리 모듈 (`src/cpoe/`)

| 모듈 | 역할 |
|---|---|
| `compoisson.py` | CMP 코어: 적응적 절단 + log-sum-exp에 의한 `log Z(λ, ν)`(잔차 허용오차 1e-12, 단조감소 연속 5회 — 논문의 log Z 식), 평균/분산, pmf, Shmueli et al.(2005)의 대-λ 점근 warm start. `config/hyperparams.yaml`을 로드. |
| `basis.py` | 직교정규 다항식계 `{ψₙ}`. 주 경로: **삼항점화** — 계수를 discretized-Stieltjes(Lanczos) 절차와 수정 Gram–Schmidt 재직교화로 산출, 전 차수에서 기계정밀도(~1e-12) 직교성 유지. 모멘트 기반 Hankel/Cholesky 경로는 저차 교차검증용으로만 보존: 조건수가 차수에 지수적으로 증가(차수 9에서 직교성 ~1e-6으로 붕괴; `task1_basis_comparison` 참조). |
| `cpoe_model.py` | 틸트 pmf `p_θ`와 투영항등식 `θₙ = E_p[ψₙ(X)]`(논문 eq. 14) — 세 추정법의 공통 앵커; `empirical_theta`가 그 plug-in 형태. |
| `feasibility.py` | 절단 지지대 위에서 실현한 폴리토프 `C_K`: 위반/여유 진단과, 실행불가능한 raw MoM 투영을 복원하는 유클리드 사영 `project_onto_C_K`. |
| `backends.py` | CMP/기저 코드를 추정기에 잇는 얇은 어댑터(`CMP_BACKEND`, `BASIS_BACKEND`, `default_M`). |
| `sampling.py` | CMP 및 정확-틸트 CPOE 분포의 표본기(시뮬레이션 시나리오가 사용). |
| `data_prep.py` | 실데이터 2종의 결정적 전처리. |
| `diagnostics.py` | `empirical_pmf`, `l1_distance`. |
| `baselines.py` | 크로스모델 비교용 적률매칭 Poisson/NB/CMP 베이스라인 pmf. |
| `order_selection.py` | 위험 기반 차수선택: 투영 통계 `(θ̂ₙ, σ̂ₙ²)`, 위험 프로파일 `R̂(K) = Σₙ₌₃^K (2σ̂ₙ²/N − θ̂ₙ²)`, `K̂_RISK = argmin R̂(K)` — 차수 간 위험 *차이*의 비편향 추정량(논문 Proposition 4). 검증 전용 동반 도구(논문 표에는 미사용)도 포함: `BIC(K)`/`K̂_BIC`, within-replication oracle `K*`, 잔여에너지 진단 `Ê^{1/2}`. |
| `estimators/mom_sandwich.py` | 검증 전용 MoM 2단계 **sandwich 공분산**: `(λ, ν, θ₃…θ_K)`의 stacked just-identified Z-추정으로 1단계 plug-in 오차를 `SE(θ̂ₙ)`에 전파; fixed-basis SE·몬테카를로 표준편차와 교차검증(`mc_tables.py` SE-check 표). |
| `estimators/mom.py` | **적률법(MoM)**: 적률매칭 시스템 `μ_CMP = X̄`, `σ²_CMP = S²`의 Newton 해 → 닫힌형 경험투영 `θ̂ₙ = N⁻¹ Σᵢ ψₙ(Xᵢ)`; raw 투영이 `C_K`를 벗어나면 유클리드 QP로 복원. 셋 중 가장 저렴. |
| `estimators/cmp_baseline.py` | 순수 CMP 최대우도(자연모수 좌표에서 오목) — Sequential 추정의 1단계. |
| `estimators/mle_sequential.py` | **Sequential(2단계, 제한정보) MLE**: CMP-MLE로 베이스라인 적합 후, 베이스라인을 고정한 채 `θ ∈ C_K` 위에서 틸트 우도를 최대화 — 공유 볼록 내부 솔버 1회 호출. |
| `estimators/mle.py` | **Joint(완전정보) MLE**: nested 프로파일 최적화. 외부 루프는 프로파일 `g(λ, ν) = max_θ ℓ_N(λ, ν, θ)`를 절단 뉴턴(TNC)으로 최대화하되 내부 최적점에서의 해석적 포락선(Danskin) 기울기를 사용; 다중 시작점(적률 기반 + Sequential 해)으로 `ℓ_Joint ≥ ℓ_Seq` 보장; 유효성 게이트(내부 실행가능 + `ℓ_N ≤ 0`)가 고차의 정규화-붕괴 퇴화해를 기각. |

수치 정책: 모든 임의 임계값(절단 허용오차, 모수 박스)은
`config/hyperparams.yaml`에 외재화 — 어떤 모듈도 영역 임계값을 하드코딩하지
않으므로 민감도 스윕은 설정 한 줄 변경으로 수행된다.

## 5. 실험 드라이버 (`experiments/`)와 논문 산출물 대응

| 스크립트 | 산출물 | 논문 산출물 | 대략 소요 |
|---|---|---|---|
| `crossmodel_eval.py` | `results/crossmodel.csv`, `papers/supp/tables/perf_crossmodel.tex` | 크로스모델 비교표 (Poisson/NB/CMP/CPOE) | ~4분 |
| `l2_eval.py` | `results/l2_*.csv`, `papers/supp/tables/l2_*.tex`, `papers/supp/figs/l2_*` | 고정차수 추정법 비교표; Supp 차수별 표 | ~10분 |
| `highk_report.py` | `papers/figs/highk_pmfbase_*.pdf`(본문), `papers/supp/figs·tables/highk_*` | pmf overlay 그림; 동일차수 추정법 overlay(Supp) | ~5분 |
| `data_pmf_overview.py` | `papers/figs/data_pmf_overview_sim.pdf`, `papers/supp/figs/data_pmf_overview_*` | 경험 pmf 개요 그림 | ~1분 |
| `compute_orderselection.py` | stdout (차수선택 표의 값; 검증 전용 `K̂_BIC`/`BIC`/`Ê^{1/2}` 열 포함) | 적응적 차수선택 표 | ~2분 |
| `mc_simulation.py --R 500` | `experiments/mc_out/main_R500.npy` | 몬테카를로 원시 기록 (6,000 적합; 선택규칙·SE-check 필드 포함). `joint` 모드는 검증 전용 Joint 축소설계 실행 | 수 시간 (아카이브 동봉) |
| `mc_tables.py` | stdout (MC 위험표; 검증 전용 selection/feasibility-동등성/SE-check 표 포함), `mc_risk_profiles.pdf` | MC 위험표·위험 프로파일 그림 | 아카이브에서 ~1분 |
| `est2nd_eval.py` | `results/tables/est2nd.tex` | (검증 전용, 논문 미수록) 세 추정법의 feasibility(in-`C_K`)·적합시간 중앙값, `K = 4, 10` | ~30분 (Joint 적합) |
| `task1_basis_comparison.py` | `results/task1_comparison.csv`, `papers/supp/tables/task1_*.tex`, 조건수 그림 | §5 기저 구성 비교의 정량 백킹 | ~1분 |
| `highk_scenarios.py`, `highk_optimal_k.py`, `_datasets.py`, `highk_plots.py`, `highk_tables.py` | (위 드라이버들이 임포트하는 라이브러리) | — | — |

몬테카를로 아카이브 `experiments/mc_out/main_R500.npy`(4.5 MB)를 동봉하므로
6,000회 적합을 재실행하지 않고도 위험표와 그림을 약 1분 만에 재생성할 수
있다; 원하면 `mc_simulation.py`로 아카이브 자체를 처음부터 재현할 수 있다
(`--R 5`는 빠른 스모크 실행).

## 6. 전체 재현

```bash
# 환경 (Python >= 3.11)
uv sync                    # 또는: pip install -e .

# 전체 재생성 (in-sample 파이프라인 총 ~20분)
python experiments/data_pmf_overview.py
python experiments/crossmodel_eval.py
python experiments/l2_eval.py
python experiments/highk_report.py
python experiments/compute_orderselection.py
python experiments/mc_tables.py            # 동봉된 MC 아카이브 사용
python experiments/task1_basis_comparison.py

# 선택: 몬테카를로 연구 자체 재실행 (수 시간)
python experiments/mc_simulation.py --R 500

# 선택: 검증 전용 도구 (논문 표에는 미사용)
python experiments/est2nd_eval.py                    # feasibility + 적합시간 표 (~30분, Joint 적합)
python experiments/mc_simulation.py joint --R 100    # Joint 축소설계 MC (검증용)
```

결정성: in-sample 표본은 고정 시드 20260606(시나리오별 오프셋), 몬테카를로는
구조적 시드맵 `default_rng([시나리오, N, 반복])`을 쓰며 모든 옵티마이저가
결정적이다 — 재생성된 CSV는 실행 간 바이트 단위로 동일하다.

Supplement 소스는 `papers/supp/supplementary.tex`이며, 자동 생성된
`papers/supp/tables/`의 표와 `papers/supp/figs/`의 그림을 `\input`하므로
파이프라인 실행 후 바로 컴파일된다.

## 7. 라이선스와 인용

- **코드**: MIT (`LICENSE` 참조).
- **데이터**: FIFA CSV는 CC BY 4.0(출처: Kaggle `martj42`), insurance CSV는
  ODbL(출처: Kaggle `mirichoi0218`)로 재배포. `data/README.md` 참조.
- **인용**: `CITATION.cff` 참조. 위 원고(심사 중; 게재 확정 시 인용 정보
  갱신 예정) 그리고/또는 이 저장소를 인용해 주기 바란다.
