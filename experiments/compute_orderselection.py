"""tab:order-selection (6 데이터셋) 계산: K̂_RISK, R̂(K̂_RISK).

[2026-07 복원] BIC companion(K̂_BIC, BIC(K̂_BIC))과 L1 잔여에너지 진단(Ê^{1/2})은
논문 축소 시 삭제되었다가 리뷰 대응용 검증 열로 복원됨. 논문 표(tab:order-selection)에는
K̂_RISK·R̂(K̂_RISK)만 실린다. 표본은 현행 fixed-seed 규약(_load, SEED=20260606)을 따른다.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/"src", ROOT/"experiments"):
    sys.path.insert(0, str(p))
import warnings; warnings.simplefilter("ignore")
import numpy as np
from cpoe.backends import CMP_BACKEND, BASIS_BACKEND, default_M
from cpoe.estimators import mom as mom_mod
from cpoe.estimators.cmp_baseline import cmp_mle_fit
from cpoe.estimators.mle import solve_theta_at_baseline, _neg_log_lik_smoothed, _LOG_CLIP, _pad_theta
from cpoe import order_selection as osel
from highk_report import _load

KMAX = 14
N_SIM = 1500
SEED = 20260606  # in-sample 공통 fixed-seed 표본 (highk_report/l2_eval/crossmodel_eval과 동일)

def get_sample(name):
    return _load(name, N_SIM, SEED)

def order_select(sample):
    sample = np.asarray(sample, int); N=len(sample); M=default_M(sample)
    r0 = mom_mod.fit_mom(sample, K=KMAX, project=True)
    th, s2 = osel.momproj_stats(sample, float(r0.lam_hat), float(r0.nu_hat), KMAX, basis=BASIS_BACKEND, M=M)
    kr = osel.select_k_risk(th, s2, N, KMAX)
    R = osel.risk_profile(th, s2, N, KMAX)[kr]
    E = osel.residual_energy_sqrt(th, s2, N, kr, KMAX)
    c = cmp_mle_fit(sample); lam_s=c["lam_hat"]; nu_s=c["nu_hat"]
    loglik={0: float(-_neg_log_lik_smoothed(sample, np.array([lam_s,nu_s]),0,cmp_mod=CMP_BACKEND,basis=BASIS_BACKEND,M=M,log_clip=_LOG_CLIP))}
    for K in range(3,KMAX+1):
        thK,ok = solve_theta_at_baseline(sample, lam_s, nu_s, K, basis=BASIS_BACKEND, M=M)
        eta=np.concatenate([[lam_s,nu_s],_pad_theta(thK,K)])
        loglik[K]=float(-_neg_log_lik_smoothed(sample,eta,K,cmp_mod=CMP_BACKEND,basis=BASIS_BACKEND,M=M,log_clip=_LOG_CLIP))
    kb = osel.select_k_bic(loglik, N)
    BIC = osel.bic_values(loglik, N)[kb]
    return N, kr, kb, R, BIC, E

print(f"{'dataset':18s} N   K_RISK K_BIC  Rhat(Kr)   BIC(Kb)    E^0.5")
rows={}
for name in ["FIFA","Insurance-smoker","S1","S2","S3","S4"]:
    N,kr,kb,R,BIC,E = order_select(get_sample(name))
    rows[name]=(kr,kb,R,BIC,E)
    print(f"{name:18s} {N:4d} {kr:4d} {kb:5d}  {R:9.5f} {BIC:10.1f} {E:8.4f}")
# LaTeX rows (논문 tab:order-selection에는 K̂_RISK, R̂만 사용)
print("\n--- LaTeX (paper: K_RISK & Rhat) ---")
for name in ["FIFA","Insurance-smoker","S1","S2","S3","S4"]:
    kr,kb,R,BIC,E=rows[name]
    print(f"{name:16s} & {kr} & {R:.4f} \\\\")
print("\n--- LaTeX (full, review-only: K_RISK & K_BIC & Rhat & BIC & E^0.5) ---")
for name in ["FIFA","Insurance-smoker","S1","S2","S3","S4"]:
    kr,kb,R,BIC,E=rows[name]
    print(f"{name:16s} & {kr} & {kb} & {R:.4f} & {BIC:.1f} & {E:.4f} \\\\")
