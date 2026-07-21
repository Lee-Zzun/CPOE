"""tab:order-selection (6 데이터셋) 계산: K̂_RISK, R̂(K̂_RISK)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/"src", ROOT/"experiments"):
    sys.path.insert(0, str(p))
import warnings; warnings.simplefilter("ignore")
import numpy as np
from cpoe.backends import BASIS_BACKEND, default_M
from cpoe.estimators import mom as mom_mod
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
    return N, kr, R

print(f"{'dataset':18s} N   K_RISK  Rhat(Kr)")
rows={}
for name in ["FIFA","Insurance-smoker","S1","S2","S3","S4"]:
    N,kr,R = order_select(get_sample(name))
    rows[name]=(kr,R)
    print(f"{name:18s} {N:4d} {kr:4d}  {R:9.5f}")
# LaTeX rows
print("\n--- LaTeX ---")
for name in ["FIFA","Insurance-smoker","S1","S2","S3","S4"]:
    kr,R=rows[name]
    print(f"{name:16s} & {kr} & {R:.4f} \\\\")
