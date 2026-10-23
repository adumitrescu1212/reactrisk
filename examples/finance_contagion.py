"""Financial contagion in a network of banks, with the same algorithm as the glioma model.

Ground truth comes from a *different* mechanism than the model (a threshold cascade with random capital
buffers and loss rates), so the comparison is not circular.
Task: given the banks hit first, which k banks should a regulator watch? Score = share of the banks that
really fail later that fall inside the k chosen banks.

    python examples/finance_contagion.py
"""
import numpy as np
from reactrisk.core.ranking import tie_aware_coverage, graph_hop_distance
from reactrisk.glioma.evaluate import bootstrap_ci, paired_wilcoxon
from reactrisk.finance import random_bank_network, exposure_operator, shock_vector, rank_exposed_banks

N, K, N_SCEN = 300, 30, 200
rng = np.random.default_rng(42)
W, degree = random_bank_network(N, m=3, seed=7)
Wd = W.toarray()
A = exposure_operator(W)


def true_cascade(shocked, rng):
    """Threshold cascade: a bank fails when losses on its exposures to failed banks exceed its capital."""
    capital = rng.uniform(0.3, 1.2, N) * Wd.sum(1).mean() * 0.5
    lgd = rng.uniform(0.4, 1.0, N)
    failed = np.zeros(N, bool); failed[shocked] = True
    for _ in range(20):
        loss = Wd @ (failed * lgd)
        new = (loss > capital) & ~failed
        if not new.any():
            break
        failed |= new
    failed[shocked] = False
    return failed


scen = []
while len(scen) < N_SCEN:
    shocked = rng.choice(N, size=3, replace=False)
    failed = true_cascade(shocked, rng)
    if failed.sum() >= 3:
        scen.append((shocked, failed))
train, test = scen[:N_SCEN // 2], scen[N_SCEN // 2:]


def score_model(shocked, failed, rho, horizon):
    mask = np.ones(N, bool); mask[shocked] = False
    _, u = rank_exposed_banks(A, shock_vector(N, shocked), rho, horizon, K)
    return tie_aware_coverage(u, failed, K, mask)


def score_baselines(shocked, failed):
    mask = np.ones(N, bool); mask[shocked] = False
    u0 = shock_vector(N, shocked)
    adj = (W > 0).astype(float)
    return {"hops": tie_aware_coverage(graph_hop_distance(adj, np.isin(np.arange(N), shocked)), failed, K, mask),
            "direct_exposure": tie_aware_coverage(Wd @ u0, failed, K, mask),
            "size (degree)": tie_aware_coverage(degree.astype(float), failed, K, mask),
            "static (shock only)": tie_aware_coverage(u0, failed, K, mask)}


# choose rho and horizon on the first half of the scenarios, report on the second half
grid = [(r, h) for r in (0.05, 0.2, 0.5, 1.0) for h in (1, 3, 6, 12)]
best = max(grid, key=lambda g: np.mean([score_model(s, f, *g) for s, f in train]))
print(f"{N} banks, {K} banks watched, {len(test)} test scenarios; rho={best[0]}, horizon={best[1]} chosen on train")
model = [score_model(s, f, *best) for s, f in test]
base = {k: [] for k in ("hops", "direct_exposure", "size (degree)", "static (shock only)")}
for s, f in test:
    for k, v in score_baselines(s, f).items():
        base[k].append(v)
m, lo, hi = bootstrap_ci(model)
print(f"{'reaction-diffusion':22s} coverage {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
for k, v in base.items():
    mm, l, h = bootstrap_ci(v)
    print(f"{k:22s} coverage {mm:.3f}  95% CI [{l:.3f}, {h:.3f}]   paired Wilcoxon p vs model = {paired_wilcoxon(model, v):.3g}")
