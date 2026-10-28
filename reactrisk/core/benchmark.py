"""One test bench for every network domain (finance, social, economy).

The truth comes from a *different* mechanism than the model (threshold or independent cascade),
so a good score is not circular. Parameters (rho, horizon) are chosen on the first half of the
scenarios and the scores are reported on the second half.
"""
import numpy as np
import scipy.sparse as sp
from .operators import graph_operator
from .solver import IMEXSolver
from .ranking import tie_aware_coverage, graph_hop_distance
from .eikonal import eikonal_score
from .stats import bootstrap_ci, paired_wilcoxon


def normalised_operator(W):
    """-Laplacian of W, scaled so that the node with the largest total weight has weight 1 per unit time."""
    A = graph_operator(W)
    return A * (1.0 / float(np.max(-A.diagonal())))


def model_score(A, u0, rho, horizon, dt=0.25):
    return IMEXSolver(A, rho, dt, tol=1e-9).run(u0, int(round(horizon / dt)))


def scenarios(n, cascade, n_scen, n_seeds, rng, min_hit=3):
    out = []
    while len(out) < n_scen:
        seeds = rng.choice(n, size=n_seeds, replace=False)
        hit = cascade(seeds, rng)
        if hit.sum() >= min_hit:
            out.append((seeds, hit))
    return out


def run(W, cascade, k, n_scen=200, n_seeds=3, seed=0, level=0.8,
        rhos=(0.05, 0.2, 0.5, 1.0), horizons=(1, 3, 6, 12)):
    """Returns {method: (mean, lo, hi, p_vs_model)} and the chosen (rho, horizon)."""
    W = sp.csr_matrix(W, dtype=float)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    A = normalised_operator(W)
    deg = np.asarray((W > 0).sum(1)).ravel().astype(float)
    adj = (W > 0).astype(float)
    sc = scenarios(n, cascade, n_scen, n_seeds, rng)
    train, test = sc[: n_scen // 2], sc[n_scen // 2:]

    def u0_of(seeds):
        u0 = np.zeros(n); u0[seeds] = level; return u0

    def cov(score, seeds, hit):
        mask = np.ones(n, bool); mask[seeds] = False
        return tie_aware_coverage(score, hit, k, mask)

    grid = [(r, h) for r in rhos for h in horizons]
    best = max(grid, key=lambda g: np.mean([cov(model_score(A, u0_of(s), *g), s, f) for s, f in train]))
    rho_e = best[0]
    res = {m: [] for m in ("model", "eikonal", "hops", "direct", "degree", "static")}
    for s, f in test:
        u0 = u0_of(s)
        res["model"].append(cov(model_score(A, u0, *best), s, f))
        res["eikonal"].append(cov(eikonal_score(A, rho_e, u0), s, f))
        res["hops"].append(cov(graph_hop_distance(adj, np.isin(np.arange(n), s)), s, f))
        res["direct"].append(cov(W @ u0, s, f))
        res["degree"].append(cov(deg, s, f))
        res["static"].append(cov(u0, s, f))
    out = {}
    for m, v in res.items():
        mean, lo, hi = bootstrap_ci(v)
        out[m] = (mean, lo, hi, 1.0 if m == "model" else paired_wilcoxon(res["model"], v))
    return out, best, len(test)


def report(title, out, best, n_test, k, n):
    print(f"{title}: {n} nodes, budget k={k}, {n_test} test scenarios, rho={best[0]}, horizon={best[1]} (chosen on train)")
    for m, (mean, lo, hi, p) in out.items():
        tail = "" if m == "model" else f"   p vs model = {p:.2g}"
        print(f"  {m:8s} coverage {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{tail}")
