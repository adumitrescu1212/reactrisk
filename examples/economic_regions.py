"""An economic shock that travels between regions through trade.

Regions sit on a map; trade between two regions follows the gravity law w_ij ~ P_i P_j / d_ij^2
(each region trades with its 6 nearest neighbours). Truth = linear threshold model (Granovetter 1978):
a region enters recession when the share of its trade with regions already in recession exceeds its
own threshold theta_i ~ U(0.1, 0.4). Task: given the first 3 regions hit, which k regions should a
government support first?

    python examples/economic_regions.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from reactrisk.core.benchmark import run, report

N, K = 250, 25
rng0 = np.random.default_rng(5)
xy = rng0.random((N, 2))
P = rng0.lognormal(0.0, 1.0, N)                          # size of the regional economy
_, nb = cKDTree(xy).query(xy, k=7)
W = sp.lil_matrix((N, N))
for i in range(N):
    for j in nb[i, 1:]:
        d = np.linalg.norm(xy[i] - xy[j]) + 1e-3
        W[i, j] = W[j, i] = P[i] * P[j] / d ** 2
W = W.tocsr(); W = W / W.data.mean()
Wd = W.toarray(); share = Wd / Wd.sum(1, keepdims=True)


def true_cascade(seeds, rng):
    theta = rng.uniform(0.1, 0.4, N)
    hit = np.zeros(N, bool); hit[seeds] = True
    for _ in range(50):
        new = (share @ hit > theta) & ~hit
        if not new.any():
            break
        hit |= new
    hit[seeds] = False
    return hit


if __name__ == "__main__":
    out, best, nt = run(W, true_cascade, K, seed=21)
    report("Economy (regions, gravity trade, threshold)", out, best, nt, K, N)
