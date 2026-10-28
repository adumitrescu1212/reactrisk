"""Spread of information (or adoption of a product) in a social network with communities.

Truth = independent cascade (Kempe, Kleinberg, Tardos 2003): every newly active person activates each
friend once, with probability 1 - exp(-beta * w_ij), where w_ij is the strength of the tie.
Task: given the first 3 people who share a message, which k people should a campaign (or a moderator)
reach first?

    python examples/social_cascade.py
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from reactrisk.core.benchmark import run, report

N, K, BETA = 300, 30, 0.13
rng0 = np.random.default_rng(3)
G = nx.stochastic_block_model([60] * 5, [[0.10 if a == b else 0.004 for b in range(5)] for a in range(5)], seed=3)
W = sp.lil_matrix((N, N))
for i, j in G.edges():
    W[i, j] = W[j, i] = rng0.lognormal(0.0, 0.7)       # tie strength
W = W.tocsr()
Wd = W.toarray()


def true_cascade(seeds, rng):
    active = np.zeros(N, bool); active[seeds] = True
    frontier = list(seeds)
    while frontier:
        nxt = []
        for u in frontier:
            nb = np.flatnonzero(Wd[u])
            p = 1 - np.exp(-BETA * Wd[u, nb])
            for v in nb[(rng.random(nb.size) < p) & ~active[nb]]:
                active[v] = True; nxt.append(v)
        frontier = nxt
    active[seeds] = False
    return active


if __name__ == "__main__":
    out, best, nt = run(W, true_cascade, K, seed=11)
    report("Social (communities, independent cascade)", out, best, nt, K, N)
