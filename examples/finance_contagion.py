"""Financial contagion in a network of banks, with the same algorithm as the glioma model.

Truth = threshold cascade (a bank fails when its losses on exposures to failed banks exceed its capital),
a different mechanism than the model, so the comparison is not circular.
Task: given the banks hit first, which k banks should a supervisor watch?

    python examples/finance_contagion.py
"""
import numpy as np
from reactrisk.core.benchmark import run, report
from reactrisk.finance import random_bank_network

N, K = 300, 30
W, _ = random_bank_network(N, m=3, seed=7)
Wd = W.toarray()


def true_cascade(shocked, rng):
    capital = rng.uniform(0.3, 1.2, N) * Wd.sum(1).mean() * 0.5
    lgd = rng.uniform(0.4, 1.0, N)                       # loss given default
    failed = np.zeros(N, bool); failed[shocked] = True
    for _ in range(20):
        new = (Wd @ (failed * lgd) > capital) & ~failed
        if not new.any():
            break
        failed |= new
    failed[shocked] = False
    return failed


if __name__ == "__main__":
    out, best, nt = run(W, true_cascade, K, seed=42)
    report("Finance (banks, threshold cascade)", out, best, nt, K, N)
