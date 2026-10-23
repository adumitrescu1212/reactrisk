"""Clinical factors versus time to first progression: Kaplan-Meier median, log-rank test, Cox model.
Inputs are plain arrays (time, event) so it works with any cohort; no external survival package needed."""
import numpy as np
from scipy import stats, optimize


def km_median(t, e):
    t, e = np.asarray(t, float), np.asarray(e, int)
    S = 1.0
    for u in np.unique(t[e == 1]):
        n = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        S *= 1 - d / n
        if S <= 0.5:
            return float(u)
    return float("nan")


def logrank_p(t1, e1, t2, e2):
    t, e = np.r_[t1, t2], np.r_[e1, e2]
    g = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    O = E = V = 0.0
    for u in np.unique(t[e == 1]):
        at = t >= u
        n, n1 = at.sum(), (at & (g == 0)).sum()
        d, d1 = ((t == u) & (e == 1)).sum(), ((t == u) & (e == 1) & (g == 0)).sum()
        O += d1
        E += d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    return float(1 - stats.chi2.cdf((O - E) ** 2 / V, 1))


def cox_fit(X, t, e, iters=50):
    """Cox proportional hazards by Newton-Raphson on the partial likelihood (Breslow ties).
    Standard errors from the exact observed information. Returns (coef, p-values)."""
    X, t, e = np.asarray(X, float), np.asarray(t, float), np.asarray(e, int)
    p = X.shape[1]
    b = np.zeros(p)
    times = np.unique(t[e == 1])
    for _ in range(iters):
        w = np.exp(X @ b)
        grad, hess = np.zeros(p), np.zeros((p, p))
        for u in times:
            r = t >= u
            ev = (t == u) & (e == 1)
            d = ev.sum()
            S0, S1 = w[r].sum(), (w[r, None] * X[r]).sum(0)
            S2 = (w[r, None, None] * X[r, :, None] * X[r, None, :]).sum(0)
            grad += X[ev].sum(0) - d * S1 / S0
            hess -= d * (S2 / S0 - np.outer(S1, S1) / S0 ** 2)
        step = np.linalg.solve(-hess, grad)
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    se = np.sqrt(np.diag(np.linalg.inv(-hess)))
    return b, 2 * (1 - stats.norm.cdf(np.abs(b / se)))
