"""Paired statistics used by every application."""
import numpy as np
from scipy import stats


def bootstrap_ci(values, n=5000, alpha=0.05, seed=0):
    """Mean and percentile bootstrap confidence interval."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank p-value (1.0 if all differences are zero)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)
