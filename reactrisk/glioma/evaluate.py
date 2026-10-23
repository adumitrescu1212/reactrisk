"""Evaluation protocol: at equal volume, how much of the real recurrence falls in the predicted zone?

Protocol A: start from the pre-operative tumour, target = first progressive disease (PD, RANO).
Protocol B: start from the first control after surgery, target = first PD.
"""
import numpy as np
from scipy import stats
from ..core.operators import grid_operator
from ..core.solver import IMEXSolver
from ..core.ranking import top_k, coverage, tie_aware_coverage, consensus, grid_distance_score
from ..core.jump import grid_jump_operator, apply_jump
from .preprocess import core_mask, initial_condition, diffusion_map, treatment_domain, H_MM, VOXEL_MM3

VOLUMES_CM3 = (20, 50, 100)
DT_DAYS = 7.0
RHO = 0.02
CONFIGS = {"D0.03": (0.03, None), "D0.1": (0.1, None), "D0.3": (0.3, None),
           "J1": (0.1, (0.002, 10)), "J2": (0.1, (0.006, 10))}      # jump: (rate per day, sigma in voxels)


def simulate(tissue, mask, u0, d_white, rho, days, ratio=10.0, dt=DT_DAYS, jump=None):
    """Run the model for ``days`` days and return u on the full grid."""
    D = diffusion_map(tissue, d_white, ratio)
    A, _ = grid_operator(D, mask, H_MM)
    solver = IMEXSolver(A, rho, dt, tol=1e-8)
    after = None
    if jump:
        K = grid_jump_operator(mask, jump[1])
        after = lambda u: apply_jump(u, K, jump[0], dt)
    u = solver.run(u0[mask], int(round(days / dt)), after_step=after)
    out = np.zeros(mask.shape)
    out[mask] = u
    return out


def run_case(tissue, pre, post, obs, start, target_week, configs=CONFIGS, volumes=VOLUMES_CM3):
    """One (patient, protocol) pair. ``obs`` maps week -> label volume; ``start`` is 'pre' or a week.
    Returns a dict of coverages, or None if the recurrence is empty inside the domain."""
    L0, T0 = (pre, 0) if start == "pre" else (obs[start], start)
    removed = core_mask(pre if start == "pre" else post)
    mask = treatment_domain(tissue, removed)
    u0 = initial_condition(L0, mask)
    O = core_mask(obs[target_week]) & mask
    if O.sum() == 0:
        return None
    days = (target_week - T0) * 7.0
    dist = grid_distance_score(core_mask(L0), H_MM)
    out = {"start": str(start), "target": target_week, "recurrence_cm3": float(O.sum() * VOXEL_MM3 / 1000), "coverage": {}}
    for V in volumes:
        k = int(round(V * 1000 / VOXEL_MM3))
        out["coverage"][f"static_{V}"] = tie_aware_coverage(u0, O, k, mask)
        out["coverage"][f"margin_{V}"] = tie_aware_coverage(dist, O, k, mask)
    m15 = (-dist <= 15) & mask
    out["margin15mm"] = coverage(m15, O)
    out["margin15mm_cm3"] = float(m15.sum() * VOXEL_MM3 / 1000)
    scores = {}
    for name, (dw, jp) in configs.items():
        scores[name] = simulate(tissue, mask, u0, dw, RHO, days, jump=jp)
        for V in volumes:
            out["coverage"][f"{name}_{V}"] = tie_aware_coverage(scores[name], O, int(round(V * 1000 / VOXEL_MM3)), mask)
    for V in volumes:
        k = int(round(V * 1000 / VOXEL_MM3))
        out["coverage"][f"consensus_{V}"] = coverage(consensus([scores[c] for c in ("D0.03", "D0.1", "D0.3")], k, 2, mask), O)
    return out


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
