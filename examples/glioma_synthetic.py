"""Self-contained demo (no patient data): a toy 'brain', a tumour, and a recurrence that grew along white matter."""
import numpy as np
from reactrisk.glioma.preprocess import initial_condition, treatment_domain
from reactrisk.glioma.evaluate import simulate
from reactrisk.core.ranking import tie_aware_coverage, grid_distance_score

n = 40
tissue = np.full((n, n, n), 2, np.uint8)            # grey matter everywhere
tissue[:, :, 18:22] = 3                              # a white-matter tract along x
labels = np.zeros_like(tissue); labels[10:14, 18:22, 18:22] = 2
mask = treatment_domain(tissue, np.zeros(tissue.shape, bool))
u0 = initial_condition(labels, mask)
truth_u = simulate(tissue, mask, u0, 0.1, 0.02, 300)         # "reality": slow spread, mostly along the tract
rec = truth_u >= 0.3                                           # what appears on the later MRI
k = int(rec.sum())
dist = grid_distance_score(labels == 2, 2.0)
pred = simulate(tissue, mask, u0, 0.1, 0.02, 300)
print(f"recurrence = {rec.sum() * 8e-3:.1f} cm3")
print(f"static zone      : {tie_aware_coverage(u0, rec, k, mask):.1%}")
print(f"uniform margin   : {tie_aware_coverage(dist, rec, k, mask):.1%}")
print(f"reaction-diffusion: {tie_aware_coverage(pred, rec, k, mask):.1%}   (same model as the truth, so this is an upper bound)")
