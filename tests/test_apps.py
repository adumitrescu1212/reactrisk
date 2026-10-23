import numpy as np
from reactrisk.glioma.preprocess import initial_condition, diffusion_map, treatment_domain, tissue_from_t1
from reactrisk.glioma.evaluate import simulate, bootstrap_ci, paired_wilcoxon
from reactrisk.glioma.factors import km_median, logrank_p, cox_fit
from reactrisk.finance import random_bank_network, exposure_operator, shock_vector, rank_exposed_banks


def toy_brain():
    tissue = np.full((14, 14, 14), 3, np.uint8)
    labels = np.zeros_like(tissue); labels[5:8, 5:8, 5:8] = 2; labels[4:9, 4:9, 4:9][labels[4:9, 4:9, 4:9] == 0] = 3
    return tissue, labels


def test_glioma_simulation_grows_and_stays_bounded():
    tissue, labels = toy_brain()
    mask = treatment_domain(tissue, np.zeros(tissue.shape, bool))   # no resection
    u0 = initial_condition(labels, mask)
    u = simulate(tissue, mask, u0, 0.1, 0.02, 70)
    assert 0 <= u.min() and u.max() <= 1 + 1e-9
    assert (u >= 0.16).sum() > (u0 >= 0.16).sum()


def test_domain_excludes_cavity():
    tissue, labels = toy_brain()
    m = treatment_domain(tissue, labels == 2)
    assert not m[6, 6, 6] and m[0, 0, 0]


def test_diffusion_map():
    D = diffusion_map(np.array([2, 3]), 0.1, 10)
    assert np.allclose(D, [0.01, 0.1])


def test_kmeans_orders_classes():
    rng = np.random.default_rng(0); t1 = rng.choice([30., 100., 180.], (20, 20, 20)) + rng.normal(0, 3, (20, 20, 20))
    tis = tissue_from_t1(t1, np.ones(t1.shape, bool), np.zeros(t1.shape))
    assert np.all(tis[t1 < 60] == 1) and np.all(tis[t1 > 150] == 3)


def test_stats_helpers():
    rng = np.random.default_rng(0)
    t1, t2 = rng.exponential(20, 60), rng.exponential(45, 60); t = np.r_[t1, t2]; e = np.ones(120, int)
    assert km_median(t1, e[:60]) < km_median(t2, e[:60])
    assert logrank_p(t1, e[:60], t2, e[:60]) < 0.01
    coef, p = cox_fit(np.r_[np.zeros(60), np.ones(60)][:, None], t, e)
    assert coef[0] < 0 and p[0] < 0.01
    m, lo, hi = bootstrap_ci([0.2, 0.4, 0.6, 0.8]); assert lo <= m <= hi
    assert paired_wilcoxon([1, 2, 3], [1, 2, 3]) == 1.0


def test_finance_shock_spreads_and_ranking_prefers_neighbours():
    W, deg = random_bank_network(120, 3, seed=1)
    A = exposure_operator(W)
    u0 = shock_vector(120, [0, 1], 0.8)
    zone, u = rank_exposed_banks(A, u0, rho=0.2, horizon=6, k=20)
    assert zone.sum() == 20 and zone[0] and zone[1]
    assert 0 <= u.min() and u.max() <= 1 + 1e-9
    assert (u > 0.01).sum() > 2
