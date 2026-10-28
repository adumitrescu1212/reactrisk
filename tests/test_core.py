import numpy as np, scipy.sparse as sp
from reactrisk import (grid_operator, graph_operator, IMEXSolver, explicit_stability_limit,
                       top_k, coverage, tie_aware_coverage, consensus)


def line(n=60, D=1.0, h=1.0):
    A, _ = grid_operator(np.full(n, D), np.ones(n, bool), h)
    return A


def test_rows_sum_to_zero_and_symmetric():
    rng = np.random.default_rng(0)
    D = rng.uniform(0.01, 1, (6, 7, 5)); mask = rng.random(D.shape) > 0.2
    A, idx = grid_operator(D, mask, 2.0)
    assert abs(A.sum(axis=1)).max() < 1e-12
    assert abs(A - A.T).max() < 1e-12
    assert A.shape[0] == mask.sum() and (idx >= 0).sum() == mask.sum()


def test_homogeneous_matches_second_difference():
    A = line(10, D=0.5, h=2.0).toarray()
    assert np.isclose(A[3, 3], -2 * 0.5 / 4) and np.isclose(A[3, 4], 0.5 / 4)
    assert np.isclose(A[0, 0], -0.5 / 4)                      # Neumann border: one neighbour only


def test_harmonic_mean_across_interface():
    D = np.array([0.1, 0.01]); A, _ = grid_operator(D, np.ones(2, bool), 1.0)
    assert np.isclose(A[0, 1], 2 * 0.1 * 0.01 / 0.11)          # ~0.018, not the arithmetic mean 0.055


def test_graph_operator_properties():
    W = sp.random(30, 30, density=0.2, random_state=1); W = (W + W.T).tocsr()
    A = graph_operator(W)
    assert abs(A.sum(axis=1)).max() < 1e-12 and abs(A - A.T).max() < 1e-12


def test_M_is_spd_and_condition_bound():
    A = line(50, D=0.3, h=1.0); dt = 7.0
    M = (sp.identity(50) - dt * A).toarray()
    ev = np.linalg.eigvalsh(M)
    assert ev.min() >= 1 - 1e-9
    assert ev.max() / ev.min() <= 1 + 4 * dt * 0.3 / 1.0 + 1e-9    # 1D version of the bound (4 instead of 12)


def test_pure_diffusion_conserves_mass_and_cg_equals_lu():
    A = line(80, D=0.4); u0 = np.zeros(80); u0[35:45] = 1
    a = IMEXSolver(A, 0.0, 1.0, tol=1e-12, method="cg").run(u0, 30)
    b = IMEXSolver(A, 0.0, 1.0, method="lu").run(u0, 30)
    assert np.allclose(a, b, atol=1e-8) and np.isclose(a.sum(), u0.sum())


def test_bounds_preserved_when_dt_rho_le_1():
    A = line(80, D=0.4); rng = np.random.default_rng(2); u0 = rng.random(80)
    u = IMEXSolver(A, 0.14 / 7, 7.0).run(u0, 50, snapshots=True)
    assert min(x.min() for x in u) >= -1e-9 and max(x.max() for x in u) <= 1 + 1e-9


def test_front_speed_matches_fisher_kpp_theory():
    """Asymptotic speed is 2 sqrt(D rho); at finite time the front lags (Bramson correction):
    x(t) = 2 sqrt(D rho) t - 1.5 sqrt(D / rho) ln t, so the speed between t1 and t2 is
    2 sqrt(D rho) - 1.5 sqrt(D / rho) ln(t2 / t1) / (t2 - t1)."""
    D, rho, h, dt = 0.5, 0.05, 0.5, 0.5
    n = int(300 / h); A = line(n, D, h); u0 = np.zeros(n); u0[:int(5 / h)] = 1
    s = IMEXSolver(A, rho, dt, tol=1e-10)
    pos = [np.argmax(s.run(u0, int(T / dt)) < 0.5) * h for T in (150, 300)]
    speed = (pos[1] - pos[0]) / 150
    expected = 2 * np.sqrt(D * rho) - 1.5 * np.sqrt(D / rho) * np.log(2) / 150
    assert abs(speed - expected) / expected < 0.03


def test_explicit_limit():
    A = line(20, D=1.0, h=1.0)
    assert np.isclose(explicit_stability_limit(A), 0.5)


def test_ranking_and_coverage():
    s = np.array([5., 4, 3, 2, 1]); t = np.array([1, 0, 1, 0, 0], bool)
    z = top_k(s, 2); assert z.tolist() == [True, True, False, False, False]
    assert coverage(z, t) == 0.5


def test_tie_aware_coverage_uniform_scores_equals_fraction():
    t = np.zeros(100, bool); t[:10] = True
    assert np.isclose(tie_aware_coverage(np.ones(100), t, 30), 0.3)


def test_consensus():
    a, b, c = np.array([3., 2, 1, 0]), np.array([3., 1, 2, 0]), np.array([0., 3, 2, 1])
    assert consensus([a, b, c], 2, 2).tolist() == [True, True, True, False]


def test_eikonal_arrival_time_on_a_line():
    from reactrisk.core.eikonal import arrival_time
    D, rho, h = 0.1, 0.02, 2.0
    A = line(40, D=D, h=h)
    u0 = np.zeros(40); u0[0] = 0.8
    T = arrival_time(A, rho, u0)
    c = 2 * np.sqrt(D * rho)
    assert np.allclose(T, np.arange(40) * h / c, rtol=1e-9, atol=1e-6)   # T = distance / speed
    u0[20] = 0.16                                                         # a weaker source starts later
    T2 = arrival_time(A, rho, u0)
    assert np.isclose(T2[20], np.log(0.8 / 0.16) / rho)


def test_network_benchmark_runs():
    import networkx as nx
    from reactrisk.core.benchmark import run
    G = nx.path_graph(30); W = nx.to_scipy_sparse_array(G, format="csr").astype(float)
    def cascade(seeds, rng):
        hit = np.zeros(30, bool)
        for s in seeds:
            hit[max(0, s - 3):s + 4] = True
        hit[seeds] = False
        return hit
    out, best, nt = run(W, cascade, k=6, n_scen=20, n_seeds=1, seed=0, rhos=(0.5,), horizons=(3,))
    assert out["model"][0] > out["degree"][0]
