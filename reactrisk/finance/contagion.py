"""Same algorithm, different world: financial contagion in a network of banks.

Mapping from the glioma model:
  voxel                -> bank
  cell density u       -> level of distress of the bank, 0 (healthy) .. 1 (failed)
  diffusion between    -> losses transmitted through interbank exposures W_ij
  neighbouring voxels
  proliferation rho    -> self-reinforcement of distress (fire sales, loss of confidence)
  tumour at diagnosis  -> initial shock (banks hit first)
  zone of V cm^3       -> the k banks a regulator can monitor or recapitalise with a given budget
  recurrence           -> banks that really fail later
  'uniform margin'     -> hops from the shocked banks (distance-based baseline)
"""
import numpy as np
import scipy.sparse as sp
from ..core.operators import graph_operator
from ..core.solver import IMEXSolver
from ..core.ranking import top_k


def random_bank_network(n=200, m=3, seed=0):
    """Scale-free interbank network (Barabasi-Albert) with log-normal exposures.
    Returns a symmetric sparse weight matrix W and the list of bank sizes."""
    import networkx as nx
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    W = sp.lil_matrix((n, n))
    for i, j in G.edges():
        W[i, j] = W[j, i] = rng.lognormal(0.0, 0.8)
    return W.tocsr(), np.array([G.degree(i) for i in range(n)])


def exposure_operator(W, scale=1.0):
    """Diffusion operator from exposures, normalised so that the strongest bank has total weight ``scale`` per unit time."""
    A = graph_operator(W)
    return A * (scale / float(np.max(-A.diagonal())))


def shock_vector(n, shocked, level=0.8):
    u0 = np.zeros(n)
    u0[list(shocked)] = level
    return u0


def simulate_cascade(A, u0, rho, horizon, dt=0.25):
    """Distress level of every bank at the horizon (same IMEX solver as the glioma model)."""
    s = IMEXSolver(A, rho, dt, tol=1e-9)
    return s.run(u0, int(round(horizon / dt)))


def rank_exposed_banks(A, u0, rho, horizon, k, dt=0.25):
    """The k banks with the highest predicted distress: the 'zone' to monitor."""
    u = simulate_cascade(A, u0, rho, horizon, dt)
    return top_k(u, k), u
