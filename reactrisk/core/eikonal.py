"""Eikonal (geometric-optics) limit of Fisher-KPP: arrival time of the front, without solving the PDE.

For du/dt = A u + rho u (1 - u) the front moves locally with speed c = 2 sqrt(D rho).
On a grid w_ij = D_ij / h^2, so the time to cross the edge (i, j) is h / c_ij = 1 / (2 sqrt(rho w_ij)).
The same formula works on any graph whose operator A has off-diagonal weights w_ij.

Arrival time T = shortest-path distance (Dijkstra) from the source set with these edge costs.
Cells that already hold a lower density start late: from level a to level b the logistic phase
takes about ln(b / a) / rho, which is used as a start delay.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph


def edge_times(A, rho):
    """Sparse matrix of crossing times 1 / (2 sqrt(rho w_ij)) for every off-diagonal w_ij > 0 of A."""
    W = sp.csr_matrix(A, dtype=float).copy()
    W.setdiag(0)
    W.eliminate_zeros()
    W.data = 1.0 / (2.0 * np.sqrt(rho * W.data))
    return W


def arrival_time(A, rho, u0, level=None):
    """Arrival time of the front at every cell.

    A     : diffusion operator (grid_operator or graph_operator), n x n
    rho   : growth rate
    u0    : initial density in [0, 1]; cells with u0 > 0 are sources
    level : density that defines 'arrived' (default: max(u0)); a source with density a < level
            starts with delay ln(level / a) / rho.
    Returns an array of length n (inf where unreachable).
    """
    u0 = np.asarray(u0, float)
    n = u0.size
    lev = float(u0.max()) if level is None else float(level)
    src = np.flatnonzero(u0 > 0)
    delay = np.log(lev / np.minimum(u0[src], lev)) / rho
    G = edge_times(A, rho)
    # super-source with index n, connected to every source with its delay
    rows = np.concatenate([G.tocoo().row, np.full(src.size, n)])
    cols = np.concatenate([G.tocoo().col, src])
    vals = np.concatenate([G.tocoo().data, delay + 1e-12])
    Gs = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    d = csgraph.dijkstra(Gs, directed=True, indices=n)
    return d[:n]


def eikonal_score(A, rho, u0, level=None):
    """Risk score for top_k: earlier arrival = higher risk."""
    T = arrival_time(A, rho, u0, level)
    return -np.where(np.isfinite(T), T, 1e30)
