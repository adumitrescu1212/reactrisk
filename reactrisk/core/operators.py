"""Diffusion operators A with zero row sums (no flux through the boundary)."""
import numpy as np
import scipy.sparse as sp


def grid_operator(D, mask, h):
    """Finite-volume operator for div(D grad u) on the voxels where ``mask`` is True.

    (A u)_i = sum_j w_ij (u_j - u_i),  w_ij = 2 D_i D_j / (D_i + D_j) / h^2
    (harmonic mean of D across each face, Neumann condition at the border of the domain).

    D    : array, same shape as ``mask`` (diffusivity, length^2 / time)
    mask : bool array, voxels that belong to the domain
    h    : voxel size (length)
    Returns (A, idx): sparse (n, n) matrix and an array that maps voxel -> row (-1 outside).
    """
    mask = np.asarray(mask, bool)
    D = np.asarray(D, float)
    idx = -np.ones(mask.shape, dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    n = int(mask.sum())
    rows, cols, vals = [], [], []
    for ax in range(mask.ndim):
        a = [slice(None)] * mask.ndim
        b = [slice(None)] * mask.ndim
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        both = mask[a] & mask[b]
        Da, Db = D[a][both], D[b][both]
        w = 2 * Da * Db / (Da + Db) / h ** 2
        ia, ib = idx[a][both], idx[b][both]
        rows += [ia, ib, ia, ib]
        cols += [ib, ia, ia, ib]
        vals += [w, w, -w, -w]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, idx


def graph_operator(W):
    """Diffusion operator on a weighted graph: (A u)_i = sum_j W_ij (u_j - u_i).

    W : symmetric sparse (or dense) matrix of non-negative weights, zero diagonal.
    This is minus the graph Laplacian, so it has the same properties as ``grid_operator``.
    """
    W = sp.csr_matrix(W, dtype=float)
    if (abs(W - W.T) > 1e-12).nnz:
        raise ValueError("W must be symmetric")
    if W.data.size and W.data.min() < 0:
        raise ValueError("weights must be non-negative")
    W.setdiag(0)
    W.eliminate_zeros()
    deg = np.asarray(W.sum(axis=1)).ravel()
    return (W - sp.diags(deg)).tocsr()
