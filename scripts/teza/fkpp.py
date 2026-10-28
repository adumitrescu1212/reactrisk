"""Solver Fisher-KPP 3D: diferente finite pe voxeli, difuzie implicita (Euler), reactie explicita."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def build_operator(D, mask, h):
    """A ~ div(D grad .) pe voxelii din `mask`, flux nul la frontiera (Neumann).
    D: array 3D (mm^2/zi), mask: bool 3D, h: pas (mm). Interfete: medie armonica."""
    idx = -np.ones(mask.shape, dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    rows, cols, vals = [], [], []
    n = int(mask.sum())
    for ax in range(3):
        sl_a = [slice(None)]*3; sl_b = [slice(None)]*3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        sl_a = tuple(sl_a); sl_b = tuple(sl_b)
        both = mask[sl_a] & mask[sl_b]
        Da, Db = D[sl_a][both], D[sl_b][both]
        w = 2*Da*Db/(Da+Db)/h**2          # medie armonica
        ia, ib = idx[sl_a][both], idx[sl_b][both]
        rows += [ia, ib, ia, ib]; cols += [ib, ia, ia, ib]
        vals += [w, w, -w, -w]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, idx

class Solver:
    def __init__(self, D, mask, h, rho, dt, use_cg=True, tol=1e-10):
        self.A, self.idx = build_operator(D, mask, h)
        self.mask, self.rho, self.dt = mask, rho, dt
        n = self.A.shape[0]
        self.M = (sp.identity(n, format='csr') - dt*self.A).tocsr()
        self.use_cg, self.tol = use_cg, tol
        if use_cg:
            self.P = sp.diags(1.0/self.M.diagonal())
        else:
            self.lu = spla.splu(self.M.tocsc())
        self.iters = []
    def step(self, u):
        rhs = u + self.dt*self.rho*u*(1-u)
        if not self.use_cg:
            return self.lu.solve(rhs)
        cnt = [0]
        def cb(_): cnt[0] += 1
        x, info = spla.cg(self.M, rhs, x0=u, rtol=self.tol, atol=0, M=self.P, callback=cb)
        assert info == 0
        self.iters.append(cnt[0]); return x
    def run(self, u0, T):
        u = u0.copy(); n = int(round(T/self.dt))
        for _ in range(n): u = self.step(u)
        return u
    def to_vol(self, u):
        v = np.zeros(self.mask.shape); v[self.mask] = u; return v
