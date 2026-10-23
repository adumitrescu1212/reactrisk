"""IMEX Euler for  du/dt = A u + rho u (1 - u):
diffusion implicit, reaction explicit,   (I - dt A) u^{n+1} = u^n + dt rho u^n (1 - u^n).

Properties (proved in the thesis): the matrix M = I - dt A is a symmetric positive definite M-matrix,
so the scheme is unconditionally stable for the diffusion part and keeps u in [0, 1]
whenever dt * rho <= 1 and u^0 is in [0, 1].
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def explicit_stability_limit(A):
    """Largest dt for which the fully explicit scheme is stable (Gershgorin bound): 1 / max|A_ii|."""
    return 1.0 / float(np.max(-A.diagonal()))


class IMEXSolver:
    def __init__(self, A, rho, dt, tol=1e-8, method="cg"):
        n = A.shape[0]
        self.A, self.rho, self.dt, self.tol, self.method = A, float(rho), float(dt), tol, method
        self.M = (sp.identity(n, format="csr") - dt * A).tocsr()
        if method == "cg":
            self.P = sp.diags(1.0 / self.M.diagonal())      # Jacobi preconditioner
        elif method == "lu":
            self.lu = spla.splu(self.M.tocsc())
        else:
            raise ValueError("method must be 'cg' or 'lu'")
        self.iters = []

    def step(self, u):
        rhs = u + self.dt * self.rho * u * (1 - u)
        if self.method == "lu":
            return self.lu.solve(rhs)
        count = [0]

        def cb(_):
            count[0] += 1
        try:
            x, info = spla.cg(self.M, rhs, x0=u, rtol=self.tol, atol=0, M=self.P, callback=cb)
        except TypeError:                                    # older SciPy
            x, info = spla.cg(self.M, rhs, x0=u, tol=self.tol, M=self.P, callback=cb)
        if info != 0:
            raise RuntimeError("PCG did not converge")
        self.iters.append(count[0])
        return x

    def run(self, u0, n_steps, after_step=None, snapshots=False):
        """Advance n_steps. ``after_step(u) -> u`` can add a term (e.g. the jump model).
        With snapshots=True returns the list [u^0, ..., u^n]."""
        u = np.array(u0, float)
        snaps = [u.copy()] if snapshots else None
        for _ in range(int(n_steps)):
            u = self.step(u)
            if after_step is not None:
                u = after_step(u)
            if snapshots:
                snaps.append(u.copy())
        return snaps if snapshots else u
