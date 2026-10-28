"""Marginea geodezica (limita eikonala a ecuatiei Fisher-KPP), evaluata cu acelasi protocol ca modelul.

Timpul de sosire T(x) = distanta minima pe graful voxelilor din domeniu, cu costul unei muchii
h / c_ij, c_ij = 2*sqrt(D_ij*rho), D_ij = media armonica.  Zona = cei K voxeli cu T minim.
Variante:  geo_tis  (viteze diferite alb/cenusiu, ca modelul)
           geo_dom  (aceeasi viteza peste tot, dar drumuri doar prin tesut: efectul frontierei)
"""
import numpy as np, json, sys, glob
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy import ndimage as ndi
sys.path.insert(0, '.'); sys.path.insert(0, '/home/claude/reactrisk')
from eval_all import load, corem, tie_cov
H = 2.0; RHO = 0.02; DW = 0.1


def geodesic_time(nodes, D, sources, late=None, tau=0.0):
    """nodes: bool volum (graful), D: difuzivitate, sources: bool volum. Intoarce T pe tot volumul (inf in afara)."""
    idx = -np.ones(nodes.shape, np.int64); idx[nodes] = np.arange(nodes.sum()); n = int(nodes.sum())
    r, c, w = [], [], []
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3; a[ax] = slice(0, -1); b[ax] = slice(1, None)
        a, b = tuple(a), tuple(b)
        both = nodes[a] & nodes[b]
        Da, Db = D[a][both], D[b][both]
        Dij = 2 * Da * Db / (Da + Db)
        cost = H / (2 * np.sqrt(Dij * RHO))
        r.append(idx[a][both]); c.append(idx[b][both]); w.append(cost)
    # nod suplimentar n = sursa comuna: muchie de cost ~0 spre nucleu, cost tau spre edem (pornire intarziata)
    s0 = idx[sources & nodes]; r.append(np.full(s0.size, n)); c.append(s0); w.append(np.full(s0.size, 1e-9))
    if late is not None:
        s1 = idx[late & nodes & ~sources]; r.append(np.full(s1.size, n)); c.append(s1); w.append(np.full(s1.size, tau + 1e-9))
    G = sp.csr_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n + 1, n + 1))
    d = csgraph.dijkstra(G, directed=False, indices=n)
    T = np.full(nodes.shape, np.inf); T[nodes] = d[:n]
    return T


def run_pair(p, start, target):
    tissue, pre, post, obs = load(p)
    L0 = pre if start == 'pre' else obs[int(start)]
    cav = ndi.binary_dilation(corem(pre) if start == 'pre' else corem(post), iterations=1)
    mask = (tissue >= 2) & ~cav
    O = corem(obs[target]) & mask
    if O.sum() == 0:
        return None
    core = corem(L0)
    nodes = (mask | cav | core) & (tissue >= 1)   # domeniul + cavitatea (legatura cu sursele)
    Dtis = np.where(tissue == 3, DW, DW / 10).astype(float); Dtis[core | cav] = DW
    Dcon = np.full(tissue.shape, DW)
    out = {}
    edema = (L0 == 3)
    tau = np.log(0.8 / 0.16) / RHO        # timpul de crestere logistica de la 0,16 la 0,8 (liniarizat)
    for name, D, late in (('geo_tis', Dtis, None), ('geo_dom', Dcon, None), ('geo_e', Dtis, edema)):
        T = geodesic_time(nodes | edema, D, core, late, tau)
        s = -np.where(np.isfinite(T), T, 1e9)
        for V in (20, 50, 100):
            out[f'{name}{V}'] = tie_cov(s, mask, O, V)
    # varianta exacta din pachet: acelasi operator A ca modelul, surse = u0 din domeniu
    from reactrisk.core.operators import grid_operator
    from reactrisk.core.eikonal import eikonal_score
    u0 = np.zeros(tissue.shape); u0[L0 == 3] = 0.16; u0[core] = 0.8; u0[~mask] = 0
    A, _ = grid_operator(np.where(tissue == 3, DW, DW / 10).astype(float), mask, H)
    sc = np.full(tissue.shape, -1e31); sc[mask] = eikonal_score(A, RHO, u0[mask], level=0.8)
    for V in (20, 50, 100):
        out[f'eik{V}'] = tie_cov(sc, mask, O, V)
    return out


if __name__ == '__main__':
    res = {}
    for f in sorted(glob.glob('ev_Patient-*.json')):
        p = f[3:-5]
        for r in json.load(open(f)):
            q = run_pair(p, r['start'], r['target'])
            if q is None:
                continue
            q.update({k: r['Q'][k] for k in r['Q']}); q['Ovol'] = r['Ovol']
            res[f"{p}|{r['start']}"] = q
        print(p, flush=True)
    json.dump(res, open('geo_results.json', 'w'))
