"""From a risk score to a zone of fixed size, and how to judge the zone."""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csgraph


def top_k(score, k, mask=None):
    """Boolean array that marks the k entries with the largest score (inside ``mask``)."""
    score = np.asarray(score)
    mask = np.ones(score.shape, bool) if mask is None else np.asarray(mask, bool)
    idx = np.flatnonzero(mask.ravel())
    order = np.argsort(-score.ravel()[idx], kind="stable")
    out = np.zeros(score.size, bool)
    out[idx[order[:int(k)]]] = True
    return out.reshape(score.shape)


def coverage(zone, target):
    """Fraction of the target set that falls inside the zone."""
    target = np.asarray(target, bool)
    return float((np.asarray(zone, bool) & target).sum() / max(target.sum(), 1))


def tie_aware_coverage(score, target, k, mask=None):
    """Coverage of the top-k zone, ties at the threshold split proportionally
    (so a baseline with many equal scores is not helped or hurt by sorting order)."""
    score = np.asarray(score, float)
    mask = np.ones(score.shape, bool) if mask is None else np.asarray(mask, bool)
    idx = np.flatnonzero(mask.ravel())
    s, t = score.ravel()[idx], np.asarray(target, bool).ravel()[idx]
    k = min(int(k), len(s))
    thr = np.sort(s)[::-1][k - 1]
    hi, eq = s > thr, s == thr
    frac = (k - hi.sum()) / max(eq.sum(), 1)
    return float((t & hi).sum() + frac * (t & eq).sum()) / max(t.sum(), 1)


def consensus(scores, k, min_votes, mask=None):
    """Zone of entries that are in the top-k of at least ``min_votes`` of the given scores
    (e.g. the same model run with three values of D)."""
    votes = sum(top_k(s, k, mask).astype(int) for s in scores)
    return votes >= min_votes


def grid_distance_score(source, spacing=1.0):
    """Baseline 'uniform margin' on a grid: minus the Euclidean distance to the source set."""
    return -ndi.distance_transform_edt(~np.asarray(source, bool)) * spacing


def graph_hop_distance(adjacency, sources):
    """Baseline 'uniform margin' on a graph: minus the number of hops to the nearest source."""
    sources = np.flatnonzero(np.asarray(sources, bool))
    d = csgraph.dijkstra(adjacency, directed=False, unweighted=True, indices=sources, min_only=True)
    return -np.where(np.isinf(d), 1e9, d)
