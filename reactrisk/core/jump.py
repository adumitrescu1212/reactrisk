"""Optional non-local term (long jumps): u <- (1 - mu) u + mu K u, with K row-stochastic.
Because K is row-stochastic and 0 <= mu <= 1, u stays in [0, 1]."""
import numpy as np
from scipy import ndimage as ndi


def grid_jump_operator(mask, sigma):
    """K = normalised Gaussian blur restricted to the domain ``mask`` (sigma in voxels)."""
    mask = np.asarray(mask, bool)
    den = ndi.gaussian_filter(mask.astype(float), sigma, mode="constant") + 1e-12

    def K(u):
        x = np.zeros(mask.shape)
        x[mask] = u
        return (ndi.gaussian_filter(x, sigma, mode="constant") / den)[mask]
    return K


def apply_jump(u, K, rate, dt):
    mu = min(1.0, rate * dt)
    return (1 - mu) * u + mu * K(u)
