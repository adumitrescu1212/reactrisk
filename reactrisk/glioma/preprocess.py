"""From segmented MRI to model inputs (labels: 1 necrosis, 2 enhancing, 3 edema)."""
import numpy as np
from scipy import ndimage as ndi

H_MM = 2.0
VOXEL_MM3 = H_MM ** 3          # 8 mm^3


def core_mask(labels):
    """Tumour core = necrosis + enhancing tumour."""
    labels = np.asarray(labels)
    return (labels == 1) | (labels == 2)


def initial_condition(labels, mask, core_value=0.8, edema_value=0.16):
    """u0: 0.8 in the core, 0.16 in the edema, 0 elsewhere (and outside the domain)."""
    u0 = np.zeros(np.shape(labels))
    u0[np.asarray(labels) == 3] = edema_value
    u0[core_mask(labels)] = core_value
    u0[~np.asarray(mask, bool)] = 0
    return u0


def diffusion_map(tissue, d_white=0.1, ratio=10.0):
    """D = d_white in white matter (tissue == 3), d_white / ratio in grey matter."""
    return np.where(np.asarray(tissue) == 3, d_white, d_white / ratio).astype(float)


def treatment_domain(tissue, removed_core, dilate=1):
    """Voxels where cells can live: brain tissue (grey/white, tissue >= 2) minus the resected core
    (dilated by ``dilate`` voxels, to cover the cavity wall)."""
    cav = ndi.binary_dilation(np.asarray(removed_core, bool), iterations=dilate) if dilate else np.asarray(removed_core, bool)
    return (np.asarray(tissue) >= 2) & ~cav


def tissue_from_t1(t1, brain_mask, tumour_labels, iters=30):
    """Tissue classes by 1-D k-means on T1 intensities: 1 CSF < 2 grey < 3 white;
    voxels that belong to the segmented tumour are assigned to white matter (3)."""
    t1 = np.asarray(t1, float)
    bm = np.asarray(brain_mask, bool)
    v = t1[bm]
    c = np.percentile(v, [15, 50, 85]).astype(float)
    for _ in range(iters):
        lab = np.abs(v[:, None] - c[None]).argmin(1)
        c = np.array([v[lab == k].mean() if (lab == k).any() else c[k] for k in range(3)])
    rank = np.empty(3, int)
    rank[np.argsort(c)] = np.arange(3)
    tissue = np.zeros(bm.shape, np.uint8)
    tissue[bm] = rank[lab] + 1
    tissue[(np.asarray(tumour_labels) > 0) & bm] = 3
    return tissue
