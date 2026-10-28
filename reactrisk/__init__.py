"""reactrisk: ranking zones of risk with a reaction-diffusion model.

The core (``reactrisk.core``) is domain independent: it works on any sparse
diffusion operator (a voxel grid, a network of banks, a road graph).
``reactrisk.glioma`` and ``reactrisk.finance`` are two applications of it.
"""
from .core.operators import grid_operator, graph_operator
from .core.solver import IMEXSolver, explicit_stability_limit
from .core.ranking import (top_k, coverage, tie_aware_coverage, consensus,
                           grid_distance_score, graph_hop_distance)
from .core.jump import grid_jump_operator, apply_jump
from .core.eikonal import arrival_time, eikonal_score

__all__ = ["grid_operator", "graph_operator", "IMEXSolver", "explicit_stability_limit",
           "top_k", "coverage", "tie_aware_coverage", "consensus",
           "grid_distance_score", "graph_hop_distance", "grid_jump_operator", "apply_jump",
           "arrival_time", "eikonal_score"]
__version__ = "0.1.0"
