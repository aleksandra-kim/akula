import numpy as np
import bw_processing as bwp

from akula.parameterization import PARAMS_DTYPE


def get_mask(all_indices, use_indices, is_params=False):
    """Creates a `mask` such that `all_indices[mask]=use_indices`."""
    if is_params:
        use_indices = np.array(use_indices, dtype=PARAMS_DTYPE)
    else:
        use_indices = np.array(use_indices, dtype=bwp.INDICES_DTYPE)
    mask = np.zeros(len(all_indices), dtype=bool)
    for indices in use_indices:
        mask_current = all_indices == indices
        mask = mask | mask_current
    return mask
