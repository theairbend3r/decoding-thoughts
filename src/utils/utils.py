"""
Utility functions.
"""

import numpy as np


def filter_by_roi(all_data: dict, roi: int):
    """Filter data by voxel roi.

    Parameters
    ----------
    all_data:
        Dictionary of numpy arrays which contains all data.
    roi:
        Integer denoting the roi type. Values lie between [1, 7].


    Returns
    -------
    np.ndarray
        A numpy array with indices corresponding to voxel roi.

    Raises
    ------
    ValueError
        If `roi` does not lie between [1, 7].
    """
    if roi not in range(1, 8):
        raise ValueError("roi should like between [1, 7].")

    return np.where(all_data["roi"] == roi)[0]
