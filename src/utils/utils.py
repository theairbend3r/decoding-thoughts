"""
Utility functions.
"""

import numpy as np


def filter_by_roi(all_data: dict, roi: int) -> np.ndarray:
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


def convert_arr_to_img(stimulus_img_arr: np.ndarray) -> np.ndarray:
    """Change the range of a normalized image to [0-255].

    Parameters
    ----------
    stimulus_img_arr:
        Normalized image array.

    Returns
    -------
    np.ndarray
        Un-normalized image array.

    Raises
    ------
    ValueError
        If `stimulus_img_arr` is not a numpy array.
    """

    # Change scale of stimulus to [0-255]
    img_transformed = np.zeros((stimulus_img_arr.shape[0], 128, 128))
    for i in range(stimulus_img_arr.shape[0]):
        img = stimulus_img_arr[i, :, :]
        img = ((img - np.min(img)) * 255 / (np.max(img) - np.min(img))).astype(int)
        img_transformed[i, :, :] = img

    return img_transformed
