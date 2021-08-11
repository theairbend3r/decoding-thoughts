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


def filter_data_by_class(
    all_data: dict, data_subset: str, class_ignore_list: list, label_level: int
) -> np.ndarray:
    """Filter data by output class label.

    Parameters
    ----------
    all_data:
        Dictionary of numpy arrays which contains all data.
    data_subset:
        A string value that denotes the train/test subset. Can only
        be "train" or "test".
    class_ignore_list:
        A list of class labels to ignore.
    label_level:
        An integer that denotes the label hierarchy level.
        Lies between [0,3].

    Returns
    -------
    np.ndarray
        A numpy array with boolean indices corresponding to
        output class labels.

    Raises
    ------
    ValueError
        If `label_level` does not lie between [0, 3].
    """

    if label_level not in range(0, 4):
        raise ValueError("label_level can only be 0, 1, 2, or 3.")

    if data_subset not in ["train", "test"]:
        raise ValueError("data_subset can only be 'train' or 'test'.")

    if len(class_ignore_list) == 0:
        raise ValueError("class_ignore_list must have atleast 1 element.")

    key = "train_labels" if data_subset == "train" else "test_labels"

    bool_idx_arr = np.ones_like(all_data[key][:, label_level], dtype="bool")

    for i in range(len(class_ignore_list)):
        bool_idx = all_data[key][:, label_level] != class_ignore_list[i]
        bool_idx_arr = bool_idx_arr * bool_idx

    return bool_idx_arr


def prepare_data_arrays(
    all_data: dict, data_subset: str, class_ignore_list: list, label_level: int
) -> tuple:
    """Prepare data for modelling.

    First filters data by class. Then rescales the image to [0-255] scale.

    Parameters
    ----------
    all_data:
        Dictionary of numpy arrays which contains all data.
    data_subset:
        A string value that denotes the train/test subset. Can only
        be "train" or "test".
    class_ignore_list:
        A list of class labels to ignore.
    label_level:
        An integer that denotes the label hierarchy level.
        Lies between [0,3].

    Returns
    -------
    tuple
        A tuple of numpy arrays which contains image and output class labels.

    Raises
    ------
    ValueError
        If `label_level` does not lie between [0, 3].
    """

    if label_level not in range(0, 4):
        raise ValueError("label_level can only be 0, 1, 2, or 3.")

    if data_subset not in ["train", "test"]:
        raise ValueError("data_subset can only be 'train' or 'test'.")

    if len(class_ignore_list) == 0:
        raise ValueError("class_ignore_list must have atleast 1 element.")

    bool_idx = filter_data_by_class(
        all_data=all_data,
        data_subset=data_subset,
        class_ignore_list=class_ignore_list,
        label_level=label_level,
    )

    x_key = "stimuli" if data_subset == "train" else "stimuli_test"
    y_key = "train_labels" if data_subset == "train" else "test_labels"

    x = convert_arr_to_img(stimulus_img_arr=all_data[x_key][bool_idx])
    y = all_data[y_key][:, 0][bool_idx]

    return x, y
