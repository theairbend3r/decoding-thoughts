"""
Utility functions.
"""

import numpy as np


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
    img_transformed = np.zeros(
        (
            stimulus_img_arr.shape[0],
            stimulus_img_arr.shape[1],
            stimulus_img_arr.shape[2],
        )
    )

    for i in range(stimulus_img_arr.shape[0]):
        img = stimulus_img_arr[i, :, :]
        img = ((img - np.min(img)) * 255 / (np.max(img) - np.min(img))).astype(int)
        img_transformed[i, :, :] = img

    return img_transformed


def filter_voxel_by_roi(all_data: dict, roi_list: list) -> np.ndarray:
    """Filter voxel data based on roi.

    Parameters
    ----------
    all_data:
        Dictionary of numpy arrays which contains all data.
    roi_list:
        List of integers denoting the roi type. Values lie between [1, 7].

    Returns
    -------
    np.ndarray
        A numpy array with indices corresponding to voxel roi.

    Raises
    ------
    ValueError
        If `roi` does not lie between [1, 7].
    """
    for i in roi_list:
        if i not in range(1, 8):
            raise ValueError("roi should lie between [1, 7].")

    final_idx_list = []
    for i in roi_list:
        idx_list = np.where(all_data["roi"] == i)[0]
        final_idx_list.extend(idx_list.tolist())

    return np.where(final_idx_list)[0]


def filter_stimulus_by_class(
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

    # filter boolean idx and perform element-wise multiplication to
    # obtain the resultant boolean idx.
    for i in range(len(class_ignore_list)):
        bool_idx = all_data[key][:, label_level] != class_ignore_list[i]
        bool_idx_arr = bool_idx_arr * bool_idx

    return bool_idx_arr


def prepare_stimulus_data(
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
        A tuple of 2d numpy arrays which contains input image and output class labels.

    Raises
    ------
    ValueError
        If `label_level` does not lie between [0, 3].
    ValueError
        If `data_subset` does not contain either "train" or "test".
    ValueError
        If `class_ignore_list` is empty.
    """

    if label_level not in range(0, 4):
        raise ValueError("label_level can only be 0, 1, 2, or 3.")

    if data_subset not in ["train", "test"]:
        raise ValueError("data_subset can only be 'train' or 'test'.")

    if len(class_ignore_list) == 0:
        raise ValueError("class_ignore_list must have atleast 1 element.")

    bool_idx = filter_stimulus_by_class(
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


def prepare_fmri_data(
    all_data: dict,
    data_subset: str,
    class_ignore_list: list,
    label_level: int,
    roi_select_list: list,
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
    roi_select_list:
        List of integers between [1, 7] that contain the roi id to use.


    Returns
    -------
    tuple
        A tuple of numpy arrays which contains image and output class labels.

    Raises
    ------
    ValueError
        If `label_level` does not lie between [0, 3].
    ValueError
        If `data_subset` does not contain either "train" or "test".
    ValueError
        If `class_ignore_list` is empty.
    """

    if label_level not in range(0, 4):
        raise ValueError("label_level can only be 0, 1, 2, or 3.")

    if data_subset not in ["train", "test"]:
        raise ValueError("data_subset can only be 'train' or 'test'.")

    if len(class_ignore_list) == 0:
        raise ValueError("class_ignore_list must have atleast 1 element.")

    bool_idx = filter_stimulus_by_class(
        all_data=all_data,
        data_subset=data_subset,
        class_ignore_list=class_ignore_list,
        label_level=label_level,
    )

    x_key = "responses" if data_subset == "train" else "responses_test"
    y_key = "train_labels" if data_subset == "train" else "test_labels"

    x = all_data[x_key][bool_idx]
    y = all_data[y_key][:, 0][bool_idx]

    # filter voxels
    roi_idx = filter_voxel_by_roi(all_data=all_data, roi_list=roi_select_list)
    x = x[:, roi_idx]

    return x, y
