"""
Module to summarise data.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_hierarchical_labels(
    all_data: dict, label_level: int, plot: bool = True, fig_size: tuple = (15, 7)
) -> np.ndarray:
    """Bar plots for hierarchical labels.

    0 is the least granular level while 3 is the most granular level.

    Parameters
    ----------
    all_data:
        Dictionary of numpy arrays which contains all data.

    label_level:
        Integer denoting the label level. Possible values are 0, 1, 2 and 3.

    plot:
        If True, plots the graph. Else, returns the label array.

    fig_size: optional
        The size of the plot.

    Returns
    -------
    np.ndarray
        A numpy array with labels of the given level.

    Raises
    ------
    ValueError
        If `label_level` is not 0, 1, 2, or 3.
    """

    if label_level not in [0, 1, 2, 3]:
        raise ValueError("label_level has to be 0, 1, 2, or 3.")

    labels = all_data["train_labels"][:, label_level]

    if plot:
        plt.figure(figsize=fig_size)
        sns.countplot(y=labels)
        plt.title(f"Labels: Level-{label_level}")
        plt.xlabel("Label")

    else:
        return labels
