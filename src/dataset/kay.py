"""
Functions to access the dataset.
"""

import os
import requests
import numpy as np
from tqdm import tqdm


def download_data(data_path: str):
    """Download the dataset.

    Parameters
    ----------
    data_path: str
        Path to save the dataset.

    """
    fnames = ["kay_labels.npy", "kay_labels_val.npy", "kay_images.npz"]
    fnames = [f"{data_path}" + file for file in fnames]

    urls = [
        "https://osf.io/r638s/download",
        "https://osf.io/yqb3e/download",
        "https://osf.io/ymnjv/download",
    ]

    if len(os.listdir(data_path)) == len(fnames):
        print("The files are already downloaded.")
    else:
        for i in tqdm(range(len(urls))):
            if not os.path.isfile(fnames[i]):
                try:
                    r = requests.get(urls[i])
                except requests.ConnectionError:
                    print("Download failed.")
                else:
                    if r.status_code != requests.codes.ok:
                        print("Download failed.")
                    else:
                        with open(fnames[i], "wb") as fid:
                            fid.write(r.content)


def load_dataset(data_path: str):
    """Load the dataset.

    Parameters
    ----------
    data_path: str
        Path to load the dataset from.

    """
    # download data if not already present.
    if len(os.listdir(data_path)) == 0:
        download_data(data_path)

    with np.load(f"{data_path}/kay_images.npz") as dobj:
        all_data = dict(**dobj)

    train_labels = np.load(f"{data_path}/kay_labels.npy")
    val_labels = np.load(f"{data_path}/kay_labels_val.npy")

    all_data["train_labels"] = train_labels.T
    all_data["test_labels"] = val_labels.T

    return all_data
