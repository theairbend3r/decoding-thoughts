import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class StimulusDataset(Dataset):
    """Torch dataset for stimulus images.

    Parameters
    ----------
    x_img:
        A 3-d numpy array of input stimulus image.
    y_lbl:
        A 1-d numpy array of output class labels.
        be "train" or "test".
    img_transforms:
        A dictionary of torchvision transforms.
    class2idx:
        A dictionary that maps class string to integers.

    Attributes
    ----------
    x_img: np.ndarray
        A 3-d numpy array of input stimulus image.
    y_lbl: np.ndarray
        A 1-d numpy array of output class labels.
        be "train" or "test".
    img_transforms: dict
        A dictionary of torchvision transforms.
    class2idx: dict
        A dictionary that maps class string to integers.
    """

    def __init__(
        self, x_img: np.ndarray, y_lbl: np.ndarray, img_transform: dict, class2idx: dict
    ):
        self.x_img = x_img
        self.y_lbl = y_lbl
        self.img_transform = img_transform
        self.class2idx = class2idx

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_img[idx].astype(np.uint8))
        y = torch.tensor(self.class2idx[self.y_lbl[idx]], dtype=torch.long)

        if self.img_transform:
            x = self.img_transform(x)

        return x, y

    def __len__(self):
        assert len(self.x_img) == len(self.y_lbl)
        return len(self.x_img)
