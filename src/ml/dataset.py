import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


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
        x = Image.fromarray(self.x_img[idx].astype(np.uint8)).convert("RGB")
        y = torch.tensor(self.class2idx[self.y_lbl[idx]], dtype=torch.long)

        if self.img_transform:
            x = self.img_transform(image=np.array(x))["image"]

        return x, y

    def __len__(self):
        assert len(self.x_img) == len(self.y_lbl)
        return len(self.x_img)


def calculate_mean_std(dataset):
    """Calculate dataset mean and standard deviation.

    Parameters
    ----------
    dataset:
        Pytorch dataset object.

    Returns
    -------
    tuple
        A tuple that consists a tuple of mean and std for 3 channels (mean, std).
    """

    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    img_size = dataset[0][0].shape[1]

    pixel_sum = torch.tensor([0.0, 0.0, 0.0])
    pixel_sum_squared = torch.tensor([0.0, 0.0, 0.0])

    for img, _ in tqdm(data_loader):
        pixel_sum += img.sum(axis=[0, 2, 3])
        pixel_sum_squared += (img ** 2).sum(axis=[0, 2, 3])

    total_num_pixels = img_size * img_size * len(data_loader)
    total_mean = pixel_sum / total_num_pixels
    total_var = (pixel_sum_squared / total_num_pixels) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean, total_std


class FMRIDataset(Dataset):
    """Torch dataset for stimulus images.

    Parameters
    ----------
    x_data:
        A 2-d numpy array of input fmri data.
    y_data:
        A 1-d numpy array of output class labels.
    class2idx:
        A dictionary that maps class string to integers.

    Attributes
    ----------
    x_data:
        A 2-d numpy array of input fmri data.
    y_data:
        A 1-d numpy array of output class labels.
    class2idx:
        A dictionary that maps class string to integers.
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, class2idx: dict):
        self.class2idx = class2idx
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data, dtype=torch.float)[idx]
        y = torch.tensor(self.class2idx[self.y_data[idx]], dtype=torch.long)

        return x, y

    def __len__(self):
        assert len(self.x_data) == len(self.y_data)
        return len(self.x_data)
