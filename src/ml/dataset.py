import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


class StimulusDataset(Dataset):
    """Torch dataset for stimulus images.

    Parameters
    ----------
    x_data:
        A 3-d numpy array of input stimulus image.
    y_data:
        A 1-d numpy array of output class labels.
        be "train" or "test".
    img_transforms:
        A dictionary of torchvision transforms.
    class2idx:
        A dictionary that maps class string to integers.

    Attributes
    ----------
    x_data: np.ndarray
        A 3-d numpy array of input stimulus image.
    y_data: np.ndarray
        A 1-d numpy array of output class labels.
        be "train" or "test".
    img_transforms: dict
        A dictionary of torchvision transforms.
    class2idx: dict
        A dictionary that maps class string to integers.
    """

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        img_transform: dict,
        class2idx: dict,
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.img_transform = img_transform
        self.class2idx = class2idx

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].astype(np.uint8)).convert("RGB")
        y = torch.tensor(self.class2idx[self.y_data[idx]], dtype=torch.long)

        if self.img_transform:
            x = self.img_transform(image=np.array(x))["image"]

        return x, y

    def __len__(self):
        assert len(self.x_data) == len(self.y_data)
        return len(self.x_data)


def calculate_mean_std(dataset: Dataset):
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
    """Torch dataset for fMRI data.

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


def create_weighted_sampler(y_data: np.ndarray, class2idx: dict):
    """
    Create weighted random sampler.

    Parameters
    ----------
    y_data:
        The output class labels.
    class2idx:
        A dictionary to convert class strings to integer.
    """
    y_data = np.array([class2idx[t] for t in y_data])
    class_count_list = np.array(
        [len(np.where(y_data == t)[0]) for t in np.unique(y_data)]
    )
    class_weight_list = 1.0 / class_count_list
    samples_weight_list = [class_weight_list[t] for t in y_data]
    class_weight_tensor = torch.tensor(samples_weight_list)
    weighted_random_sampler = WeightedRandomSampler(
        class_weight_tensor, len(class_weight_tensor)
    )

    return weighted_random_sampler
