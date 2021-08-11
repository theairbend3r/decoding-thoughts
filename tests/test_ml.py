import torch
import numpy as np
from torchvision import transforms
from src.ml.dataset import StimulusDataset
from src.ml.model import StimulusClassifier
from src.ml.utils import calc_multi_acc

num_output_classes = 10
x = np.random.randint(low=0, high=255, size=(1300, 128, 128))
y = np.random.randint(low=0, high=num_output_classes, size=(1300,))

dataset_mean = 0.5
dataset_std = 0.5

img_transform = {
    "train": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)]
    ),
    "test": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)]
    ),
}

class2idx = {k: i for i, k in enumerate(np.unique(y))}
idx2class = {v: k for k, v in class2idx.items()}

x_tensor = torch.rand((32, 1, 128, 128))
y_tensor = torch.rand((32, 1))

y_pred = torch.tensor(
    [[0.1218, 0.2507, 0.3613, 0.0625, 0.1943], [0.2231, 0.8644, 0.1703, 0.2545, 0.4967]]
)

y_true = torch.tensor([2, 1], dtype=torch.long)


class TestDataset:
    def test_stimulus_dataset(self):
        stimulus_dataset = StimulusDataset(
            x_img=x,
            y_lbl=y,
            img_transform=img_transform["train"],
            class2idx=class2idx,
        )

        assert type(stimulus_dataset[0]) == tuple


class TestModel:
    def test_stimulus_model(self):
        stimulus_model = StimulusClassifier(num_classes=len(class2idx))

        assert stimulus_model(x_tensor).shape == (32, len(class2idx))


class TestUtils:
    def test_multi_acc(self):
        assert int(calc_multi_acc(y_pred, y_true).item()) == 100
