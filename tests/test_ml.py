import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.ml.utils import calc_multi_acc
from src.ml.model import StimulusClassifier
from src.ml.dataset import FMRIDataset, StimulusDataset, calculate_mean_std


batch_size = 32
num_output_classes = 10


####################################
# Classification Output Data
####################################

y_lbl = np.random.randint(low=0, high=num_output_classes, size=(1300,))

y_pred = torch.tensor(
    [[0.1218, 0.2507, 0.3613, 0.0625, 0.1943], [0.2231, 0.8644, 0.1703, 0.2545, 0.4967]]
)

y_true = torch.tensor([2, 1], dtype=torch.long)


class2idx = {k: i for i, k in enumerate(np.unique(y_lbl))}
idx2class = {v: k for k, v in class2idx.items()}

####################################
# Stimulus Input Data
####################################

# prepare_stimulus_data() output
x_img = np.random.randint(low=0.0, high=255.0, size=(1300, 128, 128))

img_transform = {
    "train": A.Compose(
        [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
    ),
    "test": A.Compose(
        [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
    ),
}

####################################
# Stimulus fMRI Data
####################################

# prepare_fmri_data() output
x_fmri = np.random.randint(low=-1.0, high=1.0, size=(1300, 8000))


# pytorch dataloader output
# (batch, channel, height, width)
x_tensor = torch.rand((batch_size, 3, 128, 128))
# (batch,)
y_tensor = torch.rand((batch_size, 1))


class TestDataset:
    def test_stimulus_dataset(self):
        stimulus_dataset = StimulusDataset(
            x_data=x_img,
            y_data=y_lbl,
            img_transform=img_transform["train"],
            class2idx=class2idx,
        )

        # dataset must return a tuple.
        assert type(stimulus_dataset[0]) == tuple

        # output tuple should have 2 elements
        assert len(stimulus_dataset[0]) == 2

        # tuple must contain tensors
        assert stimulus_dataset[0][0].dtype == torch.float
        assert stimulus_dataset[0][1].dtype == torch.long

        # output tensors in tuple are image (3d) and its corresponding label (0d)
        assert len(stimulus_dataset[0][0].shape) == 3
        assert len(stimulus_dataset[0][1].shape) == 0

    def calculate_mean_std(self):
        stimulus_dataset = StimulusDataset(
            x_data=x_img,
            y_data=y_lbl,
            img_transform=img_transform["train"],
            class2idx=class2idx,
        )

        # output should be a tuple
        assert type(calculate_mean_std(stimulus_dataset)) == tuple

        # output tuple should have 2 elements
        assert len(calculate_mean_std(stimulus_dataset)) == 2

        # tuple must contain tensors
        assert calculate_mean_std(stimulus_dataset)[0].dtype == torch.tensor
        assert calculate_mean_std(stimulus_dataset)[1].dtype == torch.tensor

        # output tensors in tuple must have 3 numbers, one for each dimension
        assert len(calculate_mean_std(stimulus_dataset)[0]) == 3
        assert len(calculate_mean_std(stimulus_dataset)[1]) == 3

    def test_fmri_dataset(self):
        fmri_dataset = FMRIDataset(
            x_data=x_fmri,
            y_data=y_lbl,
            class2idx=class2idx,
        )

        # dataset must return a tuple.
        assert type(fmri_dataset[0]) == tuple

        # output tuple should have 2 elements
        assert len(fmri_dataset[0]) == 2

        # tuple must contain tensors
        assert fmri_dataset[0][0].dtype == torch.float
        assert fmri_dataset[0][1].dtype == torch.long

        # output tensors in tuple are fmri (1d) and its corresponding label (0d)
        assert len(fmri_dataset[0][0].shape) == 1
        assert len(fmri_dataset[0][1].shape) == 0


class TestModel:
    def test_stimulus_model(self):
        stimulus_model = StimulusClassifier(num_channel=3, num_classes=len(class2idx))

        # model output shape should be (batch_size, num_classes)
        assert stimulus_model(x_tensor).shape == (batch_size, len(class2idx))

        # model output should be a tensor
        assert stimulus_model(x_tensor).shape == (batch_size, len(class2idx))


class TestUtils:
    def test_multi_acc(self):
        assert int(calc_multi_acc(y_pred, y_true).item()) == 100
