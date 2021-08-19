"""
Function for noise analysis.
"""

import torch
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.ml.test import test_model
from src.dataset.kay import load_dataset
from src.ml.dataset import StimulusDataset
from src.ml.model import StimulusClassifier
from src.utils.util import prepare_stimulus_data


def stim_noise_test(blur_level_list, config, class2idx) -> list:
    """
    Runs the test loop with increasing levels of noise in the
    dataset.

    Parameters
    ----------
    blur_limit_list:
        List of blur levels.
    config:
        Configuration object (for fmri or stmim).
    class2idx:
        A mapping between class and it's corresponding integer idx.

    Returns
    -------
    list
        A list of tuples containing (blur_level, model_name, model_acc).
    """
    blur_model_acc_list = []

    # load data
    all_data = load_dataset(data_path="./../data/")

    # test data
    x_test, y_test = prepare_stimulus_data(
        all_data=all_data,
        data_subset="test",
        class_ignore_list=config.class_ignore_list,
        label_level=config.label_level,
    )

    # Calculate blur level for all models given a blur value.
    for blur in blur_level_list:

        # define image transform
        blur_transform = A.Compose(
            [
                A.Blur(blur_limit=blur, p=1),
                A.Normalize(mean=0.4599, std=0.2172),
                ToTensorV2(),
            ]
        )

        # define dataset and dataloader.
        test_dataset = StimulusDataset(
            x_data=x_test,
            y_data=y_test,
            img_transform=blur_transform,
            class2idx=class2idx,
        )

        # dataloader with a batch size of 1.
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

        # loop through all model names.
        for model_name in tqdm(config.model_names):

            # load model on CPU in eval() mode.
            model = StimulusClassifier(
                num_classes=len(class2idx), model_name=model_name
            )
            model.load_state_dict(
                torch.load(
                    f"./../models/stimulus_classifier/stim_classifier_model_{model_name}.pth",
                    map_location="cpu",
                )
            )

            model.eval()

            # calculate accuracy for stimulus model predictions
            y_true_list, y_pred_list = test_model(
                model=model, test_loader=test_loader, device="cpu"
            )

            model_acc = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)

            # append values to list and return list
            blur_model_acc_list.append((blur, model_name, model_acc))

    return blur_model_acc_list
