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


def noise_test(blur_limit_list, config, class2idx):
    blur_model_acc_list = []

    all_data = load_dataset(data_path="./../data/")

    # test data
    x_test, y_test = prepare_stimulus_data(
        all_data=all_data,
        data_subset="test",
        class_ignore_list=config.class_ignore_list,
        label_level=config.label_level,
    )

    for blur in blur_limit_list:
        blur_transform = A.Compose(
            [
                A.Blur(blur_limit=blur),
                A.Normalize(mean=0.4599, std=0.2172),
                ToTensorV2(),
            ]
        )

        test_dataset = StimulusDataset(
            x_data=x_test,
            y_data=y_test,
            img_transform=blur_transform,
            class2idx=class2idx,
        )

        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

        for model_name in tqdm(config.model_names):

            # load model
            stim_model = StimulusClassifier(
                num_classes=len(class2idx), model_name=model_name
            )
            stim_model.load_state_dict(
                torch.load(
                    f"./../models/stimulus_classifier/stim_classifier_model_{model_name}.pth",
                    map_location="cpu",
                )
            )
            stim_model.eval()

            # calculate accuracy for stimulus model predictions
            y_true_list, y_pred_list = test_model(
                model=stim_model, test_loader=test_loader, device="cpu"
            )

            stim_model_acc = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)

            blur_model_acc_list.append((blur, model_name, stim_model_acc))

    return blur_model_acc_list
