"""
Configuration classes for models and datasets.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


class StimulusClassifierConfig:
    """
    Configuration file for the stimulus classifier.
    """

    def __init__(self):
        self.class_ignore_list = ["person", "fungus", "fruit", "plant", "entity"]
        self.label_level = 0
        self.epochs = 30
        self.learning_rate = 0.001
        self.batch_size = 128
        self.latent_emb_size = 100
        self.model_names = [
            "vgg-11",
            "resnext50",
            "resnet-50",
            "densenet121",
            "mobilenet_v3_large",
        ]

        self.img_transform = {
            "og": A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), ToTensorV2()]),
            "train": A.Compose(
                [
                    A.HorizontalFlip(p=0.1),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=0.05, p=0.1
                    ),
                    A.Normalize(mean=0.4599, std=0.2172),
                    ToTensorV2(),
                ]
            ),
            "test": A.Compose(
                [
                    A.Normalize(mean=0.4599, std=0.2172),
                    ToTensorV2(),
                ]
            ),
        }


class FMRIClassifierConfig:
    """
    Configuration file for the fMRI classifier.
    """

    def __init__(self):
        self.roi_select_list = [1, 2, 3, 4, 5, 6, 7]
        self.class_ignore_list = ["person", "fungus", "fruit", "plant", "entity"]
        self.label_level = 0
        self.epochs = 50
        self.learning_rate = 0.001
        self.batch_size = 128
        self.latent_emb_size = 100
