import albumentations as A
from albumentations.pytorch import ToTensorV2


class StimulusClassifierConfig:
    def __init__(self):
        self.class_ignore_list = ["person", "fungus", "plant"]
        self.label_level = 0
        self.validation_dataset_size = 0.2
        self.model_input_num_channel = 3
        self.epochs = 2
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.og_transform = A.Compose(
            [A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), ToTensorV2()]
        )

        self.img_transform = {
            "train": A.Compose(
                [
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
    def __init__(self):
        self.class_ignore_list = ["person", "fungus", "plant"]
        self.label_level = 0
        self.validation_dataset_size = 0.2
        self.roi_select_list = [1, 2, 3]
        self.epochs = 2
        self.learning_rate = 0.0001
        self.batch_size = 128
