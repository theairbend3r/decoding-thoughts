"""
Model definitions for fMRI and Stimulus dataset.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.ml.config import StimulusClassifierConfig, FMRIClassifierConfig


class StimulusClassifier(nn.Module):
    """Torch CNN model."""

    def __init__(self, num_classes, model_name):
        super(StimulusClassifier, self).__init__()

        self.config = StimulusClassifierConfig()

        if model_name == "vgg-11":
            self.feature_extractor = models.vgg11_bn(pretrained=True)
        elif model_name == "resnet-50":
            self.feature_extractor = models.resnet50(pretrained=True)
        elif model_name == "resnext50":
            self.feature_extractor = models.resnext50_32x4d(pretrained=True)
        elif model_name == "mobilenet_v3_large":
            self.feature_extractor = models.mobilenet_v3_large(pretrained=True)
        elif model_name == "densenet121":
            self.feature_extractor = models.densenet121(pretrained=True)
        else:
            raise ValueError("Incorrect model name.")

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(in_features=1000, out_features=self.config.latent_emb_size)

        self.fc2 = nn.Linear(
            in_features=self.config.latent_emb_size, out_features=num_classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        y = self.forward(x)
        y_softmax = torch.log_softmax(y, dim=1)
        best_y_idx = torch.argmax(y_softmax, dim=1).item()

        return best_y_idx

    def get_latent_rep(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)

        return x


class FMRIClassifier(nn.Module):
    """Torch DNN model."""

    def __init__(self, num_features, num_classes):
        super(FMRIClassifier, self).__init__()

        self.config = FMRIClassifierConfig()
        self.feature_extractor = nn.Sequential(
            self.lin_block(f_in=num_features, f_out=512),
            self.lin_block(f_in=512, f_out=512),
        )
        self.fc1 = self.lin_block(f_in=512, f_out=self.config.latent_emb_size)
        self.fc2 = nn.Linear(
            in_features=self.config.latent_emb_size, out_features=num_classes
        )

    def lin_block(self, f_in, f_out):
        return nn.Sequential(
            nn.Linear(in_features=f_in, out_features=f_out),
            nn.BatchNorm1d(f_out),
            nn.ReLU(),
            nn.Dropout(0.6),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        y = self.forward(x)
        y_softmax = torch.log_softmax(y, dim=1)
        best_y_idx = torch.argmax(y_softmax, dim=1).item()

        return best_y_idx

    def get_latent_rep(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    fmri_model = FMRIClassifier(num_features=8000, num_classes=3)
    summary(fmri_model, input_size=(2, 8000))
    print()
    print("=" * 50)
    print()
    # stim_model_1 = StimulusClassifier(num_classes=3, model_name="vgg-11")
    # stim_model_2 = StimulusClassifier(num_classes=3, model_name="resnet-50")
    # summary(stim_model_1, input_size=(2, 3, 128, 128))
    # summary(stim_model_2, input_size=(2, 3, 128, 128))
