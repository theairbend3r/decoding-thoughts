import torch
import torch.nn as nn
from src.ml import StimulusClassifierConfig, FMRIClassifierConfig


class StimulusClassifier(nn.Module):
    """Torch cnn model."""

    def __init__(self, num_channel, num_classes):
        super(StimulusClassifier, self).__init__()

        self.config = StimulusClassifierConfig()

        self.feature_extractor = nn.Sequential(
            self.conv_block(
                c_in=num_channel, c_out=16, kernel_size=3, stride=1, padding=1
            ),
            self.conv_block(c_in=16, c_out=32, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=32, c_out=64, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=64, c_out=128, kernel_size=3, stride=1, padding=1),
        )

        self.fc1 = nn.Linear(
            in_features=4 * 4 * 128, out_features=self.config.latent_emb_size
        )
        self.fc2 = nn.Linear(
            in_features=self.config.latent_emb_size, out_features=num_classes
        )

    def conv_block(self, c_in, c_out, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)
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
        x = x.flatten(start_dim=1)
        x = self.fc1(x)

        return x


class FMRIClassifier(nn.Module):
    """Torch dnn model."""

    def __init__(self, num_features, num_classes):
        super(FMRIClassifier, self).__init__()

        self.config = FMRIClassifierConfig()
        self.fc1 = self.lin_block(f_in=num_features, f_out=self.config.latent_emb_size)
        self.fc2 = self.lin_block(f_in=self.config.latent_emb_size, f_out=num_classes)

    def lin_block(self, f_in, f_out):
        return nn.Sequential(
            nn.Linear(in_features=f_in, out_features=f_out),
            nn.ReLU(),
            nn.Dropout(p=0.7),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        y = self.forward(x)
        y_softmax = torch.log_softmax(y, dim=1)
        best_y_idx = torch.argmax(y_softmax, dim=1).item()

        return best_y_idx

    def get_latent_rep(self, x):
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    stim_model = StimulusClassifier(num_channel=3, num_classes=5)
    fmri_model = FMRIClassifier(num_features=8000, num_classes=5)
    batch_size = 16
    print(summary(stim_model, input_size=(batch_size, 3, 128, 128)))
    print()
    print("=" * 50)
    print()
    print(summary(fmri_model, input_size=(batch_size, 8000)))
