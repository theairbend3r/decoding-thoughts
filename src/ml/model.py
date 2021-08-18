import torch
import torch.nn as nn
from src.ml import StimulusClassifierConfig, FMRIClassifierConfig
import torchvision.models as models

# class StimulusClassifier(nn.Module):
#     """Torch cnn model."""

#     def __init__(self, num_channel, num_classes):
#         super(StimulusClassifier, self).__init__()

#         self.config = StimulusClassifierConfig()

#         self.feature_extractor = nn.Sequential(
#             self.conv_block(num_channel, 64),
#             self.transition_block(64, 32),
#             self.conv_block(32, 64),
#             self.transition_block(64, 32),
#             self.conv_block(32, 64),
#             self.transition_block(64, 32),
#             self.conv_block(32, 64),
#             self.transition_block(64, 32),
#             nn.AvgPool2d(kernel_size=8),
#         )

#         self.fc1 = nn.Linear(in_features=32, out_features=self.config.latent_emb_size)
#         self.fc2 = nn.Linear(
#             in_features=self.config.latent_emb_size, out_features=num_classes
#         )

#         self.relu = nn.ReLU()

#     def conv_block(self, c_in, c_out):
#         return nn.Sequential(
#             nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_features=c_in),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_features=c_out),
#             nn.ReLU(),
#             nn.Dropout2d(p=0.1),
#         )

#     def transition_block(self, c_in, c_out):
#         return nn.Sequential(
#             # Pointwise Convolution to reduce number of channels
#             nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)),
#             # Depthwise Convolution with stride=2 to reduce the channel size to half
#             nn.Conv2d(
#                 in_channels=c_out,
#                 out_channels=c_out,
#                 kernel_size=(3, 3),
#                 padding=1,
#                 stride=2,
#                 groups=c_out,
#                 bias=False,
#             ),
#         )

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = x.flatten(start_dim=1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x

#     def predict(self, x):
#         y = self.forward(x)
#         y_softmax = torch.log_softmax(y, dim=1)
#         best_y_idx = torch.argmax(y_softmax, dim=1).item()

#         return best_y_idx

#     def get_latent_rep(self, x):
#         x = self.feature_extractor(x)
#         x = x.flatten(start_dim=1)
#         x = self.fc1(x)

#         return x


class StimulusClassifier(nn.Module):
    """Torch cnn model."""

    def __init__(self, num_classes, model_name):
        super(StimulusClassifier, self).__init__()

        self.config = StimulusClassifierConfig()

        if model_name == "vgg-11":
            self.feature_extractor = models.vgg11_bn(pretrained=True)

        elif model_name == "resnet-50":
            self.feature_extractor = models.resnet50(pretrained=True)

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
    """Torch dnn model."""

    def __init__(self, num_features, num_classes):
        super(FMRIClassifier, self).__init__()

        self.config = FMRIClassifierConfig()
        self.fc1 = self.lin_block(f_in=num_features, f_out=512)
        self.fc2 = self.lin_block(f_in=512, f_out=self.config.latent_emb_size)
        self.fc3 = nn.Linear(
            in_features=self.config.latent_emb_size, out_features=num_classes
        )

    def lin_block(self, f_in, f_out):
        return nn.Sequential(
            nn.Linear(in_features=f_in, out_features=f_out),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def predict(self, x):
        y = self.forward(x)
        y_softmax = torch.log_softmax(y, dim=1)
        best_y_idx = torch.argmax(y_softmax, dim=1).item()

        return best_y_idx

    def get_latent_rep(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    stim_model_1 = StimulusClassifier(num_classes=5, model_name="vgg-11")
    stim_model_2 = StimulusClassifier(num_classes=5, model_name="resnet-50")
    fmri_model = FMRIClassifier(num_features=8000, num_classes=5)
    batch_size = 16
    summary(stim_model_1, input_size=(batch_size, 3, 128, 128))
    summary(stim_model_2, input_size=(batch_size, 3, 128, 128))
    print()
    print("=" * 50)
    print()
    # summary(fmri_model, input_size=(batch_size, 8000))
