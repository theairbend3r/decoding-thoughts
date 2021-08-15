import torch
import torch.nn as nn


class StimulusClassifier(nn.Module):
    """Torch cnn model."""

    def __init__(self, num_channel, num_classes):
        super(StimulusClassifier, self).__init__()

        self.block1 = nn.Sequential(
            self.conv_block(
                c_in=num_channel, c_out=8, kernel_size=3, stride=1, padding=1
            ),
            self.conv_block(c_in=8, c_out=8, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=8, c_out=8, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=8, c_out=8, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=8, c_out=16, kernel_size=3, stride=1, padding=1),
        )

        self.block2 = nn.Sequential(
            self.conv_block(c_in=16, c_out=16, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=16, c_out=16, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=16, c_out=16, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=16, c_out=16, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=16, c_out=32, kernel_size=3, stride=1, padding=1),
        )

        self.block3 = nn.Sequential(
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=32, c_out=64, kernel_size=3, stride=1, padding=1),
        )

        self.block4 = nn.Sequential(
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=64, c_out=128, kernel_size=3, stride=1, padding=1),
        )

        self.fc1 = nn.Linear(in_features=8 * 8 * 128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)

        x = self.block2(x)
        x = self.maxpool(x)

        x = self.block3(x)
        x = self.maxpool(x)

        x = self.block4(x)
        x = self.maxpool(x)

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
        x = self.block1(x)
        x = self.maxpool(x)

        x = self.block2(x)
        x = self.maxpool(x)

        x = self.block3(x)
        x = self.maxpool(x)

        x = self.block4(x)
        x = self.maxpool(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )


class FMRIClassifier(nn.Module):
    """Torch dnn model."""

    def __init__(self, num_features, num_classes):
        super(FMRIClassifier, self).__init__()

        self.block_1 = self.lin_block(f_in=num_features, f_out=64)
        self.block_2 = self.lin_block(f_in=64, f_out=num_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)

        return x

    def lin_block(self, f_in, f_out):
        return nn.Sequential(
            nn.Linear(in_features=f_in, out_features=f_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

    def predict(self, x):
        y = self.forward(x)
        y_softmax = torch.log_softmax(y, dim=1)
        best_y_idx = torch.argmax(y_softmax, dim=1).item()

        return best_y_idx

    def get_latent_rep(self, x):
        x = self.block_1(x)

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
