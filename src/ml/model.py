import torch.nn as nn


class StimulusClassifier(nn.Module):
    """Torch model to classify stimulus images.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    """

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
        self.block2 = self.conv_block(
            c_in=16, c_out=32, kernel_size=3, stride=1, padding=1
        )
        self.block3 = self.conv_block(
            c_in=32, c_out=64, kernel_size=3, stride=1, padding=1
        )
        self.block4 = self.conv_block(
            c_in=64, c_out=128, kernel_size=3, stride=1, padding=1
        )
        self.fc = nn.Linear(in_features=8 * 8 * 128, out_features=num_classes)

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
        x = self.fc(x)

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )


if __name__ == "__main__":
    from torchinfo import summary

    model = StimulusClassifier(num_channel=3, num_classes=5)
    batch_size = 16
    print(summary(model, input_size=(batch_size, 3, 128, 128)))
