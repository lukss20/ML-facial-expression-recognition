class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))


class ResidualCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super(ResidualCNN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            ResidualBlock(32, 64, downsample=True),
            ResidualBlock(64, 64),

            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128),

            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return self.classifier(x)