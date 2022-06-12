import torch.nn as nn
import torch


'''
Residuals
References: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''


class Residual(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x) + x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm_layer=nn.Identity, activation=nn.GELU,
                 groups=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            activation(),
            norm_layer(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, groups=groups,
                      padding=kernel_size // 2),
            activation(),
            norm_layer(out_channels),
        )

        self.identity = nn.Identity()
        # identity becomes a downsample if channels differ or stride isn't 1
        if in_channels != out_channels or stride != 1:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                activation(),
                norm_layer(out_channels),
            )

    def forward(self, features):
        return self.identity(features) + self.block(features)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_dims=3, output_dims=512):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_dims, 16, kernel_size=7, stride=2, padding=0),
            nn.GELU(),
            nn.LayerNorm((16, 117, 157)),
        )
        self.l1 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0)
        self.l2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, dilation=2, padding=2)
        self.main = nn.Sequential(
            nn.GELU(),
            ResBlock(64, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, dilation=4, padding=0),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_dims),
            nn.ReLU(True),
        )

    def forward(self, observations):
        observations = self.initial(observations)
        features1 = self.l1(observations)
        features2 = self.l2(observations)
        features = torch.cat((features1, features2), dim=1)
        return self.main(features)


class NatureCNN(nn.Module):
    def __init__(self, input_dims=3, output_dims=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dims, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.rand((1, input_dims, 120, 160)).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, output_dims), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))
