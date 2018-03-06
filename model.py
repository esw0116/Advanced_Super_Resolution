import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class small_residual_block(nn.Module):
    def __init__(self, chan_in):
        super(small_residual_block, self).__init__()
        self.srb = nn.Sequential(
            nn.Conv2d(in_channels=chan_in, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=64)
        )

    def forward(self, x):
        initial = x
        x = self.srb(x)
        x = torch.add(x, initial)
        return x


class big_residual_block(nn.Module):
    def __init__(self):
        super(big_residual_block, self).__init__()
        self.residual = self.make_srb(16)
        self.post_residual = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=64)
        )

    def make_srb(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(small_residual_block(64))
        return nn.Sequential(*layers)

    def forward(self, x):
        initial = x
        x = self.residual(x)
        x = self.post_residual(x)
        x = torch.add(x, initial)
        return x


class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet, self).__init__()
        self.before_brb = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 9), padding=4),
            nn.PReLU()
        )
        self.brb = big_residual_block()
        self.after_brb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        x = self.before_brb(x)
        x = self.brb(x)
        x = self.after_brb(x)
        return x



