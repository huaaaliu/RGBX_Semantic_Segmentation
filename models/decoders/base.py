import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F


class BaseHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], num_classes=40, norm_layer=nn.BatchNorm2d, dropout_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes

        # deconv1 1/8
        self.deconv1 = nn.ConvTranspose2d(in_channels[3], in_channels[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = norm_layer(in_channels[2])
        self.relu1 = nn.ReLU()

        # deconv1 1/4
        self.deconv2 = nn.ConvTranspose2d(in_channels[2], in_channels[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = norm_layer(in_channels[1])
        self.relu2 = nn.ReLU()

        # deconv1 1/2
        self.deconv3 = nn.ConvTranspose2d(in_channels[1], in_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = norm_layer(in_channels[0])
        self.relu3 = nn.ReLU()

        # deconv1 1/1
        self.deconv4 = nn.ConvTranspose2d(in_channels[0], in_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = norm_layer(in_channels[0])
        self.relu4 = nn.ReLU()

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.classifier = nn.Conv2d(in_channels[0], num_classes, kernel_size=1)

    def forward(self, inputs):
        f1, f2, f3, f4 = inputs
        
        y = self.bn1(self.relu1(self.deconv1(f4)) + f3)

        y = self.bn2(self.relu2(self.deconv2(y)) + f2)

        y = self.bn3(self.relu3(self.deconv3(y)) + f1)

        y = self.bn4(self.relu4(self.deconv4(y)))

        y = self.dropout(self.classifier(y))

        return y