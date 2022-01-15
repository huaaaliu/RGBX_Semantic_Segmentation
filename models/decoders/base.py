import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F


class BaseHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], num_classes=40, norm_layer=nn.BatchNorm2d, dropout_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.conv4 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels[3], out_channels=in_channels[2], kernel_size=3),
                            norm_layer(in_channels[2]),
                            nn.ReLU(inplace=True)
                            )
        self.conv3 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels[2], out_channels=in_channels[1], kernel_size=3),
                            norm_layer(in_channels[1]),
                            nn.ReLU(inplace=True)
                            )
        self.conv2 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[0], kernel_size=3),
                            norm_layer(in_channels[0]),
                            nn.ReLU(inplace=True)
                            )
        self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=1),
                            norm_layer(in_channels[0]),
                            nn.ReLU(inplace=True)
                            )
                        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.classifier = nn.Conv2d(in_channels[0], num_classes, kernel_size=1)

    def forward(self, inputs):
        f1, f2, f3, f4 = inputs
        
        f4 = self.conv4(f4)
        f4 = F.interpolate(f4, size=f3.size()[2:], mode='bilinear')

        f3 = self.conv3(f3 + f4)
        f3 = F.interpolate(f3, size=f2.size()[2:], mode='bilinear')
        
        f2 = self.conv2(f2 + f3)
        f2 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear')

        f1 = self.conv1(f1 + f2)

        y = self.dropout(self.classifier(f1))

        return y