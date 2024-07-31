from src.models.utils import MODEL_REGISTRY
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
import torch.optim as optim


def conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
    )


def max_pool(kernel_size=2, stride=2):
    return nn.MaxPool2d(kernel_size, stride)


@MODEL_REGISTRY.register("fcn_8")
class FCN8s(nn.Module):
    """Using pre-trained VGG for base model
    Source model: https://arxiv.org/pdf/1411.4038"""

    def __init__(self, num_classes=1):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG16 model
        vgg = models.vgg16(pretrained=True)

        # Encoder: VGG16 layers up to pool5
        self.features = vgg.features

        # Fully convolutional layers
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Transposed convolutions for upsampling
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4
        )
        self.upscore16 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=32, stride=16, bias=False
        )

        # Scoring layers for skip connections
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass through VGG16 feature layers
        pool3 = self.features[:17](x)  # up to relu4_3
        pool4 = self.features[17:24](pool3)  # up to relu5_3
        pool5 = self.features[24:](pool4)  # up to maxpool5
        # Stop here for FCN32s

        # Fully convolutional layers
        fc6 = self.fc6(pool5)
        fc6 = self.relu6(fc6)
        fc6 = self.drop6(fc6)

        fc7 = self.fc7(fc6)
        fc7 = self.relu7(fc7)
        fc7 = self.drop7(fc7)

        score_fr = self.score_fr(fc7)  # Output fo final scoring layer

        # Has twice the spatial resolution of score_fr
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = F.interpolate(
            score_pool4, size=upscore2.shape[2:], mode="bilinear", align_corners=True
        )
        upscore_pool4 = upscore2 + score_pool4  # Check if necessary
        # Stop here for FCN16s

        upscore_pool16 = self.upscore16(upscore_pool4)
        score_pool3 = self.score_pool3(pool3)
        score_pool3 = F.interpolate(
            score_pool3,
            size=upscore_pool16.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        upscore8 = self.upscore8(upscore_pool16 + score_pool3)
        out = F.interpolate(
            upscore8,
            size=(x.size()[2], x.size()[3]),
            mode="bilinear",
            align_corners=True,
        )
        out = F.sigmoid(out)
        # min_val, max_val = (torch.min(out), torch.max(out))
        # out = (out - min_val) / (max_val - min_val)

        # out = upscore8[:, 31 : 31 + , 31 : 31 + x.size()[3]]
        # print(out.shape)
        return out
