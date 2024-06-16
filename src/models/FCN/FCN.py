"""Partial source: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py"""
"""Using pre-trained VGG for base model"""
"""Og paper: https://arxiv.org/pdf/1411.4038"""

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
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
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

        # Scoring layers for skip connections
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass through VGG16 feature layers
        pool3 = self.features[:17](x)  # up to relu4_3
        pool4 = self.features[17:24](pool3)  # up to relu5_3
        pool5 = self.features[24:](pool4)  # up to maxpool5

        # Fully convolutional layers
        fc6 = self.fc6(pool5)
        fc6 = self.relu6(fc6)
        fc6 = self.drop6(fc6)

        fc7 = self.fc7(fc6)
        fc7 = self.relu7(fc7)
        fc7 = self.drop7(fc7)

        score_fr = self.score_fr(fc7)

        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4) # Check if necessary

        score_pool3 = self.score_pool3(pool3)[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        upscore8 = self.upscore8(upscore_pool4 + score_pool3)

        return upscore8


def test():
    model = FCN8s(num_classes=21)
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image
    output = model(input_tensor)
    print(output.shape)  # Expected output shape: (1, num_classes, 224, 224)