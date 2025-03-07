from torch import nn
import torch

from src.models.utils import MODEL_REGISTRY

import src.models.DeepLabv3Plus.network as network

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


@MODEL_REGISTRY.register("deeplabv3plus")
class DeepLabv3Plus(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=8, num_classes=2, pretrained_backbone=True, separable_conv=False, mode=False):
        super(DeepLabv3Plus, self).__init__()

        self.model = network.modeling._load_model('deeplabv3plus', backbone, num_classes, output_stride, pretrained_backbone)
        self.backbone = self.model.backbone
        self.classifier = self.model.classifier
        if separable_conv:
            network.convert_to_separable_conv(self.classifier)
        set_bn_momentum(self.backbone, momentum=0.01)
    
    def forward(self, x):
        mae_input = None
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            mae_input = x.clone().detach()

        out = self.model(x)
        out = torch.sigmoid(out[:, 0, :, :].unsqueeze(1))

        return out if mae_input is None else (out, mae_input)
