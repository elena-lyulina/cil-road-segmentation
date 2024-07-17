from torch import nn

from src.models.utils import MODEL_REGISTRY

import src.models.DeepLabv3Plus.network as network


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


@MODEL_REGISTRY.register("deeplabv3plus")
class DeepLabv3Plus(nn.Module):

    # TODO: add option for disabling pretrained backbone
    def __init__(self, backbone='resnet101', output_stride=8, num_classes=2, pretrained_backbone=True, separable_conv=False):
        super(DeepLabv3Plus, self).__init__()

        model = network.modeling._load_model('deeplabv3plus', backbone, num_classes, output_stride, pretrained_backbone)
        self.backbone = model.backbone
        self.classifier = model.classifier
        if separable_conv:
            network.convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)

    def forward(self, x):
        return self.classifier(self.backbone(x))
