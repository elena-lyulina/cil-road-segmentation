"""
Partially taken from https://www.towardsdeeplearning.com/dinov2-for-custom-dataset-segmentation-a-comprehensive-tutorial/
"""
from torch import nn

from src.models.utils import MODEL_REGISTRY
import torch
import torch.nn.functional as F
from torchvision import models

from functools import partial
from transformers import Dinov2Model, Dinov2PreTrainedModel

nonlinearity = partial(F.relu, inplace=True)


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DLinkNetClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=384, tokenH=384, num_classes=1):
        super(DLinkNetClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH

        self.init_upsampling1 = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=384, stride=7,
                                                   kernel_size=14, padding=2)
        self.init_upsampling2 = nn.ConvTranspose2d(in_channels=384, out_channels=96, stride=2,
                                                   kernel_size=4, padding=1)
        self.linear = nn.Conv2d(in_channels=96, out_channels=3, stride=1,
                                kernel_size=3, padding=1)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, inputs):
        (embeddings, pixel_values) = inputs

        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        x = self.init_upsampling1(embeddings)
        x = self.init_upsampling2(x)
        x = self.linear(x)

        x = torch.cat((x, pixel_values), dim=1)

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class Dinov2(torch.nn.Module):
    def __init__(self, config):
        super(Dinov2, self).__init__()
        self.dinov2 = Dinov2Model.from_pretrained(config)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False):
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions, )
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        return (patch_embeddings, pixel_values)


@MODEL_REGISTRY.register("dino_plus_dlinknet")
class Dinov2ForSemanticSegmentation(torch.nn.Module):
    def __init__(self, config="facebook/dinov2-base", hidden_size=768, num_labels=1):
        super().__init__()
        self.dino = Dinov2(config)

        self.classifier = DLinkNetClassifier(hidden_size, 27, 27)
        self._freeze_dinov2_parameters()

        self.model = nn.Sequential(self.dino, self.classifier)

    def _freeze_dinov2_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False

    def forward(
            self,
            pixel_values,
            output_hidden_states=False,
            output_attentions=False,
            labels=None,
    ):

        out = self.model(pixel_values)
        return out
