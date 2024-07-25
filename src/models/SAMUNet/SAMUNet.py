from src.constants import DEVICE
from pathlib import Path

import torch
from torch import nn

from src.models.small_UNet.small_UNet import UNet
from src.models.utils import MODEL_REGISTRY

from transformers import SamModel, SamProcessor


class ConvBlock(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch, pad1=1, pad2=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class UNetClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=384, tokenH=384, num_labels=1, chs=(6, 64, 128, 256, 512, 1024)):
        super(UNetClassifier, self).__init__()

        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH

        self.init_upsampling1 = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=384, stride=7, kernel_size=14, padding=2)
        self.init_upsampling2 = nn.ConvTranspose2d(in_channels=384, out_channels=96, stride=2,
                                                   kernel_size=4, padding=1)
        self.linear = nn.Conv2d(in_channels=96, out_channels=3, stride=1,
                                                   kernel_size=3, padding=1)

        self.enc_blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(2)
        self.upconvs = nn.ModuleList([
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ])
        # deconvolution
        self.dec_blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks

        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output-

    def forward(self, inputs):
        (embeddings, pixel_values) = inputs

        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        x = self.init_upsampling1(embeddings)
        x = self.init_upsampling2(x)
        x = nn.functional.pad(x, (1, 1, 1, 1))
        x = self.linear(x)

        x = torch.cat((x, pixel_values), dim=1)

        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(
                self.dec_blocks, self.upconvs, enc_features[::-1]
        ):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


@MODEL_REGISTRY.register("SAMUNet")
class SAMUNet(torch.nn.Module):
    def __init__(self):
        super(SAMUNet, self).__init__()

        # check large and huge model as well. If inference is too long, possibly do it once for each image in data loading; Requires change in dataloader though
        print('Loading SAM model.')
        self.sam = SamModel.from_pretrained('facebook/sam-vit-base')
        self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
        self._freeze_sam_encoder()

        self.classifier = UNetClassifier()

    def _freeze_sam_encoder(self):
        for name, param in self.sam.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def forward(
            self,
            pixel_values
    ):

        num_batch, ch, width, height = pixel_values.shape
        # input shapes for SAM
        originals = torch.empty([num_batch, ch, 1024, 1024]).to(DEVICE)
        masks = torch.empty([num_batch, 1, 256, 256]).to(DEVICE)
        for i in range(num_batch):
            preprocessed = self.processor(pixel_values[i])
            image = torch.tensor(preprocessed['pixel_values'][0])
            seg = torch.tensor(preprocessed['labels'][0])
            originals[i] = image
            masks[i] = seg

        outputs = self.sam.get_image_embeddings(pixel_values=originals)

        pred = self.classifier((outputs, pixel_values))

        return pred
