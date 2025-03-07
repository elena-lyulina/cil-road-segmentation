"""
Partially taken from https://www.towardsdeeplearning.com/dinov2-for-custom-dataset-segmentation-a-comprehensive-tutorial/
"""
from torch import nn

from src.models.utils import MODEL_REGISTRY
import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel


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


class Dinov2(torch.nn.Module):
    def __init__(self, config):
        super(Dinov2, self).__init__()
        self.dinov2 = Dinov2Model.from_pretrained(config)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False):
        outputs = self.dinov2(pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        return (patch_embeddings, pixel_values)

@MODEL_REGISTRY.register("dino_plus_unet")
class Dinov2ForSemanticSegmentation(torch.nn.Module):
    def __init__(self, config="facebook/dinov2-base", hidden_size=768, num_labels=1):
        super().__init__()
        self.dino = Dinov2(config)

        self.classifier = UNetClassifier(
            hidden_size, 28, 28, num_labels
        )
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
