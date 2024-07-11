from src.constants import DEVICE
from pathlib import Path

import torch
import torchvision
from torch import nn

from src.models.utils import MODEL_REGISTRY

from transformers import SamModel, SamImageProcessor
from segment_anything import SamPredictor, sam_model_registry


class ConvBlock(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
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


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(
            2
        )  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
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


@MODEL_REGISTRY.register("SAM")
class SAM(torch.nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

        # check large and huge model as well. If inference is too long, possibly do it once for each image in data loading; Requires change in dataloader though
        print('Loading SAM model.')
        path = Path(__file__).resolve().parent / 'models' / 'sam_vit_b_01ec64.pth'
        self.sam = sam_model_registry["vit_b"](checkpoint=path)
        # self.sam = SamModel.from_pretrained('facebook/sam-vit-base')
        self.predictor = SamPredictor(self.sam)
        self.UNet = UNet(chs=(3, 64, 128, 256, 512, 1024))
        self._freeze_sam_encoder()

    def _freeze_sam_encoder(self):
        for name, param in self.sam.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def forward(
            self,
            pixel_values
    ):

        prompt = self.UNet(pixel_values)
        prompt = torchvision.transforms.functional.resize(prompt, (256, 256)).squeeze(0)
        pixel_values = pixel_values.squeeze().permute(1, 2, 0).cpu().numpy()
        self.predictor.set_image(pixel_values)
        mask, _, _ = self.predictor.predict(mask_input=prompt, multimask_output=False)

        return torch.tensor(mask, dtype=torch.float).unsqueeze(0).to(DEVICE)
