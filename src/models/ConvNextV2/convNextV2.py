import torch
from transformers import ViTMAEModel, ViTMAEConfig
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm



from src.models.utils import MODEL_REGISTRY


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Define the upsampling layers with channel reduction
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(2816, 1408, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1408),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(1408, 704, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(704),
            nn.ReLU()
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(704, 352, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(352),
            nn.ReLU()
        )
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(352, 176, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(176),
            nn.ReLU()
        )
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(176, 88, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(88),
            nn.ReLU()
        )
        # Final adjustment to reach 400x400 and reduce to 2 channels
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(88, 2, kernel_size=3, stride=1, padding=1),  # Adjust to exact spatial resolution
            nn.Tanh()  # Final activation
        )

    def forward(self, x):
        # (4, 2816, 12, 12)
        x = self.upconv1(x)
        # (4, 1408, 24, 24)
        x = self.upconv2(x)
        # (4, 352, 96, 96)
        x = self.upconv3(x)
        # (4, 176, 192, 192)
        x = self.upconv4(x)
        # (4, 88, 384, 384)
        x = self.upconv5(x)
        # (4, 2, 384, 384)
        x = self.upconv6(x)
        x = F.interpolate(x, size=(400, 400), mode='bilinear', align_corners=False)
        return x



@MODEL_REGISTRY.register("convNextV2")
class ConvNextV2(nn.Module):
    def __init__(self, num_classes=2): 
        super(ConvNextV2, self).__init__()

        self.model = timm.create_model("hf_hub:timm/convnextv2_huge.fcmae_ft_in22k_in1k_512", pretrained=True)

        self.model.stages.requires_grad_(False)  # Freeze the encoder
        self.decoder = Decoder()

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outputs = self.model.forward_features(x)

        segmentation_map = self.decoder(outputs)

        return F.sigmoid(segmentation_map[:, 0, :, :]).unsqueeze(1)
