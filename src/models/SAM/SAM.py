from src.constants import DEVICE
from pathlib import Path

import torch
import torchvision
from torch import nn

from src.models.small_UNet.small_UNet import UNet
from src.models.utils import MODEL_REGISTRY

from transformers import SamModel, SamProcessor


@MODEL_REGISTRY.register("SAM")
class SAM(torch.nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

        # check large and huge model as well. If inference is too long, possibly do it once for each image in data loading; Requires change in dataloader though
        print('Loading SAM model.')
        self.sam = SamModel.from_pretrained('facebook/sam-vit-base')
        self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
        self.UNet = UNet(chs=(3, 64, 128, 256, 512, 1024))
        checkpoint = torch.load(Path(__file__).resolve().parent / 'models' / 'small_unet.pth')
        self.UNet.load_state_dict(checkpoint['model_state_dict'])
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
        prompt = prompt > 0.5

        num_batch, ch, width, height = pixel_values.shape
        # input shapes for SAM
        originals = torch.empty([num_batch, ch, 1024, 1024]).to(DEVICE)
        masks = torch.empty([num_batch, 1, 256, 256]).to(DEVICE)
        for i in range(num_batch):
            preprocessed = self.processor(pixel_values[i], segmentation_maps=prompt[i])
            image = torch.tensor(preprocessed['pixel_values'][0])
            seg = torch.tensor(preprocessed['labels'][0])
            originals[i] = image
            masks[i] = seg

        # inputs = self.processor(pixel_values.squeeze(0), segmentation_maps=prompt.squeeze(0))
        # pixel_values = torch.tensor(inputs['pixel_values'][0]).unsqueeze(0).to(DEVICE)
        # input_mask = torch.tensor(inputs['labels'][0]).unsqueeze(0).to(DEVICE)
        outputs = self.sam(pixel_values=originals, multimask_output=False)

        masks = self.processor.post_process_masks(outputs.pred_masks, [(width, height)]*4, [(1024, 1024)]*4, binarize=False)
        masks = torch.cat(masks, 0)
        # min_val = masks.min()
        # max_val = masks.max()
        # pred = (masks - min_val) / (max_val - min_val)

        # pred = torch.clamp(masks, min=0., max=1.)
        pred = torch.nn.functional.sigmoid(masks)

        return pred
