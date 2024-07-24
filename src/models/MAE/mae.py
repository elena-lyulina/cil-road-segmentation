import torch
from transformers import ViTMAEModel, ViTMAEConfig
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F


from src.models.utils import MODEL_REGISTRY


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsample from 8x8 to 50x50
        self.upsample1 = nn.ConvTranspose2d(768, 256, kernel_size=5, stride=5)
        # Upsample from 50x50 to 100x100
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=5)
        # Upsample from 100x100 to 200x200
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Upsample from 200x200 to 400x400
        self.upsample4 = nn.ConvTranspose2d(64, 2, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = x.view(-1, 768, 8, 8)  # Assuming x is the input of shape (batch_size, 64, 768)
        x = self.upsample1(x)
        # print(x.shape)
        x = self.upsample2(x)
        # print(x.shape)
        x = self.upsample3(x)
        # print(x.shape)
        x = self.upsample4(x)
        # print(x.shape)
        return x

# Example usage
# decoder = Decoder()
# input_patches = torch.randn(1, 64, 768)  # Example input
# output = decoder(input_patches)
# print(output.shape)  # Should print torch.Size([1, 2, 400, 400])







# Modify the ViT MAE model for segmentation by using only the encoder
@MODEL_REGISTRY.register("mae")
class ViTMAESegmentation(nn.Module):
    def __init__(self, num_classes=2):  # assuming 21 classes for segmentation (e.g., Pascal VOC)
        super(ViTMAESegmentation, self).__init__()

        model_name = 'facebook/vit-mae-base'
        config = ViTMAEConfig(image_size=400, patch_size=25, num_channels = 1)  # Adjust image_size to 400
        self.model = ViTMAEModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

        self.model.encoder.requires_grad_(True)  # Freeze the encoder
        # Define a simple decoder for segmentation
        self.decoder = Decoder()

    def forward(self, x):
        # ViT MAE encoder forward pass
        outputs = self.model(pixel_values=x)
        # Extract the last hidden states
        last_hidden_states = outputs.last_hidden_state
        # Calculate height and width of the patch grid
        batch_size, num_patches, hidden_size = last_hidden_states.shape
        grid_size = int(num_patches ** 0.5)
        last_hidden_states = last_hidden_states[:, 1:, :]
        last_hidden_states = last_hidden_states.permute(0, 2, 1).reshape(batch_size, hidden_size, grid_size, grid_size)

        # Apply the decoder to get segmentation map
        segmentation_map = self.decoder(last_hidden_states)

        return F.sigmoid(segmentation_map[:, 1, :, :]).unsqueeze(1)

# Instantiate the modified model
# num_classes = 2  # Adjust based on your dataset
# segmentation_model = ViTMAESegmentation(num_classes=num_classes)

# # Example input image (dummy image for illustration)
# input_image = torch.randn(1, 400, 400)  # channels, height, width

# # Add a batch dimension
# input_image = input_image.unsqueeze(0)

# # Forward pass through the segmentation model
# output = segmentation_model(input_image)
# print(output.shape)  # should output the segmentation map with shape (batch_size, num_classes, height, width)