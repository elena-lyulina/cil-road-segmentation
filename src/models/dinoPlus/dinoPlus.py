"""
Partially taken from https://www.towardsdeeplearning.com/dinov2-for-custom-dataset-segmentation-a-comprehensive-tutorial/
"""

from src.models.utils import MODEL_REGISTRY
import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=27, tokenH=27, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


@MODEL_REGISTRY.register("dino_plus")
# class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
class Dinov2ForSemanticSegmentation(torch.nn.Module):
    def __init__(self, config="facebook/dinov2-base", hidden_size=768, num_labels=1):
        super().__init__()

        self.dinov2 = Dinov2Model.from_pretrained(config)
        self.classifier = LinearClassifier(
            hidden_size, 27, 27, num_labels
        )  # 27 because DINO returns 729 patches
        self._freeze_dinov2_parameters()

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
        # use frozen features
        outputs = self.dinov2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        out = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )
        min_val, max_val = (torch.min(out), torch.max(out))
        out = (out - min_val) / (max_val - min_val)

        return out


# # We can instantiate the model as follows:

# model = Dinov2ForSemanticSegmentation.from_pretrained(
#     "facebook/dinov2-base", id2label=id2label, num_labels=len(id2label)
# )

# # Important: we don't want to train the DINOv2 backbone, only the linear classification head. Hence we don't want to track any gradients for the backbone parameters. This will greatly save us in terms of memory used:

# for name, param in model.named_parameters():
#     if name.startswith("dinov2"):
#         param.requires_grad = False

# # Let's perform a forward pass on a random batch, to verify the shape of the logits, verify we can calculate a loss:

# outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
# print(outputs.logits.shape)
# print(outputs.loss)
