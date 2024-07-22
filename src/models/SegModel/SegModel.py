import segmentation_models_pytorch as smp
import lightning as pl
import torch
from torchvision.transforms import Pad

class SegModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, padding_mode='edge', classes=2, decoder_use_batchnorm=True, decoder_attention_type=None, activation=None, aux_params=None, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            classes=classes,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type = decoder_attention_type,
            activation=activation,
            aux_params=aux_params,
            **kwargs
        )
        self.padding_mode = padding_mode

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, x):
        pad = Pad(8, padding_mode=self.padding_mode)
        x = pad((x - self.mean) / self.std)
        mask = self.model(x)
        mask = mask[:, 0, 8:-8, 8:-8].unsqueeze(1)
        return mask