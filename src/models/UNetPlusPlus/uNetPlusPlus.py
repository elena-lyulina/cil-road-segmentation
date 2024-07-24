from src.models.SegModel.SegModel import SegModel

from src.models.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register("unetplusplus")
class UNetPlusPlus(SegModel):
    def __init__(self, encoder_name, padding_mode='edge', classes=2, decoder_use_batchnorm=True, decoder_attention_type=None, activation=None, aux_params=None, **kwargs):
        super().__init__('UnetPlusPlus', encoder_name, padding_mode, classes, decoder_use_batchnorm, decoder_attention_type, activation, aux_params, **kwargs)
