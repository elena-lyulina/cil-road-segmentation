from src.models.SegModel.SegModel import SegModel

from src.models.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register("FPN")
class UNetPlusPlus(SegModel):
    def __init__(self, encoder_name, padding_mode='edge', classes=2, activation=None, aux_params=None, **kwargs):
        super().__init__('FPN', encoder_name, padding_mode, classes, activation, aux_params, **kwargs)
