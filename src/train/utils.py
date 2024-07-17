import torch
from torch import nn
from src.train.loss import *

### TRAIN CONFIG UTIL ###

DEFAULT_TRAIN_CONFIG = {
    "n_epochs": 5,
    "optimizer": {"name": "Adam", "params": {"lr": 5e-4}},
    "loss": {"name": "BCELoss", "params": {}},
    "clip_grad": None,
}


def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    params = config["train"]["optimizer"]["params"]
    model_params = model.parameters() if config["model"]["name"] != "SAM" else list(model.sam.mask_decoder.parameters()) # + list(model.UNet.parameters())
    match config["train"]["optimizer"]["name"]:
        case "Adam":
            return torch.optim.Adam(model_params, **params)
        case "SGD":
            return torch.optim.SGD(model_params, **params)


def get_loss(config: dict):
    params = config["train"]["loss"]["params"]
    llambda = params.get("lambda", 1.0)

    def create_combined_loss(loss1, loss2, lambda_param):
        return CombinedLoss(loss1, loss2, lambda_param)

    match config["train"]["loss"]["name"]:
        case "BCELoss":
            return nn.BCELoss(**params)
        case "sDice":
            return SoftDiceLoss(**params)
        case "lcDice":
            return LogCoshDiceLoss(**params)
        case "sqDice":
            return SquaredDiceLoss(**params)
        case "clDice":
            return CenterlineDiceLoss(**params)
        case "ft":
            return FocalTverskyLoss(**params)
        case "DiceBCELoss":
            return create_combined_loss(SoftDiceLoss(**params), nn.BCELoss(), llambda)
        case "lcDiceBCELoss":
            return create_combined_loss(
                LogCoshDiceLoss(**params), nn.BCELoss(), llambda
            )
        case "sqDiceBCELoss":
            return create_combined_loss(
                SquaredDiceLoss(**params), nn.BCELoss(), llambda
            )
        case "clDiceBCELoss":
            return create_combined_loss(
                CenterlineDiceLoss(**params), nn.BCELoss(), llambda
            )
        case "ftBCE":
            return create_combined_loss(
                FocalTverskyLoss(**params), nn.BCELoss(), llambda
            )
