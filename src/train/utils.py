import torch
from torch import nn
from src.train.loss import *

### TRAIN CONFIG UTIL ###

DEFAULT_TRAIN_CONFIG = {
    "n_epochs": 5,
    "optimizer": {"name": "Adam", "params": {"lr": 5e-4}},
    "loss": {"name": "BCELoss", "params": {}},
    "clip_grad": None,
    "n_gpus": 1
}


def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    params = config["train"]["optimizer"]["params"]
    model_params = model.parameters() if config["model"]["name"] != "SAM" else list(model.sam.mask_decoder.parameters()) # + list(model.UNet.parameters())
    match config["train"]["optimizer"]["name"]:
        case "Adam":
            if model.__class__.__name__ == "DeepLabv3Plus":
                return torch.optim.Adam(params=[
                    {'params': model.backbone.parameters(), 'lr': 0.1 * params["lr"]},
                    {'params': model.classifier.parameters(), 'lr': params["lr"]},
                ], lr=params["lr"])
            else:
                return torch.optim.Adam(model.parameters(), **params)
        case "SGD":
            return torch.optim.SGD(model_params, **params)


def get_loss(config: dict):
    # params = config["train"]["loss"]["params"]
    # llambda = params.get("lambda", 1.0)
    llambda = 1.0

    def create_combined_loss(loss1, loss2, lambda_param):
        return CombinedLoss(loss1, loss2, lambda_param)

    match config["train"]["loss"]:
        case "BCELoss":
            return nn.BCELoss()
        case "sDice":
            return SoftDiceLoss()
        case "lcDice":
            return LogCoshDiceLoss()
        case "sqDice":
            return SquaredDiceLoss()
        case "clDice":
            return CenterlineDiceLoss()
        case "ft":
            return FocalTverskyLoss()
        case "DiceBCELoss":
            return create_combined_loss(SoftDiceLoss(), nn.BCELoss(), llambda)
        case "lcDiceBCELoss":
            return create_combined_loss(
                LogCoshDiceLoss(), nn.BCELoss(), llambda
            )
        case "sqDiceBCELoss":
            return create_combined_loss(
                SquaredDiceLoss(), nn.BCELoss(), llambda
            )
        case "clDiceBCELoss":
            return create_combined_loss(
                CenterlineDiceLoss(), nn.BCELoss(), llambda
            )
        case "ftBCE":
            return create_combined_loss(
                FocalTverskyLoss(), nn.BCELoss(), llambda
            )
