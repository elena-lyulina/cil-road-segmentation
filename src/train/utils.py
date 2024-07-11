import torch
from torch import nn
import src.train.loss as loss

### TRAIN CONFIG UTIL ###

DEFAULT_TRAIN_CONFIG = {
    "n_epochs": 5,
    "optimizer": {"name": "Adam", "params": {"lr": 5e-4}},
    "loss": {"name": "BCELoss", "params": {}},
    "clip_grad": None,
}


def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    params = config["train"]["optimizer"]["params"]
    model_params = model.parameters() # if config["model"]["name"] != "SAM" else list(model.sam.mask_decoder.parameters()) + list(model.UNet.parameters())
    match config["train"]["optimizer"]["name"]:
        case "Adam":
            return torch.optim.Adam(model_params, **params)
        case "SGD":
            return torch.optim.SGD(model_params, **params)


def get_loss(config: dict):
    params = config["train"]["loss"]["params"]
    match config["train"]["loss"]["name"]:
        case "BCELoss":
            return nn.BCELoss(**params)
        case "DiceBCELoss":
            return loss.dice_bce_loss()
        case "FocalTverskyLoss":
            return loss.FocalTverskyLoss(**params)
