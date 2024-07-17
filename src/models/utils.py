import torch
from torch import multiprocessing
from torch._utils import _get_all_device_indices
import torch.nn as nn

from src.constants import DEVICE
from src.experiments.registry import Registry

MODEL_REGISTRY = Registry()


def get_model_class(model_name: str):
    # Returns model's class registered under the given name
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f'Model {model_name} is not registered')
    return cls


def get_model(config: dict):
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]
    model = get_model_class(model_name)(**model_params).to(DEVICE)
    ngpus = min(config["train"]["n_gpus"], torch.cuda.device_count())
    if ngpus > 1:
        multiprocessing.set_start_method('spawn', force=True)
        print(f"Using {ngpus} GPUs.")
        device_ids = _get_all_device_indices()[:ngpus]
        model = nn.DataParallel(model, device_ids)

    print(f'Running on {DEVICE}')
    return model
