from torch import nn

from src.registry import Registry

MODEL_REGISTRY = Registry()


def get_model_class(model_name: str):
    # Returns model's class registered under the given name
    model_init_fn = MODEL_REGISTRY.get(model_name)
    if model_init_fn is None:
        raise ValueError(f'Model {model_name} is not registered')
    return model_init_fn
