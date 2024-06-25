from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import DataLoader

from src.experiments.registry import Registry

DATAHANDLER_REGISTRY = Registry()


def get_datahandler_class(dataset_name: str):
    # Returns data_handler's class registered under the given name
    cls = DATAHANDLER_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(f'DataHandler {dataset_name} is not registered')
    return cls


class DataHandler(ABC):
    # A helper class to manage different dataset splitting for different datasets
    # Might become useless if we start implementing all datasets in the same format but for now let's use it

    @abstractmethod
    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # A method to implement the correct dataset splitting for the current dataset.
        # If you need to pass some parameters, pass it to the __init__ method and save as attributes to use here
        raise NotImplementedError("Please implement this method")
