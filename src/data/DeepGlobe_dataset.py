from glob import glob
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
import torch
import random

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.roadseg_dataset import RoadSegDataset
from src.data.utils import DATASET_REGISTRY, np_to_tensor


@DATAHANDLER_REGISTRY.register("DeepGlobe")
class DeepGlobeDataHandler(DataHandler):
    dataset_path_DeepGlobe = DATA_PATH.joinpath("DeepGlobe")

    images_path = dataset_path_DeepGlobe.joinpath("images")
    masks_path = dataset_path_DeepGlobe.joinpath("masks")

    def __init__(self, batch_size=4, num_workers=4, shuffle=True, resize_to=(400, 400), augment=None, masking_params = {
            "num_zero_patches": 8,
            "zero_patch_size": 50,
            "num_flip_patches": 25,
            "flip_patch_size": 16,
        }):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.resize_to = resize_to
        self.augment = augment if augment else []
        self.masking_params = masking_params


        images_paths = [f for f in sorted(glob(str(self.images_path) + "/*.png"))]
        masks_paths = [f for f in sorted(glob(str(self.masks_path) + "/*.png"))]

        (
            self.train_image_paths,
            self.val_image_paths,
            self.train_mask_paths,
            self.val_mask_paths,
        ) = train_test_split(images_paths, masks_paths, test_size=0.2, random_state=42)

    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:

        train_dataset = DeepGlobeDataset(
            self.train_image_paths,
            self.train_mask_paths,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
        )

        val_dataset = DeepGlobeDataset(
            self.val_image_paths,
            self.val_mask_paths,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
        )

        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)

        return train_dataloader, val_dataloader


@DATASET_REGISTRY.register("DeepGlobe")
class DeepGlobeDataset(RoadSegDataset):
    # dataset class that deals with loading the data and making it available by index.

    def __getitem__(self, item):
        img_path, _ = self.items[item]

        image_tensor, mask_tensor = super().__getitem__(item)

        return image_tensor, mask_tensor